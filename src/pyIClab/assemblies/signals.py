#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[2]:


##### Built-in import #####

import warnings
from typing import Literal, Callable
from types import MethodType

##### External import #####

import pandas as pd
import numpy as np
import phreeqpython
from numpy import ndarray
from pyEQL import Solution
from deprecated import deprecated
from pint import Unit

##### Local import #####

import pyIClab.ions as ions_
from pyIClab.units import ParseQuantities
from pyIClab.errors import (
    ProfileError, SuppressorOverVoltageError,
    )
from pyIClab.utils.beadedbag import is_notebook
from pyIClab.assemblies.generics import GenericAccessory

##### Conditional import #####

if is_notebook():
    import tqdm.notebook as tqdm
else:
    import tqdm

# --------------------------------------------------------------------------------


# In[3]:


def _convert_phreeqpython_formulas(
    ions,
    *,
    _reserved=('CO2', 'H2O', 'O2',),
    ) -> list:
        
    pp_formulas = []
    for ion in ions:
        if ion in _reserved:
            pp_formulas.append(ion)
            continue
        ion_obj = ions_.IonLibrary.from_formula(ion)
        if ion_obj.pp_compat == 'unsupported':
            raise ProfileError(ion)
        else:
            pp_formulas.append(ion_obj.formula_pp)
    return pp_formulas

def _extract_phreeqpython_compat_ions(
    ions,
    *,
    _reserved=('CO2', 'H2O', 'O2',),
    ):
    
    compat = []
    for ion in ions:
        if ion in _reserved:
            continue
        ion_obj = ions_.IonLibrary.from_formula(ion)
        if ion_obj.pp_compat == 'compatible':
            compat.append(ion)
            
    return compat
    
# --------------------------------------------------------------------------------


# In[4]:


class Detector(GenericAccessory):
    
    __identifier__ = 'detector'
    _unique = True
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /, name: str,
        *,
        freq: str | float | int =None,
        ) -> None:
        '''
        Parameters:
        ----------
        name: str
            The name of the detector.
        freq: str | float | int, optional.
            Sampling frequenzy in Hz. Defaults to '10 Hz'.
        '''
        super().__init__(name=name)
        freq = '10 Hz' if freq is None else freq
        self._freq = ParseQuantities.parse_string_as(freq, 'Hz') * Unit('Hz')
    
    # --------------------------------------------------------------------------------
    
    def __repr__(self):
        
        return f'<Detector "{self.name}">'
    
    # --------------------------------------------------------------------------------
    
    @property
    def freq(self):
        
        return self._freq
    
    # --------------------------------------------------------------------------------
        
    def get_signals(self, /,
        tmax: str | float | int =None,
        *,
        signal_type: Literal['concentration', 'conductivity'] =None,
        algorithm: Literal['ideal', 'realistic'] =None,
        ) -> pd.DataFrame:
        '''
        Retrieves the signal data up to a maximum time specified by `tmax`, 
        for either concentration or conductivity.
        Returns a DataFrame containing the time series data.
        
        Parameters:
        ----------
        tmax: str | float | int, optional
            The maximum time up to which the signal data is to be retrieved in minute.
            defaults to the current time of detector.
        signal_type: str, optional
            The type of signal data to retrieve. 
            Can be 'concentration', 'conductivity', defaults to 'concentration'.
            Collecting concentration signals is almost instant, 
            whereas gathering conductivity signals demands extensive computational effort.
        algorithm: str, optional.
            The algorithm to use for calculating conductivity signals. Only applicable
            for conductivity signals. 
            Can be 'ideal', 'realistic', defaults to 'realistic'.
            (1) Algorithm 'realistic'
                Based on the phreeqpython engine. 
                This algorithm equilibrates the input eluent automatically,
                considering dissociation and oxidation-reduction equilibriums. 
                It offers fast computation due to the underlying C library. 
                However, it has limitations:
                - Compromised accuracy to handle ions with significant reduction or oxidation
                activity - They decompose during equilibrating.
                - Limited ion support. 
                Use `IonLibrary.from_formula(ion).pp_compat` to check compatibility.
            (2) Algorithm 'ideal'
                Utilizes the "ideal" engine from the pyEQL package,
                ignoring the dissociation equilibriums of ions in aqueous solutions. 
                This algorithm is considerably slower compared to the 'realistic' one
                but avoids the limitations associated with ion decomposition 
                and limited ion support.
        Note:
        -----------
        When employing the 'realistic' engine, equilibrium computations are performed
            in isolation, ensuring that components in the effluents remain unchanged.
        '''
        tmax = 0.9999 * self.current_time if tmax is None else tmax
        tmax = ParseQuantities.as_min(tmax)
        assert tmax >= 0
        
        signal_type = 'concentration' if signal_type is None else signal_type
        algorithm = 'realistic' if algorithm is None else algorithm
        signal_type = signal_type.strip().lower()
        algorithm = algorithm.strip().lower()
        
        match signal_type:
            case 'concentration':
                return self._get_signals_concentration(tmax)
            case 'conductivity':
                return self._get_signals_cd(tmax, algorithm)
            case _:
                raise NotImplementedError(
                    f'''{signal_type} is not supported. Options for signal_type: '''
                    '''"concentration", "conductivity".''')
        
    def _get_signals_concentration(self, tmax: float) -> pd.DataFrame:
        
        t = np.linspace(0, tmax, round(self.freq.magnitude * tmax * 60))
            
        return pd.DataFrame(
            data=dict(time=t, signal=np.sum(list(self(t).values()), axis=0)),
            dtype=np.float64
            )
    
    def _get_cd_ideal(self, df: pd.DataFrame) -> ndarray:
        
        desc = f'Calculating eluent conductivity on {self}...'
        cd = []
        for i, data in tqdm.tqdm(df.iterrows(), 
            desc=desc, total=len(df), leave=False):
            data = data[data>0].apply(lambda c: str(c) + ' mmol/L')
            solution_profile = data.to_dict()
            s = Solution(solution_profile, engine='ideal')
            cd.append(s.conductivity.magnitude * 1e4) # S/m -> uS/cm
        return np.array(cd)
    
    def _get_cd_realistic(self, df: pd.DataFrame) -> ndarray:
        
        desc = f'Calculating eluent conductivity on {self}...'
        cd = []
        pp = phreeqpython.PhreeqPython()
        for i, data in tqdm.tqdm(df.iterrows(),
            desc=desc, total=len(df), leave=False):
            data = data[data>0]
            solution_profile = data.to_dict()
            s = pp.add_solution_simple(solution_profile, temperature=30)
            cd.append(s.sc)
            
            if not i % 1000: # garbage collection
                pp.remove_solutions(pp.get_solution_list())
                
        return np.array(cd)
        
    def _get_signals_cd(self, tmax: float, algorithm: str) -> pd.DataFrame:
        
        t = np.linspace(0, tmax, round(self.freq.magnitude * tmax * 60))
        df = pd.DataFrame(data=self(t))
        match algorithm:
            case 'ideal':
                cd = self._get_cd_ideal(df)
            case 'realistic':
                # in case that realistic algorithm does not work
                # roll back to ideal algorithm
                try:
                    pp_formulas = _convert_phreeqpython_formulas(df.columns)
                except ProfileError as err:
                    warnings.warn(
                        f'''Ion {err} is not compatible with the engine. '''
                        f'''Algorithm "ideal" is used instead.''')
                    cd = self._get_cd_ideal(df)
                else:
                    compat = _extract_phreeqpython_compat_ions(df.columns)
                    if compat:
                        warnings.warn(
                            '''Compromised accuracy in the conductivity profiles for the '''
                            f'''following analytes: {", ".join(compat)}.'''
                            )
                    df.columns = pp_formulas
                    cd = self._get_cd_realistic(df)
            case _:
                raise NotImplementedError(
                    f'''{algorithm} is not supported. Options for algorithm: '''
                    '''"ideal", "realistic".''')
            
        return pd.DataFrame(data=dict(time=t, signal=cd), dtype=np.float64)
            
# --------------------------------------------------------------------------------


# In[5]:


# --------------------------------------------------------------------------------
class Suppressor(GenericAccessory):
    '''
    Represents a suppressor accessory for IC, designed to reduce the background
        conductivity of the eluent, enhancing detection sensitivity for analyte ions. 
    '''
    
    __identifier__ = 'suppressor'
    _unique = True
    
    # --------------------------------------------------------------------------------

    def __init__(self, /,
        name: str,
        kind: Literal['anion', 'cation'] =None,
        *,
        _volume: str | float | int =None,
        ):
        '''
        Parameters:
        ----------
        name : str
            The name of the suppressor, uniquely identifying it within a IC system.
        kind : str, optional
            The type of suppressor, either 'anion' or 'cation'. 
            Defaults to 'anion'.
        _volume : str | float | int, optional
            An internal parameter representing the unit volume that the suppressor
            can handle for one loop. 
            Note: this is not the genuine void volume of the suppressor. It is
            automatically set and should not be modified by users. 
            Defaults to '0.25 uL'.
        '''
        
        super().__init__(name=name)
        
        kind = 'anion' if kind is None else kind.strip().lower()
        assert kind in ('anion', 'cation')
        _volume = '0.25 uL' if _volume is None else _volume
        self._volume = ParseQuantities.parse_string_as(_volume, 'uL') * Unit('uL')
        self._kind = kind
        
    # --------------------------------------------------------------------------------
    
    def __repr__(self):
        
        return f'<Suppressor "{self.name}">'
    
    # --------------------------------------------------------------------------------
    
    @property
    def kind(self) -> str:
        
        return self._kind
    
    # --------------------------------------------------------------------------------
    
    @property
    def flow_direction(self) -> int:
        
        d = super().flow_direction
        if not d:
            raise SuppressorOverVoltageError(self)

        return d
    
    # --------------------------------------------------------------------------------
    
    def flush(self) -> None:
        
        raise NotImplementedError
    
# --------------------------------------------------------------------------------


# In[6]:


class QuickSuppressor(Suppressor):
    '''
    A simplified implementation of the Suppressor class designed for quick suppression of
        with hydroxide and hydronium eluents. 
    It sets the concentrations of eluent competing ions to 1e-4 mM, offering a fast 
        approximation of suppression effects. However, its performance may 
        be distorted when handling carbonate buffers.
    Use .__supported_eluents__ to check the suported eluent types.

    '''
    
    __supported_eluents__ = ('hydroxides', 'hydroniumns')
    _reserved_components = ('H[+1]', 'OH[-1]', 'CO3[-2]', 'HCO3[-1]', 'CO2',)
    
    # --------------------------------------------------------------------------------
    
    def _suppress(self, input_datasets: dict[str, Callable]) -> dict[str, Callable]:
        
        datasets = input_datasets.copy()
        g = lambda t: np.zeros_like(t)
        for component in tuple(datasets.keys()):
            try:
                ioninfo = ions_.get_ioninfo(component)
            except ValueError:
                # escape possible neutral compounds.
                continue
            else:
                charge = ioninfo.charge
                if (self.kind == 'anion' and charge > 0) or (
                    self.kind == 'cation' and charge < 0):
                    datasets.pop(component)
                else:
                    if component in self._reserved_components:
                        datasets[component] = lambda t: np.ones_like(t) * 1e-4
                    # -----------
                    f = datasets[component]
                    g = lambda t, *, f0=g, f1=f, charge=charge: f0(t) + abs(charge)*f1(t)
                    
        if self.kind == 'anion':
            datasets['H[+1]'] = g
        else:
            datasets['OH[-1]'] = g
            
        return datasets
    
    # --------------------------------------------------------------------------------
    
    def flush(self):
        
        backup = self.prev.output
        self.prev._output = self._suppress(backup)
        super(Suppressor, self).flush()
        self.prev._output = backup

    


# In[7]:


class PhreeqcSuppressor(Suppressor):
    '''
    A more complex implementation of the Suppressor class.
    The suppressor initially replaces counterions in the eluent with H+ (for AEC) 
        and OH- (for CEC), then uses the PHREEQC engine to equilibrate the eluent, 
        offering more accurate suppression results.
    Provides more realistic suppression outcomes compared to QuickSuppressor
        but requires significantly more computational effort.
    Can be annoyingly time-consuming when handling long-period separations (fixed)...
    Use .__supported_eluents__ to check the suported eluent types.
    '''
    
    __supported_eluents__ = ('hydroxides', 'hydroniums', 'carbonates',)
    _reserved_components = ('H[+1]', 'OH[-1]', 'CO3[-2]', 'HCO3[-1]', 'CO2',)
    
    # --------------------------------------------------------------------------------
    
    def _suppress(self, input_datasets: dict[str, Callable]) -> dict[str, Callable]:
        '''
        Convert all the cations into H+ as for AEC, anions into OH- as for CEC...
        '''
        
        datasets = input_datasets.copy()
        g = lambda t: np.zeros_like(t)
        for component in tuple(datasets.keys()): 
            # tuple(.keys()), use .copy().keys(), otherwise RuntimeError
            try:
                ioninfo = ions_.get_ioninfo(component)
            except ValueError:
                # in case of presence of CO2
                continue
            else:
                charge = ioninfo.charge
                if (self.kind == 'anion' and charge > 0) or (
                    self.kind == 'cation' and charge < 0):
                    # discard counterions in-place
                    f = datasets.pop(component)
                    # accumulate charge, charge=charge is essential
                    g = lambda t, *, f0=g, f1=f, charge=charge: f0(t) + abs(charge)*f1(t)
        
        if self.kind == 'anion':
            datasets['H[+1]'] = g
        else:
            datasets['OH[-1]'] = g
            
        return datasets

    # --------------------------------------------------------------------------------
    
    def _make_pseudo_eluent(self,
        suppressed_datasets: dict[str, Callable]) -> dict[str, Callable]:
        '''
        Phreeqc engine could decompose unstable ions... Replace all the analyte
            ions with Cl- or K+ (they are considerably stable)
            before equilibrating the eluent.
        '''
    
        pseudo_datasets = {}
        g = lambda t: np.zeros_like(t)
        for component, f in suppressed_datasets.items():
            if component in self._reserved_components:
                pseudo_datasets[component] = f
            else:
                charge = ions_.get_ioninfo(component).charge
                # double-check if the eluent has been suppressed
                assert (self.kind == 'anion' and charge < 0) or (
                    self.kind == 'cation' and charge > 0)
                g = lambda t, *, f0=g, f1=f, charge=charge: f0(t) + abs(charge)*f1(t)
                
        if self.kind == 'anion':
            pseudo_datasets['Cl[-1]'] = g
        else:
            pseudo_datasets['K[+1]'] = g
            
        return pseudo_datasets
    
    @deprecated(reason='see docstring')
    def _pseudo_equilibrate(self, df: pd.DataFrame) -> dict[str, list[float]]:
        '''
        A more comprehensive method to do pseudo equilibriumns despite its
            relatively poor performance.
        Takes ~24s for 20,000 loops on my Old Mac.
        Still need to find out why it's getting slower and slower while looping (Fixed).
        
        -----------
        #Upadte#
        2024/04/13: clear the cache of pp engine on every 1000 iterations...
        '''
        
        desc = f'Suppressing eluent on {self}...'
        pseudo_eq_profile = {}
        pp = phreeqpython.PhreeqPython()
        tp = template = pd.DataFrame(index=range(len(df)), dtype=np.float64)
        for i, data in tqdm.tqdm(df.iterrows(),
            desc=desc, total=len(df), leave=False):
            data = data[data>0]
            solution_profile = data.to_dict()
            s = pp.add_solution_simple(solution_profile, temperature=30)
            for component, c_M in s.species.items():
                if component in self._reserved_components:
                    # in case of CO2
                    formula = component
                else:
                    try:
                        formula = ions_.get_ioninfo(component).formula
                    except ValueError:
                        # discard irrelavant neutral components
                        continue
                tp.loc[i, formula] = c_M * 1000
                
            if not i % 1000: # garbage collection
                pp.remove_solutions(pp.get_solution_list())
                
        pp.remove_solutions(pp.get_solution_list())       
        tp.fillna(0.0, inplace=True)
     
        return tp.to_dict(orient='list')

    def _pseudo_equilibrate(self, df: pd.DataFrame) -> dict[str, list[float]]:
        '''
        Don't feel like using it but it's faster...
        Takes ~18s for 20,000 loops on my Old Mac...
        It's also running slower and slower (issue fixed)...
        
        -----------
        #Upadte#
        2024/04/13: clear the cache of pp engine on every 1000 iteration...
        '''
        
        desc = f'Suppressing eluent on {self}...'
        pp = phreeqpython.PhreeqPython()
        component_table = {
            'K[+1]': 'K+',
            'Cl[-1]': 'Cl-',
            'H[+1]': 'H+',
            'OH[-1]': 'OH-',
            'HCO3[-1]': 'HCO3-',
            'CO3[-2]': 'CO3-2',
            'CO2': 'CO2',
            }
        
        possible_components = list(component_table.keys())
        
        # Compatibility statement for InsufficientPhreeqcSuppressor
        if not isinstance(self, InsufficientPhreeqcSuppressor):
            if self.kind == 'anion':
                possible_components.remove('K[+1]')
            else:
                possible_components.remove('Cl[-1]')
        
        # Compatibility statement for ContaminatedPhreeqcSuppressor
        if not isinstance(self, ContaminatedPhreeqcSuppressor):
            if not {'HCO3[-1]', 'CO3[-2]', 'CO2'} & set(df.columns):
                possible_components.remove('HCO3[-1]')
                possible_components.remove('CO3[-2]')
                possible_components.remove('CO2')
            
        pseudo_eq_profile = {component: [] for component in possible_components}
        for i, data in tqdm.tqdm(df.iterrows(),
            desc=desc, total=len(df), leave=False):
            solution_profile = data.to_dict()
            s = pp.add_solution_simple(solution_profile, temperature=30)
            for component, c in pseudo_eq_profile.items():
                c.append(s.species.get(component_table[component], 0.0) * 1000)
            
            if not i % 1000: # garbage collection
                pp.remove_solutions(pp.get_solution_list())
                
        pp.remove_solutions(pp.get_solution_list())
        
        return pseudo_eq_profile
    
    # --------------------------------------------------------------------------------
    
    def flush(self) -> None:
        
        t0 = self.current_time.magnitude
        t1 = self.prev.current_time.magnitude
        assert t1 > t0
        
        suppressed_datasets = self._suppress(self.prev.output)
        pseudo_datasets = self._make_pseudo_eluent(suppressed_datasets)
        
        n = round(
            ((self.fr * ((t1-t0) * Unit('min'))) / self._volume).to('dimensionless').magnitude)
        n = max(n, 100)
        t = np.linspace(t0, t1, n)
        # A concentration dataframe for performing equilibriums...
        df = pd.DataFrame(
            data={component: f(t) for component, f in pseudo_datasets.items()})
        try:
            pp_formulas = _convert_phreeqpython_formulas(df.columns)
        except ProfileError as err:
            # Technically, this won't happen...
            # Something must go wrong if this error is raised...
            msg = f'Component {err} is not compatible with the suppressor engine.' 
            raise ProfileError(error_msg) from err
        else:
            # Convert columns to that phreeqpython recognizes...
            df.columns = pp_formulas
            pseudo_eq_profile = self._pseudo_equilibrate(df)
            # discard the placeholder ions.
            pseudo_eq_profile.pop('K[+1]', None)
            pseudo_eq_profile.pop('Cl[-1]', None)
            # interpolate t-c data into funcs
            pseudo_eq_datasets = {
                component: lambda t, *, x=t, y=c: np.interp(t, x, y) for (
                    component, c) in pseudo_eq_profile.items()}
            
        # update the equalized funcs to suppressed datasets.
        suppressed_datasets.update(pseudo_eq_datasets)
        
        # merge the ion t-c funcs
        for component, f1 in suppressed_datasets.items():
            f0 = self.output.setdefault(component, lambda t: np.zeros_like(t))
            self.output[component] = self._update_func_time_series((t0, t1), f0, f1)
        
        # Don't forget this...
        self._cur_time = t1 * Unit('min')
        
# --------------------------------------------------------------------------------


# In[8]:


@deprecated(reason='It performs like a broken one...')
class InsufficientPhreeqcSuppressor(PhreeqcSuppressor):
    '''
    This subclass of PhreeqcSuppressor allows for the simulation of the void dip and
        baseline drifts obseraved in IC chromatograms by incomplete suppression 
        of eluents, a feature not available in the standard PhreeqcSuppressor. 
    It's marked as deprecated because it might distort analyte peaks, 
        performing like a malfunctioning suppressor.
    Use .__supported_eluents__ to check the suported eluent types.
    '''
    # --------------------------------------------------------------------------------
    
    def __init__(self, /,
        name: str,
        kind: Literal['anion', 'cation'] =None,
        *,
        _volume: str | float | int =None,
        _efficiency: float =0.99999,
        ):
        '''
        Parameters:
        ----------
        name : str
            The name of the suppressor, uniquely identifying it within a IC system.
        kind : str, optional
            The type of suppressor, either 'anion' or 'cation'. 
            Defaults to 'anion'.
        _volume : str | float | int, optional
            An internal parameter representing the unit volume that the suppressor
            can handle for one loop. 
            Note: this is not the genuine void volume of the suppressor. It is
            automatically set and should not be modified by users. 
            Defaults to '0.25 uL'.
        _efficiency : float, optional
            An internal parameter indicating the efficiency of suppression, 
            where 1.0 is 100% efficiency. 
            Note: This parameter is used for back-end testing and 
            not supposed to be altered by users. 
            Defaults to 0.99999.
        '''
        
        super().__init__(name=name, kind=kind, _volume=_volume)
        self._efficiency = float(_efficiency)
        assert 0 <= self._efficiency <= 1
        
    # --------------------------------------------------------------------------------
        
    def _suppress(self, input_datasets: dict[str, Callable]) -> dict[str, Callable]:
        
        ef = self._efficiency
        r = 1 - self._efficiency
        datasets = input_datasets.copy()
        g = lambda t: np.zeros_like(t)
        for component, f in datasets.copy().items(): 
            # tuple(.keys()), or use .copy().keys(), otherwise RuntimeError
            try:
                ioninfo = ions_.get_ioninfo(component)
            except ValueError:
                # in case of presence of CO2
                continue
            else:
                charge = ioninfo.charge
                if (self.kind == 'anion' and charge > 0) or (
                    self.kind == 'cation' and charge < 0):
                    datasets[component] = (
                        lambda t, *, f0=f, r=r: r * f0(t))
                    g = lambda t, *, f0=g, f1=f, ef=ef, charge=charge: (
                        f0(t) + abs(charge)*ef*f1(t))
        
        if self.kind == 'anion':
            h = datasets.get('H[+1]', lambda t: np.zeros_like(t))
            datasets['H[+1]'] = lambda t, *, f0=h, f1=g: f0(t) + f1(t)
        else:
            h = datasets.get('OH[-1]', lambda t: np.zeros_like(t))
            datasets['OH[-1]'] = lambda t, *, f0=h, f1=g: f0(t) + f1(t)
            
        return datasets
    
    # --------------------------------------------------------------------------------
    
    def _make_pseudo_eluent(self,
        suppressed_datasets: dict[str, Callable]) -> dict[str, Callable]:
    
        pseudo_datasets = {}
        ga = lambda t: np.zeros_like(t)
        gc = lambda t: np.zeros_like(t)
        for component, f in suppressed_datasets.items():
            if component in self._reserved_components:
                pseudo_datasets[component] = f
            else:
                charge = ions_.get_ioninfo(component).charge
                if charge > 0:
                    gc = lambda t, *, f0=gc, f1=f, charge=charge: (
                        f0(t) + charge * f1(t))
                else: # charge < 0
                    ga = lambda t, *, f0=ga, f1=f, charge=charge: (
                        f0(t) - charge * f1(t))
        
        pseudo_datasets['Cl[-1]'] = ga
        pseudo_datasets['K[+1]'] = gc
            
        return pseudo_datasets


# In[9]:


class ContaminatedPhreeqcSuppressor(PhreeqcSuppressor):
    '''
    A subclass of PhreeqcSuppressor for simulating CO2 contamination effects in ion
        chromatography. It exposes eluents to a designated CO2 level, adding carbonate 
        impurities. This approach provides realistic simulations of water dips, elevated 
        conductivity backgrounds, and baseline drifts.
    Note: The suppressor introduces undesired carbonate species into the eluents. 
        Caution is advised when installing this suppressor in valve-switching systems. 
        Unexpected errors may occur if the effluent of the suppressor is used as the 
        feed eluent for other accessories (such as columns) processed with IC models. 
        These scenarios have not been fully tested, posing potential risks of bugs.
    Use .__supported_eluents__ to check the suported eluent types.
    '''
    
    __supported_eluents__ = ('hydroxides', 'carbonates')
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /,
        name: str,
        kind: Literal['anion', 'cation'] =None,
        *,
        _volume: str | float | int =None,
        _CO2_level: float =1e-4,
        ):
        '''
        Parameters:
        ----------
        name : str
            The name of the suppressor, uniquely identifying it within a IC system.
        kind : str, optional
            Type of suppressor, 'anion' or 'cation'. Defaults to 'anion'.
        _volume : str | float | int, optional
            Internal parameter for the suppressor's unit volume that can be handled 
            per loop. Not the actual void volume. Defaults to '0.25 uL'.
        _CO2_level : float, optional
            Designated CO2 contamination level for eluent equilibration. For backend
            testing, not reflecting real IC operation CO2 levels. Defaults to 1e-4.
        '''
        
        super().__init__(
            name=name, kind=kind, _volume=_volume)
        self._CO2_level = float(_CO2_level)
        
    # --------------------------------------------------------------------------------
    
    def _make_pseudo_eluent(self, *args, **kwargs):
        
        pseudo_datasets = super()._make_pseudo_eluent(*args, **kwargs)
        f0 = pseudo_datasets.get('CO2', lambda t: np.zeros_like(t))
        f1 = pseudo_datasets.get('OH[-1]', lambda t: np.zeros_like(t))
        pseudo_datasets['CO2'] = lambda t: f0(t) + f1(t)*self._CO2_level
        
        return pseudo_datasets
        


# In[10]:


class ContaminatedPhreeqcSuppressorBeta(ContaminatedPhreeqcSuppressor):
    
    def _make_pseudo_eluent(self, *args, **kwargs):
        
        pseudo_datasets = super(
            ContaminatedPhreeqcSuppressor, self)._make_pseudo_eluent(*args, **kwargs)
        f0 = pseudo_datasets.get('CO2', lambda t: np.zeros_like(t))
        f1 = pseudo_datasets.get('OH[-1]', lambda t: np.zeros_like(t))
        pseudo_datasets['CO2'] = lambda t: f0(t) + f1(t)**0.5 * self._CO2_level
        
        return pseudo_datasets
    


# In[11]:


__all__ = [
    'Detector',
    'QuickSuppressor',
    'PhreeqcSuppressor',
    'InsufficientPhreeqcSuppressor',
    'ContaminatedPhreeqcSuppressor',
    'ContaminatedPhreeqcSuppressorBeta',
    ]

