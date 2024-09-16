#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[2]:


##### Built-in import #####

from typing import Callable, Self, Literal
from copy import deepcopy

##### External import #####

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray
from pint import Unit

##### Local import #####

import pyIClab.ions as ions_
from pyIClab.units import ParseQuantities
from pyIClab.assemblies.generics import GenericAccessory, Side
from pyIClab.errors import (
    ConfigurationError, CounterPumpingError, DuplicatedIonError,
    ProfileError,
    )  
from pyIClab.utils.beadedbag import mpl_custom_rcconfig, is_notebook

# --------------------------------------------------------------------------------


# In[3]:


class Eluent(GenericAccessory):
    '''
    Represents the isocratic / gradient profile for an IC eluent.
    Alias: EluentGenerator
    '''
    __identifier__ = 'eluent'
    _unique = True
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /,
        name: str,
        profile: dict[str, tuple | str | float | int],
        *,
        fr: str | float | int =None,
        ):
        '''
        Parameters:
        -----------
        profile: dict, a dictionary containing information about the eluent profile.
            The keys of `profile` refer to the competing ions in the eluent,
            represented as strings (e.g., 'OH-', 'hydronium', 'CO3-2', 'HCO3-', ...).
            Note: the charges must be specified at the end if the ion is present as an
            unrecognizable formula or alias. e.g.: 'citrate[-3]'. 
            Use IonLibrary to see the ions registered.
            
            The values represent the eluent profile for each competing ion and can be 
            one of the following:
            - A string, float, or int: Represents an isocratic concentration in 
                default units ('mM').
            - A tuple: Specifies a time-concentration profile for the competing ion as a list 
                of (time, concentration) tuples.
        name: str.
            The name of the eluent.
        fr: str | float | int, optional, the flow rate for the eluent. 
            Defaults to '1 mL/min'.
            
        Examples:
        ----------
        - Creating an isocratic 10 mM KOH eluent, with a flow rate of 1.2 mL/min:
            elnt = Eluent(name='KOH', profile={'OH-': 10}, fr='1.2 mL/min') 
            elnt = Eluent(name='KOH', profile={'OH-': '10 mM'}, fr=1.2) or
            elnt = Eluent(name='KOH', profile={'hydroxide': ((0.0, 10), (10, 10))}, fr='1.2')
            elnt = Eluent.HydroxideIsocratic('10.0 mM', fr='1.2mL/minute')
        - Creating a gradient KOH eluent, starting at 10 mM, holding for 10 min,
            and increasing to 20 mM within 10 min with a linear ramp:
            g = Eluent(name='KOH', profile={
                'OH-':((0.0, 10), (10, 10), (20, 20)),
                })
        - Creating an isocratic carbonates buffer, with 2.4 mM carbonate and 
            0.8 mM bicarbonate:
            elnt = Eluent(
                name='Carbobate-Buffer', profile={'CO3-2': '2.4 mmol/L', 'HCO3-': '.8 mmol/L'})
            elnt = Eluent.Carbonates(carbonate='2.4 mM', bicarbonate='.8 mM')
        - Use '+' operator to merge two eluent profiles:
            g1 = Eluent.HydroxideIsocratic(2.5)
            g2 = Eluent(
                name='EG',
                profile={
                'OH-': (('6min', '5mM'), ('16min', '28mM'), ('20min', '65mM'),
                    ('25min', '65mM'), ('25.1min', '5mM'))})
            g3 = g1 + g2
            -----
            elnt1 = Eluent(name='K2CO3', profile={'CO3-2': 2.4})
            elnt2 = Eluent(name='KHCO3', profile={'HCO3-': '.8mM'})
            elnt3 = elnt1 + elnt2

        Note:
        -----------
        If the eluent profile exhibits an electrical imbalance, K+ and Cl- 
            will be employed to restore electrical neutrality.
        '''
        
        super().__init__(name=name)
        del self.left
        
        fr = '1.0 mL/min' if fr is None else fr
        self.fr = ParseQuantities.parse_string_as(fr, 'mL/min') * Unit('mL/min')
        
        ions_.check_duplicated_ions_from_sequence(list(profile.keys()))
        self._parse_gradient(profile)
        self._neutralize_profile()
        self._drop_profile_with_zero_concentration()
        self._gradient = self.isgradient()
    
    # --------------------------------------------------------------------------------
    
    def __repr__(self):
        
        t = self._tmax
        if self.gradient:
            label = f'<Eluent "{self.name}" Gradient('
            for ion, f in self.output.items():
                label += f'{ion}: '
                label += '{:.1f}'.format(f(0))
                label += ' ~ '
                label += '{:.1f}'.format(f(t))
                label += ' mM, '
        else:
            label = f'<Eluent "{self.name}" Isocratic('
            for ion, f in self.output.items():
                label += f'{ion}: {f(0):.1f} mM, '
                
        label = label[:-2]
        label += f') in {t:.0f} min>'
        
        return label
    
    # --------------------------------------------------------------------------------
    
    def __add__(self, other):
        
        if type(other) != type(self) :
            raise TypeError(
                f'''Cannot join {type(other)} to {type(self)}.''')
            
        if self.fr != other.fr:
            raise ConfigurationError(
                f'''Cannot merge eluent profiles with different flow rates.'''
                )
        
        if self.occupied_slots:
            raise ConfigurationError(
                f'''Disassemble {self} before merging.''')

        new = deepcopy(self)
        
        new._tmax = max(self._tmax, other._tmax)
        
        output_other = deepcopy(other.output)
        for ion, f1 in new.output.items():
            f2 = output_other.pop(ion, None)
            if f2 is not None:
                new.output[ion] = (
                    lambda t, *, f1=f1, f2=f2: f1(t) + f2(t))
        
        new.output.update(output_other)
        new._gradient = new.isgradient()
        
        return new
    
    # --------------------------------------------------------------------------------
    
    @property
    def gradient(self) -> bool:
        
        return self._gradient
    
    # --------------------------------------------------------------------------------
    
    @property
    def ions(self) -> tuple:
        
        return tuple(self.output.keys())
    
    def get_competing_ions(self,
        ion_exchange_kind: Literal['anion', 'cation']) -> ndarray:
        
        match ion_exchange_kind.lower():
            
            case 'anion':
                return tuple(
                    ion for ion in self.ions if ions_.get_ioninfo(ion).charge < 0)
            case 'cation':
                return tuple(
                    ion for ion in self.ions if ions_.get_ioninfo(ion).charge > 0)
            case _:
                raise ValueError(
                    f'''expect "anion", "cation", got {ion_exchange_kind}.''')
        
    def get_counterions(self,
        ion_exchange_kind: Literal['anion', 'cation']) -> ndarray:
        
        return tuple(
            set(self.ions) - set(self.get_competing_ions(ion_exchange_kind)))
    
    # --------------------------------------------------------------------------------
    
    def assemble(self,
        other: GenericAccessory,
        sides: tuple[Side, Side] | list[Side] =None,
        ) -> None:
        
        if type(self) == type(other):
            raise CounterPumpingError(self)
        
        return super().assemble(other, sides)
    
    # --------------------------------------------------------------------------------
    
    def mean(self, /, *args) -> dict[str, float]:
        '''
        Calculates the average concentration of the ions in profile in a given
            time interval.
        Usage:
        - mean() -> 0 - tmax
        - mean(end) -> 0 - end minutes
        - mean(start, end) -> start - end minutes
        '''
        
        if len(args) > 2:
            raise TypeError(
            f'''{type(self)}.mean expected at most 2 argments, got {len(args)}.'''
                )
            
        if not args:
            start, end = 0.0, self._tmax
        elif len(args) == 1:
            start, (end,) = 0.0, args
        else:
            start, end = args
            
        t = np.arange(start, end, 0.001)
        
        return {ion: np.mean(c) for ion, c in self(t).items()}
    
    # --------------------------------------------------------------------------------
    
    def isgradient(self, /, *args) -> bool:
        '''
        Usage:
        - isgradient() -> 0 - tmax
        - isgradient(end) -> 0 - end minutes
        - isgradient(start, end) -> start - end minutes
        
        Determine if the eluent has a gradient within a specified time interval.
        '''
        if len(args) > 2:
            raise TypeError(
            f'''{type(self)}.isgradient expected at most 2 argments, got {len(args)}.'''
                )
            
        if not args:
            start, end = 0.0, self._tmax
        elif len(args) == 1:
            start, (end,) = 0.0, args
        else:
            start, end = args
        
        t = np.arange(start, end, .001)
        c = np.array(list(self(t).values()))
        cmax = np.max(c, axis=1)
        cmin = np.min(c, axis=1)
        cmean = np.mean(c, axis=1)
        mask = (cmax-cmin) / cmean >= 0.01
        
        return np.any(mask)
        
    # --------------------------------------------------------------------------------
    
    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        
        sns.set()
        plt.rcParams.update(mpl_custom_rcconfig)
        fig, ax = plt.subplots()
        tmax = np.round(1.1 * self._tmax)
        x = np.linspace(0, tmax, 1000)
        ymax = 0.0
        
        for ion, f in self.output.items():
            charge = ions_.get_ioninfo(ion).charge
            y = f(x)
            ymax = max(ymax, *y)
            if np.any(y>1e-5):
                ax.plot(x, y, 
                    linewidth=0.75, 
                    linestyle='--' if charge < 0 else '-.',
                    label=ions_.get_formula_latex(ion),
                    )
        
        ax.set_xlabel('time, min', fontsize=10, fontweight='bold')
        ax.set_ylabel('[E], mM', fontsize=10, fontweight='bold')
        ax.legend()
        ax.set(xlim=(0, tmax), ylim=(0, 1.1*ymax))
        
        if not is_notebook():
            plt.show()
            
        return fig, ax
    
    # --------------------------------------------------------------------------------
    
    def _parse_gradient(self, profile):
        
        output = {}
        tmax = 0.0
        
        for ion, p in profile.items():
            
            ioninfo = ions_.get_ioninfo(ion)
            formula = ioninfo.formula
                
            if isinstance(p, str | float | int):
                cE = ParseQuantities.as_mM(p)
                f = lambda t, *, cE=cE: np.interp(t, [0.0, 10.0], [cE, cE])
                output[formula] = f
                tmax = max(tmax, 10.0)
            
            else:
                times, concentrations = zip(*p)
                times = np.array([ParseQuantities.as_min(t) for t in times])
                concentrations = np.array(
                    [ParseQuantities.as_mM(c) for c in concentrations])
                f = lambda t, *, x=times, y=concentrations: (
                    np.interp(t, x, y))
                output[formula] = f
                tmax = max(*times, tmax)
        
        self.output.update(output)
        self._tmax = tmax
        
    # --------------------------------------------------------------------------------
    
    def _neutralize_profile(self) -> None:
        
        def _cum_charge(g, ion, f) -> Callable:
            
            ioninfo = ions_.get_ioninfo(ion)
            charge = ioninfo.charge
            
            g_cum = lambda t, g=g, f=f: g(t) + f(t)*charge
            
            return g_cum
        
        # -----------
        
        def _get_compensation(compensation_ion: str, g):
            
            charge = ions_.get_ioninfo(compensation_ion).charge
            
            def _fc(t, g=g, charge=charge):
                
                c0 = np.zeros_like(t)
                charge_cum = g(t)
                mask = charge_cum>0 if charge < 0 else charge_cum<0
                c1 = np.where(mask, -charge_cum/charge, c0)
                
                return c1
            
            return _fc
        # -----------
        
        g = lambda t: np.zeros_like(t)
        for ion, f in self.output.items():
            g = _cum_charge(g, ion, f)
            
        fK = _get_compensation('K[+1]', g)
        fCl = _get_compensation('Cl[-1]', g)
        
        merge = lambda f1, f2: (lambda t, *, f1=f1, f2=f2: f1(t) + f2(t))
        fK0 = self.output.get('K[+1]', lambda t: np.zeros_like(t))
        self.output['K[+1]'] = merge(fK, fK0)
        fCl0 = self.output.get('Cl[-1]', lambda t: np.zeros_like(t))
        self.output['Cl[-1]'] = merge(fCl, fCl0)
        
    # --------------------------------------------------------------------------------
    
    def _drop_profile_with_zero_concentration(self) -> None:
        
        t = np.arange(0, self._tmax, .001)
        
        for ion, f in self.output.copy().items():
            if all(f(t)<1e-5):
                self.output.pop(ion)
                
    # --------------------------------------------------------------------------------
    
    def flush(self, *args, **kwargs):
        
        raise NotImplementedError
    
    # --------------------------------------------------------------------------------
    
    @classmethod
    def HydroxideIsocratic(cls, c, name=None, **kwargs):
        '''
        Create an instance with an isocratic hydroxide (OH-) profile.
        '''
        kwargs.setdefault('name', name or 'KOH')
    
        return cls(profile={'OH-': c}, **kwargs)
    
    @classmethod
    def HydroniumIsocratic(cls, c, name=None, **kwargs):
        '''
        Create an instance with an isocratic hydronium (H+) profile.
        '''
        kwargs.setdefault('name', name or 'HCl')
        
        return cls(profile={'H+': c}, **kwargs)
    
    @classmethod
    def Carbonates(cls, *, carbonate, bicarbonate, name: str=None, **kwargs):
        '''
        Create an instance with iscoratic carbonate (CO3-2) and 
        bicarbonate (HCO3-) buffer profiles.
        '''
        kwargs.setdefault('name', name or 'CarbonatesBuffer')
        
        return cls(
            profile=dict(carbonate=carbonate, bicarbonate=bicarbonate),
            **kwargs)
    
    @classmethod
    def DIWater(cls, name: str=None, **kwargs):
        '''
        Create a DI water flow.
        '''
        kwargs.setdefault('name', name or 'DIWater')
        
        return cls(profile={'OH-': 1e-4, 'H+': 1e-4}, **kwargs)
    
    @classmethod
    def from_funcs(cls,
        name: str,
        profile: dict[str, Callable[float, float]], 
        tmax: float,
        **kwargs):
        '''
        Accepts a gradient profile with ion as the keys and 
            time(in min)-concentration(in mM) functions as the values.
        A parameter `tmax` in float must be specified for the profile.
        Example:
        >> eluent = Eluent.from_funcs(
            name='MyCustomizedGradient',
            profile=dict(hydroxide=lambda t: 0.5*t**2 + 10),
            tmax=10.0)
        '''
        tmax = ParseQuantities.as_min(tmax)
        assert tmax > 0
        t = np.linspace(0, tmax, min(max(int(60*tmax), 60), 3600), endpoint=True)
        g_prf = {}
        for ion, f in profile.items():
            c = [f(_) for _ in t]
            if min(c) < 0:
                raise ProfileError(
                    f'''Negative concentration in profile for {ion}.''')
            g_prf[ion] = zip(t, c)
            
        return cls(name=name, profile=g_prf, **kwargs)
        
        
    # --------------------------------------------------------------------------------   


# In[4]:


EluentGenerator = Eluent

# ---------------------------------------------------------------


# In[5]:


__all__ = ['Eluent', 'EluentGenerator',]

