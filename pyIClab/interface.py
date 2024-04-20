#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Built-in import #####

import warnings
import inspect
from dataclasses import dataclass
from typing import Callable
from abc import abstractmethod, update_abstractmethods

##### External import ######

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import linregress
from scipy.optimize import curve_fit
from numpy import ndarray

##### Loacal import ######

import pyIClab.ions as ions_
from pyIClab.beadedbag import proj2hull
from pyIClab._baseclasses import (
    BaseModel, BaseConstructor, BaseIonChromatograph,
    )
from pyIClab.assemblies.columns import (
    GenericTubing, Column, PEEKTubing,
    )
from pyIClab.engines.models import (
    GenericDiscontinousSegmentedModel,
    DSM_CompleteEquilibriums,
    DSM_SimpleEquilibriums,
    builtin_no_retain_kmap,
    builtin_diffusion_method,
    builtin_fill_column_with_eluent,
    builtin_bypass_method,
    builtin_init_vessel_with_injection,
    )

# --------------------------------------------------------------------------------


# In[2]:


@dataclass(frozen=True)
class Constructor(BaseConstructor):
        
    host: GenericTubing
    ic: BaseIonChromatograph
    analyte: str
            
    def __post_init__(self):
            
        object.__setattr__(self, 'analyte', ions_.get_ioninfo(self.analyte).formula)
        host = self.host
        ic = self.ic
        assert isinstance(host, GenericTubing)
        assert host in ic.accessories
            
    @property
    def competing_ions(self):
        
        return self.ic.competing_ions
    
    
# --------------------------------------------------------------------------------


# In[3]:


class ModelManager:
    '''
    A factory class to dynamically decorate a ModelConstructor abstract class based on 
        Model you have just passed.
    >> Model should be a subclass of _baseclasses.BaseModel
    >> Model.__init__ should accept only kw-args except for `self`.
    '''
    # --------------------------------------------------------------------------------
    
    def __init__(self, Model: type):
        
        assert issubclass(Model, BaseModel)
        self._Model = Model
        self._init_kwargs = self._get_init_kwargs()
        
    def _get_init_kwargs(self)-> tuple:
        
        signature = inspect.signature(self._Model.__init__)
        assert all(
            v.kind is inspect.Parameter.KEYWORD_ONLY for v in list(
                signature.parameters.values())[1:]
            )
        return tuple(signature.parameters.keys())[1:]
        
    # --------------------------------------------------------------------------------
    
    @update_abstractmethods    
    def __call__(self, Constructor: type) -> type:
         
        def __call__(constr):
                
            return constr.Model(**constr.parameters)
        
        def get_params(constr):
            
            params = {
                kw: getattr(constr, f'set_{kw}')() for kw in self._init_kwargs
                }
            return {kw: v for kw, v in params.items() if v is not None}
        
        setattr(Constructor, 'parameters', property(fget=get_params))
        Constructor.__call__ = __call__
        Constructor.Model = self._Model
            
        for kw in self._init_kwargs:
            setattr(Constructor, f'set_{kw}', self._create_abstract_method(kw))
            
        Constructor.__name__ = Constructor.Model.__name__ + 'Constructor'
            
        return Constructor
           
    def _create_abstract_method(self, kw):
        
        @abstractmethod
        def _abstract_method(self, *args, **kwargs):
            pass
        
        _abstract_method.__name__ = f'set_{kw}'
        
        return _abstract_method


# In[4]:


@ModelManager(GenericDiscontinousSegmentedModel)
class AbstractDSMConstructor(Constructor):
    pass

@dataclass(frozen=True)
class DSMConstructor(AbstractDSMConstructor):
    '''
    Model constructor for `GenericDiscontinuousSegmentedModel`.
    
    Abstract methods still need implementation:
        set_N, set_distribute, set_distribute_params, set_kmap, 
        set_progress_bar, set_tqdm_trange_params, set_post_distribute,
        set_post_distribute_params
    Use class attribute __abstractmethods__ to check the methods
        needing implemention...
    
    Bound methods or properties you may override:
        __post_init__
        @property Model
    
    Note:
    - Assuming all the abstract methods are implemented, calling this constructor via
      __call__ will create a `GenericDiscontinuousSegmentedModel` instance.
    - Modify the @property model to apply this constructor to other DSM subclass models.
    '''
    
    def set_analyte(self) -> str:
        
        return self.analyte
    
    def set_x(self) -> str:
        
        return ions_.get_ioninfo(self.analyte).charge
    
    def set_competing_ions(self) -> tuple[str, ...]:
        
        return self.competing_ions
    
    def set_y(self) -> tuple[int, ...]:
        
        return tuple(ions_.get_ioninfo(ion).charge for ion in self.competing_ions)
    
    def set_Vm(self) -> float:
        
        return self.host.Vm.to('mL').magnitude
    
    def set_fr(self) -> float:
        
        if not self.host.flow_direction:
            return 0.0
        else:
            return abs(self.host.fr.to('mL/min').magnitude)
        
    def set_backflush(self) -> bool:
        
        return self.host.flow_direction == -1
    
    def set_phase_ratio(self) -> float:
        
        return self.host.phase_ratio
    
    def set_Q(self) -> float: # umol
        
        return self.host.Q.to('umol').magnitude
    
    def set_length(self) -> float:
        
        return self.host.length.to('cm').magnitude
    
    def set_ID(self) -> float:
        
        return self.host.ID.to('mm').magnitude
    
    def set_ignore_tiny_amount(self) -> float:
        
        return 1e-9
    
    def _set_init_vessel_and_params(self) -> tuple[Callable, dict]:
        
#         assert self.ic.current_time.magnitude == 0.0
        df = self.ic.injection_table
        if self.host.flow_direction:
            eluent = self.host.flow_head
            cE_fill = [eluent(0).get(ion) for ion in self.competing_ions]
            init_vessel_params = dict(cE_fill=cE_fill)
            return builtin_fill_column_with_eluent, init_vessel_params
        elif self.host not in df['accessory'].to_list():
            # No injection into the tubing
            # Default for models is to fill the tubing with eluent (c = 1e-4 mM)
            # No Need to specify init_vessel method
            return builtin_bypass_method, {}
        else:
            profile = df[df['accessory']==self.host].iloc[0]  # DataSeries
            if self.analyte not in profile.index or np.isnan(profile[self.analyte]):
                # Injection into the tubing however containing no model analyte
                # use cE_fill method
                cE_fill = [profile[ion] if ion in profile.index else (
                    1e-4) for ion in self.competing_ions]
                init_vessel_params = dict(cE_fill=cE_fill)
                return builtin_fill_column_with_eluent, init_vessel_params
            else:
                # Injection containing model analyte..
                cA = profile[self.analyte]
                cE = [profile[ion] if ion in profile.index else (
                    1e-4) for ion in self.competing_ions]
                return builtin_init_vessel_with_injection, dict(cA=cA, cE=cE)
                     
    def set_init_vessel(self):
        
        return self._set_init_vessel_and_params()[0]
        
    def set_init_vessel_params(self):
        
        return self._set_init_vessel_and_params()[1]
    
# --------------------------------------------------------------------------------


# In[5]:


@dataclass(frozen=True)
class DSMConstrutorForTubings(DSMConstructor):
    '''
    An integral model constructor for tubings (those without stationary phases).
        Only diffusion will take place.
    All abstract motheds are implemented.
    '''
    
    def __post_init__(self):
        
        super().__post_init__()
        host = self.host
        ic = self.ic
        # Assertion removed: 2024-03-21
        # assert isinstance(host, PEEKTubing)
        assert host in ic.accessories
        
    def set_N(self) -> float:
        
        Vm = self.host.Vm.to('mL').magnitude # mL
        
        return _builtin_get_plates_from_constant_dV(Vm=Vm)
    
    def set_kmap(self) -> Callable:
        
        return builtin_no_retain_kmap
    
    def set_distribute(self) -> Callable:
        # Use default: bypass
        
        return 
    
    def set_distribute_params(self) -> dict:
        # Use default: {}
        
        return
    
    def set_post_distribute(self) -> Callable:
        
        return builtin_diffusion_method
    
    def set_post_distribute_params(self) -> dict:
        
        A_diff = ions_.get_diff(self.analyte)
        E_diff = tuple(
            ions_.get_diff(ion) for ion in self.competing_ions
            )
        return dict(A_diff=A_diff, E_diff=E_diff)
    
    def set_progress_bar(self) -> bool:
        
        return True
    
    def set_tqdm_trange_params(self) -> bool:
        
        return dict(
            desc=f'Processing {self.analyte} on {self.host}',
            leave=False,
            )
    

# --------------------------------------------------------------------------------


# In[6]:


@dataclass(frozen=True)
class DSMConstrutorForColumns(DSMConstructor):
    '''
    Model constructor for columns.
    
    This class attempts to retreive the value of N from `ic.Ntable`. 
        If this attempt fails, N will be inferred from the column's stationay phase
        Hdata and the average eluent concentration over the time interval (0 -> tmax).
    Additionally, kmap method value will be inferred from the column's 
        stationary phase kdata based on the Linear Solvation Strength Model (LSSM).
    
    Bound methods or properties you may override:
        __post_init__
        @property Model
    
    Note:
    - The inferred N may not accurately represent the plate height when using a gradient eluent.
    - Ensure that the column was assembled after an eluent, and the column.sp.Hdata attribute
        has the integral datasets to obtain the plate height value under the given conditions.
    - The bulitin set_kmap func now only supports single-species eluents and carbonate
        buffers (beta). For those using unconventional eluents, you may need to override
        the `set_kmap` method.
    - Although all the abstract methods are implemented for this constructor, do not 
        use it to create model instances directly, because the bound model as its property
        is `GenericDiscontinuousSegmentedModel`, whose default `distribute` method is to
        do nothing...
    - Subclass constructor you may use: 
        DSM_SEConstrutor -- Model -> DSM_SimpleEquilibriums
        DSM_CEConstrutor -- Model -> DSM_CompleteEquilibriums
    '''
    
    # --------------------------------------------------------------------------------
    
    def __post_init__(self):
        
        super().__post_init__()
        host = self.host
        ic = self.ic
        assert isinstance(host, Column)
        assert host in ic.accessories
        
    # --------------------------------------------------------------------------------
        
    def set_N(self):
        
        N = self._retreive_N_from_designated_table()
        if not pd.isna(N):
            return N
        else:
            return self._find_N_from_Hdata_under_average_eluent_condition()
        
    def _retreive_N_from_designated_table(self) -> int | None:
        
        df = self.ic.Ntable.copy()
        assert self.analyte in df.columns
        return df.loc[
            df['module_instance']==self.host, self.analyte].to_list()[0]
        
    def _find_N_from_Hdata_under_average_eluent_condition(self) -> int:
        
        assert self.host.flow_direction
        eluent = self.host.flow_head
        c_mean = eluent.mean()
        
        return _builtin_get_plates_for_isocratic_eluents(
            Hdata=self.host.sp.Hdata,
            analyte=self.analyte,
            competing_ions=self.ic.competing_ions,
            concentrations=tuple(c_mean.get(ion, 1e-4) for ion in self.competing_ions),
            fr=self.set_fr(),
            length=self.set_length(),
            )
    
    # --------------------------------------------------------------------------------
    
    def set_kmap(self):
        
        return _builtin_get_kmap(
            kdata=self.host.sp.kdata,
            analyte=self.analyte,
            competing_ions=self.competing_ions)
    
    # --------------------------------------------------------------------------------
    
    def set_distribute(self): # Use default
        
        return
    
    def set_distribute_params(self) -> dict: # Use default
        
        return
    
    # --------------------------------------------------------------------------------
    
    def set_post_distribute(self) -> Callable:
        
        return builtin_diffusion_method
    
    def set_post_distribute_params(self) -> dict:
        
        A_diff = None # bypass diffusion upon analyte
        E_diff = tuple(
            ions_.get_diff(ion) for ion in self.competing_ions
            )
        return dict(A_diff=A_diff, E_diff=E_diff)
    
    # --------------------------------------------------------------------------------
    
    def set_progress_bar(self):
        
        return True
    
    def set_tqdm_trange_params(self):
        
        return dict(
            desc=f'Processing {self.analyte} on {self.host}',
            leave=False,
            )
        
# --------------------------------------------------------------------------------


# In[7]:


@dataclass(frozen=True)
class DSM_SEConstrutor(DSMConstrutorForColumns):
    '''
    A constructor for creating DSM_SimpleEquilibriums models.
        Inherits functionality from DSMConstrutorForColumns.

    This constructor is tailored specifically for DSM_SimpleEquilibriums models.
        When an instance of this constructor is called, it creates 
        a DSM_SimpleEquilibriums model instance.
        
    For detailed information about the DSM constructor, 
        refer to the documentation of DSMConstrutorForColumns.
    '''
    
    @property
    def Model(self):
        
        return DSM_SimpleEquilibriums
    
# --------------------------------------------------------------------------------

@dataclass(frozen=True)
class DSM_CEConstrutor(DSMConstrutorForColumns):
    __doc__ = DSM_SEConstrutor.__doc__.replace(
        'DSM_SimpleEquilibriums', 'DSM_CompleteEquilibriums')
    
    @property
    def Model(self):
        
        return DSM_CompleteEquilibriums


# In[8]:


# Functions to get plate numbers from stationary phase database...

def _builtin_get_plates_for_isocratic_eluents(
    *,
    Hdata: pd.DataFrame,
    analyte: str,
    competing_ions: tuple[str, ...],
    concentrations: tuple[float, ...],
    fr: float,
    length: float
    ):
    
    df = Hdata.get(competing_ions).copy()
    df = df[df['analyte']==analyte]
    
    if len(df) < 3:
        raise ValueError(f'Not enough data points for {analyte}.')
        
    if len(competing_ions) == 1 and len(df[df['fr']==fr]) >= 3:
        return _get_plates_from_1Dinterpolator(
            df=df,
            ion=competing_ions[0],
            c=concentrations[0],
            fr=fr,
            length=length,
            )
    elif len(df[df['fr']==fr]) >= 3:
        return _get_plates_from_griddata_constant_fr(
            df=df,
            competing_ions=competing_ions,
            concentrations=concentrations,
            fr=fr,
            length=length,
            )

    else:
        return _get_plates_from_griddata(
            df=df,
            competing_ions=competing_ions,
            concentrations=concentrations,
            fr=fr,
            length=length,
            )
    
def _get_plates_from_1Dinterpolator(
    *,
    df: pd.DataFrame,
    ion: str,
    c: float,
    fr: float,
    length: float,
    ):
    
    df = df[df['fr']==fr]
    df = df.sort_values(by=ion, ignore_index=True)
    cdata = df.loc[:, ion].to_numpy()
    Hdata = df.loc[:, 'H'].to_numpy()
    Hi = np.interp(c, cdata, Hdata)
    Ni = int(np.round(length / Hi * 10, -1))
    
    return Ni

def _get_H_from_griddata(
    point: ndarray | list,
    xydata: ndarray | list,
    Hdata: ndarray | list,
    ) -> float:
    
    point = np.array(point, dtype=np.float64)
    xydata = np.array(xydata, dtype=np.float64)
    Hdata = np.array(Hdata, dtype=np.float64)
    point = proj2hull(point, xydata)
    
    return np.squeeze(LinearNDInterpolator(xydata, Hdata)(point))
    
def _get_plates_from_griddata_constant_fr(
    df: pd.DataFrame,
    competing_ions: tuple[str, ...],
    concentrations: tuple[float, ...],
    fr: float,
    length: float,
    ) -> int:
    
    df = df[df['fr']==fr]
    xy_keys = list(competing_ions)
    xydata = df[xy_keys].to_numpy()
    Hdata = df['H'].to_numpy()
    point = np.array(concentrations)
    H = _get_H_from_griddata(point, xydata, Hdata)
    
    return int(np.round(10 * length / H, -1))
    
def _get_plates_from_griddata(
    df: pd.DataFrame,
    competing_ions: tuple[str, ...],
    concentrations: tuple[float, ...],
    fr: float,
    length: float,
    ):
    
    xy_keys = ['fr', *competing_ions]
    xydata = df[xy_keys].to_numpy()
    Hdata = df['H'].to_numpy()
    point = np.array([fr, *concentrations])
    H = _get_H_from_griddata(point, xydata, Hdata)
    
    return int(np.round(10 * length / H, -1))
      
def _builtin_get_plates_from_constant_dV(
    *,
    Vm: float,
    dV: float =None, 
    ):
    '''
    Vm: float, in mL.
    dV: float, optional, defaults to 0.1, in uL.
    '''
    
    dV = 0.1 if dV is None else dV
    Vm *= 1000
    
    return max(10, int(np.round(Vm / dV, -1)))


# In[9]:


# --------------------------------------------------------------------------------
# Functions to get func kmap from stationary phase database...

def _builtin_get_kmap(
    *,
    kdata: pd.DataFrame,
    analyte: str,
    competing_ions: tuple[str, ...],
    ) -> Callable[[ndarray,], ndarray]:
    
    df = kdata.get(competing_ions).copy()
    df = df[df['analyte']==analyte]
    
    if len(competing_ions) == 1 and len(df) >= 3:
        return _builtin_get_kmap_single_eluent(
            df=df, analyte=analyte, ion=competing_ions[0])
    elif set(competing_ions) == {'CO3[-2]', 'HCO3[-1]'}:
        return _builtin_get_kmap_carbonates(df=df)
    else:
        raise NotImplementedError
    
def _builtin_get_kmap_single_eluent(
    *,
    df: pd.DataFrame,
    analyte: str,
    ion: str,
    ) -> Callable[[ndarray,], ndarray]:
    '''
    logk = a + b*logc
    => Returns: f = 10**(a + b*log10(c))
    '''
    
    logc = np.log10(df[ion])
    logk = np.log10(df['k'])
    
    b, a, r_value, p_value, std_err = linregress(
        x=logc, y=logk)
    
    if r_value**2 <= 0.9:
        warnings.warn(
            f'''Poor fitting of logk-logc obtained for {analyte} '''
            f'''with R-square: {r_value**2:4f}.''')
    
    # np.squeeze is essential, because cE is in shape(M, N).
    # For single-species eluents M = 1.
    return lambda cE: a + b*np.log10(np.squeeze(cE))

def _builtin_get_kmap_carbonates(*, df: pd.DataFrame) -> Callable[[ndarray,], ndarray]:
    '''
    reference:
    Analytical Chemistry, Vol. 74, No. 23, December 1, 2002
    '''
    # Update 2024/04/09
    def fit_func(logc, A, B, C):
        logCO3, logHCO3 = logc
        return A + B*logCO3 + C*logHCO3

    carb, bicarb = 'CO3[-2]', 'HCO3[-1]'
    logCO3 = np.log10(df[carb])
    logHCO3 = np.log10(df[bicarb])
    logc = np.array([logCO3, logHCO3])
    logk = np.log10(df['k'])
    initial_guess = [1, -0.5, -0.05]
    
    (A, B, C), pcov = curve_fit(fit_func, logc, logk, p0=initial_guess)
    
    def kmap(cE: ndarray):
        
        cE = np.array(cE, dtype=np.float64)
        cE[cE<1e-4] = 1e-4
        
        return A + np.sum(np.log10(cE)*np.array([[B], [C]]), axis=0)
    
    return kmap
        
# --------------------------------------------------------------------------------
    


# In[10]:


__all__ = [
    'ModelManager',
    'DSMConstructor',
    'DSMConstrutorForTubings',
    'DSMConstrutorForColumns',
    'DSM_SEConstrutor',
    'DSM_CEConstrutor',
    ]

