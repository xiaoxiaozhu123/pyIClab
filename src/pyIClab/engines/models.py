#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[10]:


##### Built-in import #####

import warnings
from dataclasses import dataclass, fields, field
from functools import cached_property
from typing import Self, Callable
from types import MethodType

##### External import #####

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray
from deprecated import deprecated

##### Local import #####F

import pyIClab.ions as ions_
from pyIClab.units import ParseQuantities
from pyIClab._baseclasses import BaseModel 
from pyIClab.utils.beadedbag import is_notebook, mpl_custom_rcconfig
from pyIClab.utils.optimize import find_K_LSSM
from pyIClab.assemblies.eluents import Eluent
from pyIClab.utils.methods import (
    kmap_for_GDSM_testing, no_retain_kmap, bypass_method,
    analyte_diffusion_method, eluent_diffusion_method, diffusion_method,
    total_mix_eluent, total_mix_analyte, fill_column_with_eluent,
    init_vessel_with_injection, simple_equilibrium_distribution_method,
    complete_equilibrium_distribution_method, bypass_hmap,
    )

##### Conditional import #####

if is_notebook():
    import tqdm.notebook as tqdm
else:
    import tqdm

##### Compatibility ##### DO NOT REMOVE IT!!

_builtin_kmap_for_GDSM_testing = kmap_for_GDSM_testing
builtin_no_retain_kmap = no_retain_kmap
builtin_bypass_method = bypass_method
builtin_analyte_diffusion_method = analyte_diffusion_method
builtin_eluent_diffusion_method = eluent_diffusion_method
builtin_diffusion_method = diffusion_method
_total_mix = total_mix_eluent
_total_mix_analyte = total_mix_analyte
builtin_fill_column_with_eluent = fill_column_with_eluent
builtin_init_vessel_with_injection = init_vessel_with_injection
_builtin_simple_equilibrium_distribution_method = simple_equilibrium_distribution_method
_builtin_complete_equilibrium_distribution_method = complete_equilibrium_distribution_method

# --------------------------------------------------------------------------------


# In[3]:


@dataclass(kw_only=True, repr=False)
class SegmentedColumnContainer:

    '''
    a SegmentedColumnContainer (SCC) object with N discrete segments and
        one analyte eluted by M different competing ions in the column MP
        as model.vessel...
    
    Parameters:
    ----------
    host: GenericDiscontinousSegmentedModel, the host model instance.
    cAm: ndarray, shape(N,), concentration of the analyte in each segment MP, mM.
    nAs: ndarray, shape(N,), amount of analyte in each segment SP, nmol.   
    cEm: ndarray, shape(M, N), concentration of competing ions in each 
        segment MP, mM.
    nEs: ndarray, shape(M, N), amount of competing ions in each 
        segment SP, nmol.
    '''
        
    host: BaseModel
    cAm: ndarray
    nAs: ndarray
    cEm: ndarray
    nEs: ndarray
        
    # --------------------------------------------------------------------------------
    # Constant properties
    @cached_property
    def unit_table(self) -> dict:
        
        return {
            'cAm': 'mM',
            'cAs': 'mM',
            'cEm': 'mM',
            'cEs': 'mM',
            'dQ': 'nmol',
            'dVm': 'uL',
            'dVs': 'uL',
            'dV': 'uL',
            'phase_ratio': 'dimensionless',
            'logk': 'dimensionless',
            'k': 'dimensionless',
            'nAm': 'nmol',
            'nAs': 'nmol',
            'nEm': 'nmol',
            'nEs': 'nmol',
            'fixed_nA': 'nmol',
            'fixed_cA': 'mM',
            'fixed_nE': 'nmol',
            'fixed_nA': 'mM',
            'β': 'dimensionless',
            'Φ': 'dimensionless',
            }
    
    @cached_property
    def phase_ratio(self) -> ndarray:
        
        return np.ones(self.host.N) * self.host.phase_ratio
    
    @cached_property
    def Φ(self) -> ndarray:
        
        return self.phase_ratio
    
    @cached_property
    def β(self) -> ndarray:
        
        return 1 / self.phase_ratio
    
    @cached_property
    def dQ(self) -> ndarray: # nmol
        
        return np.ones(self.host.N) * self.host.Q / self.host.N * 1000
    
    @cached_property
    def dVm(self) -> ndarray: # uL
        
        return np.ones(self.host.N) * self.host.dVm
    
    @cached_property
    def dVs(self) -> ndarray: # uL
        
        return self.phase_ratio * self.dVm
    
    @cached_property
    def dV(self) -> ndarray: # uL
        
        return self.dVm + self.dVs

    # --------------------------------------------------------------------------------
    # Mutable properties
    @property
    @deprecated(reason='issue fixed on 2024/04/10')
    def k(self):
        
        return 10**self.logk
    
    @property
    def logk(self):
        
        return self.host.kmap(self.cEm)
    
    @property
    def cAs(self): # mM
        
        return self.nAs / self.dVm / self.phase_ratio
    
    @property
    def cEs(self): # mM
        
        return self.nEs / self.dVm / self.phase_ratio
    
    @property
    def fixed_nA(self): # nmol
        
        return self.nAs + self.nAm
    
    @property
    def fixed_nE(self): # nmol
        
        return self.nEm + self.nEs
    
    @property
    def fixed_cA(self): # mM
        
        return self.fixed_nA / self.dV
    
    @property
    def fixed_cE(self): # mM
        
        return self.fixed_nE / self.dV
    
    @property
    def nAm(self): # nmol
        
        return self.cAm * self.dVm
    
    @property
    def nEm(self): # nmol
        
        return self.cEm * self.dVm
    
    # --------------------------------------------------------------------------------
    
    def init(self, **kwargs) -> None:
        
        self.host.init_vessel(**kwargs)
    
    # --------------------------------------------------------------------------------
    
    def distribute(self, **kwargs) -> None:
        '''
        Implements the distribution of an analyte across each column segment 
            in a segmented time flow. The distribution is modified in-place 
            along the spatial axis of the column segments.
            also see .post_distribute
        '''
             
        self.host.distribute(**kwargs)
        
    def post_distribute(self, **kwargs) -> None:
        
        self.host.post_distribute(**kwargs)
    
    # --------------------------------------------------------------------------------
    
    @deprecated
    def inherit(self, new_model: BaseModel) -> None:
        '''
        Used as an init_vessel method when a model inherted from an old one.
        Deprecated due to no need to inherit from a model.
        '''
        N0 = self.host.N
        N1 = new_model.N
        vessel = new_model.vessel
        if N0 == N1:
            vessel.cAm = self.cAm.copy()
            vessel.nAs = self.nAs.copy()
            vessel.cEm = self.cEm.copy()
            vessel.nEs = self.nEs.copy()
            return
        
        def _convert_array(arr, N0, N1) -> ndarray:
            
            x0 = np.arange(N0)
            x1 = np.linspace(0, N0, N1)
            return np.interp(x1, x0, arr, left=arr[0], right=arr[-1])
        
        vessel.cAm = _convert_array(self.cAm, N0, N1)
        vessel.nAs = _convert_array(self.nAs, N0, N1) * N0 / N1 # essential
        vessel.cEm = np.array([_convert_array(c, N0, N1) for c in self.cEm])
        vessel.nEs = np.array([_convert_array(n, N0, N1)  * N0 / N1 for n in self.nEs])
 
    # --------------------------------------------------------------------------------
  


# In[11]:


@dataclass(frozen=False, kw_only=True)
class _EluentFlow:
    '''
    Used to store model input/output data...
    '''
    cA: ndarray
    cE: ndarray
    
# --------------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, repr=False)
class GenericDiscontinousSegmentedModel(BaseModel):
    '''
    A generic framework for modeling of IC based on
        discontinuous plate model.
    The column is assumed homogenous.
    '''

    # --------------------------------------------------------------------------------
    
    N: int 
    analyte: str
    x: int | float
    competing_ions: tuple[str, ...]
    y: tuple[int | float, ...]
    Vm: float
    fr: float
    backflush: bool
    phase_ratio: float
    Q: float
    length: float
    ID: float
    ignore_tiny_amount: float =1e-20
    kmap: Callable[[ndarray,], ndarray] =no_retain_kmap
    hmap: Callable[[float, ndarray,], ndarray] =bypass_hmap
    distribute: Callable[[BaseModel, ...], None] =bypass_method
    post_distribute: Callable[[BaseModel, ...], None] =bypass_method
    init_vessel: Callable[[BaseModel, ...], None] =bypass_method
    distribute_params: dict =field(default_factory=dict)
    post_distribute_params: dict =field(default_factory=dict)
    init_vessel_params: dict =field(default_factory=dict)
    verbose: bool =True
    tqdm_trange_params: dict =field(default_factory=dict)
    
    def __post_init__(self):
        
        object.__setattr__(self, 'analyte', ions_.get_ioninfo(self.analyte).formula)
        object.__setattr__(self, 'competing_ions', tuple(
            ions_.get_ioninfo(ion).formula for ion in self.competing_ions))
        object.__setattr__(self, 'ignore_tiny_amount', max(self.ignore_tiny_amount, 1e-20))
        object.__setattr__(self, 'distribute', MethodType(self.distribute, self))
        object.__setattr__(self, 'post_distribute', MethodType(self.post_distribute, self))
        object.__setattr__(self, 'init_vessel', MethodType(self.init_vessel, self))
        object.__setattr__(self, 
            'input', _EluentFlow(cA=np.array([]), cE=np.empty((self.M, 0))))
        object.__setattr__(self, 
            'output', _EluentFlow(cA=np.array([]), cE=np.empty((self.M, 0))))
        
        dQ = self.Q / self.N * 1000
        vessel = SegmentedColumnContainer(
            host=self,
            cAm=np.zeros(self.N),
            nAs=np.zeros(self.N),
            cEm=np.ones((self.M, self.N)) * 1e-4,
            nEs=np.ones((self.M, self.N)) * dQ / sum([abs(y) for y in self.y]),
            )
        object.__setattr__(self, 'vessel', vessel)
        
        for attr in self._mutables:
            
            def _set_attr(self, value, *, _attr=attr):
                self._drop()
                return object.__setattr__(self, _attr, value)
            
            object.__setattr__(self, f'_set_{attr}', MethodType(_set_attr, self))
            getattr(self, f'_set_{attr}').__func__.__name__ = f'_set_{attr}'
        
        self.clear()
        self._inspect_validity()

    
    # --------------------------------------------------------------------------------
    
    def __hash__(self):
        
        return hash((
            type(self),
            self.N, self.analyte, self.x, self.competing_ions,
            self.y, self.Vm, self.phase_ratio, self.Q,
            self.length, self.ID, self.ignore_tiny_amount,
            ))
    
    # --------------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        
        return (
            f'''<{type(self).__name__} of {self.analyte} '''
            f'''under {"/ ".join(self.competing_ions)} eluetion>''')
    
    # --------------------------------------------------------------------------------
    
    def _drop(self):
        
        for attr in self._drop_when_updated:
            try:
                del self.__dict__[attr]
            except KeyError:
                pass
            
    # --------------------------------------------------------------------------------
        
    def _inspect_validity(self):
        
        assert len(self.competing_ions) == len(self.y)
        assert self.input.cA.shape[-1] == self.input.cE.shape[-1]
        assert self.output.cA.shape[-1] == self.output.cE.shape[-1]

    # --------------------------------------------------------------------------------
    
    @cached_property
    def unit_table(self) -> dict:
        
        return {
            'Vm': 'mL',
            'fr': 'mL/min',
            'phase_ratio': 'dimensionless',
            'Q': 'umol',
            'β': 'dimensionless',
            'Φ': 'dimensionless',
            'length': 'cm',
            'ID': 'mm',
            'dL': 'cm',
            'S': 'mm**2',
            }
    
    @cached_property
    def _drop_when_updated(self) -> tuple:
        
        return ('dt',)
    
    @cached_property
    def _mutables(self) -> tuple:
        
        return ('fr', 'backflush',)
    
    @cached_property
    def dVm(self) -> float: #uL
    
        return self.Vm / self.N * 1000
    
    @cached_property
    def dt(self) -> float: # min NOTE: mutable with self.fr!
  
        return self.Vm / self.fr / self.N
    
    @cached_property
    def dL(self) -> float: # cm
        
        return self.length / self.N
    
    @cached_property
    def S(self) -> float: # cm**2
        
        return (self.ID/2)**2 * np.pi / 100
    
    @cached_property
    def M(self) -> int: 
        
        return len(self.competing_ions)
    
    @property
    def integral_A(self) -> float:
        '''
        the total amount of analyte in nmol, including the portion flushed out and 
            that still retained.
        '''
        amount_out = np.sum(self.output.cA) * self.dVm
        amount_retained = np.sum(self.vessel.fixed_nA)
        
        return amount_retained + amount_out
    
    @property
    def integral_E(self) -> ndarray:
        '''
        the total amounts of competing ions in nmol, including the portion flushed out and 
            that still retained.
        '''
        amount_out = np.sum(self.output.cE, axis=1) * self.dVm
        amount_retained = np.sum(self.vessel.fixed_nE, axis=1)
        
        return amount_retained + amount_out
    
    # --------------------------------------------------------------------------------
    
    def clear(self) -> None:
        
        self._drop()
        self.input.cA = np.array([])
        self.input.cE = np.empty((self.M, 0))
        self.output.cA = np.array([])
        self.output.cE = np.empty((self.M, 0))
    
    # --------------------------------------------------------------------------------
    
    def input_data(self, cAm: list | ndarray, cEm: list | ndarray) -> None:
        
        self.input.cA = cAm
        self.input.cE = cEm
        self._inspect_validity()
        
    # --------------------------------------------------------------------------------
    
    def plot(self) -> tuple[plt.Figure, plt.Axes]:
        
        sns.set()
        plt.rcParams.update(mpl_custom_rcconfig)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        t = np.arange(len(self.output.cA)) * self.dt
        analyte_latex = ions_.get_formula_latex(self.analyte)
        amount = np.sum(self.output.cA) * self.dVm
        amount_label = '{:.5f}'.format(amount) if amount > 0.01 else '{:.5e}'.format(amount)
        
        ax1.plot(t, self.output.cA, 
            label=f'{analyte_latex}: {amount_label} nmol',
            color='blue',
            )
        for i, ion in enumerate(self.competing_ions):
            ax2.plot(t, self.output.cE[i],
            label=ions_.get_formula_latex(ion),
            linewidth=0.5,
            linestyle=':',
            )
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        ax1.set_xlabel('Time, min')
        ax1.set_ylabel('[A], mM')
        ax2.set_ylabel('[E], mM')
            
        if not is_notebook():
            plt.show()
            
        return fig, ax1
        
    # --------------------------------------------------------------------------------
    
    def standby(self):
        '''
        Calls the bound method 'init_vessel'. The initial state of the model
            will be updated, meaning the model is ready to run.
        '''
        
        self.vessel.init(**self.init_vessel_params)
    
    # --------------------------------------------------------------------------------
    
    def activate(self) -> None:
        
        for field in fields(self.input):
            data = getattr(self.input, field.name)
            if not data.size:
                raise ValueError('Empty input data detected.')
        if self.verbose:
            myrange = lambda n, *, kw=self.tqdm_trange_params: tqdm.trange(n, **kw)
        else:
            myrange = range

        vessel = self.vessel
        # Preallocate space
        if not self.backflush:
            cA = np.concatenate([np.flip(self.input.cA, axis=-1), vessel.cAm], axis=-1)
            cE = np.concatenate([np.flip(self.input.cE, axis=-1), vessel.cEm], axis=-1)
        else:
            cA = np.concatenate([vessel.cAm, self.input.cA], axis=-1)
            cE = np.concatenate([vessel.cEm, self.input.cE], axis=-1)
        
        T = len(self.input.cA) # total iterration steps
        N = self.N
        skip_distribution = False
        
        # input array of analyte amount
        nA = self.input.cA * self.dVm
        (non_ignoring_indices,) = np.where(nA>=self.ignore_tiny_amount)
        
        # When step == init_skipping, initializing skipping inspection
        if not non_ignoring_indices.size:
            init_skipping = 0
        else:
            init_skipping = non_ignoring_indices[-1] + 1
        
        # Put flow direction inspection outside the loops
        if not self.backflush:
            for i in myrange(T):
                if not skip_distribution and i >= init_skipping and (
                    np.all(vessel.fixed_nA<self.ignore_tiny_amount)):
                    skip_distribution = True
                
                if not skip_distribution:
                    vessel.cAm = cA[T-i-1:T-i-1+N]
                    vessel.cEm = cE[:, T-i-1:T-i-1+N]
                    vessel.distribute(**self.distribute_params)
                    
                vessel.post_distribute(**self.post_distribute_params)
                
        elif self.backflush:
            for i in myrange(T):
                if not skip_distribution and i >= init_skipping and (
                    np.all(vessel.fixed_nA<self.ignore_tiny_amount)):
                    skip_distribution = True
                
                if not skip_distribution:
                    vessel.cAm = cA[i+1:i+1+N]
                    vessel.cEm = cE[:, i+1:i+1+N]
                    vessel.distribute(**self.distribute_params)
                    
                vessel.post_distribute(**self.post_distribute_params)
                
        self.input.cA = np.array([])
        self.input.cE = np.empty((self.M, 0))
        
        if not self.backflush:
            self.output.cA = cA[N:][::-1]
            self.output.cE = cE[:, N:][:, ::-1]
        else:
            self.output.cA = cA[:T].copy()
            self.output.cE = cE[:, :T].copy()
        
        vessel.cAm = vessel.cAm.copy()
        vessel.cEm = vessel.cEm.copy()
            
    # --------------------------------------------------------------------------------
    
    @deprecated
    def inherit(self, new_host):
        '''
        Deprecated due to no need to inherit from a model as
            models..
        '''
        
        return self.vessel.inherit(new_host)
            
# --------------------------------------------------------------------------------

GenericDiscontinousSegmentedModel.__init__.__doc__ = '''
    Parameters:
    -----------
    N: int, number of column segments.
    analyte: str.
    x: int | float, charge of the analyte.
    competing_ions: tuple[str, ...], shape(M,).
    y: tuple[int | float, ...], charges of the competing ions, shape(M,).
    Vm: float, total dead volumn of the column, in mL.
    fr: float, eluent (MP) flow rate, in mL/min.
    backflush: bool. True if eluent flows from right to left.
    phase_ratio: float, Φ = Vs/Vm of the column.
    Q: float, ion-exchange capacity of the column, in umol.
    length: float, length of the column, in cm.
    ID: float, diameter of the column tube, in mm.
    ignore_tiny_amount: float, if the amount of analyte or competing ions
        in a segemented unit below `ignore_tiny_amount`, distribution of 
        the analyte / competing_ions between MP/SP will be ignored.
        Defaults to 1e-20 (in nmol).
        Note: mannually setting `ignore_tiny_amount` below 1e-20 will be
        ignored to avoid potential risks of numerical overflow...
    kmap: Callable[[ndarray,], ndarray].
        a mappale function modeling the logarithm of k to the base 10 
        of the analyte based on the concentrations of competing ions. 
        It accepts only one positional arguments: cE, which are 
        the concentrations of competing ions in shape(M, T).
        If not given, the default kmap always returns -inf regardless of 
            eluent concentrations.
    hmap: Callable[[float, ndarray,], ndarray].
        A function detailing the plate height H (in mm) as a function of fr and cE.
        Used for incomplete equibibrating models, Can be bypassed in complete 
        equilibrating models as N is already designated.
    distribute: Callable[[BaseModel, ...], None].
        A function to compute the distribution of an analyte after a segmented 
        time flow. It should accept the a model object and kw-arguments (if needed)
        and modify the model.vessel in-place, where vessel is the container to store the 
        concentration data of the analyte and competing ions along the column.
        If Not given, the default method does not distribute the analyte at all.
    distribute_params: dict, optional.
        Kwargs forward to distribute method.
    post_distribute: Callable[[BaseModel, ...], None]
        A function to deal with the process (e.g., diffusion) with the model after 
        distribution.
    post_distribute_params: dict, optional.
        Kwargs forward to post_distribute_params method.
    init_vessel: Callable[[BaseModel, ...], None].
        A function to equilibrate the initial state of the model vessel.
        Same to `distribute` method, accepts a model as the first positional argument
        and modify its vessel in-place.
        See `SegmentedColumnContainer` for vessel defination.
    init_vessel_params: dict, optional.
        Kwargs forward to init_vessel method.
    verbose: bool, optional.
        If True, show a progress bar while running distribution itteration, which
        is implemented via tqdm.trange method.
    tqdm_trange_params: dict, optional.
        forward to tqdm.trange method.
    '''
# --------------------------------------------------------------------------------


# In[5]:


@dataclass(kw_only=True, frozen=True, repr=False)
class DSM_SimpleEquilibriums(GenericDiscontinousSegmentedModel):
    '''
    Implements a simplified version of the discontinuous plates model 
        as proposed by M. Novic et al / J. Chromatogr. A 922 (2001) 1–11. 
    This model adaptation focuses on chromatographic processes where 
        the redistribution of eluent competing ions (competing_ions) between 
        the column's Stationary Phase (SP) and Mobile Phase (MP) is ignored. 
        Consequently, the capacity factor (k) of the analyte 
        is determined solely by the input concentrations of the competing ions.
    Alias: DSM_SE, DSM_SEQ
        
    '''
    distribute: Callable[[BaseModel, ...], None] =None
    
    def __post_init__(self):
        
        if self.distribute is not None:
            warnings.warn(
                '''\nKw-argument `distribute` is ignored because '''
                '''the distribution method of this model is predetermined.''')
            
        object.__setattr__(self,
            'distribute', simple_equilibrium_distribution_method)
        super().__post_init__() # Important: self.distribute -> Bound method in __post_init__
        
    # --------------------------------------------------------------------------------
    
    def __hash__(self):
        
        return super().__hash__()
        
# --------------------------------------------------------------------------------

DSM_SimpleEquilibriums.__init__.__doc__ = '''
    Use help(GenericDiscontinousSegmentedModel) to see the parameters.
    Note:
    `distribute` method has been designated for this model. Do not pass it.
    '''
# --------------------------------------------------------------------------------

DSM_SE = DSM_SEQ = DSM_SimpleEquilibriums

# --------------------------------------------------------------------------------


# In[6]:


@dataclass(kw_only=True, frozen=True, repr=False)
class DSM_CompleteEquilibriums(GenericDiscontinousSegmentedModel):
    '''
    A "harder" version of DSM_SimpleEquilibriums model. This model takes
        into consideration the redistribution of both analyte and eluent 
        ion (competing ion) between column MP and SP.
        It is 30-50% as slower for double-charged ones, and only applies well for 
        single/double-charged analytes and single-spiecies single-charged eluents.
        When passing a non-integer charge, this model will degrade and become
        unbearably slow... (need to solve non-linear equations)
    This model is able to model column overloading.
    Alias: DSM_CE DSM_CEQ
    
    reference: M. Novic et al / J. Chromatogr. A 922 (2001) 1–11.
    '''
    distribute: Callable[[BaseModel, ...], None] =None
        
    def __post_init__(self):
        
        if self.distribute is not None:
            warnings.warn(
                '''\nKw-argument `distribute` is ignored because '''
                '''the distribution method of this model is predetermined.''')
            
        object.__setattr__(self,
            'distribute', complete_equilibrium_distribution_method)
        
        super().__post_init__() # Important: self.distribute -> Bound method in __post_init__
        
        if self.M != 1 or abs(self.y[0]) != 1:
            raise ValueError(
                'This model only supports single-spiecies single-charged eluents.')
            
        r = self.x / self.y[0]
        if not (0.99 <= r <= 1.01 or 1.99 <= r <= 2.01):
            warnings.warn(
                '''\nThis model supports single/double-charged analytes.\n'''
                '''Processing an analyte & an elueting ion with an effective charge ratio '''
                f'''x/y = {r:.3f} on this model is not stable. '''
                '''You may encounter low computational efficiency and numerical issues.'''
                )
            
    # --------------------------------------------------------------------------------
    
    def __hash__(self):
        
        return super().__hash__()
    
    # --------------------------------------------------------------------------------
            
    @cached_property
    def K(self) -> float:
        
        a = self.kmap([1])
        
        return find_K_LSSM(
            a, self.phase_ratio, self.Q, self.Vm, abs(self.x), abs(self.y[0]))
    
# --------------------------------------------------------------------------------
    
DSM_CompleteEquilibriums.__init__.__doc__ = '''
    Use help(GenericDiscontinousSegmentedModel) to see the parameters.
    Note:
    `distribute` method has been designated for this model. Do not pass it.
    '''      
# --------------------------------------------------------------------------------

DSM_CE = DSM_CEQ = DSM_CompleteEquilibriums

# --------------------------------------------------------------------------------


# In[7]:


@dataclass(kw_only=True, frozen=True, repr=False)
class DSM_IncompleteSimpleEquilibriums(DSM_SimpleEquilibriums):
    '''
    To.Do.
    Expected to handle peak widths more accurately for gradient eluents.
    '''
    
    hmap: Callable[[ndarray], ndarray]
    
    
    def __post_init__(self):
        
        object.__setattr__(self, 'distribute', ...) # To do modify 'distribute'
        
        # Important: self.distribute -> Bound method in __post_init__
        super().__post_init__()
        
    # --------------------------------------------------------------------------------
    
    @cached_property
    def _drop_when_updated(self) -> tuple:
        
        return ('dt',)
    
    @cached_property
    def _mutables(self) -> tuple:
        
        return ('fr', 'backflush',)

# --------------------------------------------------------------------------------


# In[8]:


# Helper functions:
@deprecated(reason='No longer needed. Will be removed soon...')
def load_input_data_for_test(
    model: GenericDiscontinousSegmentedModel,
    eluent: Eluent,
    V_sample: str | float | int,
    cA_sample: str | float | int,
    cE_sample: ndarray | tuple | list,
    tmax: str | float | int,
    ):
    
    V_sample = ParseQuantities.parse_string_as(V_sample, 'uL') # uL
    cA_sample = ParseQuantities.parse_string_as(cA_sample, 'mM') # mM
    cE_sample = np.array(
        [ParseQuantities.parse_string_as(c, 'mM') for c in cE_sample])
    tmax = ParseQuantities.as_min(tmax)
    
    dVm = model.dVm
    N_total = int(np.round(tmax / model.dt))
    assert N_total > 100
    
    N_sample = int(np.round(V_sample / dVm))
    N_sample = max(N_sample, 1)
    cA_sample *= V_sample / (dVm * N_sample)
    sample_arrays = (
        np.ones(N_sample) * cA_sample,
        np.array([np.ones(N_sample) * cE for cE in cE_sample])
        )
    
    t = np.arange(N_total) * model.dt
    eluent_arrays = (
        np.zeros(N_total),
        np.array([eluent(t).get(ion) for ion in model.competing_ions])
        )
    
    input_arrays = (
        np.concatenate((sample_arrays[0], eluent_arrays[0]))[:N_total],
        np.concatenate(
            (sample_arrays[1], eluent_arrays[1]), axis=1
            )[:,:N_total]
        )
    
    model.clear()
    model.input_data(*input_arrays)


# In[9]:


__all__ = [
    'GenericDiscontinousSegmentedModel',
    'DSM_SimpleEquilibriums',
    'DSM_SE',
    'DSM_SEQ',
    'DSM_CompleteEquilibriums',
    'DSM_CE',
    'DSM_CEQ',
    ]

