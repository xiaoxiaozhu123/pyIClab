#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[2]:


##### Built-in import #####

from functools import lru_cache

##### External import #####

import numpy as np
from numpy import ndarray

##### Local import #####

from pyIClab._baseclasses import BaseModel
from pyIClab.engines.equilibriums import (
    complete_simple_equilibrium, complete_equilibrium,
    )

# --------------------------------------------------------------------------------


# In[3]:


### kmap ###

def kmap_for_GDSM_testing(cEm: ndarray):

    # LSSM logk = 1.6 - 1.0 * logc
        
    return 1.6 - np.log10(np.squeeze(cEm))

# --------------------------------------------------------------------------------

def no_retain_kmap(cEm: ndarray) -> ndarray:
    '''
    The default kmap is defined here...
    You can customize the kmap by either passing a new `kmap` func when 
    initializing the model or applying stationary phase properties.
    Modify model.vessel in-place...
    '''
    
    cEm = np.array(cEm)
    
    return -np.ones(cEm.shape[-1]) * np.inf

# --------------------------------------------------------------------------------


# In[4]:


### kmap ###

def bypass_hmap(cEm: ndarray) -> ndarray:
    
    cEm = np.array(cEm)
    
    return np.zeros(cEm.shape[-1])

# --------------------------------------------------------------------------------


# In[5]:


### distribution method ###

def bypass_method(model: BaseModel, /, **kwargs):
    '''
    The default method is defined here...
    This method doesn't do nothing...it serves as a placeholder...
    You can customize the model by passing new method funcs when 
    initializing the model.
    Note:
    (1) New funcs should modify model.vessel in-place...
    (2) New funcs should take `model` intance as the first
        positional argument followed by kw arguments if any.
    (3) New funcs should support vectorized computations...
    '''
    pass

# --------------------------------------------------------------------------------

def simple_equilibrium_distribution_method(
    model: BaseModel,
    ) -> None:
    
    tiny = model.ignore_tiny_amount
    vessel = model.vessel
    
    A = vessel.fixed_nA
    tinyA = (A<tiny)
    mask = ~tinyA
    logk = vessel.logk[mask]
    cAm = vessel.cAm[mask]
    nAs = vessel.nAs[mask]
    dVm = vessel.dVm[mask] 
    vessel.cAm[mask], vessel.nAs[mask] = complete_simple_equilibrium(logk, cAm, nAs, dVm)
    
# --------------------------------------------------------------------------------

def complete_equilibrium_distribution_method(
    model: BaseModel,
    ) -> None:
    
    tiny = model.ignore_tiny_amount
    x, y = abs(model.x), abs(model.y[0])
    K = model.K
    vessel = model.vessel
    
    A = vessel.fixed_nA
    B = vessel.fixed_nE[0, :]
    mask = (A>=tiny) & (B>=tiny)
    
    Q = vessel.dQ[mask]
    Vm = vessel.dVm[mask]
    Vs = vessel.dVs[mask]
    nAm_ = vessel.nAm[mask]
    nAs_ = vessel.nAs[mask]
    nEm_ = vessel.nEm[0, mask]
    nEs_ = vessel.nEs[0, mask]

    nAm, nAs, nEm, nEs = complete_equilibrium(
        K, Q, Vm, Vs, x, y, nAm_, nAs_, nEm_, nEs_)
        
    vessel.cAm[mask] = nAm / Vm
    vessel.nAs[mask] = nAs
    vessel.cEm[0, mask] = nEm / Vm
    vessel.nEs[0, mask] = nEs

# --------------------------------------------------------------------------------


# In[6]:


### post distribution method ###
@lru_cache(maxsize=None)
def _get_diffusion_factor_A(model: BaseModel, A_diff: float) -> ndarray:
    
    vessel = model.vessel
    D = A_diff
    dL = model.dL / 2
    dV = vessel.dVm / 2 # shape(N,)
    dt = model.dt
    S = model.S / (1+vessel.phase_ratio) # shape(N,)
    
    return D*S[:-1]*dt / (dL*dV[:-1]) * 6e4


def analyte_diffusion_method(model: BaseModel, /, *,
    A_diff: float=None) -> None:
    '''
    A built-in method to simulate analyte diffusion.
    Do not apply it unless when loading a highly concentrated sample...
    Modify model.vessel in-place...
    
    A_diff: the diffusion coefficient of the analyte in cm**2/sec.
    '''
    if A_diff is None: return
    
    vessel = model.vessel
    ΔcAm = np.diff(vessel.cAm)
    Δc = _get_diffusion_factor_A(model, A_diff) * ΔcAm
    overflow = ((ΔcAm>0) & (Δc>ΔcAm/2)) | ((ΔcAm<0) & (Δc<ΔcAm/2))
    Δc = np.where(overflow, ΔcAm/2, Δc)
    Δ2c = np.diff(Δc)
    
    vessel.cAm[1:-1] += 0.5 * Δ2c
    vessel.cAm[0] += 0.5 * Δc[0]
    vessel.cAm[-1] -= 0.5 * Δc[-1]
    
# --------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _get_diffusion_factor_E(model: BaseModel, E_diff: tuple) -> ndarray:
    
    vessel = model.vessel
    
    D = np.reshape(E_diff, (-1, 1))
    dL = model.dL / 2
    dV = vessel.dVm / 2 # shape(N,)
    dt = model.dt
    S = model.S / (1+vessel.phase_ratio) # shape(N,)
    
    return D*S[:-1]*dt / (dL*dV[:-1]) * 6e4 # shape(M, N-1) 

def eluent_diffusion_method(model: BaseModel, /, *,
    E_diff: tuple | list | ndarray =None) -> None:
    '''
    A built-in method to simulate eluent diffusion.
    Modify model.vessel in-place...
    
    E_diff: shape(M,), the diffusion coefficients of the competing ions in cm**2/sec.
    If the diffusion coefficients is not given, model will bypass this method.
    '''
    E_diff = np.array(E_diff)
    if np.all(np.isnan(E_diff)):
        return
    else:
        E_diff[np.isnan(E_diff)] = 0.0
    
    vessel = model.vessel
    ΔcEm = np.diff(vessel.cEm) # shape(M, N-1)
    Δc = _get_diffusion_factor_E(model, tuple(E_diff)) * ΔcEm # shape(M, N-1) 
    overflow = ((ΔcEm>0) & (Δc>ΔcEm/2)) | ((ΔcEm<0) & (Δc<ΔcEm/2))
    Δc = np.where(overflow, ΔcEm/2, Δc) # shape(M, N-1) in case of overflow
    Δ2c = np.diff(Δc) # shape(M, N-2)
    
    vessel.cEm[:,1:-1] += 0.5 * Δ2c
    vessel.cEm[:,0] += 0.5 * Δc[:,0]
    vessel.cEm[:,-1] -= 0.5 * Δc[:,-1]
    
# --------------------------------------------------------------------------------
    
def diffusion_method(model: BaseModel, /, *,
    A_diff: float =None,
    E_diff: tuple | list | ndarray =None,
    ):
    
    analyte_diffusion_method(model, A_diff=A_diff)
    eluent_diffusion_method(model, E_diff=E_diff)
    
# --------------------------------------------------------------------------------

def total_mix_eluent(model: BaseModel) -> None:
    
    vessel = model.vessel
    cEm = vessel.cEm
    c0 = 0.75*cEm[:, 0] + 0.25*cEm[:, 1]
    c1 = 0.75*cEm[:, -1] + 0.25*cEm[:, -2]
    cEm[:, 1:-1] = 0.25 * (cEm[:, :-2] + 2*cEm[:, 1:-1] + cEm[:, 2:])
    cEm[:, 0] = c0
    cEm[:, -1] = c1
    
def total_mix_analyte(model: BaseModel) -> None:
    
    vessel = model.vessel
    cAm = vessel.cAm
    c0 = 0.75*cAm[0] + 0.25*cAm[1]
    c1 = 0.75*cAm[-1] + 0.25*cAm[-2]
    cAm[1:-1] = 0.25 * (cAm[:-2] + 2*cAm[1:-1] + cAm[2:])
    cAm[0] = c0
    cAm[-1] = c1

# --------------------------------------------------------------------------------
    


# In[7]:


### init vessel ###
def fill_column_with_eluent(
    model: BaseModel, /, *, cE_fill: list | tuple | ndarray):
    '''
    cE_fill: shape(M,), concentrations of competing ions to fill  in the column,
        usually the inital state of the eluent.
    Support single-spieces eluents;
    Compatible with multi-spieces eluents.
    Modify model.vessel in-place.
    '''
    
    cEm = np.array([np.ones(model.N) * c for c in cE_fill])
    nEs = np.reshape(
        np.ones(model.N * model.M) * model.Q / model.N / abs(sum(model.y)) * 1000,
        (model.M, model.N)) # nmol
    
    model.vessel.cEm = cEm
    model.vessel.nEs = nEs
    
# --------------------------------------------------------------------------------
    
def init_vessel_with_injection(
    model: BaseModel, /, *, cA: float, cE: list | tuple | ndarray
    ):
    '''
    Fill the tubing void with analyte and competing ions.
    No equilibriums between SP and MP will be performed.
    Should be only used for tubings without SP, such as in the scenerio of injection.
    '''
    cAm = np.ones(model.N) * cA
    cEm = np.ones((model.M, model.N)) * np.reshape(cE, (model.M, 1))
    
    model.vessel.cAm = cAm
    model.vessel.cEm = cEm


# In[ ]:




