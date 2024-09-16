#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[2]:


##### Built-in import #####

from typing import Callable

##### External import #####

import numpy as np
import pandas as pd
from numpy import ndarray
from deprecated import deprecated
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.stats import linregress
from scipy.optimize import curve_fit

##### Local import #####

from pyIClab.utils.beadedbag import proj2hull

# --------------------------------------------------------------------------------


# In[3]:


### Generic funcs ###

def find_x_LSSM(LSSM_kmap: Callable, y: int | float) -> int | float:
    
    c = np.arange(5, 66, 10).reshape(1, -1)
    logc = np.log10(c).flatten()
    logk = LSSM_kmap(c)
    
    x = np.polyfit(logc, logk, 1)[0] * (-y)
    
    return int(np.round(x)) if abs(np.round(x) - x) < 0.01 else x
    
# --------------------------------------------------------------------------------

def find_K_LSSM(
    a: float,
    phase_ratio: float,
    Q: float, # umol
    Vm: float, # mL
    x: int | float,
    y: int | float,
    ) -> float:

    logK = y*a + (x-y)*np.log10(phase_ratio) + x*np.log10(Vm/Q*y)
            
    return 10**logK


# In[4]:


### for model constructor parameters ###
## functions to get plate numbers from stationary phase database ##

def get_plates_from_constant_dV(
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

# --------------------------------------------------------------------------------


# In[5]:


### Functions to get hmap: h(fr, cE) ###

def _get_hmap_from_1Dinterpolator(
    df: pd.DataFrame,
    ion: str,
    fr: float, 
    ) -> Callable[[ndarray], ndarray]:
    
    df = df[df['fr']==fr]
    df = df.sort_values(by=ion, ignore_index=True)
    cdata = df[ion].to_numpy()
    hdata = df['H'].to_numpy()
    
    def _interpolator(cE: ndarray) -> ndarray: # -> shape(N,)
        
        return np.interp(cE, cdata, hdata)
    
    return _interpolator

def _get_hmap_from_griddata_constant_fr(
    df: pd.DataFrame,
    competing_ions: tuple[str, ...],
    fr: float
    ) -> Callable[[ndarray], ndarray]:
    
    df = df[df['fr']==fr]
    xydata = df[[*competing_ions]].to_numpy()
    hdata = df['H'].to_numpy()
    
    interpolator_linear = LinearNDInterpolator(points=xydata, values=hdata)
    interpolator_nearest = NearestNDInterpolator(x=xydata, y=hdata)
    
    def _interpolator(cE: ndarray) -> ndarray: # -> shape(N,)
        
        points = cE.T
        hdata = interpolator_linear(points)
        mask = np.isnan(hdata)
        
        if np.any(mask):
            hdata = np.where(mask, interpolator_nearest(points[mask]), hdata)
            
        return hdata
    
    return _interpolator

def _get_hmap_from_griddata(
    df: pd.DataFrame,
    competing_ions: tuple[str, ...],
    fr: float
    ) -> Callable[[ndarray], ndarray]:
    
    xydata = df[['fr', *competing_ions]].to_numpy()
    hdata = df['H'].to_numpy()
    
    interpolator_linear = LinearNDInterpolator(points=xydata, values=hdata)
    interpolator_nearest = NearestNDInterpolator(x=xydata, y=hdata)
    
    def _interpolator(cE: ndarray) -> ndarray: # -> shape(N,)
        
        points = cE.T
        hdata = interpolator_linear(points)
        mask = np.isnan(hdata)
        
        if np.any(mask):
            hdata = np.where(mask, interpolator_nearest(points[mask]), hdata)
            
        return hdata
    
    return _interpolator   
    
    
def get_hmap(
    *, 
    Hdata: dict,
    analyte: str,
    competing_ions: tuple[str, ...],
    fr: float,
    ) -> Callable[[float, ndarray], ndarray]:
    
    df = Hdata.get(competing_ions).copy()
    df = df[df['analyte']==analyte]
    
    if len(df) < 3:
        raise ValueError(f'Not enough data points for {analyte}.')
    
    if len(competing_ions) == 1 and len(df[df['fr']==fr]) >= 3:
        
        return _get_hmap_from_1Dinterpolator(
            df=df,
            ion=competing_ions[0],
            fr=fr
            )
    
    elif len(df[df['fr']==fr]) >= 3: # for multi-species eluents
        
        return _get_hmap_from_griddata_constant_fr(
            df=df,
            competing_ions=competing_ions,
            fr=fr
            )
    else: # fr lies outside the grid data.
        
        return _get_hmap_from_griddata(
            df=df,
            competing_ions=competing_ions,
            fr=fr
            )

# --------------------------------------------------------------------------------


# In[6]:


# --------------------------------------------------------------------------------
# Functions to get func kmap from stationary phase database...

def _get_kmap_single_eluent(
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

def _get_kmap_carbonates_simple_LSSM(*, df: pd.DataFrame) -> Callable[[ndarray,], ndarray]:
    '''
    A simple LSSM for carbonate buffer, kmap is described as:
    logk = A + B*logCO3 + C*logHCO3
    It is the default method in PyICLab for carbonate buffers, may loss accuraccy on some occasions.
    '''
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

def get_kmap(
    *,
    kdata: dict,
    analyte: str,
    competing_ions: tuple[str, ...],
    ) -> Callable[[ndarray,], ndarray]:
    
    df = kdata.get(competing_ions).copy()
    df = df[df['analyte']==analyte]
    
    if len(competing_ions) == 1 and len(df) >= 3:
        return _get_kmap_single_eluent(
            df=df, analyte=analyte, ion=competing_ions[0])
    elif set(competing_ions) == {'CO3[-2]', 'HCO3[-1]'}:
        return _get_kmap_carbonates_simple_LSSM(df=df)
    else:
        raise NotImplementedError

# --------------------------------------------------------------------------------

def get_kmap_carbonates_LSSM_EA(*, df: pd.DataFrame) -> Callable[[ndarray,], ndarray]:
    '''
    reference:
    Analytical Chemistry, Vol. 74, No. 23, December 1, 2002
    
    I think they've got the wrong Equation(4) in Trends in Analytical Chemistry 80 (2016) 625–635.
    [Et] -> log[Et]
    '''
    
    def fit_func(cE, f1, f2, f3, f4):
        
        c1 = cE[0, :]
        Et = np.sum(cE, axis=0)
        logEt = np.log10(Et)
        
        return (f1 + f2*logEt) + (f3 + f4*logEt)*np.log10(c1)
    
    carb, bicarb = 'CO3[-2]', 'HCO3[-1]'
    cE = np.array([df[carb].to_numpy(), df[bicarb].to_numpy()])
    logk = np.log10(df['k'])
    initial_guess = [1, -0.5, 1, -0.5]
    
    constants, pcov = curve_fit(fit_func, cE, logk, p0=initial_guess)
    
    def kmap(cE: ndarray):
        
        cE = np.array(cE, dtype=np.float64)
        cE[cE<1e-4] = 1e-4
        
        return fit_func(cE, *constants)
    
    return kmap

# --------------------------------------------------------------------------------

