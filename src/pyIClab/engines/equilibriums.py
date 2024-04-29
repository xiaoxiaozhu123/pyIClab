#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[2]:


##### Built-in import #####

from typing import Tuple, Callable

##### External import #####

import numpy as np
from numpy import ndarray
from scipy.optimize import fsolve
from deprecated import deprecated

##### Local import #####

from pyIClab.beadedbag import cubic_solver

# --------------------------------------------------------------------------------


# In[3]:


def complete_simple_equilibrium(
    logk: float | ndarray,
    cAm: ndarray,
    nAs: ndarray,
    Vm: float | ndarray,
    ) -> Tuple[ndarray]:
    '''
    Equilibrium: A(MP) <=> A(SP)
    Eq(1): k = nAs / nAm = nAs / [cAm * Vm] 
    Eq(2): nAs + cAm * Vm = initial{nAs + cAm * Vm}
    
    Returns a tuple containing the solution of Eq(1) and Eq(2): (cAm, nAs)
        
    Parameters:
    ----------
    k: float | ndarray, shape(n,), retention factor(s).
    cAm: ndarray, shape(n,), concentrations of the analyte in MP, mM.
    nAs: ndarray, shape(n,), amounts of the analyte in SP, nmol.
    Vm: float | ndarray, shape(n,), volumn(s) of MP (segments), uL.
    '''
    k = 10**logk
    cAm = (cAm*Vm + nAs) / (1+k) / Vm
    nAs = cAm * k * Vm
    
    return cAm, nAs

# --------------------------------------------------------------------------------


# In[4]:


def complete_equilibrium(
    K: float | ndarray,
    Q: float | ndarray,
    Vm: float | ndarray,
    Vs: float | ndarray,
    x: int | float,
    y: int | float,
    nAm: ndarray,
    nAs: ndarray,
    nEm: ndarray,
    nEs: ndarray,
    ):
    '''
    Equilibrium:
        yA(MP) + xE(SP)  <=>  yA(SP) + xE(MP)
         nAm      nEs          nAs      nEm
    Eq(1): K = {(nAs / Vs)^y * (nEm / Vm)^x} / {(nAm / Vm)^y * (nEs / Vs)^x}
    Eq(2): nAm + nAs = init(nAm + nAs)
    Eq(3): nEm + nEs = init(nEm + nEs)
    Eq(4): Q = nAs + nEs
    Let  nAm  nAs  nEm  nEs
          α    β    γ    θ
    init(nAm + nAs) = A
    init(nEm + nEs) = B
    Boundary conditions:
    ### α >= 0, 0 <= β <= Q0, γ >= 0, 0 <= θ <= Q0 ### modified @20240315
    α >= 0, 0< = xβ <= Q, γ >= 0, 0<= yθ <= Q modified @20240315
    ----------
    i) If x/y = 1, we have
    (1) βγ = Kαθ
    (2) α + β = A
    (3) γ + θ = B
    (4) β + θ = Q/y => Q0
    =>(1-K)β^2 + [B+KA+(K-1)Q0]β - KAQ0 = 0
    => find β, consequently find α, γ, θ.
    ----------
    ii) If x/y = 2, we have
    (1) βγ^2 = αθ^2*{K^(1/y) * Vs/Vm} => K0*αθ^2
    (2) α + β = A
    (3) γ + θ = B
    (4) 2β + θ = Q/y => Q0
    => aβ^3 + bβ^2 + cβ + d = 0
    where 
    a = 4 * (K + 1)
    b = 4 * (B-Q) - 4*K*(A+Q)
    c = 4*A*K*Q + B**2 - 2*B*Q + K*Q**2 + Q**2
    d = -A*K*Q**2
    '''
    assert np.all(K != 1), 'The analyte ion should not be the eluting ion!'
    
    r = x / y
    if .99 <= r <= 1.01:
        A = nAm + nAs
        B = nEm + nEs
        return _complete_equilibrium_equal_charges(A, B, Q, K, y)

    elif 1.99 <= r <= 2.01:
        A = nAm + nAs
        B = nEm + nEs
        return _complete_equilibrium_double_charges(A, B, Q, K, y, Vm, Vs)
            
    else:
        # Handling arbitary charge-ratios.
        # Just leaving an API, scipy.optimize.fsolve is extremely
        # time-consuming when solving non-linear equations.
            
        return _complete_equilibrium_arbitrary_charges(
            nAm, nAs, nEm, nEs, Q, K, x, y, Vm, Vs)
            
# --------------------------------------------------------------------------------

def _complete_equilibrium_equal_charges(A, B, Q, K, y):
    
    Q0 = Q / y
    a = 1 - K
    b = B + K*A + (K-1)*Q0
    c = -K * A * Q0
    
    Δ =  np.sqrt(b**2 - 4*a*c)
    β1 = (-b + Δ) / 2 / a
    β2 = (-b - Δ) / 2 / a
    β = np.where((β1>=-1e-7) & (β1<=Q0+1e-7), β1, β2)
        
    α = A - β
    θ = Q0 - β
    γ = B - θ
    
    # deal with numeric overflows
    for arr in (α, β, θ, γ):
        mask = (arr>=-1e-7) & (arr<0)
        arr[mask] = 0.0
        
    for arr in (β, γ):
        mask = (arr>Q0) & (arr<(Q0+1e-7))
        arr = np.where(mask, Q0, arr)
        
    return α, β, γ, θ

# --------------------------------------------------------------------------------

def _complete_equilibrium_double_charges(A, B, Q, K, y, Vm, Vs):
    
    Q0 = Q / y
    K0 = K**(1/y) * Vm / Vs
    
    a = np.ones_like(A) * 4 * (K0 + 1)
    b = 4*(B-Q0) - 4*K0*(A+Q0)
    c = 4*A*K0*Q0 + B**2 - 2*B*Q0 + K0*Q0**2 + Q0**2
    d = -A * K0 * Q0**2
    
    β1, β2, β3 = cubic_solver(a, b, c, d).T
    β = np.where((β1>=-1e-7) & (β1<=Q0/2+1e-7), β1, β2)
    β = np.where((β>=-1e-7) & (β<=Q0/2+1e-7), β, β3)
    assert np.all(~np.isnan(β))
    
    α = A - β
    θ = Q0 - 2*β
    γ = B - θ
    
    # deal with numerical overflows
    for arr in (α, β, θ, γ):
        mask = (arr>=-1e-7) & (arr<0)
        arr[mask] = 0.0
    
    mask1 = (β>Q0/2) & (β<Q0/2+1e-7)
    β = np.where(mask1, Q0/2, β)
    
    mask2 = (θ>Q0) & (θ<Q0)
    θ = np.where(mask2, Q0, θ)

    
    return α, β, γ, θ

# --------------------------------------------------------------------------------
@deprecated(reason='This method would run good, but unbearably slow.')
def _complete_equilibrium_arbitrary_charges(
    nAm, nAs, nEm, nEs, Q, K, x, y, Vm, Vs):
    '''
    This method uses the fsolve function from scipy.optimize for finding the roots 
        of the equilibrium equation.
    '''
    @np.errstate(all='ignore')
    def f(β, A, B, Q0, K0, r):
        
        return K0 * (A-β) * (Q0-r*β)**r - β*(r*β + B - Q0)**r
    
    nAm = nAm.copy()
    nAs = nAs.copy()
    nEm = nEm.copy()
    nEs = nEs.copy()
    
    A = nAm + nAs
    B = nEm + nEs
    
    fast_mask = (A<1e-7) | (B<1e-7)
    
    if np.all(fast_mask): 
        return nAm, nAs, nEm, nEs
    
    A = A[~fast_mask]
    B = B[~fast_mask]
    
    # Q, Vs, Vm should be in shape(N,)
    Q = Q[~fast_mask]
    Vs = Vs[~fast_mask]
    Vm = Vm[~fast_mask]
    
    Q0 = Q / y
    r = x / y
    K0 = K**(1/y) * (Vs/Vm)**(1-r)
    β0 = nAs[~fast_mask].copy()
    
    β = fsolve(
        func=f,
        x0=β0,
        args=(A, B, Q0, K0, r),
        )
    
    α = A - β
    θ = Q0 - r*β
    γ = B - θ
    
    # deal with numerical overflows
    for arr in (α, β, θ, γ):
        mask = (arr>=-1e-7) & (arr<0)
        arr[mask] = 0.0
    
    mask1 = (β>Q/x) & (β<Q/x+1e-7)
    β = np.where(mask1, Q/x, β)
    assert np.all((β>=0) & (β<=Q/x))
    
    mask2 = (θ>Q0) & (θ<Q0)
    θ = np.where(mask2, Q0, θ)
    
    nAm[~fast_mask] = α
    nAs[~fast_mask] = β
    nEm[~fast_mask] = γ
    nEs[~fast_mask] = θ
    
    return nAm, nAs, nEm, nEs

# --------------------------------------------------------------------------------


# In[5]:


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
    x: int,
    y: int) -> float:
    '''
    Equilibrium:
        yA(MP) + xE(SP)  <=>  yA(SP) + xE(MP)
         cAm      cEs          cAs      cEm
    => K^(1/y) =  (nAs/Vs)* (Vm/nAm) * cEm^(x/y) / (cEs)^(x/y)
    => K^(1/y) = k * (1/Φ) * (cEm)^(x/y) / (cEs)^(x/y)
    => 1/y * log(K) = log(k) - log(Φ) + (x/y)*log(cEm) - (x/y)log(cEs)
    => log(k) = 1/y * log(K) + log(Φ) - (x/y)*log(cEm) + (x/y)log(cEs)
    Assuming trace conditions (nA << nE), almost all the ion-exchange sites
        are oppupied by Eluting ions.
    => cEs = (Q/y) / Vs
    => log(k) = (1/y)*log(K) + log(Φ) - (x/y)log(cEm) + (x/y)log(Q/y) - (x/y)log(Vs)
    => log(k) = (1/y)*log(K) + (x/y)*log(Q/y) + (1-x/y)*log(Φ) - 
        (x/y)*log(Vm) - (x/y)*log(cEm)
    From empirical data of LSSM: log(k) = a + b*log(cEm)
    => log(K) = y*a + (x-y)*log(Φ) + x*log(Vm) - x*log(Q/y)
    (1) for (x, y) = (1, 1):
        log(K) = a + log(Vm) - log(Q)
    (2) for (x, y) = (2, 1)
        log(K) = a + logΦ + 2*log(Vm) - 2*log(Q)
    '''
    Φ = phase_ratio

    logK = y*a + (x-y)*np.log10(Φ) + x*np.log10(Vm/Q*y)
            
    return 10**logK

# --------------------------------------------------------------------------------


# In[ ]:




