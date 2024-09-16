#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[2]:


##### Built-in import #####

import warnings
from typing import Tuple, Callable

##### External import #####

import numpy as np
from numpy import ndarray
from scipy.optimize import fsolve, newton_krylov
from deprecated import deprecated

try:
    from scipy.optimize import NoConvergence
except ImportError:
    with warnings.catch_warnings(action='ignore'):
        from scipy.optimize.nonlin import NoConvergence

##### Local import #####

from pyIClab.utils.beadedbag import cubic_solver

##### Compatibility  ##### DO NOT REMOVE !!

from pyIClab.utils.optimize import find_x_LSSM, find_K_LSSM

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
        # quite time-consuming when solving non-linear equations.
            
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
        arr[mask] = Q0[mask]
        
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

def _complete_equilibrium_arbitrary_charges_fsolve(
    nAm, nAs, nEm, nEs, Q, K, x, y, Vm, Vs):
    '''
    This method uses the fsolve function from scipy.optimize for finding the roots 
        of the equilibrium equation.
    '''
    @np.errstate(all='ignore')
    def f(β, A, B, Q0, K0, r):
        
        return K0 * (A-β) * (Q0-r*β)**r - β*(r*β + B - Q0)**r
    
    def jac(β, A, B, Q0, K0, r):
        
        term1 = (Q0 - r * β)**r
        term2 = A - β
        term3 = (r * β + B - Q0)**r
        dterm1_dβ = -r * (Q0 - r * β)**(r-1)
        dterm3_dβ = r**2 * (r * β + B - Q0)**(r-1)
        jac_values = K0 * (-term1 + term2 * dterm1_dβ) - (term3 + β * dterm3_dβ)
        
        return np.diag(jac_values)
    
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
    β0 = nAs[~fast_mask]
    
    β = fsolve(
        func=f,
        x0=β0,
        args=(A, B, Q0, K0, r),
        fprime=jac,
        )
    
    α = A - β
    θ = Q0 - r*β
    γ = B - θ
    Qx = Q / x
    
    # deal with numerical overflows
    for arr in (α, β, θ, γ):
        mask = (arr>-1e-7) & (arr<0)
        arr[mask] = 0.0
    
    mask1 = (β>Qx) & (β<Qx+1e-7)
    β = np.where(mask1, Qx, β)
    assert np.all((β>=0) & (β<=Qx))
    
    mask2 = (θ>Q0) & (θ<Q0)
    θ = np.where(mask2, Q0, θ)
    
    nAm[~fast_mask] = α
    nAs[~fast_mask] = β
    nEm[~fast_mask] = γ
    nEs[~fast_mask] = θ
    
    return nAm, nAs, nEm, nEs

# --------------------------------------------------------------------------------
def _complete_equilibrium_arbitrary_charges_Newton_Krylov(
    nAm, nAs, nEm, nEs, Q, K, x, y, Vm, Vs):
    '''
    This method uses the Newton-Krylov from scipy.optimize for finding the roots 
        of the equilibrium equation.
    Update 2024.5.15: subsitute `newton_krylov` for `fsolve`.
    '''
    @np.errstate(all='ignore')
    def f(β, A, B, Q0, K0, r):
        
        return K0 * (A-β) * (Q0-r*β)**r - β*(r*β + B - Q0)**r
    
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
    β0 = nAs[~fast_mask]
    
    β = newton_krylov(
        F=lambda beta: f(beta, A, B, Q0, K0, r),
        xin=β0,
        )
    
    α = A - β
    θ = Q0 - r*β
    γ = B - θ
    Qx = Q / x
    
    # deal with numerical overflows
    for arr in (α, β, θ, γ):
        mask = (arr>-1e-7) & (arr<0)
        arr[mask] = 0.0
    
    mask1 = (β>Qx) & (β<Qx+1e-7)
    β = np.where(mask1, Qx, β)
    assert np.all((β>=0) & (β<=Qx))
    
    mask2 = (θ>Q0) & (θ<Q0)
    θ = np.where(mask2, Q0, θ)
    
    nAm[~fast_mask] = α
    nAs[~fast_mask] = β
    nEm[~fast_mask] = γ
    nEs[~fast_mask] = θ
    
    return nAm, nAs, nEm, nEs

# --------------------------------------------------------------------------------

def _complete_equilibrium_arbitrary_charges(*args, **kwargs):
    
    try:
        return _complete_equilibrium_arbitrary_charges_Newton_Krylov(*args, **kwargs)
    except NoConvergence:
        return _complete_equilibrium_arbitrary_charges_fsolve(*args, **kwargs)

# --------------------------------------------------------------------------------


# In[ ]:




