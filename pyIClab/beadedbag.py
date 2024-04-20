#!/usr/bin/env python
# coding: utf-8

# In[2]:


'''
A junk shop of helper functions
'''

##### Built-in import #####

import warnings
import numpy as np
import functools
from time import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product

##### External import #####

from numpy import ndarray
from scipy.spatial import ConvexHull, Delaunay
from quadprog import solve_qp


# In[2]:


mpl_custom_rcconfig = {
    'figure.figsize': [8, 3],
    'figure.dpi': 300.0,
    'axes.linewidth': 1.0,
    'axes.labelsize': 14,
    'font.family': 'Arial',
    'lines.linewidth': 1.0,
    'legend.frameon': False,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'mathtext.default': 'regular',
    'mathtext.fontset': 'dejavusans',
    }

# --------------------------------------------------------------------------------


# In[3]:


# Function to determine if the environment is Jupyter
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (console, script, or something else)
    except NameError:
        return False    # Probably standard Python interpreter

# --------------------------------------------------------------------------------


# In[4]:


def timer(func):
    """A decorator function to record how much time func has cost"""
    def wrapper(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        t1 = time()
        print('Program "{0}" finished in {1:.2f}ms'.format(
            func.__name__, (t1-t0) * 1000))
        print('result: ', end='')
        return res
    return wrapper

# --------------------------------------------------------------------------------


# In[5]:


@np.errstate(divide='ignore', invalid='ignore')
def cubic_solver(a: ndarray, b: ndarray, c: ndarray, d: ndarray) -> ndarray:
    '''
    Solves cubic equations of the form ax^3 + bx^2 + cx + d = 0 for each 
        element in the input arrays.

    This function is designed to handle a batch of cubic equations simultaneously. 
    Special cases like linear (ax^2 + bx + c = 0) and quadratic equations 
        (bx + c = 0) are also handled.
    If the roots are complex, np.nan is returned for those cases.

    Parameters:
    a, b, c, d (ndarrays): Arrays of coefficients. Each element in these arrays 
        represents a coefficient for a cubic equation. The shapes of all input 
        arrays must be the same.

    Returns:
    ndarray: An array of shape (n, 3), where each row contains the 
        three roots (real or np.nan for complex roots) of the corresponding 
        cubic equation. If an equation is of lower order (linear or quadratic), 
        the non-applicable roots are filled with np.nan.
    '''
    assert a.shape == b.shape == c.shape == d.shape
    
    x1 = np.ones_like(a) * np.nan
    x2 = x1.copy()
    x3 = x1.copy()
    
    mask0 = (a==0) & (b==0) & (c==0)
    assert not np.any(mask0)
    
    # --------------------------------------------------------------------------------
    mask1 = (a==0) & (b==0)
    # cx + d = 0 => x = -d / c
    x1[mask1] = -d[mask1] / c[mask1]
    
    # --------------------------------------------------------------------------------
    mask2 = (a==0) & (~mask1)
    # bx^2 + cx + d = 0
    Δ2 = c**2 - 4*b*d
    mask3 = Δ2>=0
    mask4 = mask2 & mask3
    x1[mask4] = (-c[mask4] + np.sqrt(Δ2[mask4])) / (2*b[mask4])
    x2[mask4] = (-c[mask4] - np.sqrt(Δ2[mask4])) / (2*b[mask4])
    
    # --------------------------------------------------------------------------------
    # ax^3 + bx^2 + cx^2 + d = 0
    mask5 = (~mask1) & (~mask2)

    p = (3*a*c - b**2) / (3*(a**2))
    q = (2*(b**3) - 9*a*b*c + 27*(a**2)*d) / (27*(a**3))
    Δ3 = (q**2)/4 + (p**3)/27
     
    # Δ3 == 0
    mask6 = Δ3==0
    mask7 = mask5 & mask6
    x1[mask7] = 2 * np.cbrt(-q[mask7] / 2) - b[mask7] / a[mask7] / 3
    x2[mask7] = -2 * np.cbrt(-q[mask7] / 2) - b[mask7] / a[mask7] / 3
    
    # Δ3 > 0
    mask8 = Δ3>0
    mask9 = mask5 & mask8
    x1[mask9] = np.cbrt(-q[mask9]/2 + np.sqrt(Δ3[mask9])) + (
        np.cbrt(-q[mask9]/2 - np.sqrt(Δ3[mask9]))) - b[mask9] / a[mask9] / 3
    
    # Δ3 < 0
    mask10 = Δ3<0
    mask11 = mask5 & mask10
    θ = np.arccos((3*q) / (2*p) * np.sqrt(-3/p)) / 3
        
    x1[mask11] = 2 * np.sqrt(-p[mask11]/3) * np.cos(θ[mask11]) - (
        b[mask11]/(3*a[mask11]))
    x2[mask11] = 2 * np.sqrt(-p[mask11]/3) * np.cos(θ[mask11] + 2*np.pi/3) - (
        b[mask11]/(3*a[mask11]))
    x3[mask11] = 2 * np.sqrt(-p[mask11]/3) * np.cos(θ[mask11] + 4*np.pi/3) - (
        b[mask11]/(3*a[mask11]))
    
    return np.array([x1, x2, x3]).T

# --------------------------------------------------------------------------------


# In[6]:


def FutureWarningFilter(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            # Filter only FutureWarning
            warnings.simplefilter("ignore", FutureWarning)
            return func(*args, **kwargs)
    return wrapper

# --------------------------------------------------------------------------------


# In[4]:


# -------------------------------------------------------------------------------- 
def _inside_hull(p, points) -> bool:
    
    convex_hull = ConvexHull(points)
    delaunay = Delaunay(points[convex_hull.vertices])
    
    return delaunay.find_simplex(p) >= 0

def _adjust(point: ndarray, tolerance=1e-8) -> ndarray:
    
    dim = len(point)
    factors = np.array(list(product([1, -1], repeat=dim)))
    
    return tolerance * factors + point

def _bfs(projs, points, cache):
    
    for proj in projs:
        if _inside_hull(proj, points):
            return proj
        else:
            cache.add(tuple(proj))
    queque = [p for proj in projs for p in _adjust(proj) if tuple(p) not in cache]
    queque = np.array(queque)
    
    return _bfs(queque, points, cache)
    
def proj2hull(p: ndarray, points: ndarray) -> ndarray:
    '''
    Find the projection of `p` to the convex hull formed by `points`.
    If `p` is inside the convex hull, returns self.
    reference:
    https://stackoverflow.com/questions/42248202/find-the-projection-of-a-point-on-the-convex-hull-with-scipy
    '''
    
    G = np.eye(len(p), dtype=np.float64)
    a = np.array(p, dtype=np.float64)
    convex_hull = ConvexHull(points)
    equations = convex_hull.equations
    C = np.array(-equations[:, :-1], dtype=np.float64)
    b = np.array(equations[:, -1], dtype=np.float64)
    proj, f, xu, itr, lag, act = solve_qp(G, a, C.T, b, meq=0, factorized=True)
    
    return _bfs(np.array([proj]), points, set())
    
# -------------------------------------------------------------------------------- 


# In[8]:


def get_current_clock(strft: str ='%H:%M:%S'):
    
    return datetime.now().strftime(strft)

# -------------------------------------------------------------------------------- 

@dataclass
class Progressor:
    
    prefix: str =None
    suffix: str =None
    indentation: int =0
    separator: str =None
    timer: bool =False
        
    def __post_init__(self):
        
        self.indentation =  ' ' * self.indentation
    
    def __call__(self, func):
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            if self.prefix is not None:
                clock = get_current_clock() + ' ' if self.timer else ''
                print(self.indentation + clock + self.prefix)
                
            res = func(*args, **kwargs)
            
            if self.suffix is not None:
                clock = get_current_clock() + ' ' if self.timer else ''
                print(self.indentation + clock + self.suffix)
            if self.separator is not None:
                print(self.separator)
            
            return res
        
        return wrapper

# -------------------------------------------------------------------------------- 


# In[9]:


__all__ = [
    'mpl_custom_rcconfig',
    'is_notebook',
    'timer',
    'cubic_solver',
    'FutureWarningFilter',
    'proj2hull',
    'get_current_clock',
    'Progressor',
    ]

