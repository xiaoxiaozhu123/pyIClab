#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Built-in import ######

import re
import warnings
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal
from importlib.resources import files

##### External import ######

import pint
import pandas as pd
from numpy.typing import NDArray
from pyEQL import IonDB
from pyEQL.utils import standardize_formula

##### Local import ######

from pyIClab.units import ParseQuantities
from pyIClab.errors import (
    ConflictedIonError, NeutralIonError, DuplicatedIonError,
    )

# --------------------------------------------------------------------------------


# In[2]:


_FormulaInfo = namedtuple('IonFormulaInfo', ['formula', 'ionbody', 'charge'])
# --------------------------------------------------------------------------------
@lru_cache
def _extract_info(ion: str) -> _FormulaInfo:
    
    p = re.compile(
        r'(?P<ionbody>[A-Za-z0-9\s.]+)(?P<charge>\[(\+|-)?(\d*)\]|(\+|-)(\d*))$')
    match = re.match(p, ion)
    if match is None:
        raise ValueError(f'Cannot match formula "{ion}".')
    
    ionbody = match.group('ionbody')
    charge = match.group('charge')
    charge = re.sub(r'[\[\]]', '', charge)
    if charge in ('+', '-'):
        charge += '1'
    formula = f'{ionbody}[{int(charge):+d}]'
    
    return _FormulaInfo(formula, ionbody, int(charge))
# --------------------------------------------------------------------------------


# In[3]:


@dataclass(frozen=True, kw_only=True, repr=False)
class Ion:
    
    formula: str
    pp_compat: Literal['supported', 'compatible', 'unsupported'] ='unsupported'
    diffusion_coefficient: float | str =None
    molecular_weight: float | str =None
    
    # --------------------------------------------------------------------------------
    
    def __post_init__(self):
        
        object.__setattr__(self, 'formula', _extract_info(self.formula).formula)
        object.__setattr__(self, 'pp_compat', self.pp_compat.lower())
        assert self.pp_compat in ('supported', 'compatible', 'unsupported')
        
        if self.diffusion_coefficient is not None:
            object.__setattr__(self, 'diffusion_coefficient', (
                ParseQuantities.parse_string_as(self.diffusion_coefficient, 'cm**2/s')
                ) * pint.Unit('cm**2/s'))
            
        if self.molecular_weight is not None:
            object.__setattr__(self, 'molecular_weight', (
                ParseQuantities.parse_string_as(self.molecular_weight, 'g/mol')
                ) * pint.Unit('g/mol'))
    
    # --------------------------------------------------------------------------------
    @property
    def formula_pp(self) -> str:
        '''
        the chemical formula passed to phreeqpython engine...
        '''
        
        p = re.compile(r'\[[+-]\d+\]')
        match = re.findall(p, self.formula)[-1]
        
        return self.formula[:-len(match)] + match[1:-1]
    
    # --------------------------------------------------------------------------------
    
    @property
    def formula_latex(self) -> str:
        
        info = _extract_info(self.formula)
        ionbody = info.ionbody
        charge = info.charge
        
        for i in '0123456789':
            ionbody = ionbody.replace(i, r'$_{' + i + r'}$')
        
        c = f'{charge:+}'
        if abs(charge) == 1:
            c = c[0]
        else:
            c = c[1:] + c[0]
        
        c = r'$^{' + c + r'}$'
        
        return ionbody + c
            
    # --------------------------------------------------------------------------------
    
    @property
    def charge(self) -> bool:
        
        return _extract_info(self.formula).charge
        
    @property
    def anion(self) -> bool:
        
        return self.charge < 0
    
    @property
    def cation(self) -> bool:
        
        return self.charge > 0
    
    @property
    def organic(self) -> bool:
        
        return 'C' in self.elements and (
            'H' in self.elements) and (
            self.formula != 'HCO3[-1]')
    
    # --------------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        
        return f'{"Cation" if self.cation else "Anion"} {self.formula}'
            
    # --------------------------------------------------------------------------------
           


# In[4]:


class _IonLibary:
    
    def __init__(self, db: pd.DataFrame):
        
        self._db = db[
            ['formula', 'alias', 'pp_compat', 'diffusion_coefficient',
            'molecular_weight']
            ]
        self._registry = {
            data['alias']: Ion(
                formula=data['formula'],
                pp_compat=data['pp_compat'],
                diffusion_coefficient=data['diffusion_coefficient'],
                molecular_weight=data['molecular_weight'],
                ) for i, data in self._db.iterrows()
            }
        for alias, ion_obj in self._registry.items():
            if alias.isidentifier():
                setattr(self, f'_{alias}', ion_obj)
                setattr(
                    type(self),
                    alias,
                    property(fget=lambda self, alias=alias: getattr(self, f'_{alias}'))
                    )
                
    # --------------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        
        alias_table = [
            f'<{k.title()} {v.formula}>' for k, v in self.registry.items()]
        
        return 'Ion Library {' + ', '.join(alias_table) + '}'
    
    @property
    def registry(self) -> dict:
        
        return self._registry
    
    @property
    def f_registry(self) -> list:
        
        return sorted([v.formula for v in self.registry.values()])
    
    @lru_cache
    def from_formula(self, formula: str) -> Ion:
        '''
        Query formula from IonLibrary.
        '''
        standard_formula = get_ioninfo(formula, _ionlab=self).formula
        for alias, ion_obj in self.registry.items():
            if ion_obj.formula == standard_formula:
                return ion_obj
            
        # In case not found, search in pyEQL IonDataBase
        dc = None
        mw = None
        try:
            formula = standardize_formula(formula)
        except ValueError:
            pass
        
        db = IonDB.query_one({'formula': formula})
        if db is not None:
            mw = db.get('molecular_weight')
            dc = db.get('transport').get('diffusion_coefficient')
            if dc is not None:
                dc = dc.get('value')
        
        # We should use formula standarized by builtin get_ioninfo function
        # In case pyEQL convert the formulas into those pyIClab doesn't
        # understand (NH4[+1], HPO4[-2], etc...)
        return Ion(
            formula=standard_formula,
            diffusion_coefficient=dc,
            molecular_weight=mw
            )

# --------------------------------------------------------------------------------

db = pd.read_csv(files('pyIClab') / 'db//ionlib//IonLibrary.csv', comment='#')
IonLibrary = _IonLibary(db)


# In[5]:


@lru_cache
def get_ioninfo(ion: str, _ionlab=IonLibrary) -> _FormulaInfo:
    '''
    Extract the ion infofrom alias or formula.
    Returns a namedtuple[['formula', 'ionbody', 'charge']]
    '''
    
    if ion in _ionlab.registry.keys():
        formula = getattr(_ionlab, ion).formula
        return _extract_info(formula)
    else:
        try:
            ioninfo = _extract_info(ion)
        except ValueError as e:
            raise ValueError(
                f'''Cannot identify the ion from formula/alias: {ion}.''') from e
        else:
            if not ioninfo.charge:
                raise NeutralIonError(ion)
            try:
                ioninfo2 = get_ioninfo(ioninfo.ionbody)
            except ValueError:
                return ioninfo
            else:
                if ioninfo.charge != ioninfo2.charge:
                    raise ConflictedIonError(ion, ioninfo2.formula, 'charges')
                return ioninfo2

# --------------------------------------------------------------------------------


# In[6]:


def check_duplicated_ions_from_sequence(
    ions: list[str] | tuple[str, ...] | NDArray[str],
    ) -> None:
    
    standard_formulas = [get_ioninfo(ion) for ion in ions]
    for i, formula in enumerate(standard_formulas):
        if formula in standard_formulas[:i]:
            j = standard_formulas[:i].index(formula)
            raise DuplicatedIonError(ions[i], ions[j])
            
# --------------------------------------------------------------------------------


# In[7]:


@lru_cache
def get_formula_latex(ion: str, *, _ionlab=IonLibrary):
    
    ion_obj = _ionlab.from_formula(ion)
    
    return ion_obj.formula_latex
    
# --------------------------------------------------------------------------------


# In[8]:


@lru_cache
def get_diff(ion: str, *, _ionlab=IonLibrary) -> float:
    '''
    Query the diffusion_coefficient of the ion throught the database.
        If not find, return None.
    Units: cm**2/s
    '''
    ion_obj = _ionlab.from_formula(ion)
    diff = ion_obj.diffusion_coefficient
    if diff is None:
        return
    else:
        return pint.Quantity(diff).to('cm**2/s').magnitude
    
# --------------------------------------------------------------------------------


# In[9]:


__all__ = ['Ion', 'IonLibrary',]


# In[ ]:




