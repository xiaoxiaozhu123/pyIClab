#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024  PyICLab Kai "Kenny" Zhang
'''
##### Built-in import #####

import warnings

##### External import #####

import numpy as np
import pandas as pd

##### Local import #####

import pyIClab.ions as ions_
from pyIClab import (
    IonChromatograph, Eluent, SwitchingValve,
    Column, Dummy, PEEKTubing, 
    SampleLoop, Detector,
    )

from pyIClab.interface import Constructor
from pyIClab.assemblies.signals import  Suppressor, QuickSuppressor

# --------------------------------------------------------------------------------


# In[2]:


def PackedIC(
    name: str ='IC-Demo',
    *,
    competing_ions: tuple =('OH[-1]',),
    model_constructor_prompt: str | type ='DSM_CompleteEquilibriums', 
    eluent: Eluent =None,
    column: Column =None,
    sample_solution: dict =None,
    loop: SampleLoop =None,
    suppressor: Suppressor =None,
    ) -> IonChromatograph:
    '''
    Set up a simple suppressed anion-exchange IonChromatograph instance.
    All are set, just run start!
    '''
    # Default
    eluent = Eluent.HydroxideIsocratic('20 mM') if eluent is None else eluent
    column = Dummy.Column() if column is None else column
    sample_solution = {
        'F-': '0.1 mM',
        'Cl-': '0.1 mM',
        'NO2-': '0.1 mM',
        'Br-': '0.125 mM',
        'NO3-': '0.125 mM',
        } if sample_solution is None else sample_solution
    loop = SampleLoop('SampleLoop', V='25 uL') if loop is None else loop
    suppressor = QuickSuppressor(
        'Suppressor', kind='anion') if suppressor is None else suppressor
    valve = SwitchingValve.SixPort()
    detector = Detector('detector')
    
    # Assemble
    valve.assemble(0, eluent)
    valve.assemble(1, column)
    valve.assemble([2, 5], loop)
    column.assemble(suppressor)
    suppressor.assemble(detector)
    
    # Config
    ic = IonChromatograph(name,
        competing_ions=competing_ions, lockon=eluent)
    commands = '0.0 min, sixport, inject'
    ic.reset_commands(commands)
    ic.inject(sample_solution, loop)
    ic.set_ModelConstructor(model_constructor_prompt, column)
    
    return ic
    
# -------------------------------------------------------------------------------- 


# In[3]:


def read_csv_retention_db(files: str | list | tuple, **kwargs) -> dict:
    '''
    Load `IonExchange.retention_data` from csv files.
    kwargs forward to `pd.read_csv`.
    
    '''
    if not files:
        return {}
    if isinstance(files, list | tuple):
        f = list(files).pop()
        db0 = read_csv_retention_db(files, **kwargs)
        db1 = read_csv_retention_db(f, **kwargs)
        if tuple(db1.keys())[0] in db0.keys():
            warnings.warn(
                f'''\nDuplicated database of competing ions:{tuple(db1.keys())[0]} '''
                '''detected. Overwriting will be proceeded.'''
                )
            
        db0.update(db1)
        return db0
    
    completing_ion_table = {}
    kwargs['index_col'] = False
    df = pd.read_csv(files, **kwargs)
    for column in df.columns:
        try:
            ioninfo = ions_.get_ioninfo(column)
        except ValueError:
            pass
        else:
            completing_ion_table[column] = ioninfo.formula
    
    ions_.check_duplicated_ions_from_sequence(tuple(completing_ion_table.keys()))
    df = df.rename(columns=completing_ion_table)
    completing_ions = tuple(sorted(completing_ion_table.values()))
    df = df[['fr', *completing_ions, 'analyte', 'k', 'H']]
    df['analyte'] = df['analyte'].apply(lambda ion: ions_.get_ioninfo(ion).formula)
    dtype = {'fr': np.float64, 'analyte': 'string', 'k': np.float64,
            'H': np.float64,}
    dtype.update({ion: np.float64 for ion in completing_ions})
    
    return {completing_ions: df.astype(dtype=dtype)}
# -------------------------------------------------------------------------------- 


# In[4]:


__all__ = ['PackedIC', 'read_csv_retention_db',]

