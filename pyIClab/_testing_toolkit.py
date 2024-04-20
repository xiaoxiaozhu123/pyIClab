#!/usr/bin/env python
# coding: utf-8

# In[2]:


##### Built-in import #####

##### External import #####

##### Local import #####

import pyIClab.ions as ions_
from pyIClab import (
    IonChromatograph, Eluent, SwitchingValve,
    Column, Dummy, PEEKTubing, 
    SampleLoop, Detector,
    )

from pyIClab.interface import Constructor
from pyIClab.assemblies.signals import (
    QuickSuppressor, PhreeqcSuppressor, InsufficientPhreeqcSuppressor,
    ContaminatedPhreeqcSuppressor)

# --------------------------------------------------------------------------------


# In[10]:


def PackedSimpleIC(
    competing_ions: tuple[str, ...],
    eluent_profile: dict,
    column: Column = None,
    ColumnModelConstructorType: str | type ='DSM_CompleteEquilibriums',
    SuppressorType: type =QuickSuppressor,
    ):
    '''
    Builds up a simple suppressed anion-exchange IonChromatograph instance.
    All are set, just run start!
    '''
    if column is None: column = Dummy.Column()
    
    competing_ions = tuple(
        ions_.get_ioninfo(ion).formula for ion in competing_ions)
    eluent = Eluent('SimpleEluent', eluent_profile)
    ic = IonChromatograph('SimpleIC',
        competing_ions=competing_ions, lockon=eluent)
    valve = SwitchingValve.SixPort()
    loop = SampleLoop('sampleloop')
    suppressor = SuppressorType('suppressor', kind=ic.kind)
    detector = Detector('detector')
    valve.assemble(0, eluent)
    valve.assemble(1, column)
    valve.assemble([2, 5], loop)
    column.assemble(suppressor)
    suppressor.assemble(detector)
    
    commands =(
        '''0.0 min, sixport, inject'''
        )
    ic.reset_commands(commands)
    
    solution = {
        'F-': '0.1 mM',
        'Cl-': '0.1 mM',
        'NO2-': '0.1 mM',
        'Br-': '0.125 mM',
        'NO2-': '0.125 mM',
        }
    ic.inject(solution, loop)
    
    ic.set_ModelConstructor(ColumnModelConstructorType, column)
    
    return ic

# --------------------------------------------------------------------------------

