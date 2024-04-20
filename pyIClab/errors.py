#!/usr/bin/env python
# coding: utf-8

# In[2]:


##### Built-in import ######
from dataclasses import dataclass
from typing import Literal

##### Local import #####
from pyIClab._baseclasses import BaseAccessory

# --------------------------------------------------------------------------------


# In[3]:


ErrorStat = Literal['ignore', 'warn', 'raise']

# --------------------------------------------------------------------------------


# In[4]:


class ConfigurationError(Exception):
    '''
    For generic exceptions when configurating IC accessories.
    '''
    
# --------------------------------------------------------------------------------


# In[5]:


@dataclass
class PortPluggedError(ConfigurationError):
    '''
    Raised when an eluent flow is pumped into a plugged (dead-end) accessory.
    '''
    
    accessory: BaseAccessory
    side: Literal['left', 'right'] =None
    
    def __str__(self) -> str:
        
        if self.side is None:
            msg = (
                f'''Plugged port of {self.accessory} detected '''
                f'''on a dynamic flow condition.''')
        else:
            msg = f'''{self.accessory}'s {self.side} port is not accessible.'''
            
        return msg
    
# --------------------------------------------------------------------------------


# In[6]:


@dataclass
class CycledFlowError(ConfigurationError):
    '''
    Raised when a loop is detected on the eluent flow.
    '''
    
    accessory: BaseAccessory
    cached_accessories: list[BaseAccessory] =None
    
    def __str__(self) -> str:
        
        msg = 'A loop formed in the IC flow configuration.'
        
        if self.cached_accessories:
            msg += (
                '''\nDetails: '''
                f'''{" -> ".join([repr(a) for a in self.cached_accessories])}'''
                f''' -> {self.accessory}''')
                
        return msg
        
# -------------------------------------------------------------------------------- 


# In[7]:


@dataclass
class MultiplePortError(ConfigurationError):
    '''
    Raised when a self-pumping accessory has more than one ports, or
        a spontaneous / active flow rate (._intrinsic_fr) is attempted to be set 
        onto a two-way accessory.
    '''
    
    def __str__(self) -> str:
        
        return '''Self-pumping flow rate cannot be set onto a two-way accessory.'''

# --------------------------------------------------------------------------------
    


# In[8]:


@dataclass
class CounterPumpingError(ConfigurationError):
    '''
    Raised when two self-pumping accessories are installed at opposite ends of
        the same eluent flow line.
    '''
    
    accessory: BaseAccessory
    
    def __str__(self) -> str:
        
        msg = (
            '''Two self-pumping forces detected in the oppisite '''
            f'''ends of {self.accessory}'s flow line.''')
        
        return msg
# --------------------------------------------------------------------------------


# In[9]:


@dataclass
class NoFlowError(ConfigurationError):
    '''
    Raised when trying accessing the flow related attributes of 
        an accessory while it is placed under a static flow condition (no flow rate).
    '''
    
    accessory: BaseAccessory
    attr_name: str
    
    def __str__(self) -> str:
        
        msg = (
            f'''{self.accessory}'s {self.attr_name} is not '''
            '''available on a static flow condition.''')
        
        return msg
    
# --------------------------------------------------------------------------------


# In[10]:


@dataclass
class DisAssembleError(ConfigurationError):
    '''
    Raised when an accessory is attempt to be disassembled from nothing.
    '''
    
    msg: str
        
    def __str__(self) -> str:
        
        return self.msg
# --------------------------------------------------------------------------------


# In[11]:


@dataclass
class InterLockError(ConfigurationError):
    '''
    Raised when dealing with two inter-locked accessories.
    '''
    
    accessories: tuple[BaseAccessory, BaseAccessory]
    extra_msg: str =''
        
    def __str__(self) -> str:
        
        a1, a2 = self.accessories
        msg = f'{a1} and {a2} are interlocked.'
        
        return (msg + ' ' + self.extra_msg).strip()
    
# --------------------------------------------------------------------------------


# In[12]:


@dataclass
class LockedValvePortError(ConfigurationError):
    '''
    Raised when the switching port is attempt to reset the bonded port to
        another accessory.
    '''
    
    accessory: BaseAccessory
    valve_port: BaseAccessory
    side: Literal['left', 'right']
    
    def __str__(self) -> str:
        
        a = self.accessory
        p = self.valve_port
        s = self.side
        msg = (
            f'''Attempt to set {a} to {p}'s {s} port. '''
            f'''However, {p} is locked to built-in {p.bonded} when '''
            f'''{p.valve} is set to '{p.valve.position}'.''')
        
        return msg

# --------------------------------------------------------------------------------
    


# In[13]:


@dataclass
class SuppressorOverVoltageError(ConfigurationError):
    '''
    Raised when a suppressor is placed under a static flow condition (no flow rate).
    '''
    accessory: BaseAccessory
        
    def __str__(self) -> str:
        
        msg = (
            f'''Cannot turn on {self.accessory} due to no flow!'''
            )
        
        return msg
        
# --------------------------------------------------------------------------------


# In[ ]:


@dataclass
class BackFlushError(ConfigurationError):
    '''
    Raised when backflush an accessory which is not supposed to
        be backflushed.
    '''
    accessory: BaseAccessory
        
    def __str__(self) -> str:
        
        msg = f'''Shouldn't backflush {self.accessory}!'''
        
        return msg
        
# --------------------------------------------------------------------------------


# In[14]:


class ProfileError(Exception):
    '''
    For generic exceptions when dealing with IC profiles, 
        including eluent profiles, column chemistry datasets, etc...
    '''

class IonError(ProfileError):
    '''
    ProfileErrors when dealing with ions.
    '''
    
# --------------------------------------------------------------------------------


# In[15]:


@dataclass
class DuplicatedIonError(IonError, ValueError):
    '''
    Raised when a profile contains more than one items of an identical ion.
    e.g., a profile containing items with key 'F[+]' and 'floride'.
    '''
    
    formula1: str
    formula2: str
        
    def __str__(self) -> str:
        
        f1, f2 = self.formula1, self.formula2
        msg = (
            f'''Items with the identical ion: "{f1}" and "{f2}".''')
        
        return msg
    
# --------------------------------------------------------------------------------


# In[16]:


@dataclass
class ConflictedIonError(IonError, ValueError):
    '''
    Raised when the input data of an ion is conflicted with that in the Ion Libaray
        database.
    '''
    formula1: str
    formula2: str
    kws: str | tuple[str, ...] | list[str]
    
    def __str__(self) -> str:
        
        f1, f2, kws = self.formula1, self.formula2, self.kws
        if isinstance(kws, str): kws = (kws,)
        msg = (
            f'''Conflicted {', '.join(kws)} between the input '''
            f'''formula/alias: {f1} and that in the Ion Library: {f2}.''')
        
        return msg
        
# --------------------------------------------------------------------------------
    


# In[17]:


@dataclass
class NeutralIonError(IonError, ValueError):
    '''
    Raised when attempt to parse a neutral formula into an ion.
    '''
    
    formula: str
        
    def __str__(self) -> str:
        
        msg = (
            f'''The input formula/alias {self.formula} '''
            '''indicates a neutral compound.''')
        
        return msg

# --------------------------------------------------------------------------------


# In[18]:


class OperationError(Exception):
    '''
    Base errors for improper operations for IC
    '''
    
# --------------------------------------------------------------------------------


# In[19]:


@dataclass
class InjectionError(OperationError):
    '''
    Raised when trying to inject a sample into an accessories with
        back pressure or that should not be mannually injected with 
        samples.
    '''
    accessories: BaseAccessory | tuple[BaseAccessory, ...] | (
        list[BaseAccessory])
        
    def __str__(self) -> str:
        
        a = self.accessories
        if isinstance(a, BaseAccessory):
            a = (a,)
        else:
            a = list(a)
            
        msg = (
            '''Cannot manually inject samples into '''
            f'''{", ".join([repr(_) for _ in a])}.''')
        
        return msg
    


# In[ ]:


__all__ = [
    'ConfigurationError',
    'PortPluggedError',
    'CycledFlowError',
    'MultiplePortError',
    'CounterPumpingError',
    'NoFlowError',
    'DisAssembleError',
    'InterLockError',
    'LockedValvePortError',
    'SuppressorOverVoltageError',
    'BackFlushError',
    'ProfileError',
    'IonError',
    'DuplicatedIonError',
    'ConflictedIonError',
    'NeutralIonError',
    'OperationError',
    'InjectionError',
    ]

