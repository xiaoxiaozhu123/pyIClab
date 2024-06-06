#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024  PyICLab Kai "Kenny" Zhang
'''
##### Built-in import #####

import re
from abc import ABC, abstractmethod

# --------------------------------------------------------------------------------


# In[2]:


class NamedTrinket:
    '''
    Base class for IC Modules.
    '''

    def __init__(self, name: str):
        
        self.name = str(name).strip()
    
    @property
    def name(self):
        
        return self._name
    
    @name.setter
    def name(self, name):
        
        if re.match(r'^[a-zA-Z0-9_\-]+$', name) is not None:
            self._name = name
        else:
            raise ValueError(
                '''Name contains invalid characters. '''
                '''Only alphanumeric characters, "-" & "_" are allowed.'''
                )

# --------------------------------------------------------------------------------


# In[3]:


class BaseAccessory:
    '''
    Base class for GenericAccessory.
    '''
# --------------------------------------------------------------------------------


# In[4]:


class BaseModel(ABC):
    '''
    A base class for IC models...
    '''
    
    @property
    @abstractmethod
    def _drop_when_updated(self) -> tuple:
        '''
        A place holder for parameters that should be
            removed from the model once status of the model
            has been changed. It should be recomputed when
            needed. Implement this method by @cached_property.
        '''
    
    @property
    @abstractmethod
    def _mutables(self) -> tuple:
        '''
        A place holder for parameters that should be able
            to be set mannually.
        '''
    
    @property
    @abstractmethod
    def unit_table(self) -> dict:
        
        pass
    
    @abstractmethod
    def standby(self):
        
        pass
    
    @abstractmethod
    def activate(self):
        
        pass
# --------------------------------------------------------------------------------


# In[5]:


class BaseConstructor(ABC):
    '''
    A base class for module constructors...
    '''
# --------------------------------------------------------------------------------


# In[6]:


class BaseIonChromatograph:
    '''
    A base class for IC...
    '''
# --------------------------------------------------------------------------------

