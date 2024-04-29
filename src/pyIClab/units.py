#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[2]:


##### External import #####

import pint


# In[3]:


class ParseQuantities:
    
    @staticmethod
    def parse_string_as(
        q: pint.Quantity | float | int, default_unit: pint.Unit | str) -> float:
        
        q = pint.Quantity(q)
        
        if q.dimensionless:
            return q.magnitude
        else:
            return q.to(default_unit).magnitude
    
    # --------------------------------------------------------------------------------
    
    @classmethod
    def as_min(cls, t):
        
        return cls.parse_string_as(t, 'min')
    
    @classmethod
    def as_mM(cls, c):
    
        return cls.parse_string_as(c, 'mM')
    


# In[4]:


__all__ = [
    'ParseQuantities',
    ]

