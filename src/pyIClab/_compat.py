#!/usr/bin/env python
# coding: utf-8

# In[1]:


# phreeqpython compat
import platform
if platform.machine().startswith("arm64") and platform.system().startswith("Darwin"):
    raise OSError(
        '''Unsupported architecture: "arm64".\n'''
        '''To use PyICLab on Apple Silicon, you are advised to '''
        '''built your python3 environment with an x86 '''
        '''version of conda/miniconda.'''
        )

# --------------------------------------------------------------------------------


# In[2]:


# --------------------------------------------------------------------------------
# Bug fix when importing pyEQL
import monty.io
zopen = monty.io.zopen
monty.io.zopen = (
    lambda *args, **kwargs: zopen(
        *args, encoding='utf-8', **{k: v for k, v in kwargs.items() if k != 'encoding'}))
import pyEQL
monty.io.zopen = zopen
        
# --------------------------------------------------------------------------------


# In[ ]:




