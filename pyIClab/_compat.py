#!/usr/bin/env python
# coding: utf-8

# In[8]:


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




