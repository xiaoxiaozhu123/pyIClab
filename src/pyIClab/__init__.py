#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Compatibility #####

import pyIClab._compat

##### Built-in import #####

##### External import #####


##### Local import #####

from pyIClab.interface import *
from pyIClab.ions import *
from pyIClab.ionchromatograph import *
from pyIClab.assemblies import *
from pyIClab.engines import *
from pyIClab.updates import UpdateLogger, updates

update_logger = UpdateLogger(updates)
# --------------------------------------------------------------------------------


# In[2]:


__author__ = 'Kenny Zhang'
__version__ = '2024.9.16'
__license__ = '''
    GNU GENERAL PUBLIC LICENSE
    Version 3, 29 June 2007
    '''
__repository__ = 'https://github.com/xiaoxiaozhu123/pyIClab'

