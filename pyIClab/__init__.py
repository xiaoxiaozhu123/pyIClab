#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Compatibility adaption #####

import pyIClab._compat

##### Built-in import #####

from datetime import datetime

##### External import #####

import seaborn as sns

##### Local import #####

from pyIClab.interface import *
from pyIClab.ions import *
from pyIClab.ionchromatograph import *
from pyIClab.assemblies import *
from pyIClab.engines import *

# --------------------------------------------------------------------------------
sns.set()


# In[4]:


__version__ = '{:%Y.%m.%d}'.format(datetime(2024, 4, 19))

