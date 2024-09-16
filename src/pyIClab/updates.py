#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024  PyICLab Kai "Kenny" Zhang
'''


# In[2]:


##### Built-in import #####

from functools import namedtuple

##### External import #####

import pandas as pd

##### Local import #####

from pyIClab.utils.beadedbag import singleton

# --------------------------------------------------------------------------------


# In[3]:


UpdateInfor = namedtuple(
    'UpdateInfor',
    ['date', 'version', 'new_features', 'bug_fixes', 'improvements', 'remarks']
    )

# --------------------------------------------------------------------------------


# In[4]:


@singleton
class UpdateLogger(pd.DataFrame):
    
    def __init__(self, /, updates: list[UpdateInfor]):
        
        data = dict(
            date=[update.date for update in updates],
            version=[update.version for update in updates],
            new_features=[
                '; '.join(update.new_features).strip() for update in updates
                ],
            bug_fixes=[
                '; '.join(update.bug_fixes).strip() for update in updates
                ],
            improvements=[
                '; '.join(update.improvements).strip() for update in updates
                ],
            remarks=[
                '; '.join(update.remarks).strip() for update in updates
                ],
            )
        
        super().__init__(data=data)
        self.sort_values(by='version', inplace=True)
        
# --------------------------------------------------------------------------------


# In[5]:


updates = [
    UpdateInfor(
        date='2024-4-19',
        version='2024.4.19',
        new_features=[],
        bug_fixes=[],
        improvements=[],
        remarks=['Initial release'],
        ),
    UpdateInfor(
        date='2024-5-8',
        version='2024.5.8',
        new_features=[],
        bug_fixes=['Fixed issue: garbage collection for PHREEQPYTHON'],
        improvements=[
            'Improved effieciency for kmap',
            '`scipy.optimize.newton_krylov` instead of `fsolve` for non-linear eqution'
            ],
        remarks=['Inner release'],
        ),
    UpdateInfor(
        date='2024-6-6',
        version='2024.6.6',
        new_features=['Update readme file'],
        bug_fixes=['Fixed small bugs'],
        improvements=[],
        remarks=['Public on Github'],
        ),
    UpdateInfor(
        date='2024-6-6',
        version='2024.6.6.1',
        new_features=[],
        bug_fixes=['Fixed small bugs'],
        improvements=[],
        remarks=[],
        ),
    UpdateInfor(
        date='2024-9-16',
        version='2024.9.16',
        new_features=[],
        bug_fixes=[
            'Deprecated: `Ion.organic`',
            'Fixed small bugs',
            ],
        improvements=['Moved all helper funcs to pyIClab.utils',],
        remarks=[],
        ),
    ]

