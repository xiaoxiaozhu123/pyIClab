#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[2]:


##### Built-in import #####
import os
import re
import warnings
from typing import Callable, Iterable, Literal
from copy import deepcopy
from io import StringIO
from importlib.resources import files

##### External import #####
import numpy as np
import pandas as pd
from pint import Unit, Quantity
from numpy import ndarray
from bs4 import BeautifulSoup

##### Local import #####

import pyIClab.ions as ions_
from pyIClab.units import ParseQuantities
from pyIClab._baseclasses import (
    NamedTrinket, BaseModel,
    )
from pyIClab.assemblies.generics import GenericAccessory
from pyIClab.beadedbag import is_notebook

# --------------------------------------------------------------------------------
_LAB_PATH = str(files('pyIClab'))
__last_modification__ = '2024-04-21'

# --------------------------------------------------------------------------------


# In[3]:


class GenericTubing(GenericAccessory):
    '''
    Represents an Accessory for modelling.
    '''
    
    def __init__(self, /, name: str, *,
        length: str | float | int,
        ID: str | float | int,
        ):
        
        super().__init__(name=name)
        self._length = ParseQuantities.parse_string_as(length, 'cm') * Unit('cm')
        self._ID = ParseQuantities.parse_string_as(ID, 'mm') * Unit('mm')
        
    def __new__(cls, *args, **kwargs):
        
        if cls is GenericTubing:
            raise TypeError('GenericTubing cannot be instantiated directly.')
        else:
            return super().__new__(cls)
        
    # --------------------------------------------------------------------------------
        
    @property
    def length(self) -> Quantity:
        
        return self._length
    
    @property
    def ID(self) -> Quantity:
        
        return self._ID
    
    @property
    def V(self) -> Quantity:
        '''
        The total volumn of the Column tube, in mL
        '''
        
        return ((0.5*self.ID)**2 * np.pi * self.length).to('mL')
    
    @property
    def phase_ratio(self) -> float:
        
        return np.float64(0.0)
    
    @property
    def φ(self) -> float:

        return np.float64(self.phase_ratio)
    
    @property
    def β(self) -> float:
        
        return np.float64(1 / self.phase_ratio)
    
    @property
    def Vm(self) -> Quantity:

        return self.V
    
    @property
    def Vs(self) -> Quantity:
        
        return 0.0 * Unit('mL')
    
    @property
    def Q(self) -> Quantity:
        
        return 0.0 * Unit('umol')
    
    # --------------------------------------------------------------------------------
    
    def flush(self, *,
        models: list[BaseModel],
        ):
         
        # ban inconsistant competing ions
        competing_ions = set([model.competing_ions for model in models]).pop()
        assert len(set([model.competing_ions for model in models])) == 1
        
        # ban duplicated analytes
        analytes = [model.analyte for model in models]
        assert len(analytes) == len(set(analytes))
        
        if ions_.get_ioninfo(competing_ions[0]).charge > 0:
            kind = 'cation'
        else:
            kind = 'anion'
        
        t0 = self.current_time.magnitude
        t1 = self.prev.current_time.magnitude
        input_datasets = deepcopy(self.prev.output)
        # If the prev.input contains no ion data
        # set the input concentration always = 0
        for model in models:
            analyte = model.analyte
            competing_ions = model.competing_ions
            for ion in (analyte, *competing_ions):
                input_datasets.setdefault(ion, lambda t: np.zeros_like(t))
                
        # set the input data to models
        for model in models:
            # in case that flow changed by valve switching
            # No need to drop dt, it's already done with _set_XX method
            model._set_fr(self.fr.to('mL/minute').magnitude)
            model._set_backflush(self.flow_direction==-1)
            
            analyte = model.analyte
            competing_ions = model.competing_ions
            dt = model.dt
            t = np.arange(t0, t1, dt)
            cAm = input_datasets[analyte](t)
            cEm = np.array([input_datasets[ion](t) for ion in competing_ions])
            model.input_data(cAm, cEm)
            
        # Run the models, which have already been set to the standing-by state
        for model in models:
            model.activate()
                
        # Collect and process model outputs, counterions will be added.
        output_datasets = {}
        output_competing_ion_collection = {}
        for model in models:
            analyte = model.analyte
            competing_ions = model.competing_ions
            dt = model.dt
            t = np.arange(t0, t1, dt)
            cA_output = model.output.cA
            cE_output = model.output.cE
            integral_A = model.integral_A # sum ammount of analyte
            output_datasets[analyte] = (
                lambda t, *, x=t, y=cA_output: np.interp(t, x, y))
            for i, ion in enumerate(competing_ions):
                output_competing_ion_collection.setdefault(ion, []).append((
                    integral_A * abs(ions_.get_ioninfo(analyte).charge),
                    lambda t, *, x=t, y=cE_output[i, :]: np.interp(t, x, y)))
            model.clear()
            
        for ion, output_data in output_competing_ion_collection.items():
            cum_charge = sum(tuple(zip(*output_data))[0])
            if cum_charge == 0:
                output_datasets[ion] = output_data[0][1]
                continue
            f = lambda t: np.zeros_like(t)
            for charge, f1 in output_data:
                f = lambda t, *, f0=f, f1=f1, charge=charge: f0(t) + charge*f1(t)
            f = lambda t, *, f0=f, cum_charge=cum_charge: f0(t) / cum_charge
            output_datasets[ion] = f
            
        # maintain the charge neutrality of output profile
        g = lambda t: np.zeros_like(t)
        for ion, f in output_datasets.items():
            charge = abs(ions_.get_ioninfo(ion).charge)
            g = lambda t, *, g=g, f=f, charge=charge: g(t) + charge*f(t)
        
        counterion = 'K[+1]' if kind == 'anion' else 'Cl[-1]'
        output_datasets[counterion] = g
        
        # merge the ion t-c funcs
        for ion, f1 in output_datasets.items():
            f0 = self.output.setdefault(ion, lambda t: np.zeros_like(t))
            self.output[ion] = self._update_func_time_series((t0, t1), f0, f1)
                
        self._cur_time = t1 * Unit('min')
            
            
# --------------------------------------------------------------------------------


# In[4]:


class IonExchanger(NamedTrinket):
    '''
    Represents an stationary phase used in IC.
    Containing retention information about different analytes.
    '''
    _unique = True
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /, name: str, **kwargs):
        '''
        Parameters
        -----------
        name: str.
            The name of the ion-exchanger.
        Qv: str | float | int | Quantity
            Volumetric ion exchange capacity of the ion-exchanger, in umol/mL.
            Defaults to 0.0.
        phase_ratio: float
            The phase ratio of the ion-exchanger under general experimental conditions. 
            Defaults to 0.0.
        retention_data: dict
            Retention database of the ion exchanger.
        '''
        
        super().__init__(name=name)
        
        Qv = kwargs.pop('Qv', None)
        if Qv is not None:
            self.Qv = ParseQuantities.parse_string_as(Qv, 'umol/mL') * Unit('umol/mL')
        else:
            self.Qv = 0.0 * Unit('umol/mL')
        
        self.phase_ratio = kwargs.pop('phase_ratio', 0.0)
        self.retention_data = kwargs.pop('retention_data', {})
        
        if kwargs:
            raise AttributeError(
                    f'''{type(self)}.__init__ got '''
                    f'''{"an unexpected keyword augment" if len(
                        kwargs) == 1 else "unexpected keyword augments"}: '''
                    f'''{', '.join([f"'{attr}'" for attr in kwargs.keys()])}'''
                    )
 
    # --------------------------------------------------------------------------------
   
    @property
    def φ(self) -> float:
        '''
        Alias of phase_ratio.
        '''
        return self.phase_ratio
    
    @property
    def β(self) -> float:
        '''
        Reciprocal of phase_ratio.
        '''
        return 1 / self.φ
    
    @staticmethod
    def _get_kdata(competing_ions, df):
        
        df = df[[*competing_ions, 'analyte', 'k']]
        df = df.dropna()
        df = df.groupby([*competing_ions, 'analyte']).agg(
            {'k': lambda v: 10**(np.mean(np.log10(v)))}
            )
        
        return df.reset_index()
    
    @property
    def kdata(self) -> dict:
        
        return {competing_ions: self._get_kdata(competing_ions, df) for (
            competing_ions, df) in self.retention_data.items()}
    
    @staticmethod
    def _get_Hdata(competing_ions, df):
        
        df = df[['fr', *competing_ions, 'analyte', 'H']]
        df = df.dropna()
        df = df.groupby(['fr', *competing_ions, 'analyte']).agg({'H': 'mean'})
        
        return df.reset_index()
    
    @property
    def Hdata(self) -> dict:
        
        return {competing_ions: self._get_Hdata(competing_ions, df) for (
            competing_ions, df) in self.retention_data.items()}
        
    # --------------------------------------------------------------------------------
    
    @staticmethod
    def _convert_data_points_with_standard_formulas(data_points, escape_kws):
        
        if not isinstance(data_points, dict | pd.DataFrame):
            return data_points
        else:
            df = pd.DataFrame(data_points)
            input_ions = [c for c in df.columns if c not in escape_kws]
            ions_.check_duplicated_ions_from_sequence(input_ions)
            lookup_table = {ion: ions_.get_ioninfo(ion).formula for ion in input_ions}
            return df.rename(columns=lookup_table)
    
    def update_retention_data(self, /,
        data_points: ndarray | Iterable | dict | pd.DataFrame,
        *,
        competing_ions: tuple[str, ...],
        analyte: str,
        fr: str | int | float ='1.0 mL/min',
        ) -> None:
        
        competing_ions = tuple(sorted(
            ions_.get_ioninfo(ion).formula for ion in competing_ions))
        ions_.check_duplicated_ions_from_sequence(competing_ions)
        analyte = ions_.get_ioninfo(analyte).formula
        fr = ParseQuantities.parse_string_as(fr, 'mL/minute')
        data_points = self._convert_data_points_with_standard_formulas(
            data_points, ['k', 'H'])
        
        # convert the input data points into a DataFrame
        df = pd.DataFrame(
            data=data_points,
            columns=[*competing_ions, 'k', 'H']
            )
        df['fr'] = fr
        df['analyte'] = analyte
        dtype = {'fr': np.float64, 'analyte': 'string', 'k': np.float64,
            'H': np.float64,}
        dtype.update({ion: np.float64 for ion in competing_ions})
        df = df.astype(dtype=dtype)
        
        template = pd.DataFrame(columns=['fr', *competing_ions, 'analyte', 'k', 'H'])
        template = template.astype(dtype=dtype)
        df0 = self.retention_data.setdefault(competing_ions, template)
        # remove the stale data points for the analyte
        df0 = df0[~((df0['fr']==fr) & (df0['analyte']==analyte))]
        # update it with the new DataFrame
        df = pd.concat((df0, df), ignore_index=True).reset_index(drop=True)
        self.retention_data[competing_ions] = df
        
    # --------------------------------------------------------------------------------

    def _get_path(self, fname=None, directory=None) -> str:
        
        fname = f'{self.name}.dat' if fname is None else fname
        directory = f'{_LAB_PATH}//db//ion_exchangers/' if directory is None else directory
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        return directory + '/' + fname
    
    # --------------------------------------------------------------------------------
    
    def _dump_name(self) -> str:
        
        return f'<name>\n{self.name}\n</name>'
    
    def _dump_Qv(self) -> str:
        
        return f'<Qv>\n{self.Qv}\n</Qv>'
        
    def _dump_phase_ratio(self) -> str:
        
        return f'<phase_ratio>\n{self.phase_ratio}\n</phase_ratio>'
    
    def _dump_retention_data(self) -> str:
        
        strings = []
        for competing_ions, df in self.retention_data.items():
            
            df = df.copy()
            columns = ['fr', *competing_ions, 'analyte', 'k', 'H']
            df['k'] = df['k'].apply(lambda col: '{:.10f}'.format(col))
            df['H'] = df['H'].apply(lambda col: '{:.5f}'.format(col))
            data = df[columns].to_string(index=False)
            op_tag = (
                f'''<retention_table competing_ions="{' '.join(competing_ions)}">''')
            cl_tag = '</retention_table>'
            strings.append(f'{op_tag}\n{data}\n{cl_tag}')
            
        retention_tables = '\n'.join(strings)
        
        return f'<retention_data>\n{retention_tables}\n</retention_data>'
    
    def dump(self, /,
        fname: str =None,
        *,
        directory: str =None,
        comments: str =None,
        ) -> None:
        '''
        Dump the IonExchanger instance to a text file for storage.
    
        Parameters:
        -----------
        fname : str, optional
            The name of the pickle file to be saved. Defaults to the name of the
            IonExchanger instance with a ".dat" extension.
        directory : str, optional
            The directory where the pickle file should be saved. Defaults to 
            '..//db//ion_exchangers//'.
        comments: str, optional
            Add comments to the datasets.
        '''
        
        path = self._get_path(fname=fname, directory=directory)
        attrs = ['name', 'Qv', 'phase_ratio', 'retention_data']
        
        if comments is None:
            comments = ''
        else:
            comments = ''.join(f'# {line}' for line in comments.splitlines())
        
        data = comments + '\n' + (
            '\n'.join(getattr(self, f'_dump_{attr}')() for attr in attrs))
        
        with open(path, mode='w', encoding='utf-8') as f:
            f.write(data.strip())
    
    # --------------------------------------------------------------------------------
    
    @staticmethod
    def _load_name(bs: BeautifulSoup) -> str:
        
        return bs.find('name').string.strip()
    
    @staticmethod
    def _load_Qv(bs: BeautifulSoup) -> float:
        
        return ParseQuantities.parse_string_as(bs.qv.string, 'umol/mL')
    
    @staticmethod
    def _load_phase_ratio(bs: BeautifulSoup) -> float:
        
        return float(bs.phase_ratio.string)
        
    @staticmethod
    def _load_retention_data(bs: BeautifulSoup) -> dict:
        
        retention_data = {}
        for tag in bs.retention_data.find_all('retention_table'):
            
            competing_ions = tag.get('competing_ions').split()
            competing_ions = tuple(sorted(
                ions_.get_ioninfo(ion).formula for ion in competing_ions))
            ions_.check_duplicated_ions_from_sequence(competing_ions)
            retention_table = tag.string
            data_io = StringIO(retention_table)
            df = pd.read_csv(data_io, sep='\s+')
            for column in df.columns:
                try:
                    formula = ions_.get_ioninfo(column).formula
                except ValueError:
                    pass
                else:
                    df = df.rename(columns={column: formula})
            dtype = {'fr': np.float64, 'analyte': 'string', 'k': np.float64,
                'H': np.float64,}
            dtype.update({ion: np.float64 for ion in competing_ions})
            df['analyte'] = [ions_.get_ioninfo(ion).formula for ion in df['analyte']]
            df = df[['fr', *competing_ions, 'analyte', 'k', 'H']]
            df = df.astype(dtype=dtype)
            retention_data[competing_ions] = df
        
        return retention_data
    
    @classmethod
    def load(cls, /, fname: str, *, directory: str =None):
        '''
        Loads an ion exchanger instance from a file.
        Parameters:
        ----------
        fname: str
        directory: str, optional
            The path of the directory. If not specified, a default directory path
            '.../db/ion_exchangers/' is used.
        '''
        
        directory = f'{_LAB_PATH}//db//ion_exchangers/' if directory is None else directory
        path = directory + '/' + fname
        
        with open(path, mode='r', encoding='utf-8') as f:
            data = f.read()
            
        # discard comments and empty lines
        data = re.sub(r'#.*\n', '\n', data)
        data = '\n'.join(line.strip() for line in data.splitlines() if line.strip())
        
        # parse data using BeautifulSoup
        bs = BeautifulSoup(data, 'html.parser')
        
        attrs = ['name', 'Qv', 'phase_ratio', 'retention_data']
        params = {attr: getattr(cls, f'_load_{attr}')(bs) for attr in attrs}
            
        return cls(**params)
                     
# --------------------------------------------------------------------------------


# In[5]:


class Column(GenericTubing):
    '''
    Represents a realistic homogenous column used in IC.
    '''
    __identifier__ = 'column'
    _unique = True
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /, name: str, *,
        length: str | float | int =None,
        ID: str | float | int =None,
        ):
        '''  
        Initiate an empty column tude ..
        Parameters:
        -----------
        name: str, optional
            The name or identifier for the column. Defaults to 'Nameless'.
        length : str | float | int, optional
            The length of the column. Defaults to 15 cm.
        ID: str | float | int, optional
            The inner diameter (ID) of the column. Defaults to 4.6 mm.
            
        properties:
        -----------
        sp: IonExchanger. 
            The stationary phase in the column, containing the information
            about the chromatographic chemistry of the ion-exchanger.
            Defaults to None; you can not set a value to .sp when initalizing,
            use 'pack' method later on.
        '''
        
        length = '15.0 cm' if length is None else length
        ID = '4.6 mm' if ID is None else ID
        
        super().__init__(name, length=length, ID=ID)
        
    # --------------------------------------------------------------------------------
    # Properties
    
    @property
    def sp(self) -> IonExchanger:
        
        return self._sp
    
    @sp.setter
    def sp(self, ion_exchange):
        
        if hasattr(self, '_sp'):
            raise AttributeError('Only an empty column can be packed.')
        elif not isinstance(ion_exchange, IonExchanger | type(None)):
            raise TypeError(
                'Only an ion-exchanger can be packed into the column.')
        else:
            self._sp = ion_exchange
    
    @property
    def V(self) -> Quantity:
        '''
        The total volumn of the Column tube, in mL
        '''
        
        return ((0.5*self.ID)**2 * np.pi * self.length).to('mL')
    
    @property
    def phase_ratio(self) -> float:
        '''
        The phase ratio (Vs/Vm) of the column.
        '''
        return self.sp.phase_ratio
    
    @property
    def Vm(self) -> Quantity:
        '''
        The dead volumn of the Column, in mL
        '''
        return self.V / (1+self.φ)
    
    @property
    def Vs(self) -> Quantity:
        '''
        The volumn of the Column SP, including the substrate.
        '''
        
        return self.V - self.Vm
    
    @property
    def Q(self) -> Quantity:
        '''
        ion-exchange capacity.
        '''
        
        return self.sp.Qv * self.Vs
    
    # --------------------------------------------------------------------------------
    
    def __repr__(self):
        
        return (
            f'''<Column "{self.name}" ('''
            f'''{self.ID.to('mm').magnitude:.1f} × '''
            f'''{self.length.to('mm').magnitude:.0f} mm)>'''
            )
    
    # --------------------------------------------------------------------------------
    
    def pack(self, ion_exchanger: IonExchanger):
        
        self.sp = ion_exchanger
        
# --------------------------------------------------------------------------------       


# In[6]:


class PEEKTubing(GenericTubing):
    
    '''
    Represents a PEEK tubing in IC.
    '''
    __identifier__ = 'PEEKtubing'
    _unique = False
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /, name: str =None, *, 
        length: str | float | int =None,
        ID: str | float | int =None,
        ): 
        '''
        '''
        name = 'PEEK' if name is None else name
        length = '5.0 cm' if length is None else length
        ID = '0.18 mm' if ID is None else ID
        super().__init__(name=name, length=length, ID=ID)
        
    # --------------------------------------------------------------------------------
    
    def __repr__(self):

        return (
            f'''<Tubing "{self.name}" ('''
            f'''{self.ID.to('mm').magnitude:.1f} × '''
            f'''{self.length.to('mm').magnitude:.0f} mm)>'''
            )
    
    # --------------------------------------------------------------------------------
    
    def __add__(self, other):
        
        if type(other) != type(self):
            raise TypeError(
                f'''Cannot join {type(other)} to {type(self)}.''')
        if self.ID != other.ID:
            raise ValueError(
                f'''Cannot join two PEEK tubings with different IDs.'''
                )
        
        tubing = deepcopy(self)
        tubing.length += other.length
        
        return tubing

# --------------------------------------------------------------------------------


# In[7]:


class SampleLoop(PEEKTubing):
    
    __identifier__ = 'loop'
    _unique = True
    
    def __init__(self, /, name: str, V: str | float | int =None) -> None:
        
        V = '25 uL' if V is None else V
        V = ParseQuantities.parse_string_as(V, 'uL') * Unit('uL')
        ID = 0.75 * Unit('mm')
        length = V / (ID/2)**2 / np.pi
        super().__init__(name=name, length=length, ID=ID)
        
    def __repr__(self) -> str:
        
        return f'<Loop "{self.name}" {np.round(self.V.to("uL").magnitude):.0f} μL>'
    
# --------------------------------------------------------------------------------


# In[8]:


class Dummy:
    
    @property
    def _c(self):
        
        return [15., 20., 25., 30., 15., 20., 25., 30., 15., 20., 25., 30., 15.,
            20., 25., 30., 15., 20., 25., 30., 15., 20., 25., 30., 15., 20.,
            25., 30.]
    
    # --------------------------------------------------------------------------------
    
    @property
    def _analyte(self):
        
        return ['F[-1]', 'F[-1]', 'F[-1]', 'F[-1]', 'CH3COO[-1]', 'CH3COO[-1]',
           'CH3COO[-1]', 'CH3COO[-1]', 'Cl[-1]', 'Cl[-1]', 'Cl[-1]', 'Cl[-1]',
           'NO2[-1]', 'NO2[-1]', 'NO2[-1]', 'NO2[-1]', 'Br[-1]', 'Br[-1]',
           'Br[-1]', 'Br[-1]', 'NO3[-1]', 'NO3[-1]', 'NO3[-1]', 'NO3[-1]',
           'SO4[-2]', 'SO4[-2]', 'SO4[-2]', 'SO4[-2]']
    
    # --------------------------------------------------------------------------------
    
    @property
    def _k(self):
        
        return [
            0.9915157449191228, 0.7225485016429176, 0.5652830852945626,
            0.4625584508188677, 1.1617272540312975, 0.8588523425678186,
            0.6794585858536988, 0.5610772794854665, 2.704878914030833,
            2.022831483381597, 1.6146581582204178, 1.3430974752949507,
            3.851852547154886, 2.897212193129702, 2.3229474829803913,
            1.9393221501788611, 5.590241450962446, 4.216873803277284,
            3.3885881791469865, 2.834139156759271, 6.5413841384243865,
            4.920172216817286, 3.9449308179734355, 3.2934415316197123,
            9.198778916853527, 5.056588457957548, 3.178957940376229,
            2.17564378370687
            ]
    
    # --------------------------------------------------------------------------------
        
    @property
    def _H(self):
        
        return [0.06, 0.0375, 0.0417, 0.0472, 0.04, 0.035, 0.042, 0.045,
           0.0223, 0.0241, 0.0266, 0.0296, 0.0222, 0.0234, 0.0253, 0.0309,
           0.0197, 0.021, 0.0217, 0.0337, 0.0198, 0.0217, 0.0232, 0.0265,
           0.0346, 0.0337, 0.0372, 0.0625]
    
    # --------------------------------------------------------------------------------
    
    @classmethod
    def AnionExchanger(cls):
        
        dummy = cls()
        df = pd.DataFrame(data={
            'fr': 1.0, 
            'OH[-1]': dummy._c,
            'analyte': dummy._analyte,
            'k': dummy._k,
            'H': dummy._H},
            )
        
        df = df.astype({
            'fr': np.float64,
            'OH[-1]': np.float64,
            'analyte': 'string',
            'k': np.float64,
            'H': np.float64}
            )
        
        return IonExchanger('Dummy', 
            Qv='100 umol/mL', phase_ratio=0.99, retention_data={('OH[-1]',): df})
    
    # --------------------------------------------------------------------------------
    @classmethod
    def Column(cls):
        
        
        column = Column('Dummy')
        column.pack(cls.AnionExchanger())
        
        return column
    
    


# In[9]:


__all__ = [
    'Column', 'IonExchanger',
    'Dummy', 'SampleLoop', 'PEEKTubing',
    ]   


# In[ ]:




