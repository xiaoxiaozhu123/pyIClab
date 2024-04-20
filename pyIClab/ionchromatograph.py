#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Built-in import #####

import warnings
from typing import Literal
from collections import namedtuple

##### External import #####

import pandas as pd
import numpy as np
from pint import Unit, Quantity

##### Internal import #####

import pyIClab.ions as ions_
from pyIClab.units import ParseQuantities
from pyIClab._baseclasses import (
    NamedTrinket, BaseConstructor, BaseIonChromatograph,
    )
from pyIClab.interface import (
    DSMConstrutorForTubings, DSM_SEConstrutor, DSM_CEConstrutor
    )
from pyIClab.errors import (
    ProfileError, InjectionError,
    )
from pyIClab.beadedbag import Progressor
from pyIClab.assemblies.generics import GenericAccessory, Side
from pyIClab.assemblies.eluents import Eluent
from pyIClab.assemblies.injections import (
    SwitchingValve, SwitchingPort, SampleLoop,
    )
from pyIClab.assemblies.columns import (
    GenericTubing, Column, PEEKTubing,
    )
from pyIClab.assemblies.signals import (
    Detector, Suppressor,
    )

# --------------------------------------------------------------------------------


# In[2]:


_ICModule = (
    Eluent | SwitchingValve | Column | Detector | PEEKTubing | Suppressor)
_IA = _IndividualAccessory = (
    Eluent | Column | Detector | SwitchingPort | PEEKTubing | Suppressor)
_GA = _GeneralAccessory = _ICModule | _IA
_Injectables = PEEKTubing | SwitchingPort
_MethodParser = namedtuple('MethodPaser', ['method', 'args', 'kwargs'])

class IonChromatograph(BaseIonChromatograph, NamedTrinket):
    '''
    Class representing an Ion Chromatograph.
    '''
    _parse_commands = {
        SwitchingValve.__identifier__: {
            'SWITCH': _MethodParser('switch', (), {}),
            'LOAD': _MethodParser('switch', ('LOAD',), {}),
            'INJECT': _MethodParser('switch', ('INJECT',), {})
            }
        }
    
    _constructor_registry = {
        'DSM_SEConstrutor'.lower(): DSM_SEConstrutor,
        'DSM_SE'.lower(): DSM_SEConstrutor,
        'DSM_SimpleEquilibriums'.lower(): DSM_SEConstrutor,
        'DSM_SEQ'.lower(): DSM_SEConstrutor,
        'DSM_CEConstrutor'.lower(): DSM_CEConstrutor,
        'DSM_CE'.lower(): DSM_CEConstrutor,
        'DSM_CompleteEquilibriums'.lower(): DSM_CEConstrutor,
        'DSM_CEQ'.lower(): DSM_CEConstrutor,
        'GenericDiscontinousSegmentedModel'.lower(): DSMConstrutorForTubings,
        }
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /,
        name: str,
        competing_ions: list[str] | tuple[str, ...],
        lockon: _GA  | tuple | list | set =None,
        *,
        reset_valves: bool =True,
        ):
        '''
        Parameters:
        -----------
        name : str
            The name of the ion chromatograph.
        competing_ions: list[str] | tuple[str, ...]
            competing ions (eluent type) for IC system.
            must be a tuple or a list, e.g., ('hydroxide',) or ('CO3-2', 'HCO3-1',)
        lockon : A General Accessory for IC, or a collection of them.
            Specifies the modules or accessories to be locked onto the Ion Chromatograph.
            Defaults to None. If None, pyIClab will look into the global namespace for all
            the variables that are General Accessories and lock them onto the IC instance.
            General Accessories representing a group of Ion Chromatograph modules or accessories.
            including Eluent, SwitchingValve, Column, Suppressor, Detector 
            and SwitchingPort.
        reset_valves: bool, optional.
            If True, set all the valves to 'LOAD' when initializing. Defaults to True.
        '''
        
        super().__init__(name=name)
        
        self._convert_competing_ions_and_check_validity(competing_ions)
        
        self._locked_modules = self._to_modules(lockon)
        assert all(isinstance(m, _ICModule) for m in self.locked_modules)
        self._check_namespace()
        
        self._init_dynamic_props()
        
        if reset_valves:
            self._init_valves()
        self._valve_init_positions = {}
    
        self.reset()
        
        # cached settings, accessories / modules / analyte in these settings not 
        # necessarily parts of this IC system
        # IC will filter cached settings by its properties.
        
        self._cached_injection_profile = {}
        self._cached_designated_Ndict = {}
        self._cached_commands = {} # Dict[Valve, tuple(float, str)]
        self._cached_model_constructor_types = {}
        self._cached_models = {}
 
    # --------------------------------------------------------------------------------
    
    def _convert_competing_ions_and_check_validity(self, competing_ions):
        
        ions_.check_duplicated_ions_from_sequence(competing_ions)
        self._competing_ions = tuple(
            ions_.get_ioninfo(ion).formula for ion in competing_ions)
        charges = np.array([ions_.get_ioninfo(ion).charge for ion in self._competing_ions])
        for charge in charges[1:]:
            if charge * charges[0] < 0:
                raise ProfileError(
                    'Competing ions of IC must be either anions or cations.'
                    )
                
    # --------------------------------------------------------------------------------
    
    def _init_dynamic_props(self) -> None:
        
        for type_ in _ICModule.__args__:
            setattr(
                type(self),
                f'{type_.__identifier__}s',
                property(fget=lambda self, type_=type_: {
                    m for m in self.modules if isinstance(m, type_)})
                )
    
    # --------------------------------------------------------------------------------
    
    def _init_valves(self):
        
        for valve in [v for v in self.modules if isinstance(v, SwitchingValve)]:
            valve.switch('LOAD')
    
    # --------------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        
        return f'<IC System "{self.name}">'
    
    # --------------------------------------------------------------------------------
    
    def reset(self):
        
        for accessories in self.accessories:
            accessories.reset()
            
        for v in self.valves:
            position = self._valve_init_positions.setdefault(v, v.position)
            v.switch(position)
            
        self._cur_time = 0.0 * Unit('min')
    
    # --------------------------------------------------------------------------------
    
    def clear(self):
        
        for attr in self.__dict__.keys():
            if attr.startswith('_cached'):
                container = getattr(self, attr)
                assert isinstance(container, dict)
                container.clear()
    
    # --------------------------------------------------------------------------------
    
    def _synchronize(self, t: str | float | int =None) -> None:
        
        if t is not None:
            t = ParseQuantities.as_min(t)
            assert t >= 0
            self._cur_time = t * Unit('min')
            
        for a in self.accessories:
            a._cur_time = self.current_time
    
    # --------------------------------------------------------------------------------
    
    def _to_modules(self, accessories) -> set[_ICModule]:
        '''
        Convert accessories to a set of IC modules.
        '''
        
        if accessories is None:
            return {
                a for a in globals().values() if isinstance(a, _ICModule)}
        
        if isinstance(accessories, _GA):
            accessories = {accessories,}
        else:
            accessories = set(accessories)
        
        ports = {p for p in accessories if isinstance(p, SwitchingPort)}
        valves = {p.valve for p in ports}
        
        return (accessories-ports) | valves

    # --------------------------------------------------------------------------------
    
    def _to_accessories(self, modules) -> set[GenericAccessory]:
        '''
        Convert IC modules to a set of Generic Accessories.
        '''
        
        if modules is None:
            return {
                a for a in globals().values() if isinstance(a, _IA)}
        
        if isinstance(modules, _GA):
            modules = {modules,}
        else:
            modules = set(modules)
            
        valves = {v for v in modules if isinstance(v, SwitchingValve)}
        valves |= {p.valve for p in modules if isinstance(p, SwitchingPort)} ##
        ports = {p for v in valves for p in v}
        
        return (modules-valves) | ports
    
    # --------------------------------------------------------------------------------
    
    def _check_namespace(self):
        
        df = self.namespace[['type_identifier', 'name']].copy()
        df.loc[:, 'name'] = df['name'].str.upper()
        counter = df.groupby(by=['type_identifier', 'name']).value_counts()
        for (type_, name), count in counter.items():
            if count > 1:
                raise ProfileError(
                    '''Got IC modules with duplicated names: '''
                    f'''{type_.upper()} - "{name}"'''
                    )
        
    @property
    def namespace(self) -> pd.DataFrame:
        
        unique_modules = [
            m for m in self.modules if hasattr(type(m), '_unique') and type(m)._unique]
        data = [[m.__identifier__, m.name, m] for m in unique_modules]
        df = pd.DataFrame(
            data=data, columns=['type_identifier', 'name', 'module_instance'])
        
        return df.sort_values(by='type_identifier', ignore_index=True)
    
    # --------------------------------------------------------------------------------
    
    def from_name(self, name: str) -> _ICModule:
        '''
        Find the IC Module from its name.
        name: str, case-insenstive.
            Can be just the name if it is unique in the IC system.
            If not unique, type of module must be specified before name using
            a whitespace as the delimiter.
            e.g., "Valve column", where "column" is the name of a valve.
        '''
        df = self.namespace
        
        name_passed = name
        name = name.strip().lower()
        df.loc[:, 'name'] = df['name'].str.lower()
        condition = df['name']==name
        count = len(df[condition])
        if count == 1:
            return df.loc[condition, 'module_instance'].to_list()[0]
        elif count > 1:
            raise ValueError(
                f'''Found multiple modules named "{name_passed}". '''
                '''Specify the module type before name.\n'''
                f'''Types: {', '.join(self.namespace['type_identifier'].unique())}.'''
                )
        elif ' ' not in name:
            raise ValueError(
                f'''Could not find a module named "{name_passed}".''')
        else:
            type_ = name[:name.index(' ')]
            name_ = name[name.index(' '):].strip()
            condition = (df['type_identifier']==type_) & (df['name']==name_)
            count = len(df[condition])
            if count == 1:
                return df.loc[condition, 'module_instance'].to_list()[0]
            else:
                assert not count
                raise ValueError(
                    f'''Could not find a module named "{name_passed}".''')
                
    # --------------------------------------------------------------------------------
    
    @property
    def Ntable(self) -> pd.DataFrame:
                
        tp = template = self.namespace
        tp = tp[tp['type_identifier']=='column'].copy()
        for analyte in self.analytes:
            tp[analyte] = pd.NA
            
        for column, Ndict in self._cached_designated_Ndict.items():
            for analyte, N in Ndict.items():
                if analyte in self.analytes:
                    tp.loc[tp['module_instance']==column, analyte] = N
        tp.loc[:, list(self.analytes)] = tp.loc[:, list(self.analytes)].astype('Int64')
                
        return tp
    
    # --------------------------------------------------------------------------------
    
    @property
    def injection_table(self) -> pd.DataFrame:
        
        tp = template = pd.DataFrame(columns=['accessory'])
        
        # update the table with cached injection data
        for accessory, injection_dict in self._cached_injection_profile.items():
            if accessory in self.accessories:
                tp.loc[len(tp.index), 'accessory'] = accessory
                for ion, c in injection_dict.items():
                    if ion in tp.columns:
                        tp.at[len(tp.index)-1, ion] = c
                    else:
                        tp[ion] = np.nan
                        tp.at[len(tp.index)-1, ion] = c
                        
        # remove the injection data if accessory under an eluent flow
        tp = tp[~tp['accessory'].isin([a for fl in self.flow_lines for a in fl])]
        
        return tp

    # --------------------------------------------------------------------------------
    @property
    def ions(self) -> tuple:
        
        ions = {
            ion for ion in self.injection_table.columns if ion != 'accessory'
            } | {
            ion for eluent in self.eluents for ion in eluent.ions}
        
        return tuple(ions)
    
    @property
    def anions(self) -> tuple:
        
        return tuple(ion for ion in self.ions if ions_.get_ioninfo(ion).charge < 0)
    
    @property
    def cations(self) -> list:
        
        return tuple(ion for ion in self.ions if ions_.get_ioninfo(ion).charge > 0)
    
    @property
    def competing_ions(self) -> tuple:
        
        return self._competing_ions
    
    @property
    def analytes(self) -> tuple:
        
        return tuple(
            ion for ion in getattr(self, f'{self.kind}s') if ion not in self.competing_ions)
    
    # --------------------------------------------------------------------------------
    
    @property
    def kind(self) -> Literal['anion', 'cation']:
        
        charge = ions_.get_ioninfo(self.competing_ions[0]).charge
        if charge > 0:
            return 'cation'
        else:
            assert charge < 0
            return 'anion'
    
    # --------------------------------------------------------------------------------
    
    @property
    def locked_modules(self) -> set[_ICModule]:
        
        return self._locked_modules
    
    # --------------------------------------------------------------------------------
    
    @property
    def current_time(self):
        
        return self._cur_time
    
    # --------------------------------------------------------------------------------
    
    @property
    def locked_accessories(self) -> set[_IA]:
        
        return self._to_accessories(self.locked_modules)
    
    # --------------------------------------------------------------------------------  
        
    @property
    def accessories(self) -> set[_IA]:
        
        return self._roam(self.locked_accessories, set())
    
    @property
    def modules(self) -> set[_ICModule]:
        
        return self._to_modules(self.accessories)
    
    def _roam(self, queque, cache) -> set[GenericAccessory]:
        
        if not queque:
            return cache
        
        for a in queque.copy():
            if a not in cache:
                line = set(a.from_left())
                cache |= line
                queque |= self._to_accessories(line) - cache
                queque -= line

        return self._roam(queque, cache)
                
    # --------------------------------------------------------------------------------
    
    @property
    def lines(self) -> list[list[GenericAccessory]]:
        
        return self._roam_for_lines(self.accessories, [])
    
    @property
    def flow_lines(self) -> list[list[GenericAccessory]]:
        
        return [l[0].flow_line for l in self.lines if l[0].flow_direction]
    
    def repr_lines(self) -> None:
        
        msglist = [
            f'Line #{i+1}: {l[0].repr_line()}' for (
                i, l) in enumerate(self.lines)
            ]
        
        return f'\n{"-" * 10}\n'.join(msglist)
                
    def repr_flow_lines(self) -> None:
        
        msglist = [
            f'Flow Line #{i+1}: {fl[0].repr_flow_line()}' for (
                i, fl) in enumerate(self.flow_lines)
            ]
        
        return f'\n{"-" * 10}\n'.join(msglist)

    def _roam_for_lines(self, queque, lines):
        
        if not queque:
            return lines
        
        for a in queque.copy():
            if a not in {a_ for l in lines for a_ in l}:
                new_line = a.from_left()
                lines.append(new_line)
                queque -= set(new_line)
                
        return self._roam_for_lines(queque, lines)
      
    # --------------------------------------------------------------------------------
    
    def reset_ModelConstructor(self, column: Column | str =None):
        '''
        Clear the ModelConstructor settings for a specific column in the IC system.
        If column is not provided, clear all the settings.
        '''
        
        if column is None:
            self._cached_model_constructor_types.clear()
        else:
            if isinstance(column, str):
                column = self.from_name(column)
            try:
                del self._cached_model_constructor_types[column]
            except KeyError:
                pass
    
    def set_ModelConstructor(self,
        ModelConstructor: type | str,
        column: Column | str,
        *,
        return_dataframe: bool =False,
        ) -> pd.DataFrame | None:
        '''
        Assigns a Model Constructor to a specific column in the IC system.
    
        Parameters:
        -----------
        ModelConstructor: type | str
            The constructor class or the name identifier of the constructor to be assigned
            If a string is provided, it attempts to match it with the class 
            attribute ._constructor_registry.
        column: Column | str
            The target column (either as a Column object or its string identifier).
        return_dataframe: bool, optional
            If True, returns the property .model_params containing the parameters that
            have been set to the IC system. Defaults to False.
        '''
        
        # type authentications
        if isinstance(ModelConstructor, str):
            ModelConstructor = self._constructor_registry.get(
                ModelConstructor.strip().lower(), None)
        assert issubclass(ModelConstructor, BaseConstructor)
        
        if isinstance(column, str):
            column = self.from_name(column)
        assert isinstance(column, GenericTubing)
        
        self._cached_model_constructor_types[column] = ModelConstructor
        
        if return_dataframe:
            return self.model_params
    
    # --------------------------------------------------------------------------------
    
    @property
    def model_params(self) -> pd.DataFrame:
        
        tp = template = pd.DataFrame(columns=['host', 'analyte', 'constructor', 'Model'])
        for tubing in [a for a in self.accessories if isinstance(a, GenericTubing)]:
            ModelConstructor = self._cached_model_constructor_types.get(tubing, None)
            if ModelConstructor is None and isinstance(tubing, PEEKTubing):
                ModelConstructor = DSMConstrutorForTubings
            
            for analyte in self.analytes:
                tp.loc[len(tp.index), tp.columns] = pd.NA # add a row
                tp.loc[len(tp.index)-1, ['host', 'analyte']] = [tubing, analyte]
                if ModelConstructor is not None:
                    constructor = ModelConstructor(tubing, self, analyte)
                    tp.loc[len(tp.index)-1, ['constructor', 'Model']] = (
                        constructor, constructor.Model.__name__)
                    for param, value in constructor.parameters.items():
                        if param not in tp.columns:
                            tp[param] = pd.NA # add a column
                        tp.at[len(tp.index)-1, param] = value
                        
        return tp.sort_values(
            by=['host', 'analyte'],
            key=lambda col: col.apply(str),
            ignore_index=True)
    
    # --------------------------------------------------------------------------------
    
    @property
    def schedule(self) -> pd.DataFrame:
        '''
        A schedule DataFrame indicating the execution schedule of commands
        on different modules within the IC system.
        '''
        
        ns = self.namespace
        tp = template = pd.DataFrame(columns=['time', *ns.columns.to_list(), 'action'])
        for module, commandlist in self._cached_commands.items():
            if module not in ns['module_instance'].to_list():
                warnings.warn(
                    f'''Commands on Module {module} is ignored because '''
                    '''it has been disassembled from IC system. ''')
            else:
                type_identifier, module_name, module_instance = (
                    ns[ns['module_instance']==module].iloc[0].to_list())
                assert type_identifier in self._parse_commands.keys()
                for t, action in commandlist:
                    assert action in self._parse_commands[type_identifier].keys()
                    tp.loc[len(tp.index), tp.columns] = (
                        t, type_identifier, module_name, module_instance, action)
        return tp.sort_values(
            by=['time', 'name', 'type_identifier'],
            ignore_index=True)
    
    def reset_commands(self,
        commands: str | tuple[tuple] =None, 
        delimeter: str =','
        ) -> pd.DataFrame:
        '''
        Reschedule the IC system.

        Parameters:
        -----------
        commands: str | tuple[tuple], optional
            A string or tuple containing the commands to reset. 
            If None, clears all cached commands.
            Defaults to None.
        delimeter: str, optional
            The delimeter used to parse commands if the input is a string.
            Defaults to ",".
        Returns property `schedule` in DataFrame.
        
        Examples:
        ----------
        ic = IonChromatograph('MyIC2024', ('hydroxide',), SwicthingValve.SixPort())
        commands = """
            0.0 min, valve sixport, inject
            0.5 min, valve sixport, load
            """
        or
        commands = (
            ('0.0 min', 'valve sixport', 'inject'),
            ('0.5 min', 'valve sixport', 'load'),
            )
        ic.reset_commands(commands)
        
        Notes:
        ----------
        Will clear and modify IC's cached commands in-place.
        Check class attribute ._parse_commands to see the available commands.
        '''
        self._cached_commands.clear()
        if commands is None:
            return self.schedule
        
        if isinstance(commands, str):
            command_lines = [l.strip() for l in commands.splitlines() if l.strip()]
        else:
            command_lines = commands
            
        for line in command_lines:
            self.add_command(line, delimeter)
        
        return self.schedule[['time', 'type_identifier', 'name', 'action']]
    
    def add_command(self,
        command: str | tuple[str, str, str],
        delimeter: str =','
        ) -> pd.DataFrame:
        '''
        Adds a command to the IC system's command schedule.

        Parameters:
        -----------
        command: str | tuple[str, str, str]
            The command to add, either as a string or a tuple containing time, 
            module name, and action.
            All command segments should be passed as strings
        delimeter: str, optional
            The delimiter used to separate fields if the command is a string.
            Defaults to ",".

        Returns property `schedule` in DataFrame.
        
        Examples:
        ----------
        ic = IonChromatograph('MyIC2024', ('hydroxide',), SwicthingValve.SixPort())
        ic.add_commands('0.0 min, valve sixport, inject')
        ic.add_commands('0.5 min, valve sixport, load')
        Above actions equal to:
        ic.add_commands(('0.0 min', 'valve sixport', 'inject'))
        ic.add_commands(('0.5 min', 'valve sixport', 'load'))
        
        Notes:
        ----------
        Check class attribute ._parse_commands to see the available commands.
        '''
        
        t, module, action = self._parse_command_line(command, delimeter)
        assert t >= 0
        if not hasattr(module, '__identifier__') or (
            module.__identifier__ not in self._parse_commands.keys()):
            raise ValueError(f'{module} cannot be operated.')
        elif action not in self._parse_commands[module.__identifier__].keys():
            raise ValueError(
                f'''Cannot parse action {action} for '''
                f'''{module.__identifier__} {module}.''')
        else:
            self._cached_commands.setdefault(module, []).append((t, action))
        
        return self.schedule[['time', 'type_identifier', 'name', 'action']]
    
    def _parse_command_line(self,
        line: str | tuple,
        delimeter: str,
        ) -> tuple[float, _ICModule, str]:
        
        if isinstance(line, str):
            time, module_name, action = line.split(delimeter)
        else:
            time, module_name, action = line
        
        t = ParseQuantities.as_min(time)
        module = self.from_name(module_name)
        action = action.strip().upper()
        
        return t, module, action
            
    # --------------------------------------------------------------------------------
    
    def designate_N(self,
        Ndict: dict[str, int],
        column: Column | str,
        ) -> pd.DataFrame:
        '''
        Update designated N values to .Ntable.
        
        Parameters:
        -----------
        Ndict: dict[str, int]
            A dictionary representing the N profile, where keys are analyte ions and values
            are N values assigned to the column.
        column: Column | str
            A Column instance or the name identifier in IC's namespace.
        '''
        if isinstance(column, str):
            column = self.from_name(column)
        Ndict = self._standardize_Ndict(Ndict)
        self._cached_designated_Ndict.setdefault(column, {}).update(Ndict)
        
        df = self.Ntable
        return df.loc[:, [c for c in df.columns if c != 'module_instance']]
        
    def clear_Ndict(self, column: Column | str) -> pd.DataFrame:
        
        if isinstance(column, str):
            column = self.from_name(column)
        
        Ndict = self._cached_designated_Ndict.get(column, {})
        Ndict.clear()
        
        df = self.Ntable
        return df.loc[:, [c for c in df.columns if c != 'module_instance']]
        
    def _standardize_Ndict(self, Ndict) -> dict:
        
        ions_.check_duplicated_ions_from_sequence(list(Ndict.keys()))
        if not all(isinstance(N, float | int) and N > 100 for N in Ndict.values()):
            raise ValueError('N values must be integers and > 100')
        
        return {ions_.get_ioninfo(ion).formula: int(np.round(N, -1)) for (
            ion, N) in Ndict.items()}
            
    # --------------------------------------------------------------------------------
    
    def inject(self,
        sample_profile: dict[str, str | float | Quantity],
        module: SwitchingValve | PEEKTubing |  SampleLoop | str,
        port_number: int =None,
        ) -> None:
        '''
        Injects a sample into the specified static flow line.
        Parameters:
        -----------
        sample_profile: dict[str, str | float | Quantity]
            A dictionary representing the sample profile to be injected, where keys are ions
            and values are either molar concentration values (as strings or floats)
            or pint.Quantity.
            Units default to mM if not specified.
        module: SwitchingValve | PEEKTubing | SampleLoop | str
            Any module or accessory to inject sample into. 
            This can be either an IC module instance, or the name of the module 
            in IC's namespace. A SampleLoop instance is usually recommended.
        port_number: int, optional
            The port number must be provided if `module` is a switching port. Defaults to None.
            
        Notes:
        ------
        Must no accessories such as pumps, columns, detectors & supressors(TODO) in the line,
            otherwise InjectionError will be raised.
        K+/OH- for AEC and H+/Cl- for CEC will be added to maintain sample charge neutrality.
        The built-in models of pyIClab do not consider the interactions of the counterions. 
            As the eluent passes through generic tubing processed by a model, counterions infor
            will be discarded due to the model feature. Therefore, it is impractical 
            to designate counterions other than K+ or Cl- when injecting samples.
            It is also at your convenience to inject solutions without counterion profile.
        '''
        
        if isinstance(module, str):
            module = self.from_name(module)
        
        #
        if isinstance(module, SwitchingValve):
            accessory = module[port_number]
        else:
            accessory = module
        #
        nias = non_inject_accessories =  [
            a for a in accessory.from_left() if not isinstance(a, _Injectables)]
        if nias:
            raise InjectionError(nias)
            
            
        profile = self._convert_sample_profile_with_standard_formulas(sample_profile)
        self._cached_injection_profile.update(
            {a: profile.copy() for a in accessory.from_left() if (
                isinstance(a, GenericTubing))})
    
    # --------------------------------------------------------------------------------
    
    def _convert_sample_profile_with_standard_formulas(self,
        profile) -> dict[str, float]:
        
        standard_profile = {}
        ions_.check_duplicated_ions_from_sequence(list(profile.keys()))
        cum_charge = 0
        for ion, c in profile.items():
            ioninfor = ions_.get_ioninfo(ion)
            c_as_mM = ParseQuantities.as_mM(c)
            standard_profile[ioninfor.formula] = c_as_mM
            cum_charge += ioninfor.charge * c_as_mM
        
        if not cum_charge:
            return standard_profile
        else:
            match (cum_charge>0, self.kind):
                case (False, 'anion'):
                    counterion = 'K+'
                case (True, 'anion'):
                    counterion = 'OH-'
                case (True, 'cation'):
                    counterion = 'Cl-'
                case (False, 'cation'):
                    counterion = 'H+'
            counterioninfo = ions_.get_ioninfo(counterion)
            standard_profile.setdefault(counterioninfo.formula, 0.0)
            standard_profile[counterioninfo.formula] -= (
                cum_charge / counterioninfo.charge)
            return standard_profile
    
    # --------------------------------------------------------------------------------
    
    @Progressor(prefix='Configurating model paratemers...', indentation=4, timer=True)
    def _check_models_ready(self) -> pd.DataFrame:
        
        df = model_params = self.model_params
        df = df[pd.isna(df['constructor'])]
        msg_dict = {}
        for column, analyte in df.groupby(['host', 'analyte']).groups.keys():
            msg_dict.setdefault(column, set()).add(analyte)
            
        msg = '\n'.join(repr(column) + ': ' + ', '.join(analytes) for (
            column, analytes) in msg_dict.items())
        if len(msg_dict):
            raise ProfileError(
                '''Models haven't been assigned for columns:\n'''
                f'''{msg}''')
            
        return model_params
        
    def _get_time_intervals(self, tmax: float) -> tuple[tuple[float, float], ...]:
        
        df = self.schedule
        t = sorted({*df.loc[df['type_identifier']=='valve', 'time'], 0.0, tmax})
        i = t.index(tmax)
        return tuple((t1, t2) for t1, t2 in zip(t[:i], t[1:i+1]))
    
    @Progressor(prefix='Building models...', indentation=4, timer=True)
    def _build_models(self, model_params: pd.DataFrame =None) -> None:
        
        df = self.model_params if model_params is None else model_params
        for i, (host, constructor) in df[['host', 'constructor']].iterrows():
            self._cached_models.setdefault(host, []).append(constructor())
    
    @Progressor(
        prefix='Injecting Samples...', indentation=4, timer=True)
    def _init_models(self) -> None:
        
        for model in (
            model for models in self._cached_models.values() for model in models
            ):
            model.standby()
    
    def _execute(self, module, action) -> None:
        
        prefix = f'{self.current_time.magnitude:.1f} min: Execute Command -- {module} {action}'
        
        @Progressor(prefix=prefix, indentation=8)
        def _exc(self, module, action):
        
            type_identifier = module.__identifier__
            p = method_parser = self._parse_commands.get(type_identifier).get(action)
        
            return getattr(module, p.method)(*p.args, **p.kwargs)
        
        return _exc(self=self, module=module, action=action)
        
    # --------------------------------------------------------------------------------
    
    def activate(self, tmax: str | float | int):
        
        # reset all the accessories and synchronize them...
        self.reset() # This line must be placed ahead of getting model_params...

        schedule = self.schedule
        tmax = ParseQuantities.as_min(tmax)
        assert tmax >= 0  # TODO Check tmax... Done..
        
        # Check model_params integrality
        model_params = self._check_models_ready()
        
        # build models for columns and tubings...
        self._build_models()
        
        # init all models
        self._init_models()
        
        # get time intervals
        time_intervals = self._get_time_intervals(tmax)
        for interval in time_intervals:
            t1, t2 = interval
            commands = schedule[schedule['time']==t1]
            
            # Excecute the commands at the start of each interval
            for i, data in commands.iterrows():
                module = data['module_instance']
                action = data['action']
                self._execute(module, action)
            
            # run the models from the head of the flow lines
            for flow_line in self.flow_lines:
                flow_line[0]._cur_time = t2 * Unit('min')
                for accessory in flow_line[1:]:
                    models = self._cached_models.get(accessory)
                    if models is None:
                        accessory.flush()
                    else:
                        accessory.flush(models=models)
            
            # need to synchronize those not in a flow...
            self._synchronize(t2 * Unit('min'))
        
        # End modelling
        self._cached_models.clear()
            
    def start(self, tmax: str | float | int):
        
        return Progressor(
            prefix=f'Activating {self}...',
            suffix='IC simulation finished...\n',
            timer=True)(self.activate)(tmax)

# --------------------------------------------------------------------------------


# In[3]:


__all__ = ['IonChromatograph',]

