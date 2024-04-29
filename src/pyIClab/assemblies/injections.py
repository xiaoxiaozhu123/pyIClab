#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[1]:


##### Built-in import #####

import functools
from typing import Self, Literal

##### External import #####

import numpy as np

##### Local import #####

from pyIClab._baseclasses import NamedTrinket
from pyIClab.errors import (
    ConfigurationError, InterLockError, LockedValvePortError,
    ErrorStat,
    )
from pyIClab.assemblies.generics import GenericAccessory, Side
from pyIClab.assemblies.columns import SampleLoop

# --------------------------------------------------------------------------------


# In[2]:


class SwitchingPort(GenericAccessory):
    '''
    Represents a specialized type of Port, intended to be used within a switching valve.
        A SwitchingPort Object is locked to its counterparts in the the SwicthingValve 
        Object where it is located.
    '''
    _unique = False
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /, *args, **kwargs):
        
        super().__init__(*args, **kwargs) 
        self._locked = True
        
    # --------------------------------------------------------------------------------
    
    def __repr__(self):
        
        if hasattr(self, 'valve'):

            return f'<{repr(self.valve).strip("<").strip(">").strip()}[{self.name}]>'
        else:
            return super().__repr__()
    
    # --------------------------------------------------------------------------------
    # properties
    
    @property
    def left(self):
        
        return super().left

    @left.setter
    def left(self, accessory):
        
        if not hasattr(self, 'valve') or not self._locked:
            self._left = accessory
            return

        elif self.open_slot != 'left' and accessory is not self.bonded:
            raise LockedValvePortError(accessory, self, 'left')
            
        self._left = accessory
        
    @property
    def right(self):
        
        return super().right
    
    @right.setter
    def right(self, accessory):
        
        if not hasattr(self, 'valve') or not self._locked:
            self._right = accessory
            return

        elif self.open_slot != 'right' and accessory is not self.bonded:
            raise LockedValvePortError(accessory, self, 'right')
            
        self._right = accessory
        
    # --------------------------------------------------------------------------------
        
    @property
    def valve(self):
        
        return self._valve
    
    @valve.setter
    def valve(self, v):
        
        if not hasattr(self, '_valve'):
            self._valve = v
        else:
            raise ConfigurationError(
                f'''The valve of {self} can not be reset.'''
                )
            
    @property
    def number(self) -> int:

        return self.valve.ports.index(self)
    
    @property
    def open_slot(self) -> Side:
        '''
        str, 'left' or 'right'.
        Side of the port to the out in respect to the specified valve position.
        '''
        p = self.valve.position
        i = self.number
        
        match (p, i % 2):
            case ('LOAD', 0):
                return 'left'
            case ('LOAD', 1):
                return 'right'
            case ('INJECT', 0):
                return 'right'
            case ('INJECT', 1):
                return 'left'
            
    @property
    def bonded_slot(self) -> Side:
        
        return 'left' if self.open_slot == 'right' else 'right'
    
    @property
    def bonded(self) -> Self:
        '''
        The bonded Switching Port object in respect to the specified valve position.
        '''
        v = self.valve
        p = self.valve.position
        i = self.number
        
        match (p, i % 2):
            case ('LOAD', 0):
                return v[i+1]
            case ('LOAD', 1):
                return v[i-1]
            case ('INJECT', 0):
                return v[i-1]
            case ('INJECT', 1):
                return v[i+1]
        
    # --------------------------------------------------------------------------------
    
    def assemble(self, other: GenericAccessory, side: Side =None) -> None:
        '''
        Assemble the switching port with `other` from its open side.
        
        Parameters:
        ----------
        other: an GenericAccessory object.
        side: str, optional, 'left' or 'right'. 
            `other`'s connecting port. if None, connect to its open port,
            which defaults to 'left'.
        '''
        
        assert hasattr(self, 'valve')
            
        s1 = self.open_slot
        if side is None:
            if not other.open_slots:
                ConfigurationError(
                    f'''{other} is fully assembled.'''
                    '''Disassemble before assembling.''')
            else:
                s2 = other.open_slots[0]
        else:
            s2 = side.lower()
        
        return super().assemble(other, [s1, s2])
        
    # --------------------------------------------------------------------------------
    
    def disassemble(self, *, error: ErrorStat =None) -> None:
        '''
        Disassemble the switching port from the open side. 
            
        Parameters:
        ----------
        error: str, optional, 'ignore', 'warn', 'raise'.
            If 'ignore', do nothing when disassembling from None.
            If 'warn', send a warning message.
            If 'raise', raise a DisAssembleError.
            Defaults to 'warn'.
            
        Notes:
        ----------
        InterLockError will be also raised if swicthing port is interlocked with its bonded
            switching port if 'raise'.
        '''
        assert hasattr(self, 'valve')
        return super().disassemble(self.open_slot, error=error)
        
    # --------------------------------------------------------------------------------
        


# In[3]:


def _temporary_unlock(func):
    
    @functools.wraps(func)
    def _wrapper(valve, *args, **kwargs):
        
        valve._unlock()
        res = func(valve, *args, **kwargs)
        valve._lock()
        
        return res
    
    return _wrapper

Position = Literal['LOAD', 'INJECT']


# In[4]:


class SwitchingValve(NamedTrinket):
    
    '''
    Represents a switching valve for IC, capable of altering the flow path 
        between its ports.
    '''
    __identifier__ = 'valve'
    _unique = True
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /, name: str, n: int =None) -> None:
        '''
        Parameters:
        ----------
        name: str, optional.
            The name of the switching valve. Defaults to an empty string.
        n: int, optional, 6 or 10.
            The number of ports in the valve. Defaults to 6.
        '''
        
        super().__init__(name=name)
        n = 6 if n is None else int(n)
        assert n in (6, 10)
        
        self.ports = tuple(SwitchingPort(str(i)) for i in range(n))
        
        for port in self: 
            port.valve = self
        
        self._position = 'LOAD'
        self._init_valve_ports()
            
    # --------------------------------------------------------------------------------
    # Properties
    
    @property
    def n(self) -> int:
        return len(self.ports)
    
    @property
    def position(self) -> Position:
    
        return self._position
      
    # --------------------------------------------------------------------------------
    
    def __repr__(self):
        
        return f'<Valve "{self.name}">'
    
    # --------------------------------------------------------------------------------
    
    def __len__(self):
        
        return self.n
    
    # --------------------------------------------------------------------------------
    
    def __call__(self, i: int) -> SwitchingPort: 
        
        return self.ports[i]
    
    # --------------------------------------------------------------------------------
    
    def __getitem__(self, i: int) -> SwitchingPort:
        
        return self.ports[i % self.n]
    
    # --------------------------------------------------------------------------------
    
    def __iter__(self):
        
        return iter(self.ports)
    
    # --------------------------------------------------------------------------------
    
    def _lock(self):
        
        for port in self:
            port._locked = True
            
    def _unlock(self):
        
        for port in self:
            port._locked = False
    
    @_temporary_unlock
    def _init_valve_ports(self) -> None:
        
        for i in range(0, self.n, 2):
            p1 = self[i]
            p2 = self[i+1]
            p1._assemble(p2, ['right', 'left'])
    
    # --------------------------------------------------------------------------------
    
    def switch(self, position: Position =None) -> None:
        '''
        Changes the configuration of the valve between 'LOAD' and 'INJECT' positions 
            or toggles it if no position is specified.

        Parameters:
        ----------
        position : str, optional
            Can be 'LOAD', 'INJECT', or None. 
            If None, the valve toggles between 'LOAD' and 'INJECT'.
            Defaults to None.

        Examples:
        ----------
        >>> valve = SwitchingValve.SixPort()
        >>> valve.switch('LOAD')  # Sets the valve to LOAD position
        >>> valve.switch()        # Toggles the valve position
        >>> valve.switch('INJECT') # Sets the valve to INJECT position
        '''
        
        if position is None:
            position = 'INJECT' if self.position == 'LOAD' else 'LOAD'
            
        position = position.upper()
        if position not in ('LOAD', 'INJECT'):
            raise ValueError(
                f'''Invalid position code '{position}' specified. '''
                '''Please specify 'LOAD' or 'INJECT' as the position.''')
        else:    
            self._swicth(position)

    @_temporary_unlock 
    def _swicth(self, position) -> None:
        
        curr = self.position
        
        if curr == position:
            return
        
        # cache the open sides and their connecting ports
        outlet_accessories = [getattr(port, port.open_slot) for port in self]
        connecting_sides = []
        for accessory, port in zip(outlet_accessories, self):
            if accessory is None:
                connecting_sides.append(None)
            elif accessory in self:
                # InterLock issues are actually fixed.
                # TODO: this statement is able to be removed in the furture...
                if port.interlocked_to(accessory):
                    assert port.bonded is accessory
                    msg = ('''Use a PEEK tubing to connect '''
                        '''them from the open sides.''')
                    raise InterLockError((port, accessory), msg)
                else:
                    connecting_sides.append(accessory.bonded_slot)
            else:
                for side in accessory.slots:
                    if getattr(accessory, side) is port:
                        connecting_sides.append(side)
                        break
        
        # disassemble all the ports
        for port in self:
            port._disassemble(side='both', error='ignore')
        
        # switch position and assemble the valve ports
        self._position = position
        for i, port in enumerate(self):
            if (position, i % 2) in (('LOAD', 0), ('INJECT', 1)):
                port._assemble(port.bonded, ['right', 'left'])
            
        # reattach the open sides
        for i, (port, accessory, connecting_side) in enumerate(zip(
            self, outlet_accessories, connecting_sides)):
            # InterLock issues are actually fixed.
            # TODO: this statement is able to be removed in the furture...
            if port.bonded is accessory:
                msg = ('''Use an PEEK tubing to connect '''
                    '''them from the open sides.''')
                raise InterLockError((port, accessory), msg) 
            elif accessory is not None and port not in outlet_accessories[:i]: 
                port.assemble(accessory, connecting_side)
                   
    # --------------------------------------------------------------------------------
    
    def assemble(self,
        port_numbers: int | tuple[int, ...] | list[int],
        accessory: GenericAccessory,
        sides: Side | tuple[Side, ...] | list[Side] =None,
        ) -> None:
        '''
        Attach the accessory to the switching valve.
        
        Parameters:
        -----------
        port_numbers: int | tuple[int, ...] | list[int].
            The ports used to attach the accessory in port numbers.
            if a tuple/list with 2 elements, use both ports to attach.
        accessory: GenericAccessory.
            The accessory to be assembled to the switching valve.
        sides: Side | tuple[Side, ...] | list[Side], optional.
            The sides of the accessory to be attached to `ports`, respectively.
            If Not given, use an empty port (defaults to 'left') for one-end assembling, 
            and ['left', 'right'] for two-end assembling.
            
        Examples:
        ----------
        >>> valve = SwitchingValve.SixPort()
            eluent = Eluent.HydroxideIsocratic('15 mM') 
            loop = PEEKTubing(length='5cm') 
            column = Dummy.Column() 
        >>> valve.assemble(0, eluent)  # attach the eluent to the valve at port 0. 
        >>> valve.assemble(1, column)  # attach the column to the valve at port 1.
        >>> valve.assemble([2, 5], loop) # install the loop between port 2 and port 5.
        '''
        if isinstance(port_numbers, int):
            pn = [port_numbers,]
        else:
            assert isinstance(port_numbers, tuple | list) and (
                0 < len(port_numbers) < 3)
            pn = port_numbers
            
        if len(accessory.open_slots) < len(pn):
            raise ConfigurationError(
                f'''Not enough open ports for {accessory} to assemble.''')
            
        if sides is None:
            sides = accessory.open_slots[:len(pn)]
        elif isinstance(sides, str):
            sides = [sides.lower(),]
        else:
            assert isinstance(sides, tuple | list) and len(sides) == len(pn)
            sides = [s.lower() for s in sides]
        
        for i, side in zip(pn, sides):
            self._assemble(i, accessory, side)
         
    def _assemble(self, i, accessory, side):
        
        port = self[i]
        
        return port.assemble(accessory, side)
        
    # --------------------------------------------------------------------------------
    
    def disassemble(self,
        port_numbers: int | tuple[int, ...] | list[int] =None,
        *,
        error: ErrorStat =None,
        ) -> None:
        '''
        Detach accessories from the specified ports of the switching valve.

        Parameters:
        -----------
        port_numbers: int | tuple[int, ...] | list[int], optional.
            The ports from which the accessories are to be detached. 
            If not provided, disassembles accessories from all ports and ignores any errors.
            If an int is provided, disassembles the accessory from the specified port.
            If a tuple/list is provided, disassembles accessories from the specified ports.
        error: str, optional.
            Specifies how to handle errors during disassembly. 
            Can be 'ignore', 'warn', or 'raise'. Defaults to 'warn'.
            Note: If `port_numbers` is not provided, errors are always ignored regardless
            of the `error` parameter value.
        '''
        
        if port_numbers is None:
            return self.disassemble(range(self.n), error='ignore')
        elif isinstance(port_numbers, int):
            return self.disassemble([port_numbers,])
        else:
            for pn in port_numbers:
                self[pn].disassemble(error=error)
            return
    
    # --------------------------------------------------------------------------------
    
    @classmethod
    def SixPort(cls, name: str='SixPort'):
        '''
        Create an six-port switching valve object.
        '''
        
        return cls(name=name)
    
    @classmethod
    def TenPort(cls, name: str='TenPort'):
        '''
        Create an ten-port switching valve object.
        '''
        
        return cls(name=name, n=10)
    
# --------------------------------------------------------------------------------


# In[5]:


__all__ = ['SwitchingPort', 'SwitchingValve', 'SampleLoop']

