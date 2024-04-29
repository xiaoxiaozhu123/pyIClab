#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Copyright (C) 2024 PyICLab, Kai "Kenny" Zhang
'''


# In[2]:


##### Built-in import #####

import warnings
from typing import Callable, Self, Literal

##### External import #####

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray
from pint import Unit, Quantity

##### Local import #####

import pyIClab.ions as ions_
from pyIClab.units import ParseQuantities
from pyIClab.errors import (
    ConfigurationError, PortPluggedError, CycledFlowError,
    MultiplePortError, NoFlowError, DisAssembleError,
    InterLockError, ErrorStat,
    )
from pyIClab._baseclasses import BaseAccessory, NamedTrinket
from pyIClab.beadedbag import mpl_custom_rcconfig, is_notebook


# --------------------------------------------------------------------------------


# In[3]:


Side = Slot = Literal['left', 'right']

# --------------------------------------------------------------------------------


# In[4]:


class GenericAccessory(BaseAccessory, NamedTrinket):
    
    '''
    Represents a connecting one-port or two-port accessory with no void volumn for IC.
    '''
    
    # --------------------------------------------------------------------------------
    
    def __init__(self, /, name: str) -> None:
        '''        
        Parameters:
        ----------
        name: str 
            The identifier of the accessory.
        '''
        
        super().__init__(name=name)
        self.left = None
        self.right = None
        self.fr = None
        self._output = {}
        self._cur_time = 0.0 * Unit('min')
        
    # --------------------------------------------------------------------------------
    
    def __new__(cls, *args, **kwargs):
        
        if cls is GenericAccessory:
            raise TypeError('GenericAccessory cannot be instantiated directly.')
        else:
            return super().__new__(cls)
    
    # --------------------------------------------------------------------------------
    
    def __call__(self, t: str | float | int | list | ndarray):
        
        if isinstance(t, str | int | float):
            t = ParseQuantities.as_min(t)
        else:
            t = np.array([ParseQuantities.as_min(_) for _ in t])
            
        return {ion: f(t) for ion, f in self.output.items()}
        
    # --------------------------------------------------------------------------------
    # properties
    
    @property
    def current_time(self) -> Quantity:
        
        return self._cur_time
    
    @property
    def ions(self) -> tuple:
        
        return tuple(self.output.keys())
    
    # --------------------------------------------------------------------------------
    
    @property
    def right(self):
            
        return self._right
  
    @right.setter
    def right(self, accessory):
        
        self._right = accessory
        
    @right.deleter
    def right(self):
        
        del self._right
        
    @property
    def left(self):
        
        return self._left
    
    @left.setter
    def left(self, accessory):
        
        self._left = accessory
        
    @left.deleter
    def left(self):
        
        del self._left
        
    @property
    def output(self):
        
        return self._output
        
    @property
    def fr(self):
        
        return self._intrinsic_fr or self.flow_head.fr
    
    @fr.setter
    def fr(self, fr):
    
        if self.twoway and fr is not None:
            raise MultiplePortError()
    
        self._intrinsic_fr = fr
        
    # --------------------------------------------------------------------------------
    @property
    def slots(self) -> tuple[Side, ...]:
        
        pa = []
        if hasattr(self, 'left'):
            pa.append('left')
        if hasattr(self, 'right'):
            pa.append('right')
        
        return tuple(pa)
    
    @property
    def open_slots(self) -> tuple[Side, ...]:
        
        return tuple(
            side for side in self.slots if getattr(self, side) is None)
    
    @property
    def occupied_slots(self) -> tuple[Side, ...]:
        
        return tuple(
            side for side in self.slots if getattr(self, side) is not None)
    
    # --------------------------------------------------------------------------------
        
    @property
    def left_end(self):
        
        return self._roam('left')[-1]
    
    def from_left(self):
        
        return self._roam('left')[::-1] + self._roam('right')[1:]
        
    @property
    def right_end(self):
        
        return self._roam('right')[-1]
    
    def from_right(self):
        
        return self.from_left()[::-1]
    
    def repr_line(self) -> str:
        
        return ' -- '.join([repr(a) for a in self.from_left()])
        
    def _roam(self,
        side: Side,
        cached_accessories: list[Self] =None
        ) -> list[Self]:
        
        cached_accessories = [self] if cached_accessories is None else cached_accessories
        side = side.lower()
        
        if not hasattr(self, side) or getattr(self, side) is None:
            return cached_accessories

        other = getattr(self, side)
        if other in cached_accessories:
            raise CycledFlowError(other, cached_accessories)
            
        cached_accessories.append(other)
        try:
            return other._roam('left', cached_accessories)
        except CycledFlowError: pass
            
        return other._roam('right', cached_accessories)
        
    # --------------------------------------------------------------------------------
    
    @property
    def oneway(self) -> bool:
        
        return len(self.slots) == 1
    
    @property
    def twoway(self) -> bool:
        
        return len(self.slots) == 2
    
    # --------------------------------------------------------------------------------
    
    @property
    def moterized(self) -> bool:
        
        return self._intrinsic_fr is not None
    
    @property
    def flow_direction(self):
        
        # For self-pumping accessories
        if self.moterized:
            (side,) = self.slots
            match side:
                case 'right':
                    return 1
                case 'left':
                    return -1
            
        # For non-self-pumping accessories
        if self.oneway:
            try:
                end = self.left_end
            except PortPluggedError:
                end = self.right_end
            if end.moterized:
                raise PortPluggedError(self)
            else:
                return 0
        else:
            assert self.twoway
            l, r = self.left_end, self.right_end
            if l.moterized and r.moterized:
                raise CounterPumpingError(self)
            elif not l.moterized and not r.moterized:
                return 0
            elif l.moterized:
                return 1
            else:
                assert r.moterized #
                return -1
            
    @property
    def prev(self):

        match self.flow_direction:
            case 0:
                raise NoFlowError(self, 'prev')
            case 1:
                return self.left
            case -1:
                return self.right
        
    @property
    def next(self):
        
        match self.flow_direction:
            case 0:
                raise NoFlowError(self, 'next')
            case 1:
                return self.right
            case -1:
                return self.left
            
    @property
    def _queque(self) -> list:
        
        if self.next is None:
            return [self]
        else:
            return [self] + self.next._queque
        
             
    @property
    def flow_end(self):
        
        if not self.flow_direction:
            raise NoFlowError(self, 'flow end')
        
        if self.next is None:
            return self
        else:
            return self.next.flow_end
        
    @property
    def flow_head(self):
        
        if not self.flow_direction:
            raise NoFlowError(self, 'flow head')
            
        if self.moterized:
            return self
        else:
            return self.prev.flow_head
        
    @property
    def flow_line(self) -> list:
        
        return self.flow_head._queque
    
    def repr_flow_line(self):
        
        return (
            f'''{" -> ".join([repr(a) for a in self.flow_line])} -> <Waste>\n'''
            f'''Flow Rate: {self.fr.to("mL/min").magnitude:.1f} mL/min'''
            )
    
    # --------------------------------------------------------------------------------
    
    def _connected_to_one_way(self, other: Self) -> bool:
        
        for s1 in self.slots:
            if getattr(self, s1) is other:
                return True
        return False
    
    def connected_to(self, other: Self) -> bool:
        '''
        Detect if `self` is connected to `other` directly.
        '''
        if self._connected_to_one_way(other) and other._connected_to_one_way(self):
            return True
        elif not self._connected_to_one_way(other) and not other._connected_to_one_way(self):
            return False
        elif self._connected_to_one_way(other):
            raise ConfigurationError(
                f'''While {self} is detected to be connected {other}, '''
                f'''{other} disconnected to {self}.''')
        else:
            raise ConfigurationError(
                f'''While {other} is detected to be connected {self}, '''
                f'''{self} disconnected to {other}.''')
    
    # --------------------------------------------------------------------------------
    
    def interlocked_to(self, other: Self) -> bool:
        '''
        Detect if `self` and `other` are interlocked.
        '''
        
        if not self.twoway or not other.twoway:
            return False
        
        for s1 in self.slots:
            if getattr(self, s1) is not other:
                return False
        for s2 in other.slots:
            if getattr(other, s2) is not self:
                return False
            
        return True
    
    # --------------------------------------------------------------------------------
    
    def assemble(self,
        other: Self,
        sides: tuple[Side, Side] | list[Side] =None,
        ) -> None:
        '''
        Connect `self` to the `other`.
        
        Parameters:
        ----------
        other: a GenericAccessory object.
        sides: tuple | list, optional -> [side1, side2].
            Connect self's sides1 to other's sides2.
            If None, connect their open port repectively,
                which defaults to 'right' for self, and 'left' for other.
        
        Examples:
        ----------
            p1 = GenericAccessory()
            p2 = GenericAccessory()
        - Connect p1's right to p2's left , which is recommended:
            p1.assemble(p2)
        - Connect p1's left to p2's right:
            p1.assemble(p2, ['left', 'right'])
        '''
        if sides is None:
            for accessory in [self, other]:
                if not accessory.open_slots:
                    raise ConfigurationError(
                        f'''{accessory} is fully assembled.''' 
                        '''Dissemmble before assembling.''')
            s1, s2 = self.open_slots[-1], other.open_slots[0]
        else:
            s1, s2 =  [s.lower() for s in sides]
                    
        if self is other:     
            raise ConfigurationError(
                f'''Self conncetion is not allowed.''')
            
        for accessory, side in zip([self, other], [s1, s2]):
            if not hasattr(accessory, side):
                raise PortPluggedError(accessory, side)
            if getattr(accessory, side) is not None:
                raise ConfigurationError(
                    f'''{accessory}'s {side} already assembled. '''
                    '''Dissemmble before assembling.''')
            
        self._assemble(other, [s1, s2])
        
    def _assemble(self, other, sides):
        
        s1, s2 = sides
        setattr(self, s1, other)
        setattr(other, s2, self)

    # --------------------------------------------------------------------------------
    
    def disassemble(self,
        side: Literal['left', 'right', 'both'] =None,
        *, 
        error: ErrorStat =None) -> None:
        '''
        Disassemble the accessory from the `side` port. 
            
        Parameters:
        ----------
        side: str, optional, 'left', 'right' or 'both'.
            from which port the accessory is disassembled. If 'both', 
            desemble from the both sides. Defaults to 'both'.
        error: str, optional, 'ignore', 'warn', 'raise'.
            If 'ignore', do nothing when disassembling from None.
            If 'warn', send a warning message.
            If 'raise', raise a DisAssembleError.
            Defaults to 'warn'.
            
        Notes:
        ----------
        InterLockError will be raised if the two accessories are interlocked.
        '''
        
        side = 'both' if side is None else side.lower()
        error = 'warn' if error is None else error.lower()
        
        if side not in ('both', 'left', 'right'):
            raise ValueError(
                f'''Expect 'both', 'left' or 'right', got {side}.''')
            
        if error not in ('ignore', 'warn', 'raise'):
            raise ValueError(
                f'''Expect 'ignore', 'warn', 'raise', got {error}.''')
            
        return self._disassemble(side, error)
    
    def _disassemble(self, side, error):
        
        if side == 'both':
            for s in self.slots:
                self._disassemble(s, error=error)
            return
        
        other = getattr(self, side)
        if other is None:
            message = (
                f'''{self}'s {side} port already disassembled. '''
                '''Do you mean another side?''')
            e = DisAssembleError(message)
            match error:
                case 'ignore': return
                case 'warn': warnings.warn(f'\n{e.msg}') # return
                case 'raise': raise e
        elif self.interlocked_to(other):
            self._disassemble_from(other, error)
        else:
            (otherside,) = (
                s for s in other.slots if getattr(other, s) is self)

            setattr(self, side, None)
            setattr(other, otherside, None)
            
    # --------------------------------------------------------------------------------
    
    def disassemble_from(self, other: Self, *, error: ErrorStat =None) -> None:
        
        error = 'warn' if error is None else error.lower()
        if error not in ('ignore', 'warn', 'raise'):
            raise ValueError(
                f'''Expect 'ignore', 'warn', 'raise', got {error}.''')
            
        return self._disassemble_from(other, error)
    
    def _disassemble_from(self, other, error):
        
        if not self.connected_to(other):
            message = f'{self} and {other} are not connected.'
            e = DisAssembleError(message)
            match error:
                case 'ignore': return
                case 'warn': warnings.warn(f'\n{e.msg}') # return
                case 'raise': raise e
        elif self.interlocked_to(other):
            if error == 'raise':
                raise InterLockError((self, other))
            else:
                for s1 in self.slots:
                    setattr(self, s1, None)
                for s2 in other.slots:
                    setattr(other, s2, None)
                if error == 'warn':
                    warnings.warn(
                        f'''{self} and {other} are detached from both sides.''')
        else:
            for s1 in self.slots:
                if getattr(self, s1) is other:
                    setattr(self, s1, None)
                    break
            for s2 in other.slots:
                if getattr(other, s2) is self:
                    setattr(other, s2, None)
                    return
                
    # --------------------------------------------------------------------------------
    
    def flush(self) -> None:
        
        t0 = self.current_time.magnitude
        t1 = self.prev.current_time.magnitude
        assert t1 > t0
        
        input_datasets = self.prev.output
        for component, f1 in input_datasets.items():
            f0 = self.output.setdefault(component, lambda t: np.zeros_like(t))
            self.output[component] = self._update_func_time_series((t0, t1), f0, f1)
        
        self._cur_time = t1 * Unit('min')
        
    # --------------------------------------------------------------------------------
    
    @staticmethod
    def _update_func_time_series(
        time_interval: tuple,
        default_func: Callable,
        update_func: Callable,
        ) -> Callable:
        
        def merged_func(t, *, f0=default_func, f1=update_func):
            
            t = np.array(t)
            a, b = time_interval
            mask = (t>=a) & (t<b)
            return np.where(mask, f1(t), f0(t))
        
        return merged_func
            
    # --------------------------------------------------------------------------------
    
    def reset(self) -> None:
        
        self._cur_time = 0.0 * Unit('min')
        
        if not self.moterized:
            self.output.clear()
    
    # --------------------------------------------------------------------------------
    
    def plot(self, component: str =None) -> tuple[plt.Figure, plt.Axes]:
        '''
        Plots the concentration profiles of specified components or all components
            from the eluent outlet.
        Returns a tuple containing a `Figure` instance and an `Axes` instance.

        Parameters:
        ----------
        component : str, optional
            The specific ion or component to plot. If not specified, the method
            plots concentration profiles for all components detected.
        '''
        
        if component is not None:
            try:
                components = (ions_.get_ioninfo(component).formula,)
            except ValueError:
                components = (component,)
        else:
            components = tuple(self.output.keys())
        
        plt.rcParams.update(mpl_custom_rcconfig)
        sns.set()
        fig, ax = plt.subplots()
        tmax = 0.99 * self.current_time.magnitude
        if not tmax: return
        
        x = np.linspace(0, tmax, 1000)
        ymax = 0.0
        
        for component in components:
            f = self.output.get(component)
            if f is None: return
            try:
                charge = ions_.get_ioninfo(component).charge
            except ValueError:
                label = component
                linestyle = '-'
            else:
                label = ions_.get_formula_latex(component)
                linestyle = '--' if charge < 0 else '-.'
                
            y = f(x)
            ymax = max(ymax, *y)
            if np.any(y>1e-5):
                ax.plot(x, y, 
                    linewidth=0.75, 
                    linestyle=linestyle,
                    label=label,
                    )
                
        ax.set_xlabel('time, min', fontsize=10, fontweight='bold')
        ax.set_ylabel('c, mM', fontsize=10, fontweight='bold')
        ax.legend()
        ax.set(xlim=(0, self.current_time.magnitude),
            ylim=(-ymax * 0.05, ymax * 1.05))
        
        if not is_notebook():
            plt.show()
            
        return fig, ax
    
    


# In[ ]:




