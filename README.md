# PyICLab

PyICLab is an open-source Python package designed for in-silico simulations of ion chromatography (IC). It features an object-oriented programming (OOP) interface, providing robust tools for realistic and customized simulations.

## Features

- Implementations of IC components
- Built-in Numerical models
- Flow path management
- Suitable for complex IC setups and conditions

## Installation

PyICLab can be installed via PyPI using pip. Ensure you have Python 3.11 or higher.

For Windows:
```sh
pip install pyiclab
```
For MacOS:
```sh
pip3 install pyiclab
```
### Note for Apple Silicon Users

PyICLab currently does not support the ARM64 architecture directly. To use PyICLab on Apple Silicon, you are advised to build your Python3 environment using an x86 version of conda/miniconda. 

## Get Started

Here is a simple example to demonstrate the basic usage of PyICLab:

```python
from pyIClab._testing_toolkit import PackedIC
from pyIClab.beadedbag import mpl_custom_rcconfig
import seaborn as sns
import matplotlib.pyplot as plt

ic = PackedIC()
ic.start(tmax='10 min')

detector = ic.detectors.pop()
df = detector.get_signals()

sns.set()
plt.rcParams.update(mpl_custom_rcconfig)
fig, ax = plt.subplots()
ax.plot('time', 'signal', data=df)
ax.set_xlabel('Time, min')
ax.set_ylabel('Concentration, mM')
plt.show()
```

