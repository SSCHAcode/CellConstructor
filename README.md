# CellConstructor

Welcome to the CellConstructor python package!

## Requirements

To correnctly install and use the package, you need to have
1. python >= 2.7
2. ASE : Atomic Simulation Environment
3. numpy
4. scipy
5. A fortran compiler
6. Lapack

The fortran compiler is required to compile the fortran libraries 
from Quantum ESPRESSO.

Suggested, but not required, is the installation of spglib. 
This library is able to compute symmetries from the structure, 
and inside the symmetry module there is a convertor to let CellConstructure 
dealing with symmetries extracted with spglib.


## Installation

Once you make sure to have all the required packages installed on your system
and working, just type on the terminal

```bash
python setup.py install
```

while you are located in the same directory as the setup.py script is.

This program is also distributed upon PyPI. You can install it by typing

```bash
pip install CellConstructor
```
In this way you will not install the last developing version.


NOTE:
If you want to install the package into a system python distribution, the
installation commands should be executed as a superuser.

## GO!

To test if the installation runned properly, run the examples reported 
in the test directory. The python code in these tests should be
almost self explaining and introduce you to the potentiality of this library.

Please, note that all the function of the library have a proper numpy style
docstring.