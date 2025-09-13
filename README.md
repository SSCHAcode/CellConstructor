# CellConstructor

Welcome to the CellConstructor python package!

## What is CellConstructor?
CellConstructor is a python library that allows to easily manipulate crystal structures and phonon dynamical matrices.
It is based on quantum espresso, and wraps in python many of the utilities available in the PHonon package as post-processing tools.
It is also interfaced with ASE for I/O of common file formats (.cif, input of common calculators, etc.) and to SPGLIB for a more stable tool for symmetry identification (also the Quantum Espresso symmetry library is available).


## What can I do with CellConstructor?
CellConstructor is a general purpouse library. Some of the things you can do with one single command:

1. Compute Quasi-Harmonic free energies from a dynamical matrix at any temperature.
2. Impose symmetries on a structure, or on a dynamical matrix.
3. Impose the acoustic sum rule on a dynamical matrix
4. Compute the phonon dispersion along high-symmetry paths
5. Extract harmonic randomly displaced configurations according to the dynamical matrix.

And also many others cool features!

## Requirements

To correnctly install and use the package, you need to have
1. python >= 2.7
2. ASE : Atomic Simulation Environment (suggested but not mandatory)
3. numpy
4. scipy
5. A fortran compiler
6. Lapack

The fortran compiler is required to compile the fortran libraries 
from Quantum ESPRESSO.

Suggested, but not required, is the installation of ASE and spglib. 
The presence of a valid ASE installation will enable some more features, 
like the possibility to load structures by any ASE supported file format, 
or the possibility to export the structures into a valid ASE Atoms class.
This library is able to compute symmetries from the structure, 
and inside the symmetry module there is a convertor to let CellConstructure 
dealing with symmetries extracted with spglib. 
However, for a more carefull symmetry analisys, we suggest the use of external tools like ISOTROPY.
This package can generate ISOTROPY input files for more advanced symmetry detection.

Please, note that some fortran libraries are needed to be compiled, therefore the Python header files should be localized by the compiling process. 
This requires the python distutils and developing tools to be properly installed.
On ubuntu this can be achieved by running:
```bash
sudo apt-get install python-dev
```

If you are using anaconda or pip, it should be automatically installed.


## Installation

To install prerequisites you can use the pip installation:
```bash
pip install -r requirements.txt
```
If you are running python2, then ase will fail the installation probably (as it requires python3).
To solve the issue, you can replace the ase line inside requirements.txt with:
ase==3.16.0
This will make it work also with python2.

Be sure you have also python-dev package, otherwise you will experience error while importing Python.h.
Also, you need gfortran and lapack, blas libraries installed.

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

<!--

If the compilation of the modules fails and you are using
an anaconda module on a 64bit machine, you have to install the conda gcc version.
You can do this by typing (on Linux):

```bash
conda install gxx_linux-64
```
or (on MacOS):
```bash
conda install clangxx_osx-64
```
-->


NOTE:
If you want to install the package into a system python distribution, the
installation commands should be executed as a superuser. 
Otherwise, append the --user flag to either the setup.py or the pip installation. 
In this way no administrator privileges is required, but the installation will be effective only for the current user.
Note that some python distribution, like anaconda, does not need the superuser, as it has an installation path inside the HOME directory.

You can install also using the intel compiler.
In this case, you must edit the setup.py script so that:
- remove the lapack and blas as extra library for the SCHAModules extension.
- add a new flag: 'extra_link_args = ["-mkl"]' to the extension. 

Remember to specify the intel compiler both to the compilation and for the running:
CC="icc"
LDSHARED="icc -shared"
otherwise the C module will give an error when loaded reguarding some "_fast_memcpy_" linking.


## GO!

To test if the installation runned properly, run the examples reported 
in the test directory. The python code in these tests should be
almost self explaining and introduce you to the potentiality of this library.

Please, note that all the functions of the library have a proper numpy style
docstring.

You can test the installation using the script:
```bash
cellconstructor_test.py
```

To run the complete testsuite, you can use the pytest, running the following command:
```bash
pytest
```

For a full API documentation, you can compile the documentation inside the UserGuide directory.  
To compile it simply use the make utility followed by the frontend.
For example, if you want the html version run:
```bash
make html
```
inside the UserGuide directory. It will generate a build directory that contains the html version of the full documentation.
