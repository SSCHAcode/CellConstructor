************
Introduction
************

What is CellConstructor?
========================

The CellConstructor is a python module originally created to
manipulate atomic structures for *ab-initio* simulations.

It can be used in interactive mode, or through python scripting, and can be interfaced with other libraries like ASE (Atomic Simulation Environment) or spglib for symmetries.

The library is constituted of three main modules:
1. ``Structures``: The module class that allows the manipulation of the atomic structure, including constraining symmetries, apply deformations on the cell, extracting molecular information (averaging bond lengths).
2. ``Phonons``: The module that allows the manipulation of the harmonic dynamical matrix. It includes utilities like the Fourier transform, generating randomly distributed configurations, thermodynamic properties, computing frequencies and phonon modes, Raman and IR responce, as well as interpolating in bigger cells.
3. ``symmetries``: The module that allows the symmetry search and constraining. It allows the symmetrization of vectors, matrices and 3 or 4 rank tensors. It has its own builtin symmetry engine, but ``spglib`` can be used. It can export also input files for ``ISOTROPY`` symmetry analysis program.





Requirements
============

The CellConstructor can be installed with ``distutils``.
First of all, make sure to satisfy the requirements

1. python >= 2.7 and < 3
2. ASE : Atomic Simulation Environment (suggested but not mandatory)
3. numpy
4. scipy
5. A fortran compiler
6. Lapack

The fortran compiler is required to compile the fortran libraries from Quantum ESPRESSO.

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

.. bash:: 

   $ sudo apt-get install python-dev

   

If you are using anaconda or pip, it should be automatically installed.


Installation
============


Once you make sure to have all the required packages installed on your system
and working, just type on the terminal

.. bash::
   $ python setup.py install


while you are located in the same directory as the setup.py script is.

This program is also distributed upon PyPI. You can install it by typing

.. bash::
   $ pip install CellConstructor

In this way you will not install the last developing version.

If the compilation of the modules fails and you are using
an anaconda module on a 64bit machine, you have to install the conda gcc version.
You can do this by typing (on Linux):

.. bash::
   $ conda install gxx_linux-64

or (on MacOS):

.. bash::
   $ conda install clangxx_osx-64



NOTE:
If you want to install the package into a system python distribution, the
installation commands should be executed as a superuser. 
Otherwise, append the --user flag to either the setup.py or the pip installation. 
In this way no administrator privilages is required, but the installation will be effective only for the current user.
Note that some python distribution, like anaconda, does not need the superuser, as they have an installation path inside the HOME directory.

You can install also using the intel compiler.
In this case, you must edit the setup.py script so that:
- remove the lapack and blas as extra library for the SCHAModules extension.
- add a new flag: 'extra_link_args = ["-mkl"]' to the extension. 

Remember to specify the intel compiler both to the compilation and for the running:
CC="icc"
LDSHARED="icc -shared"
otherwise the C module will give an error when loaded reguarding some "_fast_memcpy_" linking.



Test the installation
---------------------

You can run the testsuite to test your installation as

.. highlight :: bash

    cellconstructor_test.py

The execution of the test suite can require some time. If everything is OK, then the softwere is correctly installed and working.



