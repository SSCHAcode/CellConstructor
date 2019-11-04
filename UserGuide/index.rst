.. CellConstructor documentation master file, created by
   sphinx-quickstart on Mon Nov  4 15:01:20 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CellConstructor's documentation!
===========================================

CellConstructor is a Python library to manipulate atomic
structures and to compute vibrational properties within
the harmonic approximation.

It can read harmonic dynamical matrices, interpolate,
enforce symmetrization, fourier transform, extract frequencies
and phonon polarizations, compute Raman or IR cross sections,
compute the harmonic free energy or use the quasi-harmonic approximation, transform the dynamical matrix by changing the unit cell,
compute energy and forces of a configuration within the harmonic approximation, compute the single or the two phonon density of state.
On the top of the calculation feature, it can be used also to visualize results, as it can generate the atomic video of a vibrational mode and export it in the xyz format.

In this guide all the functions are documented in detail.


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

  
The Structure Module
====================

This module is the basis of CellConstructor.
Here, the atomic structure and its method are defined.

This class is can be imported in the cellconstructor.Structure
file.

.. autoclass:: Structure.Structure
   :members:

The Methods
===========

This library contains a set of tools to
perform some general computations. For example to
pass between crystal and Cartesian units, read espresso
input namelists, compute reciprocal lattice vectors. 
      
.. automodule:: Methods
   :members:
      
The Phonon Module
=================

The Phonon module of cellconstructor deals with the dynamical
matrix.
All the harmonic properties of a crystal can be computed within this module. 

Besides the Phonons class, some method is directly callabile in the Phonons module.

.. automodule:: Phonons
   :members:

THE MANIPULATE METHODS
======================

This is an extra tool used to manipulate the Phonons and the
structures to perform some analysis. It can be used to
get video of vibrations.

.. automodule:: Manipulate
   :members:
      
THE SYMMETRY CLASS
==================

This is the symmetry class, used to perform symmetry operation.
It can be used to enforce symmetries on dynamical matrix, vectors, 3 or 4 rank tensors.
It can apply symmetries both in real space (slow) and in q space.

To recognize symmetries it has a builtin Fortran code
taken from QuantumESPRESSO software suite.
If symmetries are used in real-space it is possible to use
spglib instead.

.. autoclass:: symmetries.QE_Symmetry
   :members:

Besides the QE_Symmetry class, other methods are available in the symmetries module to operate directly on symmetries.
      
.. automodule:: symmetries
   :members:

      
