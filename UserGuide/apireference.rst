
  
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


The Manipulate Module
======================

This is an extra tool used to manipulate the Phonons and the
structures to perform some analysis. It can be used to
get video of vibrations.

.. automodule:: Manipulate
   :members:
      
      
      
The Symmetry Class
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
