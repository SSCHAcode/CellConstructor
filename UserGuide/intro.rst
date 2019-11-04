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





