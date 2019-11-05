Getting started
===============

Now that we have correctly installed the CellConstructor, we
can start testing the features.

The source code comes with a series of test and examples
available under the ``tests`` directory of the git source.

As a first example, we will load a dynamical matrix in the Quantum ESPRESSO PHonon output style, compute the eigenvalues and print them into the screen.

First of all, lets copy the file ``tests/TestPhononSupercell/dynmat1`` inside the working directory::

  import cellconstructor as CC
  import cellconstructor.Phonons
  import cellconstructor.Units
  
  # Load the dynamical matrix
  dyn = CC.Phonons.Phonons("dynmat")

  # Compute the polarization vector and the phonon frequencies
  w_freq, pols = dyn.DyagonalizeSupercell()

  # Print the frequencies on the screen
  print("\n".join("{:16.4f} cm-1".format(x * CC.Units.RY_TO_CM) for x in w_freq))


If we run this script using python, it will print for each line the frequency in cm-1 of each phonon mode at gamma.

Note that we open the dynamical matrix dynmat1 with the line::

  dyn = CC.Phonons.Phonons("dynmat")

The current dynamical matrix is a (3,3,2) supercell of ice XI.
Quantum ESPRESSO stores the matrix in fourier space, as it is block diagonal for the Block theorem (and occupies less space).
Each  blocks is a different q-point in the Brillouin zone.
In total there are 3*3*2 = 18 different q-points. However, some q points are linked by symmetries. In this particular case, only 8 q-points are indepenent. This is why we have 8 files dynmat1 ... dynmat8.

To load all of them, we need to specify the keyword ``nqirr`` to the loading line::

  dyn = CC.Phonons.Phonons("dynmat", nqirr = 8)
  # Or alternatively
  # dyn = CC.Phonons.Phonons("dynmat", 8)


In our first example, we loaded only the first file, dynmat1, as the default value of nqirr is 1.

Given the q-points, we are able to compute the supercell. For example, try to run the code::

  >>> dyn = CC.Phonons.Phonons("dynmat")
  >>> print (dyn.GetSupercell())
  [1,1,1]
  
  >>> dyn = CC.Phonons.Phonons("dynmat", 8)
  >>> print (dyn.GetSupercell())
  [3,3,2]

As you can see, CellConstructor correctly recognized the supercell mesh once after reading all the 8 irreducible q points.
If you try to print again the frequencies when you load all the 8 irreducible q-points, you will have much more modes, that are all the possible modes of the supercell.

However, there is a way to print only the modes of a particular q point::

  q_index = 0
  w_freq, pols = dyn.DyagDinQ(q_index)
  
Here, ``q_index`` is the index of the q-point that you want to diagonalize. Usually ``q_index = 0`` corresponds to the Gamma point. To print a list of all the q-points::

  print("\n".join("{:4d})  {:16.4f} {:16.4f} {:16.4f}".format(iq, *list(q)) for iq, q in enumerate(dyn.q_tot)))

This will print on the screen the q points preceeded by the respective index. Note that the number should be always equal to the number of cells.


Tutorials
=========


In this section, I will show how simple tasks can be performed using CellConstructor.


Extract an harmonic random ensemble
-----------------------------------

You can use CellConstructor to extract an harmonic esnemble.
It can be used for measuring electron-phonon properties, as you can average
dielectric function on this ensemble to get phonon mediated electronic properties.

To get a list of structures distributed according to a dynamical matrix use the following commands::

  import cellconstructor as CC
  import cellconstructor.Phonons

  # We read the dynmat1 file (quantum espresso gamma dynamical matrix)
  dyn = CC.Phonons.Phonons("dynmat")

  structures = dyn.ExtractRandomStructures(size = 10, T = 100)

Then, ``structures`` is a list of ``CC.Structures.Structures``.
We have extracted 10 structures at a temperature of 100 K.
You can save them in the .scf format, that is ready to be copied on the bottom of a quantum espresso pw.x input file for a calculation of the band structure for example::

  for i, struct in enumerate(structures):
      struct.save_scf("random_struct{:05d}.scf".format(i))

In this way, we will create 10 files named
.. code:: bash

   random_struct00000.scf
   random_struct00001.scf
   random_struct00002.scf
   random_struct00003.scf
   ...


Each of them will be ready to be used in a quantum espresso pw.x calculation with ibrav=0.

Alternatively, one can convert them into the ASE atoms object. In this way
any configured calculator can be used directly inside python::

  for i, struct in enumerate(structures):
      ase_atmoms = struct.get_ase_atoms()

      # Here the code to perform the ab-initio calculation
      # with ase

Note that we used a gamma dynamical matrix. To generate a random structure in a supercell you need to first generate a supercell real space dynamical matrix. This is covered in the next tutorial.

If you are using the python-sscha package, it has a class Ensemble that can automatically generate and store the ensemble using this function.

For more details on the structure generation, I remand to the specific documentation:

.. autoclass:: Phonons.Phonons
   :members: ExtractRandomStructures



Generate a real space supercell dynamical matrix
------------------------------------------------

Many operations are possible only when the dynamical matrix
is expressed in real space supercell, as the extraction of the
random ensemble.

So it is crucial to be able to define a dynamical matrix in the supercell.
Luckily, it is very easy::

  import cellconstructor as CC
  import cellconstructor.Phonons

  # We load the dynmat1 ... dynmat8 files
  # They are located inside tests/TestPhononSupercell
  dyn = CC.Phonons.Phonons("dynmat", nqirr = 8)

  super_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())

  # Now we can do what we wont with the super_dyn variable.
  # For example we can extract a random ensemble in the supercell

  structures = super_dyn.ExtractRandomStructures(size = 10, T = 100)
  
Also in this case, please refer to the official documentation

.. autoclass:: Phonons.Phonons
   :members: GenerateSupercellDyn



  
Force a dynamical matrix to be positive definite
------------------------------------------------

This is an important task if you want to use the dynamical matrix to extract random configurations.
Often, harmonic calculation leads to imaginary frequencies. This may happen if:

1. The calculation is not well converged.
2. The dynamical matrix comes from interpolation of a small grid
3. The structure is in a saddle point of the Born-Oppenheimer energy landscape.

In all these cases, to generate a good dynamical matrix for extracting randomly distributed configurations
