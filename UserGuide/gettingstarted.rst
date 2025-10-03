Getting started
===============

Now that we have correctly installed the CellConstructor, we can start testing the features.

The source code comes with a series of test and examples available under the ``tests`` directory of the git source.

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
  print("\n".join(["{:16.4f} cm-1".format(x * CC.Units.RY_TO_CM) for x in w_freq]))


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

In this way, we will create 10 files named *random_struct00000.scf*,
*random_struct00001.scf*, ...

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

In all these cases, to generate a good dynamical matrix for extracting randomly distributed configurations we need to define a dynamical matrix that is positive definite.

A good way to do that is to redefine our matrix as

.. math::

   \Phi_{ab}' = \sqrt{m_am_b}\sum_{\mu} e_\mu^a e_\mu^b \left|\omega_\mu^2 \right|

where :math:`\omega_\mu^2` and :math:`e_\mu` are, respectively, eigenvalues and eigenvectors of the original :math:`\Phi` matrix.

This can be achieved with just one line of code::

  dyn.ForcePositiveDefinite()

In the following tutorial, we create a random structure, associate to it a random dynamical matrix, print the frequencies of this dynamical matrix on the screen, force it to be positive definite, and the print the frequencies again.

.. code:: python

   import cellconstructor as CC
   import cellconstructor.Structure
   import cellconstructor.Phonons
   import numpy as np

   # We create a structure with 6 atoms
   struct = CC.Structure.Structure(6)

   # By default the structure has 6 Hydrogen atoms
   # struct.atoms = ["H"] * 6

   # We must setup the masses (in Ha)
   struct.masses = {"H" : 938}

   # We extract the 3 cartesian coordinates
   # Randomly for all the coordinates
   struct.coords = np.random.uniform(size = (6, 3))

   # We set a cubic unit cell
   # eye makes the identity matrix, 3 is the size (3x3)
   struct.unit_cell = np.eye(3)
   struct.has_unit_cell = True # Set the periodic boundary conditions

   # Now we create a Phonons class with the structure
   dyn = CC.Phonons.Phonons(struct)

   # We fill the gamma dynamical matrix with a random matrix
   dyn.dynmats[0] = np.random.uniform(size = (3*struct.N_atoms, 3*struct.N_atoms))

   # We impose the matrix to be hermitian
   dyn.dynmats[0] += dyn.dynmats[0].T

   # We print all the frequencies of the dynamical matrices
   freqs, pols = dyn.DyagonalizeSupercell()
   print("\n".join(["{:3d}) {:10.4f} cm-1".format(i+1, w * CC.Units.RY_TO_CM) for i, w in enumerate(freqs)]))

   # Now we Impose the dynamical matrix to be positive definite
   dyn.ForcePositiveDefinite()

   # Now we can print again the frequencies
   print("")
   print("After the  force to positive definite")
   freqs, pols = dyn.DyagonalizeSupercell()
   print("\n".join(["{:3d}) {:10.4f} cm-1".format(i+1, w * CC.Units.RY_TO_CM) for i, w in enumerate(freqs)]))


The output of this code is shown below. Note, since we randomly initialized the dynamical matrix, the absolute value of the frequencies may change.

.. code::

      1) -5800.6476 cm-1
      2) -5690.3146 cm-1
      3) -4521.1801 cm-1
      4) -4373.4443 cm-1
      5) -4184.2736 cm-1
      6) -4028.2141 cm-1
      7) -3220.6904 cm-1
      8) -2043.1963 cm-1
      9) -1073.2832 cm-1
     10)  2216.9358 cm-1
     11)  2952.8991 cm-1
     12)  3478.0255 cm-1
     13)  3801.1767 cm-1
     14)  4805.3197 cm-1
     15)  5073.9250 cm-1
     16)  5524.8080 cm-1
     17)  6162.4137 cm-1
     18) 14400.8452 cm-1

    After the  force to positive definite
      1)  1073.2832 cm-1
      2)  2043.1963 cm-1
      3)  2216.9358 cm-1
      4)  2952.8991 cm-1
      5)  3220.6904 cm-1
      6)  3478.0255 cm-1
      7)  3801.1767 cm-1
      8)  4028.2141 cm-1
      9)  4184.2736 cm-1
      10)  4373.4443 cm-1
      11)  4521.1801 cm-1
      12)  4805.3197 cm-1
      13)  5073.9250 cm-1
      14)  5524.8080 cm-1
      15)  5690.3146 cm-1
      16)  5800.6476 cm-1
      17)  6162.4137 cm-1
      18) 14400.8452 cm-1

In this tutorial we saw also how to create a random dynamical matrix. Note, the negative frequencies are imaginary: they are the square root of the negative eigenvalues of the dynamical matrix.

You may notice that this dynamical matrix at gamma does not satisfy the acoustic sum rule: it should have 3 eigenvalues of 0 frequency.
We will see in the next tutorials how to enforce all the symmetries and the acoustic sum rule on the dynamical matrix

Symmetries of the structure
---------------------------

It is very common to have a structure that violates slightly some symmetry. It can come from a relaxation that is affected from some noise, from experimental data.

Here, we will see how to use spglib and CellConstructor to enforce the symmetries on the structure. We will create a BCC structure, then we will add to the atoms a small random displacements so that spglib does not recognize any more the original group.
Then we will enforce the symmetrization with CellConstructor.

This tutorial requires spglib and the ASE packages installed.

.. code:: python

   from __future__ import print_function
   from __future__ import division
   import cellconstructor as CC
   import cellconstructor.Structure

   import spglib
   
   # We create a structure of two atoms
   struct = CC.Structure.Structure(2)

   # We fix an atom in the origin, the other in the center
   struct.coords[0,:] = [0,0,0]
   struct.coords[1,:] = [0.5, 0.5, 0.5]

   # We add an unit cell equal to the identity matrix (a cubic structure with :math:`a = 1`)
   struct.unit_cell = np.eye(3)
   struct.has_unit_cell = True # periodic boundary conditions on

   # Lets see the symmetries that prints spglib
   print("Original space group: ", spglib.get_spacegroup(struct.get_spglib_cell()))

   # The previous command should print
   # Original space group: Im-3m (299)

   # Lets store the symmetries and convert from spglib to the CellConstructor
   syms = spglib.get_symmetry(struct.get_spglib_cell())
   cc_syms = CC.symmetries.GetSymmetriesFromSPGLIB(syms)

   # We can add a random noise on the atoms
   struct.coords += np.random.normal(0, 0.01, size = (2, 3))

   # Let us print again the symmetry group
   print("Space group with noise: ", spglib.get_spacegroup(struct.get_spglib_cell()))

   # This time the code will print
   # Space group with noise: P-1 (2)
   #
   # This means that the structure lost all the symmetries

   # Now we enforce the symmetrization on the structure
   struct.impose_symmetries(cc_syms)

   # The previous command will print details on the symmetrization iterations
   print("Final group: ", spglib.get_spacegroup(struct.get_spglib_cell()))
   # Now the structure will be again in the Im-3m group.

You can pass to all spglib commands a threshold for the symmetrization. In this case you can also use a large threshold and get the symmetries of the closest larger space group. You can use them to constrain the symmetries.

Note, in some cases the symmetrization does not converge. If this happens, then the symmetries cannot be enforced on the structure.
It could also be possible that spglib identifies some symmetries with a threshold, the code impose them, but then the symmetries are still not recognized by spglib with a lower threshold. This happens when the symmetries you are imposing are not satisfied by the unit cell.
In that case, you have to manually edit the unit cell before imposing the symmetries.

  

Symmetries of the dynamical matrix
----------------------------------

In this tutorial we will create a random dynamical matrix of a high symmetry structure and enforce the symmetrization.
Constraining symmetries on the dynamical matrix can be achieved in two ways. Firstly, we will use the builtin symmetry engine from QuantumESPRESSO, then we will use the spglib.

The builtin Quantum ESPRESSO module performs the symmetrization in q space, it is much faster than spglib in large supercells. Moreover, it is installed together with CellConstructor and no additional package is required. However, spglib is much better in finding symmetries; it is a good tool to be used when the structure is already in a supercell. The CellConstructor module interfaces with spglib symmetry identification and performs the symmetrization of the dynamical matrix.

.. code:: python

   from __future__ import print_function

   # Import numpy
   import numpy as np

   # Import cellconstructor
   import cellconstructor as CC
   import cellconstructor.Structure
   import cellconstructor.Phonons
   import cellconstructor.symmetries

   # Define a rocksalt structure
   bcc = CC.Structure.Structure(2)
   bcc.coords[1,:] = 5.6402 * np.array([.5, .5, .5]) # Shift the second atom in the center
   bcc.atoms = ["Na", "Cl"]
   bcc.unit_cell = np.eye(3) * 5.6402 # A cubic cell of 5.64 A edge
   bcc.has_unit_cell = True # Setup periodic boundary conditions

   # Setup the mass on the two atoms (Ry units)
   bcc.masses = {"Na": 20953.89349715178,
   "Cl": 302313.43272048925}



   # Lets generate the random dynamical matrix
   dynamical_matrix = CC.Phonons.Phonons(bcc)
   dynamical_matrix.dynmats[0] = np.random.uniform(size = (3 * bcc.N_atoms,
   3* bcc.N_atoms))

   # Force the random matrix to be hermitian (so we can diagonalize it)
   dynamical_matrix.dynmats[0] += dynamical_matrix.dynmats[0].T
   
   # Lets compute the phonon frequencies without symmetries
   w, pols = dynamical_matrix.DiagonalizeSupercell()

   # Print on the screen the random frequencies
   print("Non symmetric frequencies:")
   print("\n".join(["{:d}) {:.4f} cm-1".format(i, w * CC.Units.RY_TO_CM)for i,w in enumerate(w)]))

   # Symmetrize the dynamical matrix
   dynamical_matrix.Symmetrize() # Use QE to symmetrize

   # Recompute the frequencies and print them in output
   w, pols = dynamical_matrix.DiagonalizeSupercell()
   print()
   print("frequencies after the symmetrization:")
   print("\n".join(["{:d}) {:.4f} cm-1".format(i, w * CC.Units.RY_TO_CM) for i,w in enumerate(w)]))



In this tutorial we first build the NaCl structure, then we generate a random force constant matrix.
After the symmetrization, the system will have 3 frequencies to zero (the acoustic modes at gamma) and 3 identical frequencies (negative or positive).
If you want to use this dynamical matrix, it is recommanded to force it to be positive definite with the command dynamical_matrix.ForcePositiveDefinite().

The only command actually required to symmetrize is:

.. code:: python

   dynamical_matrix.Symmetrize() # Use QE to symmetrize

This command will always use the espresso subroutines. You can, however, setup the symmetries from SPGLIB
and force the symmetrization in the supercell. This procedure is a bit more involved, as you need to create
and initialize manually the symmetry class.

The code to perform the whole symmetrization in spglib is

.. code:: python

   # Initialize the symmetry class
   syms = CC.symmetries.QE_Symmetry(bcc)
   syms.SetupFromSPGLIB() # Setup the espresso symmetries on spglib

   # Generate the real space dynamical matrix
   superdyn = dynamical_matrix.GenerateSupercellDyn(dynamical_matrix.GetSupercell())

   # Apply the symmetries to the real space matrix
   CC.symmetries.CustomASR(superdyn.dynmats[0])
   syms.ApplySymmetriesToV2(superdyn.dynmats[0])

   # Get back the dyanmical matrix in q space
   dynq = CC.Phonons.GetDynQFromFCSupercell(superdyn.dynmats[0], np.array(dynamical_matrix.q_tot), dynamical_matrix.structure, superdyn.structure)

   # Copy each q point of the symmetrized dynamical matrix into
   # the original one
   for i in range(len(dynamical_matrix.q_tot)):
	  dynamical_matrix.dynmats[i] = dynq[i,:,:]



The symmetrization here occurs in real space, therefore it is necessary in principle to transform the matrix in real space.
In this case it is not strictly necessary, as we have only one q-point at Gamma, therefore, the dynamical_matrix.dynmats[0] can be directly passed to the symmetrization subroutines, however, this is not true when more q points are present.

In this part we also showed how to explicitly perform a fourier transformation between real spaces dynamical matrices and q space quantities using the subroutine GetDynQFromFCSupercell.


Load and Save structures
------------------------


Cellconstructor supports many standard structure files.
For the structure, the basic format is the 'scf'. It is a format where both the unit cell and the cartesian position of all the atoms are explicitly given.
It is compatible with quantum espresso, therefore it can be appended on a espresso input file to perform a calculation on the given structure.

However, Cellconstructor can be interfaced with other libraries like ASE (Atomic Simulation Environment).
ASE provide I/O facilities for all most common DFT and MD programs, as well as the ability to read and write in many format, including 'cif', 'pdb', 'xyz' and so on.

To read and write in the native 'scf' format, use the following code:

.. code:: python

   struct = CC.Structure.Structure()
   struct.read_scf("myfile.scf")

   # --- some operation ----
   struct.save_scf("new_file.scf")

You can read all the file format supported by ASE using the read_generic_file function, however, you must have ASE installed

.. code:: python

   struct = CC.Structure.Structure()
   struct.read_generic_file("myfile.cif")


You can also directly convert an ASE Atoms into the CellConstructor Structure, and vice-versa

.. code:: python

   import ase
   import cellconstructor as CC
   import cellconstructor.Structure

   # Generate the N2 molecule using ASE
   N2_mol = ase.Atoms("2N", [(0,0,0), (0,0, 1.1)])
   
   struct = CC.Structure.Structure()
   struct.generate_from_ase_atoms(N2_mol)
   struct.save_scf("N2.scf")

Vice-versa, you can generate an ASE Atoms structure using the function get_ase_atoms() of the Structure class


.. autoclass:: Structures.Structures
   :members: get_ase_atoms


	     
Load and save the dynamical matrix
----------------------------------

CellConstructor is primarly ment for phonon calculations.
The main interface of the Phonon object is with quantum espresso.

We can read a quantum espresso dynamical matrix easily.

.. code:: python

   import cellconstructor as CC
   import cellconstructor.Phonons

   # Load the dynamical matrix in the quantum espresso format
   # ph.x of quantum espresso splits the output of the
   # dynamical matrices dividing per irreducible q points of the Brilluin zone.
   # This command will load a dynamical matrix with 3 irreducible q points.
   # The files are dyn1, dyn2 and dyn3
   dyn = CC.Phonons.Phonons("dyn", 3)

   # -- do something --

   # Now we save the dynamical matrix in the espresso format
   dyn.save_qe("new_dyn")

When loading from quantum espresso, CellConstructor will also import raman_tensor, dielectric tensor and effective charges.
The structure is read from the first dynamical matrix (usually the gamma point).
An experimental interface to phonopy is under developement.


Load phonons from Phonopy and ASE
---------------------------------

Often, you may have to load a dynamical matrix from other sources than quantum-espresso.
The two most common file formats are the Phonopy and the ASE calculation.

For phonopy, we expect you generate a FORCE_CONSTANTS file and a phonopy.yaml file containing the information about the structure and the atoms in the supercell.

NOTE: Up to version 1.0 the Phonopy importer assumes the FORCE_CONSTANTS and the structure are written in Rydberg atomic units. This will probably change in the near future.

.. code::

   import cellconstructor as CC
   import cellconstructor.Phonons

   # Load the dynamical matrix in phonopy format
   dyn = CC.Phonons.Phonons()
   dyn.load_phonopy("path/to/phonopy.yaml")

   # Save as new_dyn in quantum espresso format
   dyn.save_qe("new_dyn")

   # Save in the phonopy format
   dyn.save_phonopy("path/to/directory")

Optionally, it is possible to provide also the path to the FORCE_CONSTANTS file, which by default is assumed to be in the same directory as the phonopy.yaml.
This script can be employed to convert from Phonopy to quantum espresso file format.

In the same way, it is possible to generate the FORCE_CONSTANTS file with the method save_phonopy. Here, you just need to specify the directory.
Instead of the phonopy.yaml, the unitcell.in in quantum espresso format is generated. Phonopy is able to read this file.


The ASE calculator can be used to directly compute the dynamical matrix from finite displacements.
Once you have your ase.phonons.Phonons object, you can convert it into a CellConstructor Phonons object using the method get_dyn_from_ase_phonons.

.. code::
   
   import ase
   from ase.build import bulk
   from ase.calculators.emt import EMT
   from ase.phonons import Phonons
   
   import cellconstructor as CC, cellconstructor.Phonons

   # Here the code to get the ase.phonon.Phonons object
   # Setup crystal and EMT calculator
   atoms = bulk('Al', 'fcc', a=4.05)

   # Phonon calculator with ASE
   N = 6
   ph = Phonons(atoms, EMT(), supercell=(N, N, N), delta=0.05)
   ph.run()

   # Read forces and assemble the dynamical matrix
   ph.read(acoustic=True)
   ph.clean()

   # Convert into the cellconstructor and save in quantum espresso format
   dyn = CC.get_dyn_from_ase_phonons(ph)
   dyn.save_qe("my_dyn_computed_with_ase")




Generate a video of phonon vibrations
-------------------------------------

It could be very usefull to generate a video of a phonon mode, for post-processing reasons.

In this tutorial I will introduce the Manipulate module, that can be used for post-processing analysis.

In the following simple example, we will read the dynamical matrix ice_dyn1.
The methods that makes the video is GenerateXYZVideoOfVibrations.
It is quite self-explaining: it needs the dynamical matrix, the name of the file in witch to save the video,
the index of the vibrational mode.

You also need info on the video, the vibrational amplitude (in angstrom) the actual time step (in femtoseconds) and the number of time steps to be included.

Take in consideration that a vibration of 800 cm-1 has a period of about 40 fs.
In this case we are seeing a vibration of 3200 cm-1, whose period is about 10 fs.
Therefore we pick a dt = 0.5 fs to correctly sample the vibration and a total time of 50 fs (100 steps). In this way we will have about 5 full oscillations.

.. code:: python

   from __future__ import print_function

   import cellconstructor as CC
   import cellconstructor.Phonons
   import cellconstructor.Manipulate


   # Load a dynamical matrix that represent an ice structure (at gamma)
   dyn = CC.Phonons.Phonons("ice_dyn")

   # We dyagonalize the dynamical matrix
   w, p = dyn.DiagonalizeSupercell()

   # We pick the hardest mode
   mode_id = len(w) - 1

   # We must specify the amplitude of the vibrations (in A)
   amplitude = 0.8 #A

   # The time steps between two frams (in Femtoseconds)
   dt = 0.5

   # The total number of time steps
   N_t = 100


   # Save the video of the trajectory in a xyz file.
   CC.Manipulate.GenerateXYZVideoOfVibrations(dyn, "vibration.xyz",  mode_id, amplitude, dt, N_t)



This code will save the video as 'vibration.xyz'. You can load this file in your favorite viewer.
If you have ASE installed, you can view the video just typing in the console

.. code:: bash

   ase gui vibration.xyz



Here you will find more details about this API.


.. automodule:: Manipulate
   :members: GenerateXYZVideoOfVibrations



Radial pair distribution function
---------------------------------

A very usefull quantities, directly related to the static structure factor,
is the pair radial distribution function, defined as

.. math::

   g_{AB}(r) = \frac{\rho_{AB}^{(2)}(r)}{\rho_A(r) \rho_B(r)}


This quantity probes the how many couples of the AB atoms are inside a shell of
distance $r$ with respect to what we would expect in a non interacting system.

It is a standard quantity for liquid simulation.

In this tutorial we will take an harmonic dynamical matrix, generate
the harmonic ensemble, and compute the g(r) on it.

.. code:: python

   from __future__ import print_function
   import cellconstructor as CC
   import cellconstructor.Phonons
   import cellconstructor.Methods
   import matplotlib.pyplot as plt


   # Some info on the g(r)
   ENSEMBLE_SIZE=10000
   T = 0 # temperature in K

   # The maximum distances for computing the g(r)
   R_MIN = 0.9
   R_MAX_OH = 2
   R_MAX_HH = 3

   # The number of bins to divide the interval
   N_bins = 1000

   # The limits for the final plot
   R_LIM_OH= (0.75, 2)
   R_LIM_HH=(1, 3)


   # Load the ice XI dynamical matrix
   iceXI_dyn = CC.Phonons.Phonons("h2o.dyn", full_name = True)
   iceXI_dyn.Symmetrize() # Impose the sum rule

   # Use the dynamical matrix to generate the displacements
   print ("Generating displacements...")
   structures = iceXI_dyn.ExtractRandomStructures(ENSEMBLE_SIZE, T)

   # Get the g(r) between O and H atoms
   print ("Computing OH g(r)...")
   grOH = CC.Methods.get_gr(structures, "O", "H", R_MIN, R_MAX_OH, N_bins)
   print ("Computing HH g(r)...")
   grHH = CC.Methods.get_gr(structures, "H", "H", R_MIN, R_MAX_HH, N_bins)

   # Plot the result
   plt.plot(grOH[:,0], grOH[:,1])
   plt.xlabel("r [$\\AA$]")
   plt.ylabel("$g_{OH}(r)$")
   plt.title("O-H radial distribution function")
   plt.xlim(R_LIM_OH)
   plt.tight_layout()

   plt.figure()

   plt.plot(grHH[:,0], grHH[:,1])
   plt.xlabel("r [$\\AA$]")
   plt.ylabel("$g_{HH}(r)$")
   plt.title("H-H radial distribution function")
   plt.xlim(R_LIM_HH)
   plt.tight_layout()
   plt.show()


In this example I use a ice XI harmonic dynamical matrix at :math:`\Gamma`
computed with quantum espresso and the PBE exchange correlation functional.
You can find this file inside the tutorials/RadialDistributionFunction,
or you can replace it with your own dynamical matrix, to test it for your case.
You can decide your temperature, to test temperture effects.

The example is quite self-explaining, it will produce
two figures one for HH distance and one for the OH.


Indeed, if you are using molecular dynamics, you can load your array of structures and pass it through the get_gr functions of the Methods module.

A detailed documentation of this method is available here:

.. automodule:: Methods
   :members: get_gr
