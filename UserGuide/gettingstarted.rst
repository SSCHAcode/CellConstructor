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


