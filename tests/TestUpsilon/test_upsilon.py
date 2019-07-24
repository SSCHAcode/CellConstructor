from __future__ import print_function

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons

T = 100

# Load a simple dynamical matrix
dyn = CC.Phonons.Phonons("../TestSymmetriesSupercell/dyn.SnSe.", 3)

# Get the upsilon matrix for the supercell
superdyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
ups1 = superdyn.GetUpsilonMatrix(T)
ups2 = dyn.GetUpsilonMatrix(T)

delta = np.sqrt(np.sum( (ups1 - ups2)**2))
print("The distance is {}.".format(delta))

