# -*- coding: utf-8 -*-
from __future__ import print_function
"""
This code is an example on how to load a supercell phonon calculation at
several q points.
This is the ice XI structure of crystalline H2O
"""

import cellconstructor as CC
import cellconstructor.Phonons
import numpy as np


# Read the phonons (8 irreducible q points)
dynmat = CC.Phonons.Phonons("unitcell.in",use_Phonopy=True, nqirr=4)

# Perform compute the frequency in the second q point
print ("The q point is:")
print (dynmat.q_tot)
print ()
freq, pol_vect = dynmat.DyagDinQ(0)
print ("Frequencies:")
print (np.sort(freq))


# Test the saving method
dynmat.save_qe("QE_dyn")
