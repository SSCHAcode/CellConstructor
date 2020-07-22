# -*- coding: utf-8 -*-
from __future__ import print_function
"""
This code is an example on how to load the dynamical matrix calculated by
Phonopy and convert it to the QE format.
This is the ice XI structure of crystalline H2O
"""

import cellconstructor as CC
import cellconstructor.Phonons
import numpy as np
import os


total_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(total_path)
                             
# Read the phonons (4 irreducible q points)
dynmat = CC.Phonons.Phonons("unitcell.in",use_Phonopy=True, nqirr=4)

# Write all the q points and compute the frequency in Gamma
print ("The q points are:")
print (dynmat.q_tot)
print ()
freq, pol_vect = dynmat.DyagDinQ(0)
print ("Frequencies in Gamma:")
print (np.sort(freq))


# Test the saving method
dynmat.save_qe("QE_dyn")
