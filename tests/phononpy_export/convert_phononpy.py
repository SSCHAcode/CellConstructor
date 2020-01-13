"""
In this file we convert the dynamical matrix into the phononpy format
"""
from __future__ import print_function

import cellconstructor as CC 
import cellconstructor.Phonons

# Load the dynamical matrix
supercell = (1,1,1)
dyn = CC.Phonons.Phonons("dynmat", 1)

# Save them into phononpy
dyn.save_phononpy(supercell)

print ()
