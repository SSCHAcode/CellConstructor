"""
In this file we convert the dynamical matrix into the phononpy format
"""

import cellconstructor as CC 
import cellconstructor.Phonons

# Load the dynamical matrix
supercell = (3,3,2)
dyn = CC.Phonons.Phonons("dynmat", 8)

# Save them into phononpy
dyn.save_phononpy("FORCE_CONSTANTS", supercell)

print "Done"