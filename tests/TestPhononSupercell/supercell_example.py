# -*- coding: utf-8 -*-

"""
This code is an example on how to load a supercell phonon calculation at
several q points.
This is the ice XI structure of crystalline H2O
"""

import cellconstructor as CC
import numpy as np
from ase.visualize import view

RyToCm = 109691.40235

# Read the phonons (8 irreducible q points)
iceXI = CC.Phonons("dynmat", 8)

# Show the structure
ase_iceXI = iceXI.structure.get_ase_atoms()
view(ase_iceXI)

# Perform compute the frequency in the second q point
print "The q point is:"
print iceXI.q_tot[1]
print ""
freq, pol_vect = iceXI.DyagDinQ(1)
print "Frequencies:"
print np.sort(freq * RyToCm)
