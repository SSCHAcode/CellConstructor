# -*- coding: utf-8 -*-
import cellconstructor as CC
import numpy as np
from ase.visualize import view

RyToCm = 109691.40235

"""
In this script I will show how to read a QE dynmat into the cell constructure
generating both the structure and all the phonon information.
"""

# Load the QE dynmat
hydrogen_structure = CC.Phonons("hydrogen_dyn")

# Show the structure with the ASE viewer
ase_struct = hydrogen_structure.structure.get_ase_atoms()
view(ase_struct)

# Try to dyagonalize the matrix to check the frequencies
print hydrogen_structure.structure.masses
freq, pol_vect = hydrogen_structure.DyagDinQ(0)
print "Frequencies:"
print np.sort(freq * RyToCm)