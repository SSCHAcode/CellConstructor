# -*- coding: utf-8 -*-
from __future__ import print_function
"""
This code generates a random dynamical matrix
of a trial bcc crystal. Then it uses symmetries to
determine the non zero element of the Gamma dynamical matrix.
"""

import numpy as np
import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.Methods

import spglib
from ase.visualize import view

# Init the hydrogen mass
H_mass = 918.836054935965

# Generate the initial structure
struc = CC.Structure.Structure(2)
struc.atoms = ["H", "O"]
struc.masses = { "H" : H_mass , "O" : H_mass * 16 }

# Get a cubic unit cell
struc.unit_cell = np.eye(3) * 2
struc.has_unit_cell = True

# Place the second atom in the center of the unit cell
struc.coords[1, :] = np.ones(3) 

# Display the structure
ase_atoms = struc.get_ase_atoms()
view(ase_atoms)

# Print the symmetry group
print ("Symmetry group: ", spglib.get_spacegroup(ase_atoms))

# Load the symmetries from SPGLIB
symmetries = CC.Methods.GetSymmetriesFromSPGLIB( spglib.get_symmetry(ase_atoms))

# Prepare the Force constant matrix
dynmat = CC.Phonons.Phonons(structure = struc)

# Setup a random hermitian force constant matrix 
dynmat.dynmats[0] = np.random.normal(size = (6,6))
dynmat.dynmats[0] += np.transpose(dynmat.dynmats[0])

# Apply the symmetries
dynmat.ApplySumRule()
dynmat.ForceSymmetries(symmetries)

# Write the dynamical matrix
dynmat.save_qe("non_zero.dyn", True)
