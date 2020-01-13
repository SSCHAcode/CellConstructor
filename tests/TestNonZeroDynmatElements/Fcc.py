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
import cellconstructor.symmetries

import spglib
from ase.visualize import view

# Init the hydrogen mass
H_mass = 918.836054935965

# Generate the initial structure
struc = CC.Structure.Structure(2)
struc.atoms = ["H", "He"]
struc.masses = { "H" : H_mass , "He" : H_mass * 4 }

# Get a cubic unit cell
struc.unit_cell[0,:] = np.array([1, 1, 0])
struc.unit_cell[1,:] = np.array([1,-1, 0])
struc.unit_cell[2,:] = np.array([1, 0, 1])
struc.has_unit_cell = True

# Place the second atom in the center of the unit cell
struc.coords[1, :] = np.array([1,0,0])
struc.fix_coords_in_unit_cell()

# Display the structure
ase_atoms = struc.get_ase_atoms()
view(ase_atoms)

# Print the symmetry group
print ("Symmetry group: ", spglib.get_spacegroup(ase_atoms))

# Load the symmetries from SPGLIB
symmetries = CC.symmetries.GetSymmetriesFromSPGLIB( spglib.get_symmetry(ase_atoms))

# Prepare the Force constant matrix
#dynmat = CC.Phonons.Phonons(structure = struc)

# Setup a random hermitian force constant matrix 
#dynmat.dynmats[0] = np.random.normal(size = (6,6))
#dynmat.dynmats[0] += np.transpose(dynmat.dynmats[0])

# Load 
dynmat = CC.Phonons.Phonons("RockSalt.dyn", full_name = True)

## Apply the symmetries
dynmat.ApplySumRule()
#dynmat.ForceSymmetries(symmetries)

# Initialize the symmetries using the QE module
qe_sym = CC.symmetries.QE_Symmetry(dynmat.structure)
#qe_sym.SetupQPoint(np.array([0,0,0]))
qe_sym.InitFromSymmetries(symmetries, np.array( [0,0,0] ))
syms =  qe_sym.GetSymmetries()


# Parse the force constant matrix
#qe_sym.SymmetrizeDynQ(dynmat.dynmats[0], np.array([0,0,0]))
dynmat.ForceSymmetries(syms)

# Write the dynamical matrix
dynmat.save_qe("RockSalt.dyn_end", True)
