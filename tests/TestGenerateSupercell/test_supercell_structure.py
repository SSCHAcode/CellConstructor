# -*- coding: utf-8 -*-

"""
This simple test generates the  supercell using both the self defined and 
the quantum-espresso convention.
Then the two structure are displayed to see if they appear equal
"""

import cellconstructor as CC
import cellconstructor.Structure

from ase.visualize import view

SUPERCELL = (3,3,2)

# Load a simple structure
simple_struct = CC.Structure.Structure()
simple_struct.read_scf("unit_cell_structure.scf")


# Generate the conventional supercell
super1 = simple_struct.generate_supercell(SUPERCELL, QE_convention=False)
super2 = simple_struct.generate_supercell(SUPERCELL, QE_convention=True) # This is the default


view(super1.get_ase_atoms())
view(super2.get_ase_atoms())


# Print the displacement between the atoms
print super1.get_displacement(super2)