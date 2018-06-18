# -*- coding: utf-8 -*-

"""
This example file convert a primitive cell of ice XI into the
conventional one, showing how simple it is doing this cell
manipulation with the cellconstructor.
"""

import cellconstructor as CC
import cellconstructor.Structure
from ase.visualize import view

# Load the structure from the scf file
primitive_cell_str = CC.Structure.Structure()
primitive_cell_str.read_scf("primitive_cell.scf")

# Use ase visualizer to show the structure in the primitive cell
view(primitive_cell_str.get_ase_atoms())

# Get the conventional cell structure
conventional_cell_str = primitive_cell_str.get_strct_conventional_cell()

# Save the new structure into a scf file
conventional_cell_str.save_scf("conventional_cell.scf")
print "Generated file conventiona_cell.scf"

# View the structure in the conventional cell
view(conventional_cell_str.get_ase_atoms())