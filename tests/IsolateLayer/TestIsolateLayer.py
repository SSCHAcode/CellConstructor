# -*- coding: utf-8 -*-

"""
In this example a layer of atoms is isolated, to allow the separate
calculation of small part of the structure.
"""

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Structure

from ase.visualize import view

LAYER = [6,22,18, 10, 38, 26, 30, 34, 14, 2, 42, 46]

# Get the dynamical matrix from which to extract the structure
struct = CC.Structure.Structure()
struct.read_scf("Hydrogen.scf")

# Setup the conventional cell
new_struct = struct.generate_supercell( (2,1,2) )
new_struct.unit_cell[0,:] =  (struct.unit_cell[0,:] - struct.unit_cell[2,:])
new_struct.unit_cell[2,:] =  (struct.unit_cell[0,:] + struct.unit_cell[2,:])
new_struct.fix_coords_in_unit_cell()

# Show the structure
view(new_struct.get_ase_atoms())

# Extract the layer
layer = new_struct.IsolateAtoms(LAYER)

view(layer.get_ase_atoms())