# -*- coding: utf-8 -*-
import cellconstructor as CC
import cellconstructor.Structure

from ase.visualize import view

"""
In this example few simple structures in the QuantumESPRESSO format
are read. This program test if the (crystal) unit, alat, cell and others works
properly or not.
"""

# Load the first structure with crystal coordinates
structure1 = CC.Structure.Structure()
structure1.read_scf("crystal.scf")

# Display the structure using ase viewer
view(structure1.get_ase_atoms())