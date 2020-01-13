# -*- coding: utf-8 -*-
from __future__ import print_function
"""
This example file convert a primitive cell of ice XI into the
conventional one, showing how simple it is doing this cell
manipulation with the cellconstructor.
"""

import cellconstructor as CC
import cellconstructor.Structure
from ase.visualize import view

import sys, os

import pytest


def test_cell_convertion():

    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    # Load the structure from the scf file
    primitive_cell_str = CC.Structure.Structure()
    primitive_cell_str.read_scf("primitive_cell.scf")

    # Use ase visualizer to show the structure in the primitive cell
    #view(primitive_cell_str.get_ase_atoms())

    # Get the conventional cell structure
    conventional_cell_str = primitive_cell_str.get_strct_conventional_cell()

    # Save the new structure into a scf file
    conventional_cell_str.save_scf("conventional_cell.scf")
    print("Generated file conventiona_cell.scf for ice")

    # Do the same with the hydrogen
    pcell = CC.Structure.Structure()
    pcell.read_scf("hydrogen.scf")
    #view(pcell.get_ase_atoms())
    pcell.get_strct_conventional_cell().save_scf("hydrogen_conventiona.scf")
    #view(pcell.get_strct_conventional_cell().get_ase_atoms())

    # View the structure in the conventional cell
    #view(conventional_cell_str.get_ase_atoms())

if __name__ == "__main__":
    test_cell_convertion()
