# -*- coding: utf-8 -*-
from __future__ import print_function
"""
This simple test generates the  supercell using both the self defined and 
the quantum-espresso convention.
Then the two structure are displayed to see if they appear equal
"""

import cellconstructor as CC
import cellconstructor.Structure

from ase.visualize import view

import numpy as np

import sys, os

def test_generate_supercell():

    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    SUPERCELL = (3,3,2)

    # Load a simple structure
    simple_struct = CC.Structure.Structure()
    simple_struct.read_scf("unit_cell_structure.scf")


    # Generate the conventional supercell
    super1 = simple_struct.generate_supercell(SUPERCELL, QE_convention=False)
    super2 = simple_struct.generate_supercell(SUPERCELL, QE_convention=True) # This is the default

    # Check that the two are equal
    assert super1.N_atoms == super2.N_atoms


    for i in range(super1.N_atoms):
        is_zero = False
        for j in range(super2.N_atoms):
            # Skip different atoms
            if super2.atoms[j] != super1.atoms[i]:
                continue

            r_vec = CC.Methods.get_closest_vector(super1.unit_cell,
                                                  super1.coords[i, :] - super2.coords[j,:])

            if np.sqrt(r_vec.dot(r_vec)) < 1e-7:
                is_zero = True
                break

        if not is_zero:
            print("The atom {} has not corrispondence:".format(i+1))
            print("coord: {}".format(super1.coords[i,:]))
            print("All other coordinates:")
            print(super2.coords[:,:])

        assert is_zero, "Error, I did not find the corrispondence of one atom."
        



if __name__ == "__main__":
    test_generate_supercell()
