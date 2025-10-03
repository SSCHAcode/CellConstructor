# -*- coding: utf-8 -*-
from __future__ import print_function
import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.symmetries

try:
    import spglib
except:
    raise ValueError("Error, to run this example you need to install spglib")
    
    
"""
This code loads a dynamical matrix with the structure that barely 
satisfy a C2/c monoclinic group (with a 0.04 threshold) and
constrain the symmetries to allow programs like quantum espresso
to detect symmetries correctly.


NOTE: To recognize symmetries this example uses spglib.
"""

import sys, os

def test_impose_symmetry():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    # initialize the dynamical matrix
    dyn = CC.Phonons.Phonons("old_dyn", full_name=True)

    # Print the symmetry group at high threshold
    GROUP = spglib.get_spacegroup(dyn.structure.get_spglib_cell(), 0.05)
    s_group_expected = spglib.get_spacegroup(dyn.structure.get_spglib_cell())
    print ("Space group with high threshold:", s_group_expected)
    print ("Space group with low threshold:", GROUP)

    # Get the symmetries from the new spacegroup
    symmetries = spglib.get_symmetry(dyn.structure.get_spglib_cell(), symprec = 0.05)
    print("Number of symmetries: {}".format(len(symmetries["rotations"])))

    # Transform the spglib symmetries into the CellConstructor data type
    sym_mats = CC.symmetries.GetSymmetriesFromSPGLIB(symmetries, True)
    # Force the symmetrization
    dyn.structure.impose_symmetries(sym_mats)

    # Check once again the symetry
    s_group_after = spglib.get_spacegroup(dyn.structure.get_spglib_cell())
    print ("New space group with high threshold:", s_group_after)

    assert s_group_after == GROUP


if __name__ == "__main__":
    test_impose_symmetry()
