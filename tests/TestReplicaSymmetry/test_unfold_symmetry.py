# -*- coding: utf-8 -*-

from __future__ import print_function

"""
Here we test the symmetry unfolding of a particular configuration.
"""
import numpy as np

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.symmetries


import sys, os
import pytest


def test_unfold_symmetry():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)



    # Load the structure
    structure = CC.Structure.Structure()
    structure.read_scf("unit_cell_structure.scf")

    # Get symmetries
    qe_sym = CC.symmetries.QE_Symmetry(structure)
    qe_sym.SetupQPoint(verbose=True)
    syms, irts = qe_sym.GetSymmetries(True)
    nat = structure.N_atoms
    nsyms = len(syms)

    # Get a displaced structure
    d_structure = structure.copy()
    d_structure.coords += np.random.normal(scale = 0.1, size=np.shape(d_structure.coords))

    # Get the new pool of structures
    new_d_structures = []
    for i in range(nsyms):
        u_disp = d_structure.coords - structure.coords
        new_u_disp = CC.symmetries.ApplySymmetryToVector(syms[i], u_disp, structure.unit_cell, irts[i, :])
        tmp = structure.copy()
        tmp.coords += new_u_disp
        tmp.save_scf("replica_%d.scf" % i)
        new_d_structures.append(tmp)

    print ("Symmetry of a displaced structure:")
    qe_sym = CC.symmetries.QE_Symmetry(new_d_structures[0])
    qe_sym.SetupQPoint(verbose=True)
    print ()

    # Average all the displacements to see if the symmetries are recovered correctly
    new_structure = structure.copy()
    new_structure.coords = np.sum([x.coords for x in new_d_structures], axis = 0) / nsyms

    # Get again the symmetries
    print ("Symmetries after the sum:")
    qe_sym = CC.symmetries.QE_Symmetry(new_structure)
    qe_sym.SetupQPoint(verbose=True)
    print ()

    # Lets check if the structure is the same as before 
    # Should be 0 only if the symmeties are enaugh to have 0 force.
    print ("Difference from the first one:")
    print ( np.sqrt(np.sum((new_structure.coords - structure.coords)**2)))

if __name__ == "__main__":
    test_unfold_symmetry()
