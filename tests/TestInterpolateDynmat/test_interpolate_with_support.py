# -*- coding: utf-8 -*-

"""
Here we interpolate a dynamical matrix in a finer grid.
"""
from __future__ import print_function

import cellconstructor as CC
import cellconstructor.Phonons

import sys, os
import pytest

def test_interpolate_with_support():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dyn_prova")

    harm_coarse = CC.Phonons.Phonons("harm_support")
    harm_fine = CC.Phonons.Phonons("harm_support", 2)

    # Interpolate on a 3x3x2 supercell
    new_dyn = dyn.Interpolate((1,1,1), (2,2,1), harm_coarse, harm_fine, symmetrize=False)

    symqe = CC.symmetries.QE_Symmetry(new_dyn.structure)
    symqe.SetupQPoint(verbose=True)


    print (new_dyn.q_stars)
    q_stars, q_index = symqe.SetupQStar(new_dyn.q_tot)
    # Save the new dynamical matrix
    new_dyn.save_qe("new_dyn")

if __name__ == "__main__":
    test_interpolate_with_support()
    
