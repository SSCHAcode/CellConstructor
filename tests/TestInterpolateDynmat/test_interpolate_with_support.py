# -*- coding: utf-8 -*-

"""
Here we interpolate a dynamical matrix in a finer grid.
"""
from __future__ import print_function

import cellconstructor as CC
import cellconstructor.Phonons
import numpy as np

import sys, os
import pytest

def test_interpolate_with_support():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dyn_prova")

    harm_coarse = CC.Phonons.Phonons("harm_support")
    harm_fine = CC.Phonons.Phonons("harm_support", 2)

    harm_coarse.AdjustToNewCell(dyn.structure.unit_cell)
    harm_fine.AdjustToNewCell(dyn.structure.unit_cell)
    harm_coarse.structure.coords = dyn.structure.coords
    harm_fine.structure.coords = dyn.structure.coords

    # Interpolate on a 3x3x2 supercell
    new_dyn = dyn.Interpolate((1,1,1), (2,2,1), harm_coarse, harm_fine, symmetrize=False)
    new_dyn2 = dyn.Interpolate((1,1,1), (2,2,1), harm_coarse, harm_fine, symmetrize=False, force_old_method = True)

    w1, p1 = new_dyn.DiagonalizeSupercell()
    w2, p2 = new_dyn2.DiagonalizeSupercell()

    thr = np.max(np.abs(w1 - w2)) < 1e-8
    if not thr:
        print("Error, discrepancies between interpolations.")
        print("\n".join(["{:5d}) {:.4f} cm-1 | {:.4f} cm-1".format(i, w1[i] * CC.Units.RY_TO_CM, w2[i] * CC.Units.RY_TO_CM) for i in range(len(w1))]))
        raise ValueError("Error, the frequencies are different")

    symqe = CC.symmetries.QE_Symmetry(new_dyn.structure)
    symqe.SetupQPoint(verbose=True)


    print (new_dyn.q_stars)
    q_stars, q_index = symqe.SetupQStar(new_dyn.q_tot)
    # Save the new dynamical matrix
    new_dyn.save_qe("new_dyn")

if __name__ == "__main__":
    test_interpolate_with_support()
    
