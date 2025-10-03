import cellconstructor as CC
import cellconstructor.Phonons
import numpy as np
import sys, os
import spglib

def test_qstar_with_spglib():
    # Go to the current directory
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    cmca_dyn = CC.Phonons.Phonons("cmca", 8)

    # Adjust the Q star
    cmca_dyn.AdjustQStar(use_spglib = True)

    print("The number of irreducible q points are:")
    print(len(cmca_dyn.q_stars))
    print("The total number of q:")
    print(len(cmca_dyn.q_tot))

    print("Space group:", spglib.get_spacegroup(cmca_dyn.structure.get_spglib_cell()))
    print("Number of symmetries:")
    syms = spglib.get_symmetry(cmca_dyn.structure.get_spglib_cell())
    print(len(syms["rotations"]))

    assert len(cmca_dyn.q_stars) == 8


if __name__ == "__main__":
    test_qstar_with_spglib()
