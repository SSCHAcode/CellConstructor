import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import pytest


def test_dyn_diag_supercell_loto():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    dyn = CC.Phonons.Phonons("sscha_converged_tetra_dyn_", 3)
    dyn.ReadInfoFromESPRESSO("dielectric_calc_tetra.pho")
    dyn.DiagonalizeSupercell(lo_to_split="random")


if __name__ == "__main__":
    test_dyn_diag_supercell_loto()

