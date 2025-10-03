from __future__ import print_function
from __future__ import division


import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries

import spglib
import sys, os

import numpy as np

import ase
from ase.visualize import view

import pytest

@pytest.mark.skip(reason="Function not implemented")
def test_diag_symmetries():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Diagonalize the dynamical matrix in the supercell
    dyn = CC.Phonons.Phonons("../TestDiagonalizeSupercell/prova", 4)
    w, p = dyn.DiagonalizeSupercell()

    view(dyn.structure.get_ase_atoms())

    # Get the symmetries
    supercell_s = dyn.structure.generate_supercell(dyn.GetSupercell())
    spglib_syms = spglib.get_symmetry(dyn.structure.get_spglib_cell())
    syms = CC.symmetries.GetSymmetriesFromSPGLIB(spglib_syms)

    # Get the symmetries on the polarization vectors
    pols_syms = CC.symmetries.GetSymmetriesOnModes(syms, supercell_s, p)

    # Now complete the diagonalization of the polarization vectors
    # To fully exploit symmetries
    new_pols, syms_character = CC.symmetries.get_diagonal_symmetry_polarization_vectors(p, w, pols_syms)

    # TODO: Test if these new polarization vectors really rebuild the dynamical matrix

    # write the symmetry character
    n_modes, n_syms = syms_character.shape

    for i in range(n_modes):
        print("Mode {} | ".format(i), np.angle(syms_character[i,:], deg = True))

if __name__ == "__main__":
    test_diag_symmetries()
