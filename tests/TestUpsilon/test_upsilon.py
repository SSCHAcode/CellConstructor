from __future__ import print_function

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons


import sys, os
import pytest

def test_upsilon():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    T = 100

    # Load a simple dynamical matrix
    dyn = CC.Phonons.Phonons("../TestSymmetriesSupercell/Sym.dyn.", 3)
    w, pols = dyn.DiagonalizeSupercell()

    # Get the upsilon matrix for the supercell
    superdyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    ups1 = superdyn.GetUpsilonMatrix(T, debug = True)
    ups2 = dyn.GetUpsilonMatrix(T, debug = True)

    delta = np.max(np.abs( (ups1 - ups2)))
    assert delta < 1e-7

if __name__ == "__main__":
    test_upsilon()
