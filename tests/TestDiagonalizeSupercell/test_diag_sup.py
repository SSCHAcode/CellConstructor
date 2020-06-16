from __future__ import print_function
import cellconstructor as CC
import cellconstructor.Manipulate
import cellconstructor.Phonons
import numpy as np
import matplotlib.pyplot as plt


import sys, os
import pytest

def test_qha():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    dyn = CC.Phonons.Phonons("prova", 4)

    w, p = dyn.DiagonalizeSupercell()

    w2, p2 = dyn.GenerateSupercellDyn(dyn.GetSupercell()).DyagDinQ(0)

    diff = np.max(np.abs(w -  w2))

    assert diff < 1e-6
