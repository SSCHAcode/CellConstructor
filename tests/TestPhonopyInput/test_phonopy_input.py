import sys, os
import cellconstructor as CC
import cellconstructor.Phonons

import pytest
import numpy as np

def test_phonopy_input():

    dyn = CC.Phonons.Phonons()
    dyn.load_phonopy()

    dyn.save_qe("prova")


if __name__ == "__main__":
    test_phonopy_input()
