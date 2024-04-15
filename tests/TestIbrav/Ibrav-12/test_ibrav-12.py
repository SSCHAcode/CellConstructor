# -*- coding: utf-8 -*-

""" 
This is an harmonic dynamical matrix for H3S.
It has an ibrav= 3 to setup the bcc kind of crystal structure.
Here we test the loading of such a structure within the dynamical matrix
"""

import cellconstructor as CC
import cellconstructor.Phonons

from ase.visualize import view

import sys, os
import numpy as np
import pytest

def test_ibrav_minus_12():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dynamical-matrix-1.txt", full_name=True)

    # If the structure is loaded without errors, we are done   
    struct = CC.Structure.Structure()
    struct.read_scf("struct.scf")

    # Compare the structure of the dynamical matrix with the one of the structure
    assert np.max(np.abs(struct.coords- dyn.structure.coords)) < 1e-7

    
if __name__ == "__main__":
    test_ibrav_minus_12()
