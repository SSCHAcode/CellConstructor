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
import pytest

def test_h3s():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn_h3s = CC.Phonons.Phonons("dynq", nqirr = 3)

    
if __name__ == "__main__":
    test_h3s()
