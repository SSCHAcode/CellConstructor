# -*- coding: utf-8 -*-

"""
Here we test how to generate an input file for the findsym program from the ISOTROPY
suite. This can be used as an alternative tool to find the symmetry group of a 
given structure
"""
from __future__ import print_function

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.symmetries



import sys, os
import pytest

def test_isotropy():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    # Load the trial structure
    structure = CC.Structure.Structure()
    structure.read_scf("trial_structure.scf")

    # Prepare the ISOTROPY input file
    CC.symmetries.PrepareISOTROPYFindSymInput(structure, "isotropy_findsym.in")

    print ("File prepared in 'isotropy_findsym.in'")

if __name__ == "__main__":
    test_isotropy()
    
