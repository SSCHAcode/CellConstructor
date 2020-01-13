from __future__ import print_function
from __future__ import division

import cellconstructor as CC 
import cellconstructor.Phonons

import sys, os

import pytest

def test_add_tensor():

    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    ph_file = "phonon.pho"
    dyn_file = "dyn_"
    nqirr = 3

    INFO = """
    This script reads the dynamical matrix in the supercell and 
    adds a dielectric tensor and the effective charges computed with
    quantum espresso ph.x.
    Those are read directly from the ph.x output stored in {}

    In this case the effective charges have been computed in a supercell 
    with distorted atoms.
    We first need to generate the supercell, and then we apply the effective charges.
    """.format(ph_file)

    print(INFO)

    # Reading the dynamical matrix
    dyn = CC.Phonons.Phonons(dyn_file, nqirr)

    # Generate the dynamical matrix in the supercell
    super_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell()) 

    # Now read the effective charges in the supercell
    super_dyn.ReadInfoFromESPRESSO(ph_file)

    # Print the dielectric tensor
    print("The read dielectric tensor is:")
    print(super_dyn.dielectric_tensor)


if __name__ == "__main__":
    test_add_tensor()
