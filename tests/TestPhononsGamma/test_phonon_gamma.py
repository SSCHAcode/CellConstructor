# -*- coding: utf-8 -*-
from __future__ import print_function
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Manipulate
import numpy as np
from ase.visualize import view

import sys, os
import pytest


import sys, os
import pytest

def test_phonon_gamma():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    RyToCm = 109691.40235

    """
    In this script I will show how to read a QE dynmat into the cell constructure
    generating both the structure and all the phonon information.
    """

    # Load the QE dynmat
    hydrogen_structure = CC.Phonons.Phonons("hydrogen_dyn")

    # Show the structure with the ASE viewer
    hydrogen_structure.structure.fix_coords_in_unit_cell()
    ase_struct = hydrogen_structure.structure.get_ase_atoms()

    # Try to dyagonalize the matrix to check the frequencies
    print (hydrogen_structure.structure.masses)
    freq, pol_vect = hydrogen_structure.DyagDinQ(0)
    print ("Frequencies:")
    print (np.sort(freq * RyToCm))

    # Save a video of the 70 id vibration
    idvib=70
    print ("Saving a video of the %d vibration = %.3f cm-1" % (idvib, freq[idvib] * RyToCm))
    CC.Manipulate.GenerateXYZVideoOfVibrations(hydrogen_structure, "vibron.xyz", idvib, 0.4, 0.1, 100)
    print ("Done! Check vibron.xyz")



if __name__ == "__main__":
    test_phonon_gamma()
