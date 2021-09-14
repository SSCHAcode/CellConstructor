# -*- coding: utf-8 -*-

"""
This example uses the harmonic dynamical matrix to generate
an ensemble of configuration, and then to compute the radial distribution function.
"""
from __future__ import print_function
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Methods
import matplotlib.pyplot as plt

import pytest
import sys, os


def test_gr(plot = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    # Some info
    ENSEMBLE_SIZE=1000
    T = 0 #K
    R_MIN = 0.5
    R_MAX_OH = 2
    R_MAX_HH = 3 
    NR = 500
    R_LIM_OH= (0.5, 2)
    R_LIM_HH=(0.5, 3)


    # Load the ice XI dynamical matrix
    iceXI_dyn = CC.Phonons.Phonons("h2o.dyn", full_name = True)
    iceXI_dyn.Symmetrize() # Impose the sum rule

    # Use the dynamical matrix to generate the displacements
    print ("Generating displacements...")
    structures = iceXI_dyn.ExtractRandomStructures(ENSEMBLE_SIZE, T)

    # Get the g(r) between O and H atoms
    print ("Computing OH g(r)...")
    grOH = CC.Methods.get_gr(structures, "O", "H", R_MIN, R_MAX_OH, NR)
    print ("Computing HH g(r)...")
    grHH = CC.Methods.get_gr(structures, "H", "H", R_MIN, R_MAX_HH, NR)

    # Plot the result
    if plot:
        plt.plot(grOH[:,0], grOH[:,1])
        plt.xlabel("r [$\\AA$]")
        plt.ylabel("$g_{OH}(r)$")
        plt.title("O-H radial distribution function")
        plt.xlim(R_LIM_OH)
        plt.tight_layout()

        plt.figure()

        plt.plot(grHH[:,0], grHH[:,1])
        plt.xlabel("r [$\\AA$]")
        plt.ylabel("$g_{HH}(r)$")
        plt.title("H-H radial distribution function")
        plt.xlim(R_LIM_HH)
        plt.show()

if __name__ == "__main__":
    test_gr(True)
