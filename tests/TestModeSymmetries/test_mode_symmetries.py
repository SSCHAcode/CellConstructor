from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Phonons, cellconstructor.Timer
import numpy as np
import time

import sys, os
import pytest

def test_mode_symmetries(verbose = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("../TestPhononSupercell/dynmat")

    # Apply the symmetries
    dyn.Symmetrize()

    # Load the symmetries from the structure
    qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
    qe_sym.SetupQPoint(verbose = True)
    symmetries = qe_sym.GetSymmetries()

    # Get frequencies and polarization vectors
    w, pols = dyn.DiagonalizeSupercell()

    timer = CC.Timer.Timer(active = True)

    # Get the symmetry matrix in the polarization space
    t1 = time.time()
    sim_modes = CC.symmetries.GetSymmetriesOnModes(symmetries, dyn.structure, pols, timer, debug =True)
    t2 = time.time()
    # Exploit the old slower function to test if the implementation is correct
    sim_modes2 = CC.symmetries._GetSymmetriesOnModes(symmetries, dyn.structure, pols)
    t3 = time.time()


    assert np.max( np.abs(sim_modes2 - sim_modes)) < 1e-8

    #sim_modes2 = CC.symmetries.GetSymmetriesOnModesFast(symmetries, dyn.structure, pols)
    if verbose:
        for i in range(len(symmetries)):
            print("Symmetry:")
            print(symmetries[i])
            print("Interaction matrix:")
            print("\n".join(["{:16.4f} cm-1  | diag_value = {:10.4f}".format(w[j] * CC.Phonons.RY_TO_CM, sim_modes[i,j,j]) for j in range(len(w))]))

        print()
        print("Time for the new method: {} s".format(t2-t1))
        print("Time for the old method: {} s".format(t3-t2))

        timer.print_report()


if __name__ == "__main__":
    test_mode_symmetries(True)
