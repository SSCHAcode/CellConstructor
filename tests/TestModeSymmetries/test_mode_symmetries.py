from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Phonons, cellconstructor.Timer
import numpy as np
import time

import spglib

import sys, os
import pytest

TESTDYN="../TestSymmetriesSupercell/SnSe.dyn.2x2x2"
NQIRR = 3

def test_mode_symmetries(verbose = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons(TESTDYN, NQIRR)

    # Apply the symmetries
    dyn.Symmetrize()

    ss = dyn.structure.generate_supercell(dyn.GetSupercell())
    
    # Load the symmetries from the structure
    spglib_sym = spglib.get_symmetry(ss.get_spglib_cell())
    symmetries = CC.symmetries.GetSymmetriesFromSPGLIB(spglib_sym)

    # Select only one of the problematic symmetries
    #symmetries = [symmetries[3]]

    # Get frequencies and polarization vectors
    w, pols = dyn.DiagonalizeSupercell()


    timer = CC.Timer.Timer(active = True)

    # Get the symmetry matrix in the polarization space
    t1 = time.time()
    sim_modes = CC.symmetries.GetSymmetriesOnModes(symmetries, ss, pols, [], timer, debug =True)
    t2 = time.time()
    # Exploit the old slower function to test if the implementation is correct
    sim_modes2 = CC.symmetries._GetSymmetriesOnModes(symmetries, ss, pols)
    t3 = time.time()


    assert np.max( np.abs(sim_modes2 - sim_modes)) < 1e-8

    # Now try to get the modes using the new function that exploits the degeneracies

    sim_modes3, basis = CC.symmetries.GetSymmetriesOnModesDeg(symmetries, ss,
                                                              pols, w, timer)


    print(basis)

    
    for i, modes in enumerate(basis):

        ss = np.zeros( (len(symmetries), len(modes), len(modes)), dtype = np.double)
        print("I = {}, MODES = {}".format( i, modes))
        for j, m in enumerate(modes):
            for k, n in enumerate(modes):
                ss[:, j, k] = sim_modes2[:, m, n]
                       
        diff = ss - sim_modes3[i]


        error_bool =np.max( np.abs(diff)) < 1e-8

        if not error_bool:
            np.savetxt("errsym.dat", sim_modes2[0, :, :])
            
        assert error_bool, "Error on block {}:\n ss = {}\n new = {}\n".format(i, ss, sim_modes3[i])
    

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
