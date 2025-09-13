# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries

import sys, os
import pytest

@pytest.mark.parametrize("FILDYN, NQIRR", [("Sym.dyn.", 3), ("skydyn_", 4)])
def test_symmetries_supercell(FILDYN, NQIRR):

    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    dynmat = CC.Phonons.Phonons(FILDYN, NQIRR)
    SUPERCELL = dynmat.GetSupercell()

    # Compute the frequencies
    supercell_dyn = dynmat.GenerateSupercellDyn(SUPERCELL)
    w1, pols = supercell_dyn.DyagDinQ(0)

    # Show the modes for each q point
    for i,q in enumerate(dynmat.q_tot):
        print ("Dyagonalizing:", q)
        w, p = dynmat.DyagDinQ(i)
        print (" ".join(["%.4f cm-1 " % (x * CC.Phonons.RY_TO_CM) for x in w]))

    #dynmat.Symmetrize()
    # # Test the symmetrization
    qe_sym = CC.symmetries.QE_Symmetry(dynmat.structure)

    fc_dynmat_start = np.array(dynmat.dynmats)


    after_sym = fc_dynmat_start.copy()
    qe_sym.SymmetrizeFCQ(after_sym, dynmat.q_stars, verbose = True)
    for i,q in enumerate(dynmat.q_tot):
        dynmat.dynmats[i] = after_sym[i,:,:]

    # Show the modes for each q point
    for i,q in enumerate(dynmat.q_tot):
        print ("After Dyagonalizing:", q)
        w, p = dynmat.DyagDinQ(i)
        print (" ".join(["%.4f cm-1 " % (x * CC.Phonons.RY_TO_CM) for x in w]))

    # Print the difference between before and after the symmetrization
    print ()
    print ("Difference of the symmetrization:")
    print (np.sqrt( np.sum( (after_sym - fc_dynmat_start)**2 ) / np.sum(after_sym*fc_dynmat_start)))

    # print ""

    # Now lets try to randomize the matrix
    #new_random = np.random.uniform( size = np.shape(fc_dynmat_start)) + 1j*np.random.uniform( size = np.shape(fc_dynmat_start))

    # print "Saving a not symmetrized random matrix to Random.dyn.IQ, where IQ is the q index"
    # # Lets save the new matrix in QE format
    # for i, q in enumerate(dynmat.q_tot):
    #     dynmat.dynmats[i] = new_random[i, :, :]
    # dynmat.save_qe("Random.dyn.")

    # # Lets constrain the symmetries
    # # We use asr = crystal to force the existence of the acustic modes in Gamma
    # qe_sym.SymmetrizeFCQ(new_random, np.array(dynmat.q_stars), asr = "no")

    # # Lets save the new matrix in QE format
    # for i, q in enumerate(dynmat.q_tot):
    #     dynmat.dynmats[i] = new_random[i, :, :]

    # print "Saving a symmetrized random matrix to Sym.dyn.IQ, where IQ is the q index"
    # dynmat.save_qe("Sym.dyn.")
    # print ""

    # Compute the frequencies
    supercell_dyn = dynmat.GenerateSupercellDyn(SUPERCELL)
    w, pols = supercell_dyn.DyagDinQ(0)
    # Get the translations
    t = CC.Methods.get_translations(pols, supercell_dyn.structure.get_masses_array())

    dynmat.Symmetrize()
    # Compute the frequencies
    supercell_dyn = dynmat.GenerateSupercellDyn(SUPERCELL)
    w3, pols = supercell_dyn.DyagDinQ(0)
    # Get the translations
    t = CC.Methods.get_translations(pols, supercell_dyn.structure.get_masses_array())


    # Make the assert test
    for i, _w_ in enumerate(w):
        w2 = w3[i]

        assert np.abs(_w_ - w2) < 1e-8

    # print "Frequencies:"
    # print "\n".join(["%.4f cm-1  | %.4f cm-1  | %.4f cm-1  T: %d" % (w1[i]*CC.Phonons.RY_TO_CM, w[i]*CC.Phonons.RY_TO_CM, w3[i]*CC.Phonons.RY_TO_CM, t[i]) for i in range(len(w))])
    # print ""
    # print "Done."


if __name__ == "__main__":
    test_symmetries_supercell("Sym.dyn.", 3)
