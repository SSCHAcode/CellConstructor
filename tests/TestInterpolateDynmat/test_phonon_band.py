# -*- coding: utf-8 -*-
from __future__ import print_function
"""
In this example we interpolate between two q points the dynamical matrices
in order to get a band dispersion.
"""
import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons

import matplotlib.pyplot as plt


import sys, os
import pytest

def test_phonon_band():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)



    # Load the dynamical matrices
    NQIRR = 8
    supercell_size = (3,3,2)
    dynmats = CC.Phonons.Phonons("dynmat", nqirr = NQIRR)
    q_target = np.array(dynmats.q_tot[1], dtype = np.float64)

    # Get the force constant in real space
    dyn_fc = dynmats.GenerateSupercellDyn(supercell_size)

    # Interpolate a point in itself
    new_dynmat = CC.Phonons.InterpolateDynFC(dyn_fc.dynmats[0], supercell_size, dynmats.structure, dyn_fc.structure, q_target)

    # Compose a new phonon structure and get the frequencies
    dyn0 = CC.Phonons.Phonons("dynmat", nqirr = 1)
    dyn0.dynmats[0] = new_dynmat
    w,pols = dyn0.DyagDinQ(0)
    w2,pols = dynmats.DyagDinQ(1)
    w *= CC.Phonons.RY_TO_CM
    w2 *= CC.Phonons.RY_TO_CM
    print ("Frequencies:")
    print ("\n".join(["%d) %16.8f | %16.8f" % (i, w[i], w2[i]) for i in range(len(w))]))



    # Interpolate in between the first two q points of the mesh
    q0 = np.array( [0,0,0], dtype = np.float64)
    q1 = dynmats.q_stars[-1][0]
    N_STEPS = 10


    w_arrays = []
    for i in range(N_STEPS):
        print ("Interpolation... (step %d of %d)" % (i+1, N_STEPS))
        q_vector = q0 + i*(q1 - q0) / (N_STEPS-1)

        # Interpolate
        new_dynmat = CC.Phonons.InterpolateDynFC(dyn_fc.dynmats[0], supercell_size, 
                                                 dynmats.structure, dyn_fc.structure, 
                                                 q_vector)

        # Get the frequencies
        dyn0.dynmats[0] = new_dynmat
        ws, pols = dyn0.DyagDinQ(0)
        ws *= CC.Phonons.RY_TO_CM
        w_arrays.append(ws)

    # Prepare the plot
    freqs = np.array(w_arrays)
    n_q, n_freqs = np.shape(freqs)
    plt.figure()
    for i in range(n_freqs):
        plt.plot(freqs[:, i])

    plt.title("Phonon band")
    plt.xlabel("q point")
    plt.ylabel("freq. [cm-1]")
    plt.tight_layout()


if __name__ == "__main__":
    test_phonon_band()
    plt.show()
