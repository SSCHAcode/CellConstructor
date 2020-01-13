from __future__ import print_function

"""
This code is meant for show how to
measure the proton transfer in ICE.
In this case the Harmonic dynamical matrices are used to
extract the proton transfer.

This example has been written in order to
clearly work with MPI for parallelization.

Submit it with:
mpirun -np X python test_proton_transfer.py
to exploit paralelization (note: you must have mpi4py installed)
"""

import cellconstructor as CC 
import cellconstructor.Phonons
import cellconstructor.Manipulate
import time


import matplotlib.pyplot as plt
import numpy as np

import pytest

import sys, os

def test_proton_transfer():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # This handles the parallelization for the
    # Proton transfer measurements
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except:
        __MPI__ = False
        rank = 0

    from ase.visualize import view

    # Size of the ensemble and temperature
    N_SIZE = 1000
    T = 0
    # Setup the atoms involved in the proton transfers
    # These atoms are the OH-O  where OH is the covalent bond, 
    # H-O is the hydrogen bond.
    MOLS = [(6,7,3), (9,11,6),(0,1,9),(3,5,0)]

    # Load the Ice Dynamical matrix
    dynmat = CC.Phonons.Phonons("dynmat", nqirr = 1)
    dynmat.Symmetrize()

    #exit()

    # Generate an ensemble of configuration
    # According to the harmonic dynamical matrix
    if rank == 0:
        print ("Extracting random configurations...")
    structures = dynmat.ExtractRandomStructures(N_SIZE, T)

    if rank == 0:
        print ("Computing the proton transfer...")
    t1 = time.time()
    pt_coords = CC.Manipulate.MeasureProtonTransfer(structures, MOLS)
    t2 = time.time()

    if rank == 0:
        print ("Time elapsed:", t2 - t1)
    # Plot the histogram
    if rank == 0:
        print ("Plotting the results...")
    h, be = np.histogram(pt_coords, 100)
    x_axis = be[:-1] + np.diff(be) / 2

    # Get the ratio probability of a proton transfer to occurr
    if rank == 0:
        print ("Probability of Proton transfer: ", sum((pt_coords >= 0).astype(int)) / float(len(pt_coords)))

    if rank == 0:
        plt.figure()
        plt.fill_between(x_axis, h, 0, color= "r", alpha = 0.8)




if __name__ == "__main__":
    test_proton_transfer()
    plt.show()
