# -*- coding: utf-8 -*-
from __future__ import print_function
import cellconstructor as CC
import cellconstructor.Manipulate
import cellconstructor.Phonons
import numpy as np
import matplotlib.pyplot as plt


import sys, os
import pytest

def test_qha():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)



    """
    This example file provide the quasi harmonic approximation to compute the pressure
    by interpolating two different dynamical matrix on a supercell 3x3x2 of common ice at
    different volumes.
    """

    RyToEv=13.605698
    Ev_AngToGPa=160.21766208

    # Import the two phonons
    ph1 = CC.Phonons.Phonons("V804/dynmat", nqirr = 8)
    ph2 = CC.Phonons.Phonons("V907/dynmat", nqirr = 8)


    # Perform the interpolation
    N_points = 100
    N_T = 100
    T = np.linspace(0, 300, N_T)
    free_energy = CC.Manipulate.QHA_FreeEnergy(ph1, ph2, T, N_points)


    # Get the volumes (The determinant of the unit cell vectors)
    V0 = np.linalg.det(ph1.structure.unit_cell)
    V1 = np.linalg.det(ph2.structure.unit_cell)

    print ("The two volumes are:", V0, "Angstrom^3 and", V1, "Angstrom^3")

    # Take the derivative and compute the pressure [Ry/angstrom^3]
    pressure = np.diff(free_energy, axis = 0) / ((V0 - V1)/(N_points - 1))
    pressure *= RyToEv*Ev_AngToGPa * 10 # kbar


    # Plot the free energy
    plt.figure()
    plt.imshow(free_energy, aspect = "auto")
    plt.colorbar()


    # Plot a single graf of the pressure
    plt.figure()
    plt.title("QHA Pressure contribution")
    plt.plot(T, pressure[0,:])
    plt.xlabel("T [K]")
    plt.ylabel("P [kbar]")

if __name__ == "__main__":
    test_qha()
    plt.show()
