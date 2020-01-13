# -*- coding: utf-8 -*-
from __future__ import print_function
import cellconstructor as CC
import numpy as np
from ase.visualize import view
import matplotlib.pyplot as plt
import cellconstructor.Structure

"""
This example will load a quantum espresso structure of phase C2c-24 of
high pressure hydrogen and will compute the distances between the H-H bounded
molecules.
"""

import sys, os
import pytest

def test_structure_measurements():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # The structure has a uniform scaling with alat equal to
    alat = 2.838000235


    # Load the hydrogen structure (rescaling both the cell)
    HydIII = CC.Structure.Structure()
    HydIII.read_scf("Hydrogen.scf", alat)

    # Show the structure
    #view(HydIII.get_strct_conventional_cell().get_ase_atoms())

    # Extract the hydrogen molecules
    distance = 0.687
    tollerance = 0.2

    # Get all the hydrogen molecules
    Mols, indices = HydIII.GetBiatomicMolecules(["H", "H"], distance, tollerance, True)

    print ("Found %s molecules:" % len(Mols))

    # Now compute the H-H distances between the molecules
    dist = []
    for i, mol in enumerate(Mols):
        d = np.sqrt(np.sum((mol.coords[0,:] - mol.coords[1,:])**2))
        dist.append(d)

    print ("Here all the H-H distances:")
    print (dist)

    # Use matplotlib to plot a pretty hystogram
    plt.figure()
    plt.title("H-H distance histogram")
    plt.xlabel(r"dist [$\AA$]")
    plt.ylabel(r"freq")
    plt.hist(dist, 20)
    plt.tight_layout()


    # Now Get info about triatomic molecules
    Mols = HydIII.get_strct_conventional_cell().GetTriatomicMolecules(["H", "H", "H"], 0.71, 1.44, 144, 0.2, 10)
    print ("N mols:", len(Mols))

    # Print the angle between the molecules
    angls = []
    for i,mol in enumerate(Mols):
        angls.append( mol.get_angle(0, 1, 2))
        print ("%d) %.3f %.3f %.3f" % (i, mol.get_angle(0, 1, 2), mol.get_angle(1, 2, 0), mol.get_angle(2, 0, 1)))
        #view(mol.get_ase_atoms())

    plt.figure()
    plt.title("H-H-H angle histogram")
    plt.xlabel(r"angle [degree]")
    plt.ylabel(r"freq")
    plt.hist(angls, 20)
    plt.tight_layout()


if __name__ == "__main__":
    test_structure_measurements()
    plt.show()
