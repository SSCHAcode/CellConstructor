# -*- coding: utf-8 -*-
from __future__ import print_function
"""
In this example one configuration is strained according with a new unit cell.
Then also the dynamical matrix is strained. As a test for the correct strain
of the dynamical matrix, the probability of the configuration is checked before and
after the strain with the two dynamical matrices.
"""

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
from ase.visualize import view

import numpy as np
import sys, os
import pytest

# Test both at 0 K and at fixed temperature
@pytest.mark.parametrize("T", [0, 100])
def test_strain_config(T):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Ry to cm-1 conversion
    RyToCm=109737.37595

    EPS = 1e-6

    # Load the dynamical matrix
    dynmat = CC.Phonons.Phonons("ice.dyn", full_name = True)

    # Load the configuration (it is written in alat units from the dyn)
    # And assign it the same cell as the dynamical matrix
    config = CC.Structure.Structure()
    config.read_scf("config.scf", alat = dynmat.alat)
    config.unit_cell = dynmat.structure.unit_cell.copy()
    config.has_unit_cell = True

    # Enlarge the unit cell volume by a factor 1.2
    new_cell = dynmat.structure.unit_cell * 1.2

    # Strain the configuration
    old_config = config.copy()
    config.change_unit_cell(new_cell)

    # Strain the dynamical matrix
    new_dyn = dynmat.GetStrainMatrix(new_cell, T, EPS)

    # Compute the two displacements
    disp_old = old_config.get_displacement(dynmat.structure)
    disp_new = config.get_displacement(new_dyn.structure)


    # Get the ratio between the new probabilities (without normalization)
    print ("Strain performed.")
    print ("Old frequencies [cm-1]:")
    old_w, old_pols = dynmat.DyagDinQ(0)
    print (old_w * RyToCm)
    print ("New frequencies [cm-1]:")
    new_w, new_pols = new_dyn.DyagDinQ(0)
    print (new_w * RyToCm)
    print ("Relation between the two <u | Upsilon |  u> factors (They should be similar):")

    fact1 =  new_dyn.GetProbability(disp_new, T, normalize= False)
    fact2 = dynmat.GetProbability(disp_old, T, normalize= False)
    print ("Factor1: %e" % fact1)
    print ("Factor2: %e" % fact2)

    assert np.abs(fact1 - fact2) < 1e-7

    
    
    # Save the new dynamical matrix
    new_dyn.save_qe("strained_dyn", True)



if __name__ == "__main__":
    test_strain_config(100)
