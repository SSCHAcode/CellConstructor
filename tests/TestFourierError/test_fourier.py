from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Phonons

import numpy as np
import sys, os

def test_fourier_transform():
    # Go in the current directory
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("harmonic_", 8)

    # Check the q start
    dyn.AdjustQStar()
    
    # Check the symmetrization
    dyn.Symmetrize()

    # Check if the grid is correct
    q_grid = CC.symmetries.GetQGrid(dyn.structure.unit_cell, dyn.GetSupercell())

    # Check if the q point is contained into dyn
    bg = dyn.structure.get_reciprocal_vectors() / (2 * np.pi)
    for iq, qi in enumerate(q_grid):
        min_dist = 10
        for jq, qj in enumerate(dyn.q_tot):
            dist = CC.Methods.get_min_dist_into_cell(bg, qi, qj)
            if dist < min_dist:
                min_dist = dist

        if min_dist > 1e-4:
            raise ValueError("Error, the q points do not define the correct grid")

    # Generate the supercell
    print("Generate supercell")
    dyn_supercell = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    
    # If I'm here, everything is good!
    print("Everything is ok!")        
            
if __name__ == "__main__":
    test_fourier_transform()    
