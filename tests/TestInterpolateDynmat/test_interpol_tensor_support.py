# Import the modules to read the dynamical matrix
import cellconstructor as CC
import cellconstructor.Phonons
import numpy as np

import sys, os

def test_interpolate_support_srtio3():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    dyn = CC.Phonons.Phonons("dyn_pop3_",6)

    # Load the harmonic dynamical matrices
    harm_dyn = CC.Phonons.Phonons("matdyn", 6)
    harm_dyn_fine_grid = CC.Phonons.Phonons("matdyn_4x/matdyn", 24)
    
    new_cell = harm_dyn_fine_grid.GetSupercell()

    # Perform the interpolation
    #big_dyn = dyn.Interpolate( dyn.GetSupercell(), new_cell, support_dyn_coarse = harm_dyn, support_dyn_fine = harm_dyn_fine_grid)
    big_dyn = dyn.InterpolateMesh(new_cell, support_dyn_coarse = harm_dyn, support_dyn_fine = harm_dyn_fine_grid)
    big_dyn.save_qe("my_new_dyn")



if __name__ == "__main__":
    test_interpolate_support_srtio3()
