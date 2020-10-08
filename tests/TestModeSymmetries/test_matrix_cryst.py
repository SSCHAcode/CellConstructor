from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Methods

import sys, os
import pytest
import numpy as np
    
def test_matrix_cryst(verbose = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    random_mat = np.random.uniform(size = (3,3))
    random_v = np.random.uniform(size = 3)

    # Lets pick an uggly unit cell
    unit_cell = np.eye(3) * 5
    unit_cell[2,2] = 5

    mat_cryst = CC.Methods.convert_matrix_cart_cryst2(random_mat,
                                                      unit_cell)

    v_cryst = CC.Methods.covariant_coordinates(unit_cell, random_v)

    w_cryst = mat_cryst.dot(v_cryst)
    w_cart = unit_cell.T.dot(w_cryst)

    w = random_mat.dot(random_v)

    thr = np.max(np.abs(w_cart - w))

    if verbose:
        print("Residual of conversion {}".format(thr))
        
    assert thr < 1e-10, "Error while converting between cartesian and crystal of {}".format(thr)

    # Lets go back
    mat_cart = CC.Methods.convert_matrix_cart_cryst2(mat_cryst,
                                                     unit_cell,
                                                     cryst_to_cart = True)
    
    thr = np.max(np.abs(mat_cart - random_mat))
    if verbose:
        print("Residual of matrix conversion {}".format(thr))
    assert thr < 1e-10, "Error while converting the matrix back to cartesian of {}".format(thr)



if __name__ == "__main__":
    test_matrix_cryst(True)
