from __future__ import print_function
from __future__ import division
import Phonons
import Methods
import symmetries

import numpy as np

import itertools



"""
In this module we compute the Spectral function 
using the interpolation on the third order force constant matrix
"""


def get_static_bubble(dyn, tensor3, k_grid, q):
    """
    COMPUTE THE STATIC BUBBLE
    =========================
    
    This function computes the static bubble for a given dynamical matrix,
    the third order force constant tensor by using the Fourier interpolation
    
    
    Parameters
    ----------
        dyn : Phonons()
            The dynamical matrix
        tensor3 : ForceTensor.Tensor3()
            The centered third order force costant
        k_grid : (nk1, nk2, nk3)
            The grid of k points to be used for the integration
        q : ndarray(size = 3)
            The q point at which compute the bubble.
            
    Results
    -------
        dynq : ndarray( size = (3*nat, 3*nat), dtype = np.complex128)
            The bubble matrix at the specified q point (only bubble).
    """
    
    # Get the integration points 
    k_points = CC.symmetries.GetQGrid(dyn.structure.unit_cell, k_grid)
    
    # Generate the dynamical matrix in the supercell
    superdyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    
    m = np.tile(dyn.structure.get_masses_array(), (3,1)).T.ravel()
    
    mm_mat = np.sqrt(np.outer(m, m))
    mm_inv_mat = 1 / mm_mat
    
    
    for k in k_points:
        # we want the matrix at -q, k, q - k
        phi3 = tensor3.Interpolate(k, q - k)
        phi2 = CC.Phonons.InterpolateDynFC(superdyn.dynmats[0], dyn.GetSupercell(),
                                           dyn.structure, superdyn.structure,
                                           k)
        
        
        D2 = mm_mat * phi2
                                         
        
        
        