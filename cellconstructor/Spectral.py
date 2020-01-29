from __future__ import print_function
from __future__ import division
import Phonons
import Methods
import symmetries

import numpy as np

import itertools

import symph
import thirdorder

import cellconstructor as CC
import cellconstructor.symmetries

"""
In this module we compute the Spectral function 
using the interpolation on the third order force constant matrix
"""


def get_static_bubble(dyn, tensor3, k_grid, q, T = 0):
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
        T : float
            The tempearture of the calculation (default 0 K)
            
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
    
    # Get the dynamical matrix at -q
    phi2_mq = CC.Phonons.InterpolateDynFC(superdyn.dynmats[0], dyn.GetSupercell(),
                                         dyn.structure, superdyn.structure,
                                         -q)
    d2_mq = phi2_mq * mm_inv_mat
    
    # Diagonalize the dynamical matrix at -q
    w2_mq, pols_mq = np.linalg.eigh(d2_mq)
    # Check if the -q point is gamma
    is_mq_gamma = CC.Methods.is_gamma(dyn.structure.unit_cell, -q)
    
    if is_mq_gamma:
        w2_mq[0:3]=0.0
    assert (w2_mq >= 0.0).all()
    w_mq=np.sqrt(w2_mq)
    
    

    
    # Allocate the memory for the bubble
    tmp_bubble = np.zeros((3*dyn.structure.N_atoms, 3*dyn.structure.N_atoms),
                          dtype = np.complex128, order = "F")
    
    
    for k in k_points:
        # we want the matrix at -q, k, q - k
        phi3 = tensor3.Interpolate(k, q - k)

        np.save("phi3.npy", phi3)

        # phi2 in k
        phi2_k = CC.Phonons.InterpolateDynFC(superdyn.dynmats[0], dyn.GetSupercell(),
                                           dyn.structure, superdyn.structure,
                                           k)
        # phi2 in q mk
        phi2_q_mk = CC.Phonons.InterpolateDynFC(superdyn.dynmats[0], dyn.GetSupercell(),
                                           dyn.structure, superdyn.structure,
                                           q - k)
        
        # Divide by the masses
        d2_k = phi2_k * mm_inv_mat
        d2_q_mk = phi2_q_mk * mm_inv_mat
        
        # Diagonalize the dynamical matrices
        w2_k, pols_k = np.linalg.eigh(d2_k)
        w2_q_mk, pols_q_mk = np.linalg.eigh(d2_q_mk)
        
        is_k_gamma = CC.Methods.is_gamma(dyn.structure.unit_cell, k)
        is_q_mk_gamma = CC.Methods.is_gamma(dyn.structure.unit_cell, q-k)
        
        if is_k_gamma:
            w2_k[0:3]=0.0
        assert (w2_k >= 0.0).all()
        w_k=np.sqrt(w2_k)

        if is_q_mk_gamma:
            w2_q_mk[0:3]=0.0
        assert (w2_q_mk >= 0.0).all()
        w_q_mk=np.sqrt(w2_q_mk)

        
        # Dividing the phi3 by the sqare root of masses
        d3 = np.einsum("abc, a, b, c -> abc", phi3, 1/np.sqrt(m), 1/np.sqrt(m), 1/np.sqrt(m))

        np.save("d3_divmass.npy", d3)
        
        # Rotation of phi3 in the mode space
        d3_pols = np.einsum("abc, ai, bj, ck -> ijk", d3, pols_mq, pols_k, pols_q_mk)

        np.save("d3_pols.npy", d3_pols)

        
        
        
        # Fortran duty ====
        # The bubble out  in mode space
        tmp_bubble += thirdorder.third_order_bubble.compute_static_bubble(T,np.array([w_mq,w_k,w_q_mk]).T,
                                                                       np.array([is_mq_gamma,is_k_gamma,is_q_mk_gamma]),
                                                                       d3_pols,n_mod=3*dyn.structure.N_atoms)

        np.save("tmp_bubble.npy", tmp_bubble)
    
    # Rotate the bubble in cartesian  
    d_bubble = np.einsum("ab, ai, bj -> ij", tmp_bubble, pols_mq, np.conj(pols_mq))
    # multiply by the -1/(2N_k) factor
    d_bubble /= -8.0*len(k_points) 

    np.save("tmp_odd.npy", d_bubble)

    # add to the SSCHA dynamical matrix in q
    d2_final_q = np.conj(d2_mq) + d_bubble
    # and mutiply by the masses ( -> FC)
    phi2_final_q = d2_final_q * mm_mat
    
    return phi2_final_q
    #new_dyn = CC.Phonons.Phonons(dyn.structure)
    #new_dyn.q_tot = [q]
    #new_dyn.dynmats[0] = phi2_final_q
    #new_dyn.save_qe("dyn_plus_odd")
        
        
        
def get_static_correction(dyn, tensor3, k_grid, list_of_q_points, T):
    """
    Get the dyn + static bubble correction for the list of q points
    """
    
    dynq = np.zeros( (len(list_of_q_points), 3*dyn.structure.N_atoms, 3*dyn.structure.N_atoms), dtype = np.complex128)
    for iq, q in enumerate(list_of_q_points):
        dynq[iq, :, :] = get_static_bubble(dyn, tensor3, k_grid, np.array(q), T)
    
    return dynq
        

def get_static_correction_along_path(dyn, tensor3, k_grid, q_path, T):
    """
    Get the dyn + static bubble correction on the give path in a plottable fashon.
    
    Parameters
    ----------
        
    Results
    -------
        length : ndarray(size = len(q_path))
            The distance of the q path.
        frequencies : ndarray(size = (len(q_path), 3*nat))
            The frequencies of the corrected dynamical matrix along the q_path [in cm-1]
    """
    
    # Get the length of the q path
    x_length = np.zeros(len(q_path))
    q_tot = np.sum(np.diff(np.array(q_path), axis = 0)**2, axis = 1)
    x_lenght[1:] = q_tot
    
    frequencies = np.zeros((len(q_path), 3 * dyn.structure.N_atoms))
    
    dynq = get_static_correction(dyn, tensor3, k_grid, q_path, T)
    
    for iq in range(len(q_path)):
        tmp_dyn = dyn.Copy()
        tmp_dyn.dynmats[0] = dynq[iq,:,:]
        # Add the masses and diagonalize
        w, p = tmp_dyn.DyagDinQ(0)
        
        frequencies[iq, :] = w * CC.Units.RY_TO_CM
    
    return x_lenght, frequencies
        
    
    
    
        
        
                                         
        
        
        