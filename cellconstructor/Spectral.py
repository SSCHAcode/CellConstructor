from __future__ import print_function
from __future__ import division

import cellconstructor.Phonons as Phonons 
import cellconstructor.Methods as Methods 
import cellconstructor.symmetries as symmetries

import numpy as np

import itertools

import symph
import thirdorder

import cellconstructor as CC
import cellconstructor.symmetries
import cellconstructor.Settings

import time

from cellconstructor.Settings import ParallelPrint as print 


"""
In this module we compute the Spectral function 
using the interpolation on the third order force constant matrix
"""

def get_static_bubble(tensor2, tensor3, k_grid, q, T = 0, asr = True, verbose = False):
    """
    COMPUTE THE STATIC BUBBLE
    =========================
    
    This function computes the static bubble for a given dynamical matrix,
    the third order force constant tensor by using the Fourier interpolation
    
    
    Parameters
    ----------
        tensor2 : ForceTensor.Tensor2()
            The second order force constant
        tensor3 : ForceTensor.Tensor3()
            The centered third order force costant
        k_grid : (nk1, nk2, nk3)
            The grid of k points to be used for the integration
        q : ndarray(size = 3)
            The q point at which compute the bubble.
        T : float
            The tempearture of the calculation (default 0 K)
        asr : bool
            If true, impose the acoustic sum rule during the Fourier transform
        verbose : bool
            If true print debugging and timing info
            
    Results
    -------
        dynq : ndarray( size = (3*nat, 3*nat), dtype = np.complex128)
            The bubble matrix at the specified q point (only bubble).
    """

    structure = tensor2.unitcell_structure
    
    # Get the integration points 
    k_points = CC.symmetries.GetQGrid(structure.unit_cell, k_grid)
    
        
    # Get the phi2 in q
    phi2_q = tensor2.Interpolate(q, asr = asr)


    # dynamical matrix in q
    m = np.tile(structure.get_masses_array(), (3,1)).T.ravel()    
    mm_mat = np.sqrt(np.outer(m, m))
    mm_inv_mat = 1 / mm_mat
    #
    d2_q = phi2_q * mm_inv_mat
    
    # Diagonalize the dynamical matrix in q
    w2_q, pols_q = np.linalg.eigh(d2_q)
    
    # Check if the q point is gamma
    is_q_gamma = CC.Methods.is_gamma(structure.unit_cell, q)
    
    if is_q_gamma:
        w2_q[0:3]=0.0
    assert (w2_q >= 0.0).all()
    w_q=np.sqrt(w2_q)
    
    # Allocate the memory for the bubble
    tmp_bubble = np.zeros((3*structure.N_atoms, 3*structure.N_atoms),
                          dtype = np.complex128, order = "F")
    
    def compute_k(k):
        # phi3 in q, k, -q - k
        t1 = time.time()        
        phi3=tensor3.Interpolate(k,-q-k)
        t2 = time.time()
        # phi2 in k
        phi2_k = tensor2.Interpolate(k, asr = asr) 

        # phi2 in -q-k
        phi2_mq_mk = tensor2.Interpolate(-q -k, asr = asr)

        t3 = time.time()
        
        # dynamical matrices (divide by the masses)
        d2_k = phi2_k * mm_inv_mat
        d2_mq_mk = phi2_mq_mk * mm_inv_mat
        
        # Diagonalize the dynamical matrices
        w2_k, pols_k = np.linalg.eigh(d2_k)
        w2_mq_mk, pols_mq_mk = np.linalg.eigh(d2_mq_mk)
        
        
        is_k_gamma = CC.Methods.is_gamma(structure.unit_cell, k)
        is_mq_mk_gamma = CC.Methods.is_gamma(structure.unit_cell, -q-k)
        
        if is_k_gamma:
            w2_k[0:3]=0.0
        #assert (w2_k >= 0.0).all()
        w_k=np.sqrt(w2_k)

        if is_mq_mk_gamma:
            w2_mq_mk[0:3]=0.0
        #assert (w2_mq_mk >= 0.0).all()
        w_mq_mk=np.sqrt(w2_mq_mk)
        
        # Dividing the phi3 by the sqare root of masses
        d3 = np.einsum("abc, a, b, c -> abc", phi3, 1/np.sqrt(m), 1/np.sqrt(m), 1/np.sqrt(m))

        # d3 in mode components
        #d3_pols = np.einsum("abc, ai, bj, ck -> ijk", d3, pols_mq, pols_k, pols_q_mk)
        d3_pols = np.einsum("abc, ai -> ibc", d3, pols_q)
        d3_pols = np.einsum("abc, bi -> aic", d3_pols, pols_k)
        d3_pols = np.einsum("abc, ci -> abi", d3_pols, pols_mq_mk)
        
        t4 = time.time()
        
        
        # Fortran duty ====
        
        tmp_bubble = thirdorder.third_order_bubble.compute_static_bubble(T,np.array([w_q,w_k,w_mq_mk]).T,
                                                                       np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                                       d3_pols,n_mod=3*structure.N_atoms)        
        
            
        
        t5 = time.time()
        
        if verbose:
            print("Time to interpolate the third order: {} s".format(t2 - t1))
            print("Time to interpolate the second order: {} s".format(t3 - t2))
            print("Time to transform the tensors: {} s".format(t4 - t3))
            print("Time to compute the bubble: {} s".format(t5 - t4))
        
        return tmp_bubble
    
    
    CC.Settings.SetupParallel()
    tmp_bubble = CC.Settings.GoParallel(compute_k, k_points, reduce_op = "+")
    # divide by the N_k factor
    tmp_bubble /= len(k_points) 
    # bubble in cartesian  
    d_bubble = np.einsum("ab, ia, jb -> ij", tmp_bubble, pols_q, np.conj(pols_q))
    # add to the SSCHA dynamical matrix in q
    d2_final_q = d2_q + d_bubble 
    # and mutiply by the masses ( -> FC)
    phi2_final_q = d2_final_q * mm_mat
    
    return phi2_final_q
    #new_dyn = CC.Phonons.Phonons(dyn.structure)
    #new_dyn.q_tot = [q]
    #new_dyn.dynmats[0] = phi2_final_q
    #new_dyn.save_qe("dyn_plus_odd")

def get_static_bubble_old(dyn, tensor3, k_grid, q, T = 0):
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
        
    # Get the phi2 in q
    phi2_q = CC.Phonons.InterpolateDynFC(superdyn.dynmats[0], dyn.GetSupercell(),
                                         dyn.structure, superdyn.structure,
                                         q)
    # dynamical matrix in q
    m = np.tile(dyn.structure.get_masses_array(), (3,1)).T.ravel()    
    mm_mat = np.sqrt(np.outer(m, m))
    mm_inv_mat = 1 / mm_mat
    #
    d2_q = phi2_q * mm_inv_mat
    
    # Diagonalize the dynamical matrix in q
    w2_q, pols_q = np.linalg.eigh(d2_q)
    
    # Check if the q point is gamma
    is_q_gamma = CC.Methods.is_gamma(dyn.structure.unit_cell, q)
    
    if is_q_gamma:
        w2_q[0:3]=0.0
    assert (w2_q >= 0.0).all()
    w_q=np.sqrt(w2_q)
    
    # Allocate the memory for the bubble
    tmp_bubble = np.zeros((3*dyn.structure.N_atoms, 3*dyn.structure.N_atoms),
                          dtype = np.complex128, order = "F")
    
    def compute_k(k):
        # phi3 in q, k, -q - k
        t1 = time.time()        
        phi3=tensor3.Interpolate(k,-q-k)
        t2 = time.time()
        # phi2 in k
        phi2_k = CC.Phonons.InterpolateDynFC(superdyn.dynmats[0], dyn.GetSupercell(),
                                           dyn.structure, superdyn.structure,
                                           k)
        # phi2 in -q-k
        phi2_mq_mk = CC.Phonons.InterpolateDynFC(superdyn.dynmats[0], dyn.GetSupercell(),
                                           dyn.structure, superdyn.structure,
                                           -q - k)
        t3 = time.time()
        
        # dynamical matrices (divide by the masses)
        d2_k = phi2_k * mm_inv_mat
        d2_mq_mk = phi2_mq_mk * mm_inv_mat
        
        # Diagonalize the dynamical matrices
        w2_k, pols_k = np.linalg.eigh(d2_k)
        w2_mq_mk, pols_mq_mk = np.linalg.eigh(d2_mq_mk)
        
        
        is_k_gamma = CC.Methods.is_gamma(dyn.structure.unit_cell, k)
        is_mq_mk_gamma = CC.Methods.is_gamma(dyn.structure.unit_cell, -q-k)
        
        if is_k_gamma:
            w2_k[0:3]=0.0
        assert (w2_k >= 0.0).all()
        w_k=np.sqrt(w2_k)

        if is_mq_mk_gamma:
            w2_mq_mk[0:3]=0.0
        assert (w2_mq_mk >= 0.0).all()
        w_mq_mk=np.sqrt(w2_mq_mk)
        
        # Dividing the phi3 by the sqare root of masses
        d3 = np.einsum("abc, a, b, c -> abc", phi3, 1/np.sqrt(m), 1/np.sqrt(m), 1/np.sqrt(m))

        # d3 in mode components
        #d3_pols = np.einsum("abc, ai, bj, ck -> ijk", d3, pols_mq, pols_k, pols_q_mk)
        d3_pols = np.einsum("abc, ai -> ibc", d3, pols_q)
        d3_pols = np.einsum("abc, bi -> aic", d3_pols, pols_k)
        d3_pols = np.einsum("abc, ci -> abi", d3_pols, pols_mq_mk)
        
        t4 = time.time()
        
        
        # Fortran duty ====
        
        tmp_bubble = thirdorder.third_order_bubble.compute_static_bubble(T,np.array([w_q,w_k,w_mq_mk]).T,
                                                                       np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                                       d3_pols,n_mod=3*dyn.structure.N_atoms)        
        
            
        
        t5 = time.time()
        
        
        print("Time to interpolate the third order: {} s".format(t2 - t1))
        print("Time to interpolate the second order: {} s".format(t3 - t2))
        print("Time to transform the tensors: {} s".format(t4 - t3))
        print("Time to compute the bubble: {} s".format(t5 - t4))
        
        return tmp_bubble
    
    
    CC.Settings.SetupParallel()
    tmp_bubble = CC.Settings.GoParallel(compute_k, k_points, reduce_op = "+")
    # divide by the N_k factor
    tmp_bubble /= len(k_points) 
    # bubble in cartesian  
    d_bubble = np.einsum("ab, ia, jb -> ij", tmp_bubble, pols_q, np.conj(pols_q))
    # add to the SSCHA dynamical matrix in q
    d2_final_q = d2_q + d_bubble 
    # and mutiply by the masses ( -> FC)
    phi2_final_q = d2_final_q * mm_mat
    
    return phi2_final_q
    #new_dyn = CC.Phonons.Phonons(dyn.structure)
    #new_dyn.q_tot = [q]
    #new_dyn.dynmats[0] = phi2_final_q
    #new_dyn.save_qe("dyn_plus_odd")
        
        
def get_static_correction(dyn, tensor3, k_grid, list_of_q_points, T, asr = True):
    """
    Get the dyn + static bubble correction for the list of q points
    """
    dynq = np.zeros( (len(list_of_q_points), 3*dyn.structure.N_atoms, 3*dyn.structure.N_atoms), dtype = np.complex128 )

    # Prepare the tensor2
    tensor2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
    tensor2.SetupFromPhonons(dyn)
    tensor2.Center()
    
    for iq, q in enumerate(list_of_q_points):
        dynq[iq, :, :] = get_static_bubble(tensor2, tensor3, k_grid, np.array(q),T, asr = asr)  

    return dynq

def get_static_correction_interpolated(dyn, tensor3, T, new_supercell, k_grid):
    """
    Interpolate the dyn + the v3 tensor on a new supercell.
    The dyn and the tensor3 can be defined on different supercells

    Parameters
    ----------
        dyn : Phonons()
            The harmonic / SSCHA dynamical matrix
        tensor3 : Tensor3()
            The third order force constant matrix
        T : float
            The temperature
        new_supercell : list(len = 3)
            The new supercell on which you want to interpolate the results
        k_grid : list(len = 3)
            The integration grid on the Brilluin zone

    Results
    -------
        dyn_plus_odd : Phonons()
            The dynamical matrix that includes the static bubble.
    """

    new_dyn = Phonons.Phonons(dyn.structure)

    q_tot = symmetries.GetQGrid(dyn.structure.unit_cell, new_supercell)

    # Prepare the q points for the new dynamical matrix
    new_dyn.q_tot = q_tot 
    # For now we fill all the q point in the same star (we will adjust them later)
    new_dyn.q_stars = [ [x.copy() for x in q_tot] ]

    # Get the dynamical matrix interpolated along each q point
    dynq = get_static_correction(dyn, tensor3, k_grid, q_tot, T)

    # Add all the new computed dynamical matrix
    for iq in range(len(q_tot)):
        new_dyn.dynmats.append(dynq[iq, :, :])

    # Adjust the dynamical matrix q points and the stars
    new_dyn.AdjustQStar()


    return new_dyn


    
        

def get_static_correction_along_path(dyn, tensor3, k_grid, q_path, T):
    """
    Get the dyn + static bubble correction on the give path in a plottable fashion.
    
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
    q_tot = np.sqrt(np.sum(np.diff(np.array(q_path), axis = 0)**2, axis = 1))
    x_length[1:] = q_tot
    x_length=np.cumsum(x_length)
    x_length_exp=np.expand_dims(x_length,axis=0) 
    # Allocate frequencies array
    frequencies = np.zeros((len(q_path), 3 * dyn.structure.N_atoms))    
    # Mass matrix
    m = np.tile(dyn.structure.get_masses_array(), (3,1)).T.ravel()
    mm_mat = np.sqrt(np.outer(m, m))
    #
    dynq = get_static_correction(dyn, tensor3, k_grid, q_path, T)
    # ==============================
    for iq in range(len(q_path)):
        #
        w2, p = np.linalg.eigh(dynq[iq, :, :] / mm_mat)
        frequencies[iq,:] = np.sqrt(np.abs(w2)) * CC.Units.RY_TO_CM
        #      
    # ==============================
    return np.hstack((q_path,x_length_exp.T,frequencies))
    
    
    
        
        
                                         
        
        
        
