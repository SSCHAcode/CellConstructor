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
import cellconstructor.Methods as Methods 


import time

from cellconstructor.Settings import ParallelPrint as print 


"""
In this module we compute the Spectral function 
using the interpolation on the third order force constant matrix
"""

# ========================== STATIC ==================================================

def get_static_bubble(tensor2, tensor3, k_grid, q, T , verbose = False):
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
    phi2_q = tensor2.Interpolate(q, asr = False)


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
    if not (w2_q >= 0.0).all():
        print('q= ',q, '    (2pi/A)')
        print('w(q)= ',np.sign(w2_q)*np.sqrt(np.abs(w2_q))*CC.Units.RY_TO_CM,'  (cm-1)')
        print('Cannot continue with SSCHA negative frequencies')
        exit()
    w_q=np.sqrt(w2_q)    
    
    # Allocate the memory for the bubble
    n_mod=3*structure.N_atoms
    tmp_bubble = np.zeros((n_mod, n_mod),
                          dtype = np.complex128, order = "F")
    
    def compute_k(k):
        
        # phi3 in q, k, -q-k
        t1 = time.time()        
        phi3=tensor3.Interpolate(k,-q-k, asr = False)
        t2 = time.time()
 
        # phi2 in k
        phi2_k = tensor2.Interpolate(k, asr = False) 

        # phi2 in -q-k
        phi2_mq_mk = tensor2.Interpolate(-q-k, asr = False)

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
        if not (w2_k >= 0.0).all():
            print('k= ',k, '    (2pi/A)')
            print('w(k)= ',np.sign(w2_k)*np.sqrt(np.abs(w2_k))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_k=np.sqrt(w2_k)

        if is_mq_mk_gamma:
            w2_mq_mk[0:3]=0.0
        if not (w2_mq_mk >= 0.0).all():
            print('-q-k= ',-q-k, '    (2pi/A)')
            print('w(-q-k)= ',np.sign(w2_mq_mk)*np.sqrt(np.abs(w2_mq_mk))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_mq_mk=np.sqrt(w2_mq_mk)
        
        # Dividing the phi3 by the sqare root of masses
        d3 = np.einsum("abc, a, b, c -> abc", phi3, 1/np.sqrt(m), 1/np.sqrt(m), 1/np.sqrt(m))

        # d3 in mode components
        # d3_pols = np.einsum("abc, ai, bj, ck -> ijk", d3, pols_mq, pols_k, pols_q_mk)
        d3_pols = np.einsum("abc, ai -> ibc", d3, pols_q)
        d3_pols = np.einsum("ibc, bj -> ijc", d3_pols, pols_k)
        d3_pols = np.einsum("ijc, ck -> ijk", d3_pols, pols_mq_mk)
        
        t4 = time.time()
        
        # Fortran duty ====

        tmp_bubble = thirdorder.third_order_bubble.compute_static_bubble(T,np.array([w_q,w_k,w_mq_mk]).T,
                                                                       np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                                       d3_pols,
                                                                       n_mod=n_mod)        
        
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
    #d_bubble = np.einsum("ab, ia, jb -> ij", tmp_bubble, pols_q, np.conj(pols_q))
    
    d_bubble = np.einsum("ij, ai -> aj", tmp_bubble, pols_q)
    d_bubble = np.einsum("aj, bj -> ab", d_bubble, np.conj(pols_q))

    # add to the SSCHA dynamical matrix in q
    d2_final_q = d2_q + d_bubble  
    # and mutiply by the masses ( -> FC)
    phi2_final_q = d2_final_q * mm_mat
 
    return phi2_final_q, w_q
        
def get_static_correction(dyn, tensor3, k_grid, list_of_q_points, T, asr = False):
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
            The integration grid on the Brillouin zone

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

def get_static_correction_along_path(dyn, 
                                     tensor3, 
                                     k_grid,
                                     T=0,
                                     q_path=[0.0,0.0,0.0],                                           
                                     q_path_file=None,
                                     print_path = True,                                           
                                     filename_st="v2+d3static_freq.dat",
                                     print_dyn = False,
                                     name_dyn = "sscha_plus_odd_dyn",
                                     d3_scale_factor = None,
                                     tensor2 = None):
    """
    Get the dyn + static bubble correction along a given path and prints the SSCHA and the 
    corrected frequencies in the file filename_st (path length in 2pi/Angstrom, SSCHA frequencies (cm-1),
    SSCHA+static odd correction frequencies (cm-1)).
    If print_dyn = True, the dyn+static bubble dynamical matrices are printed in QE format.
    
    Parameters
    ----------
    
        dyn : Phonons()
            The harmonic / SSCHA dynamical matrix
        tensor3 : Tensor3()
            The third order force constant matrix
        k_grid : list(len = 3)
            The integration grid on the Brillouin zone
        
        Optional
        --------
        
        T : float
            The temperature 
            (default: 0 K)
        q_path : list of triplets
                 Path of the q-points of the Brillouin Zone, in 2pi/Anstrom units,
                 where the caculation is performed 
                 (defualt: [0.0,0.0,0.0])            
        q_path_file : string
                      Name of the file where the q_path can be read. 
                      Format: column of triples, q points in 2pi/Angstrom
                      If the name of the file is present, it has the
                      priority over the q_path value 
                      (default: None)
        print_path : logical
                     If True (and the path is composed of more then one q-point), 
                     a file 'path_len.dat' is printed.
                     Format: column of 4 values, coordinates of 
                     the q-point and path length (in 2pi/Angstrom) .
                     (default: True)  
        filename_st : string 
                      File where the result is written. 
                      Format: length of the path (in 2pi/Alat), 
                      SSCHA freq (cm-1),SSCHA+static bubble freq. (cm-1)
                      (default: "v2+d3static_freq.dat")        
        print_dyn : logical 
                    If True, the dyn+odd dynamical matrices are printed
                    for the q-points of the path (in QE format)
                    (default: False) 
        name_dyn :   string
                    Prefix of the name of the dyn+odd dynamical matrix printed
                    name: prefix#q(progressive_number)
                    (default: "sscha_plus_odd_dyn")
        d3_scale_factor : float 
                          If present, the 3rd order FC is multiplied by this factor
                          (e.g. it can be used to make tests about the perturbative limit)
                          (default: None)
        tensor2 : ndarray( size = (3*nat, 3*nat), dtype = np.float)
                  If present, this 2nd order FC overwrites the one 
                  obtained from dyn.
                  (default: None)  
        
    """
     
    print(" ") 
    print(" ====================================" ) 
    print("      Bubble static correction       " ) 
    print(" ====================================" )
    print(" ") 
    print(" T= {:>5.1f} K".format(T))
    print(" k grid= {} x {} x {} ".format(*tuple(k_grid))) 
    print(" ")    
        
    if ( tensor2 == None ):        
        # Prepare the tensor2
        tensor2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
        tensor2.SetupFromPhonons(dyn)
        tensor2.Center()    

    # Scale the FC3 ===========================================================================
    if  d3_scale_factor != None :
            print(" ")
            print(" d3 scaling : d3 -> d3 x {:>7.3f}".format(d3_scale_factor))
            print(" ")
            tensor3.tensor=tensor3.tensor*d3_scale_factor
    #  ================================== q-PATH ===============================================   
    if  q_path_file == None:
        q_path=np.array(q_path)
    else:
        print(" ")
        print(" q_path read from "+q_path_file)
        print(" ")        
        q_path=np.loadtxt(q_path_file)
    if len(q_path.shape) == 1 : q_path=np.expand_dims(q_path,axis=0)    
    # Get the length of the q path
    x_length = np.zeros(len(q_path))        
    q_tot = np.sqrt(np.sum(np.diff(np.array(q_path), axis = 0)**2, axis = 1))
    x_length[1:] = q_tot
    x_length=np.cumsum(x_length)
    x_length_exp=np.expand_dims(x_length,axis=0) 
    # print the path q-points and length
    if print_path and (q_path.shape[0] > 1) :
        fmt_txt=['%11.7f\t','%11.7f\t','%11.7f\t\t','%10.6f\t']
        result=np.hstack((q_path,x_length_exp.T))        
        np.savetxt('path_len.dat',result,fmt=fmt_txt)
        print(" ")
        print(" Path printed in path_len.dat ")
        print(" ")        
    # ==========================================================================================    
   
    # Mass matrix
    m = np.tile(dyn.structure.get_masses_array(), (3,1)).T.ravel()
    mm_mat = np.sqrt(np.outer(m, m))
    
    # Allocate frequencies array
    nat=dyn.structure.N_atoms
    n_mod=3 * nat
    frequencies = np.zeros((len(q_path), n_mod), dtype = np.float64 ) # SSCHA+odd freq
    v2_wq = np.zeros( (len(q_path), n_mod), dtype = np.float64 ) # pure SSCHA freq
        
    # =============== core calculation ===========================================
    if print_dyn:
        print(" ")
        print(" dyn+odd dynamical matrices printed in "+name_dyn+"#q")
        print(" ")        
    for iq, q in enumerate(q_path):
        dynq, v2_wq[iq,:] = get_static_bubble(tensor2=tensor2, tensor3=tensor3, 
                                              k_grid=k_grid, q=np.array(q), 
                                         T=T, verbose = False)

        w2, pol = np.linalg.eigh(dynq / mm_mat)
        frequencies[iq,:] = np.sign(w2)*np.sqrt(np.abs(w2))
    
        if print_dyn:            
            Methods.save_qe(dyn,q,dynq,frequencies[iq,:],pol,fname=name_dyn+str(iq+1))
    # ============================================================================    

    # === print result ==================================
    frequencies *= CC.Units.RY_TO_CM
    v2_wq *= CC.Units.RY_TO_CM
    result=np.hstack((x_length_exp.T,v2_wq,frequencies))     
    fmt_txt='%10.6f\t\t'+n_mod*'%11.7f\t'+'\t'+n_mod*'%11.7f\t'
      
    print(" ")
    print(" Results printed in "+filename_st)
    print(" ")
    head=("------------------------------------------------------------------------"
        "\nlen (2pi/Angstrom), sscha freq (cm-1), sscha + static bubble freq (cm-1)"  
        "\n------------------------------------------------------------------------")
    
    np.savetxt(filename_st,result,fmt=fmt_txt,header=head) 
    # ==================================================================================   
 









 
 
 # ========================= FULL DYNAMIC =========================

def get_full_dynamic_bubble(tensor2, tensor3, k_grid, q, 
                            smear_id, smear, energies,
                            T,
                            static_limit, 
                            notransl, diag_approx,
                            verbose = False ):
    

    """
    COMPUTE THE FULL DYNAMIC BUBBLE
    =========================
    
    This function computes the dynamic bubble for a given dynamical matrix,
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
    phi2_q = tensor2.Interpolate(q, asr = False)


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
    if not (w2_q >= 0.0).all():
        print('q= ',q, '    (2pi/A)')
        print('w(q)= ',np.sign(w2_q)*np.sqrt(np.abs(w2_q))*CC.Units.RY_TO_CM,'  (cm-1)')
        print('Cannot continue with SSCHA negative frequencies')
        exit()
    w_q=np.sqrt(w2_q) 
    
    # Allocate the memory for the bubble
    ne=energies.shape[0]
    nsm=smear.shape[0]
    tmp_bubble = np.zeros((ne,nsm,3*structure.N_atoms, 3*structure.N_atoms),
                          dtype = np.complex128, order = "F")

    
    def compute_k(k):
        # phi3 in q, k, -q - k
        t1 = time.time()        
        phi3=tensor3.Interpolate(k,-q-k, asr = False)
        t2 = time.time()
        # phi2 in k
        phi2_k = tensor2.Interpolate(k, asr = False) 

        # phi2 in -q-k
        phi2_mq_mk = tensor2.Interpolate(-q -k, asr = False)

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
        if not (w2_k >= 0.0).all():
            print('k= ',k, '    (2pi/A)')
            print('w(k)= ',np.sign(w2_k)*np.sqrt(np.abs(w2_k))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_k=np.sqrt(w2_k)

        if is_mq_mk_gamma:
            w2_mq_mk[0:3]=0.0
        if not (w2_mq_mk >= 0.0).all():
            print('-q-k= ',-q-k, '    (2pi/A)')
            print('w(-q-k)= ',np.sign(w2_mq_mk)*np.sqrt(np.abs(w2_mq_mk))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
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
        tmp_bubble = thirdorder.third_order_bubble.compute_dynamic_bubble(energies,smear,static_limit,T,
                                                            np.array([w_q,w_k,w_mq_mk]).T,
                                                            np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                            d3_pols,diag_approx,ne,nsm,n_mod=3*structure.N_atoms)        
        
        t5 = time.time()
        
        if verbose:
            print("Time to interpolate the third order: {} s".format(t2 - t1))
            print("Time to interpolate the second order: {} s".format(t3 - t2))
            print("Time to transform the tensors: {} s".format(t4 - t3))
            print("Time to compute the bubble: {} s".format(t5 - t4))
        
        return tmp_bubble
    
    
    CC.Settings.SetupParallel()
    d_bubble_mod = CC.Settings.GoParallel(compute_k, k_points, reduce_op = "+")
    # divide by the N_k factor
    d_bubble_mod /= len(k_points) # (ne,nsmear,3nat,3nat)
    # the self-energy bubble in cartesian coord, divided by the sqare root of masses
    d_bubble_cart = np.einsum("pqab, ia, jb -> pqij", d_bubble_mod, pols_q, np.conj(pols_q))
    # get the spectral function
    no_gamma_pick=bool(is_q_gamma*notransl)
    #
    if no_gamma_pick :
        print(" ")
        print(" The acoustic pick in Gamma is discarded ")
        print(" ")
    #   
    # 
    spectral_func=thirdorder.third_order_bubble.compute_spectralf(smear_id,
                                                                  energies,
                                                                  d2_q,
                                                                  d_bubble_cart,
                                                                  no_gamma_pick,
                                                              structure.get_masses_array(),
                                                              structure.N_atoms,ne,nsm)  
    
    return spectral_func


def get_full_dynamic_correction_along_path(dyn, 
                                           tensor3, 
                                           k_grid,  
                                           e1, de, e0,
                                           sm1, sm0, 
                                           sm1_id, sm0_id,
                                           nsm=1,
                                           T=0,
                                           q_path=[0.0,0.0,0.0],                                           
                                           q_path_file=None,
                                           print_path = True,                                           
                                           static_limit = False, 
                                           notransl = True, 
                                           diag_approx = False, 
                                           filename_sp='full_spectral_func',
                                           d3_scale_factor=None,
                                           tensor2 = None):
 
    """
    Get the spectral function for a list of energies, and several q along a given path.
    The calculations are performed for several values of smearings to calculate the self-energy
    and the Green function. The resuls is printed in the file
    filename_sp_[id_smear]_[smear].dat (path length in 2pi/Angstrom, energies (cm-1),
    spectral function (1/cm-1)).
    
    Parameters
    ----------
    
        dyn : Phonons()
            The harmonic / SSCHA dynamical matrix
        tensor3 : Tensor3()
            The third order force constant matrix
        k_grid : list(len = 3)
            The integration grid on the Brillouin zone        
        e1, de ,e0: float
                    The list of energies considered (cm-1), from e0 to e1, with interval de
        sm0, sm1 : float      
              Minimum and maximum value of the smearing (cm-1) to compute the self-energy   
        sm0_id, sm1_id : float      
              Minimum and maximum value of the smearing (cm-1) for the term of the Green function 
              proportional to the identity        

        Optional
        --------

        nsm : integer
              Number of smearings to consider         
              (default = 1)
        T : float
            The temperature 
            (default: 0 K)
        q_path : list of triplets
                 Path of the q-points of the Brillouin Zone, in 2pi/Anstrom units,
                 where the caculation is performed 
                 (defualt: [0.0,0.0,0.0])            
        q_path_file : string
                      Name of the file where the q_path can be read. 
                      Format: column of triples, q points in 2pi/Angstrom
                      If the name of the file is present, it has the
                      priority over the q_path value 
                      (default: None)
        print_path : logical
                     If True (and the path is composed of more then one q-point), 
                     a file 'path_len.dat' is printed.
                     Format: column of 4 values, coordinates of 
                     the q-point and path length (in 2pi/Angstrom) .
                     (default: True)  
        static limit : logical
                      If True the self-energy is evaluated at E=0.
                      The spectral function is given by delta peaks in correspondence
                      of the frequencies of the sscha + static bubble correction
                      (default : False)
        notransl : logical
                    If True, the contribution to the spectral function given by the acoustic
                    phonons in Gamma is discarded.
                    (defaul = True)
        diag approx : logical
                    If True, the off-diagonal terms of the slef-energy are discarded
                    (the same result can be obtained in a cheaper way by using the 
                    corresponding function)
                    (default : False)
        filename_sp  : string 
                      filename_sp_[id_smear]_[smear].dat
                      is the file where the result is written. 
                      Format: length of the path (in 2pi/Alat), 
                      energy (cm-1),spectral function (1/cm-1)
                      (default: "full_spectral_func")        
        d3_scale_factor : float 
                          If present, the 3rd order FC is multiplied by this factor
                          (e.g. it can be used to make tests about the perturbative limit)
                          (default: None)
        tensor2 : ndarray( size = (3*nat, 3*nat), dtype = np.float)
                  If present, this 2nd order FC overwrites the one 
                  obtained from dyn.
                  (default: None)  
        
    """
 
    print(" ") 
    print(" ===========================================" ) 
    print("        Bubble full dynamic correction      " ) 
    print(" ===========================================" )
    print(" ") 
    print(" T= {:>5.1f} K".format(T))
    print(" k grid= {} x {} x {} ".format(*tuple(k_grid))) 
    if static_limit :
        print(" ")
        print(" - The static limit is considered - ")
        print(" ")
    if diag_approx :
        print(" ")
        print(" - The off-diagonal terms of the self-energy are discarded - ")
        print(" ")     
 
 
    if ( tensor2 == None ):
        
        # Prepare the tensor2
        tensor2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
        tensor2.SetupFromPhonons(dyn)
        tensor2.Center()     
    
    # Scale the FC3 ===========================================================================
    if  d3_scale_factor != None :
            print(" ")
            print("d3 scaling : d3 -> d3 x {:>7.3f}".format(d3_scale_factor))
            print(" ")
            tensor3.tensor=tensor3.tensor*d3_scale_factor
    #  ================================== q-PATH ===============================================   
    if  q_path_file == None:
        q_path=np.array(q_path)
    else:
        print(" ")
        print(" q_path read from "+q_path_file)
        print(" ")        
        q_path=np.loadtxt(q_path_file)
    if len(q_path.shape) == 1 : q_path=np.expand_dims(q_path,axis=0)    
    # Get the length of the q path
    x_length = np.zeros(len(q_path))        
    q_tot = np.sqrt(np.sum(np.diff(np.array(q_path), axis = 0)**2, axis = 1))
    x_length[1:] = q_tot
    x_length=np.cumsum(x_length)
    x_length_exp=np.expand_dims(x_length,axis=0) 
    # print the path q-points and length
    if print_path and (q_path.shape[0] > 1) :
        fmt_txt=['%11.7f\t','%11.7f\t','%11.7f\t\t','%10.6f\t']
        result=np.hstack((q_path,x_length_exp.T))        
        np.savetxt('path_len.dat',result,fmt=fmt_txt)
        print(" ")
        print(" Path printed in path_len.dat ")
        print(" ")        
    # ==========================================================================================     
 
    #  ======================= Energy & Smearing ==========================================    
    # energy   in input is in cm-1
    # smearing in input is in cm-1
    # converto to Ry

    # list of energies
    energies=np.arange(e0,e1,de)/CC.Units.RY_TO_CM
    ne=energies.shape[0]
    # list of smearing
    if nsm == 1 : 
        sm1=sm0
        sm1_id=sm0_id
    smear=np.linspace(sm0,sm1,nsm)/CC.Units.RY_TO_CM
    smear_id=np.linspace(sm0_id,sm1_id,nsm)/CC.Units.RY_TO_CM
    # ==========================================================================================     
    #      
    #
    spectralf = np.zeros( (len(q_path), ne, nsm), dtype = np.float64 )
    #
    for iq, q in enumerate(q_path):
        spectralf[iq, :, :] = get_full_dynamic_bubble(tensor2, tensor3, k_grid, np.array(q),
                                                      smear_id, smear, energies, T,   
                                                      static_limit, notransl , 
                                                      diag_approx, verbose=False )    
    
    # convert from 1/Ry to 1/cm-1
    spectralf /= CC.Units.RY_TO_CM
    # reconvert to cm-1
    smear *= CC.Units.RY_TO_CM 
    smear_id *= CC.Units.RY_TO_CM 
    energies *= CC.Units.RY_TO_CM 
    

    # ==================================================================================   
 
    # print the result
    for  ism in range(nsm):
        #
        name="{:5.2f}".format(smear_id[ism]).strip()+"_"+"{:6.1f}".format(smear[ism]).strip()
        #
        filename_new=filename_sp+'_'+name+'.dat'
        with open(filename_new,'w') as f:
            f.write(" # ------------------------------------------------------------- \n")
            f.write(" # len (2pi/Angstrom), energy (cm-1), spectral function (1/cm-1) \n")
            f.write(" # ------------------------------------------------------------- \n")
            for iq,leng in enumerate(x_length):
             for ie, ene in enumerate(energies):
                 f.write("{:>10.6f}\t{:>11.7f}\t{:>11.7f}\n".format(leng,ene,spectralf[iq,ie,ism]))
             f.write("\n")
 
    print(" ")
    print(" Results printed in "+filename_sp+'_[id_smear]_[smear].dat')
    print(" ")      
 
 
 


 
# ========================= DIAGONAL SELF-ENERGY DYNAMIC CORRECTION  =========================      

def get_diag_dynamic_bubble(tensor2, 
                            tensor3, 
                            k_grid, 
                            q, 
                            smear_id, 
                            smear, 
                            energies,
                            T,
                            verbose = False ):
    
    
    structure = tensor2.unitcell_structure
    
    # Get the integration points 
    k_points = CC.symmetries.GetQGrid(structure.unit_cell, k_grid)
    
        
    # Get the phi2 in q
    phi2_q = tensor2.Interpolate(q, asr = False)


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
    if not (w2_q >= 0.0).all():
        print('q= ',q, '    (2pi/A)')
        print('w(q)= ',np.sign(w2_q)*np.sqrt(np.abs(w2_q))*CC.Units.RY_TO_CM,'  (cm-1)')
        print('Cannot continue with SSCHA negative frequencies')
        exit()
    w_q=np.sqrt(w2_q)   
    
    # Allocate the memory for the bubble
    ne=energies.shape[0]
    nsm=smear.shape[0]
    nat=structure.N_atoms
    n_mod=3*nat
    tmp_bubble = np.zeros((ne,nsm,n_mod),
                          dtype = np.complex128, order = "F")

    
    def compute_k(k):
        # phi3 in q, k, -q - k
        t1 = time.time()        
        phi3=tensor3.Interpolate(k,-q-k, asr = False)
        t2 = time.time()
        # phi2 in k
        phi2_k = tensor2.Interpolate(k, asr = False) 

        # phi2 in -q-k
        phi2_mq_mk = tensor2.Interpolate(-q -k, asr = False)

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
        if not (w2_k >= 0.0).all():
            print('k= ',k, '    (2pi/A)')
            print('w(k)= ',np.sign(w2_k)*np.sqrt(np.abs(w2_k))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_k=np.sqrt(w2_k)

        if is_mq_mk_gamma:
            w2_mq_mk[0:3]=0.0
        if not (w2_mq_mk >= 0.0).all():
            print('-q-k= ',-q-k, '    (2pi/A)')
            print('w(-q-k)= ',np.sign(w2_mq_mk)*np.sqrt(np.abs(w2_mq_mk))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
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
        
        #
        tmp_bubble  = thirdorder.third_order_bubble.compute_diag_dynamic_bubble(energies,smear,T,
                                                            np.array([w_q,w_k,w_mq_mk]).T,
                                                            np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                            d3_pols,ne,nsm,n_mod=n_mod)                

        t5 = time.time()
        
        if verbose:
            print("Time to interpolate the third order: {} s".format(t2 - t1))
            print("Time to interpolate the second order: {} s".format(t3 - t2))
            print("Time to transform the tensors: {} s".format(t4 - t3))
            print("Time to compute the bubble: {} s".format(t5 - t4))
        
        return tmp_bubble    
    
    CC.Settings.SetupParallel()

    d_bubble_mod =CC.Settings.GoParallel(compute_k, k_points, reduce_op = "+")

    # divide by the N_k factor
    d_bubble_mod /= len(k_points) # (ne,nsmear,n_mod)
    #
    #
    spectralf=thirdorder.third_order_bubble.compute_spectralf_diag(smear_id,energies,w_q,
                                                                   d_bubble_mod,
                                                                   nat,ne,nsm)    
                                                                    # (ne, n_mod, nsmear)

    w2_q_ext=w2_q[None,None,...]
    z=np.sqrt(d_bubble_mod + w2_q_ext) # (A20) PHYSICAL REVIEW B 97, 214101 (2018)

    w_q_ext=w_q[None,None,...]
    z_pert=w_q_ext+np.divide(d_bubble_mod, 2*w_q_ext, out=np.zeros_like(d_bubble_mod), where=w_q_ext!=0)
    
    return spectralf, z, z_pert, w_q

    
def get_diag_dynamic_correction_along_path(dyn, tensor3, 
                                           k_grid,                                            
                                           e1, de, e0,
                                           sm1, sm0,
                                           sm1_id=None, sm0_id=None,
                                           nsm=1,
                                           q_path=[0.0,0.0,0.0],
                                           q_path_file=None, 
                                           print_path = True,
                                           T=0.0,
                                           filename_sp       = 'spectral_func',
                                           filename_z        =  None,
                                           filename_freq_dyn = 'freq_dynamic',
                                           filename_shift_lw  = 'v2_freq_shift_hwhm',
                                           self_consist = False,
                                           iterative=False,
                                           numiter=200,                                           
                                           d3_scale_factor=None,
                                           tensor2 = None):                                           
  

    """
    Get the spectral function for a list of energies, and several q along a given path,
    in the diagonal approximation (off-diagonal terms of the self-energies are discarded).
    The calculations are performed for several values of smearings to calculate the self-energy
    and the Green function. The resuls is printed in the file
    filename_sp_[id_smear]_[smear].dat (path length in 2pi/Angstrom, energies (cm-1),
    spectral function (1/cm-1), mode components of the spectral function (1/cm-1) ).
    The Z function [PRB 97 214101 (A20)] is also printed in filename_z_[id_smear]_[smear].dat.
    The frequency shift (with respect to the SSCHA frequency) and linewidth are computed in three ways
    (one optional). 1. One shot, evaluating the Z function in the SSCHA frequency value.2. Perturbative,
    evaluating the perturbative correction. 3. (optional) solving the self-consistent relation (details
    in [PRB 97 214101 (A21)]). The corresponding Lorentzian spectral functions are then printed.
    
    Parameters
    ----------
    
        dyn : Phonons()
            The harmonic / SSCHA dynamical matrix
        tensor3 : Tensor3()
            The third order force constant matrix
        k_grid : list(len = 3)
            The integration grid on the Brillouin zone        
        e1, de ,e0: float
                    The list of energies considered (cm-1), from e0 to e1, with interval de
        sm0, sm1 : float      
              Minimum and maximum value of the smearing (cm-1) to compute the self-energy          

        Optional
        --------

        nsm : integer
              Number of smearings to consider         
              (default = 1)
        T : float
            The temperature 
            (default: 0 K)
        sm0_id, sm1_id : float      
              Minimum and maximum value of the smearing (cm-1) for the term of the Green function 
              proportional to the identity. If not present, it is  sm0_id = sm1_id = 2.0 * de
              (default: None)
        q_path : list of tripletss
                 Path of the q-points of the Brillouin Zone, in 2pi/Anstrom units,
                 where the caculation is performed 
                 (defualt: [0.0,0.0,0.0])            
        q_path_file : string
                      Name of the file where the q_path can be read. 
                      Format: column of triples, q points in 2pi/Angstrom
                      If the name of the file is present, it has the
                      priority over the q_path value 
                      (default: None)
        print_path : logical
                     If True (and the path is composed of more then one q-point), 
                     a file 'path_len.dat' is printed.
                     Format: column of 4 values, coordinates of 
                     the q-point and path length (in 2pi/Angstrom) .
                     (default: True)  
        filename_sp  : string 
                      filename_sp_[smear].dat
                      is the file where the spectral function is written. 
                      Format: length of the path (in 2pi/Alat), 
                      energy (cm-1),spectral function (1/cm-1),
                      single mode contributions to spectral function (1/cm-1)
                      (default: "spectral_func")   
        filename_z  : string 
                      if present, the file
                      filename_z_[smear].dat
                      with the z function is written
                      Format: length of the path (in 2pi/Alat), 
                      energy (cm-1), z function (cm-1),
                      (default: None)
        filename_shift_lw : string      
                            filename_shift_lw_[method]_[smear].dat
                            is the file where
                            len (2pi/Angstrom), SSCHA freq (cm-1), shift (cm-1) , HWHM (cm-1)
                            are printed. [method] are "one shot", "perturb" and "self-consist" 
                            (the last one optional)
                            (default: "v2_freq_shift_hwhm")
        filename_freq_dyn :  string                   
                             filename_freq_dyn_[method]_[smear].dat
                             is the file where
                             len (2pi/Angstrom), freq (cm-1) (sorted in ascending order), corresponding HWHM (cm-1)
                             are printed. [method] are "one shot", "perturb" and "self-consist" 
                             (the last one optional)
                             (default: "freq_dynamic")                                       
        self_consist : Logical
                       If True, the dynamical frequency is found solving the self-consistent 
                       relation [PRB 97 214101 (A21)]
                       (default: False)
        iterative    : Logical
                       If True, the self-consistent relation is found iteratively
                       (default: False)                        
        numiter      : integer
                       Number of maximum steps to find the self-consistency iteratively        
                       (default : 200)
        d3_scale_factor : float 
                          If present, the 3rd order FC is multiplied by this factor
                          (e.g. it can be used to make tests about the perturbative limit)
                          (default: None)
        tensor2 : ndarray( size = (3*nat, 3*nat), dtype = np.float)
                  If present, this 2nd order FC overwrites the one 
                  obtained from dyn.
                  (default: None)  
        
    """
 

    
    print(" ") 
    print(" ===========================================" ) 
    print("     Bubble diagonal dynamic correction     " ) 
    print(" ===========================================" )
    print(" ") 
    print(" T= {:>5.1f} K".format(T))
    print(" k grid= {} x {} x {} ".format(*tuple(k_grid))) 
    print(" ")
    print(" Smearing values: ")
    for sm in np.linspace(sm0,sm1,nsm):
        print("     sm= {:>6.2f} cm-1".format(sm))  
    print(" ") 
    print(" ===========================================" ) 
    print(" " ) 


    if sm1_id != None and sm0_id != None:
        for sm in np.linspace(sm0_id,sm1_id,nsm):
                print("     sm_id= {:>6.2f} cm-1".format(sm))  
    else:
        sm1_id=de*2.0
        sm0_id=de*2.0
            
    if ( tensor2 == None ):
        
        # Prepare the tensor2
        tensor2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
        tensor2.SetupFromPhonons(dyn)
        tensor2.Center()      
    structure = tensor2.unitcell_structure



    # Scale the FC3 ===========================================================================
    if  d3_scale_factor != None :
            print(" ")
            print("d3 scaling : d3 -> d3 x {:>7.3f}".format(d3_scale_factor))
            print(" ")
            tensor3.tensor=tensor3.tensor*d3_scale_factor
    #  ================================== q-PATH ===============================================   
    if  q_path_file == None:
        q_path=np.array(q_path)
    else:
        print(" ")
        print(" q_path read from "+q_path_file)
        print(" ")        
        q_path=np.loadtxt(q_path_file)
    if len(q_path.shape) == 1 : q_path=np.expand_dims(q_path,axis=0)    
    # Get the length of the q path
    x_length = np.zeros(len(q_path))        
    q_tot = np.sqrt(np.sum(np.diff(np.array(q_path), axis = 0)**2, axis = 1))
    x_length[1:] = q_tot
    x_length=np.cumsum(x_length)
    x_length_exp=np.expand_dims(x_length,axis=0) 
    # print the path q-points and length
    if print_path and (q_path.shape[0] > 1) :
        fmt_txt=['%11.7f\t','%11.7f\t','%11.7f\t\t','%10.6f\t']
        result=np.hstack((q_path,x_length_exp.T))        
        np.savetxt('path_len.dat',result,fmt=fmt_txt)
        print(" ")
        print(" Path printed in path_len.dat ")
        print(" ")        
    #  ======================= Energy & Smearing ==========================================      
    #
    # energy   in input is in cm-1
    # smearing in input is in cm-1
    # converto to Ry
    
    # list of energies
    energies=np.arange(e0,e1,de)/CC.Units.RY_TO_CM
    ne=energies.shape[0]
    # list of smearing
    if nsm == 1 : 
        sm1=sm0
        sm1_id=sm0_id
    smear=np.linspace(sm0,sm1,nsm)/CC.Units.RY_TO_CM
    smear_id=np.linspace(sm0_id,sm1_id,nsm)/CC.Units.RY_TO_CM
    #  
    # ========================================================================================== 
    #
    n_mod=3*dyn.structure.N_atoms
    #
    spectralf   = np.zeros( (len(q_path), ne, n_mod, nsm), dtype = np.float64 )
    z           = np.zeros( (len(q_path), ne, nsm, n_mod), dtype = np.complex128 )
    z_pert      = np.zeros( (len(q_path), ne, nsm, n_mod), dtype = np.complex128 )
    wq          = np.zeros( (len(q_path), n_mod), dtype = np.float64 )
    #
    for iq, q in enumerate(q_path):

        spectralf[iq, :, :, :], z[iq, :, :, :], z_pert[iq, :, :, :], wq[iq,:]  = get_diag_dynamic_bubble(tensor2, tensor3,
                                                         k_grid, np.array(q),
                                                         smear_id, smear, energies,
                                                         T,  verbose=False )            
    
    #
    # convert from Ry to cm-1
    smear*=CC.Units.RY_TO_CM
    smear_id*=CC.Units.RY_TO_CM
    energies*=CC.Units.RY_TO_CM
    #
    z*=CC.Units.RY_TO_CM
    z_pert*=CC.Units.RY_TO_CM
    wq*=CC.Units.RY_TO_CM
    # convert from 1/Ry to 1/cm-1
    spectralf /= CC.Units.RY_TO_CM
    
    def Lorentz(x,x0,G):
        return G/((x-x0)**2+G**2)/np.pi/2.0
    def findne(val,e0,de):
        #return int( round(  ((val-e0)/de)+1 )   )
        return int(round( (val-e0)/de  ) )    
    
    
    print(" ")
    #print(" Spectral function, in diagonal approximation, printed in "+filename_sp+"_[smear_id]_[smear].dat")
    print(" Spectral function, in diagonal approximation, printed in "+filename_sp+"_[smear].dat")    
    print(" ")
    if filename_z != None:
        print(" ")
        print(" Z function [PRB 97 214101 (A21)], printed in "+filename_z+"_[smear].dat")
        print(" ")       

    print(" ========================================= ")
    print(" Frequncies shifts and widths calculations ")
    print(" ========================================= ")
    print(" ")
    print(" Frequencies shifts and linewidths computed with perturbative approximation and one-shot calculation in: ")
    print(" ")
    #print(" "+filename_shift_lw +"_perturb_[smear_id]_[smear].dat")        
    #print(" "+filename_shift_lw +"_one_shot_[smear_id]_[smear].dat")    
    print(" "+filename_shift_lw +"_perturb_[smear].dat")        
    print(" "+filename_shift_lw +"_one_shot_[smear].dat")    
    print(" ")
    print(" ")
    print(" Dynamical frequencies sorted, with HWHM: ")    
    print(" ")
    #print(" "+filename_freq_dyn +"_perturb_[smear_id]_[smear].dat")        
    #print(" "+filename_freq_dyn +"_one_shot_[smear_id]_[smear].dat")    
    print(" "+filename_freq_dyn +"_perturb_[smear].dat")        
    print(" "+filename_freq_dyn +"_one_shot_[smear].dat")    
    print(" ")
    print(" ")
    print(" Relative spectral functions in Lorentzian approximation: ")
    print(" ")
    #print(" "+filename_sp+"_lorentz_perturb_[smear_id]_[smear].dat")        
    #print(" "+filename_sp+"_one_shot_[smear_id]_[smear].dat")        
    print(" "+filename_sp+"_lorentz_perturb_[smear].dat")        
    print(" "+filename_sp+"_one_shot_[smear].dat")    
    print(" ")
    if self_consist:
        print(" ************************************************ ")
        print(" Self-consistent search for dynamical frequencies ")
        print(" ************************************************ ")
        print(" ")
        print(" Results printed in: ")    
        print(" ")
        #print(" "+filename_shift_lw +"_[smear_id]_[smear].dat")            
        print(" "+filename_shift_lw +"_self-consist_[smear].dat")            
        print(" ")     
        print(" ")
        #print(" "+filename_freq_dyn +"_[smear_id]_[smear].dat")         
        print(" "+filename_freq_dyn +"_self-consist_[smear].dat") 
        print(" ")
        print(" ")
        #print(" "+filename_sp+"_lorentz_[smear_id]_[smear].dat")         
        print(" "+filename_sp+"_lorentz_self-consist_[smear].dat") 
        print(" ")
        print(" ")
        
    for  ism in range(nsm):
        #
        # pre-name for writing data
        #
        #name="{:5.2f}".format(smear_id[ism]).strip()+"_"+"{:6.1f}".format(smear[ism]).strip()
        name="{:6.2f}".format(smear[ism]).strip()
        #
        # write spectral and z function
        #
        # =======
        # spectral func
        # =======
        filename_new=filename_sp+'_'+name+'.dat'
        fmt="{:>10.6f}\t"+"\t{:>11.3f}"+"\t{:>11.7f}"*(n_mod+1)+"\n"
        with open(filename_new,'w') as f:
            f.write("# ---------------------------------------------------------------------------------------------------------\n")
            f.write("# len (2pi/Angstrom), energy (cm-1), spectral function (1/cm-1), spectral function mode components (1/cm-1)\n")
            f.write("# ---------------------------------------------------------------------------------------------------------\n")
            for iq,leng in enumerate(x_length):
             for ie, ene in enumerate(energies):
                 out=spectralf[iq,ie,:,ism]
                 f.write(fmt.format(leng,ene,np.sum(out),*out))
             f.write("\n")    
        # =======
        # z func
        # =======
        if filename_z != None:
            filename_new=filename_z+'_'+name+'.dat'
            fmt="{:>10.6f}\t"+"\t{:>11.3f}"+"\t{:>11.7f}"*(n_mod)+"\n"
            with open(filename_new,'w') as f:
                f.write("# ---------------------------------------------------- \n")
                f.write("# len (2pi/Angstrom), energy (cm-1), z function (cm-1) \n")            
                f.write("# ---------------------------------------------------- \n")            
                for iq,leng in enumerate(x_length):
                    for ie, ene in enumerate(energies):
                        out=z[iq,ie,ism,:]
                        f.write(fmt.format(leng,ene,*out))                                                  
        # ======================================
        # compute frequency shift and linewidth
        # ======================================
                                
        if self_consist:
            res=np.zeros((len(q_path),n_mod,2),dtype=np.float64)      #   self-consist shifted freq and  linewidth  
            res_os=np.zeros((len(q_path),n_mod,2),dtype=np.float64)   #   one-shot     shifted freq and  linewidth      
            res_pert=np.zeros((len(q_path),n_mod,2),dtype=np.float64) #   perturbative shifted freq and  linewidth   
            for iq,leng in enumerate(x_length):
                is_q_gamma = CC.Methods.is_gamma(structure.unit_cell, q_path[iq])
                for ifreq in range(n_mod):
                    done=False
                    if iterative :
                    #
                    #
                        freqold=wq[iq,ifreq]
                        freqoldold=freqold
                        for i in range(numiter):
                            x=findne(freqold,e0,de)
                            if i==0: xtriv=x    
                            freqshifted=np.real(z[iq,x-1,ism,ifreq]) # Re(z) is the shifted freq
                            if abs(freqshifted-freqold)< 2*de:
                                done=True
                                break
                            else: 
                                freqoldold=freqold
                                freqold=freqshifted
                    #
                    #
                    else:
                        xtriv=findne(wq[iq,ifreq],e0,de)
                        osval=np.real(z[iq,xtriv-1,ism,ifreq])
                        diff=np.infty
                        for x in range(ne):
                            value=np.real(z[iq,x,ism,ifreq])-energies[x]
                            if( abs(value ) < 2*de) :                        
                                if (   1.0 < abs(energies[x]) or ( is_q_gamma and ifreq < 3 ) ):
                                    done=True
                                    if ( abs( energies[x]-osval ) < diff  ):
                                        diff=abs( energies[x]-osval )
                                        freqshifted=energies[x]  
                                    
                                         
                    #               
                    if done:        
                    #
                        res[iq,ifreq,0]=freqshifted   
                        x=findne(freqshifted,e0,de)
                        res[iq,ifreq,1]=-np.imag(z[iq,x-1,ism,ifreq])
                    #
                    else:
                    #    
                        print(" Self-consistency for the {:5d}-th mode of the {:5d}-th q-point not reached. "
                               "One-shot approx. value used".format(ifreq+1,iq+1))
                        res[iq,ifreq,0]=np.real(z[iq,xtriv-1,ism,ifreq])
                        res[iq,ifreq,1]=-np.imag(z[iq,xtriv-1,ism,ifreq])                        
                    #    
                    res_os[iq,ifreq,0]=np.real(z[iq,xtriv-1,ism,ifreq])                 
                    res_os[iq,ifreq,1]=-np.imag(z[iq,xtriv-1,ism,ifreq]) 
                    #
                    res_pert[iq,ifreq,0]=np.real(z_pert[iq,xtriv-1,ism,ifreq])                 
                    res_pert[iq,ifreq,1]=-np.imag(z_pert[iq,xtriv-1,ism,ifreq])                     
        else:
            res_os=np.zeros((len(q_path),n_mod,2),dtype=np.float64)   #   one-shot     shifted freq and  linewidth              
            res_pert=np.zeros((len(q_path),n_mod,2),dtype=np.float64) #   perturbative shifted freq and  linewidth        
            for iq,leng in enumerate(x_length):
                for ifreq in range(n_mod):        
                    xtriv=findne(wq[iq,ifreq],e0,de)
                    #
                    res_os[iq,ifreq,0]=np.real(z[iq,xtriv-1,ism,ifreq])                 
                    res_os[iq,ifreq,1]=-np.imag(z[iq,xtriv-1,ism,ifreq]) 
                    #
                    res_pert[iq,ifreq,0]=np.real(z_pert[iq,xtriv-1,ism,ifreq])                 
                    res_pert[iq,ifreq,1]=-np.imag(z_pert[iq,xtriv-1,ism,ifreq])         
        
        # =======================
        # v2_freq, shift, hwhm
        # =======================
        
        if self_consist:
            filename_new=filename_shift_lw+'_self-consist_'+name+'.dat'
            fmt="{:>10.6f}\t"+"\t{:>11.7f}"*(3*n_mod)+"\n"
            with open(filename_new,'w') as f:
                f.write("# ----------------------------------------------------------------- \n")
                f.write("# len (2pi/Angstrom), SSCHA freq (cm-1), shift (cm-1) , HWHM (cm-1) \n")            
                f.write("# ----------------------------------------------------------------- \n")                
                for iq,leng in enumerate(x_length):
                    out=np.concatenate((wq[iq,:],res[iq,:,0]-wq[iq,:], res[iq,:,1]))
                    f.write(fmt.format(leng,*out))                
        #         
        filename_new=filename_shift_lw+'_one_shot_'+name+'.dat'
        fmt="{:>10.6f}\t"+"\t{:>11.7f}"*(3*n_mod)+"\n"
        with open(filename_new,'w') as f:
            f.write("# ----------------------------------------------------------------- \n")
            f.write("# len (2pi/Angstrom), SSCHA freq (cm-1), shift (cm-1) , HWHM (cm-1) \n")            
            f.write("# ----------------------------------------------------------------- \n")
            for iq,leng in enumerate(x_length):
                 out=np.concatenate((wq[iq,:],res_os[iq,:,0]-wq[iq,:], res_os[iq,:,1]))
                 f.write(fmt.format(leng,*out))                         
        #         
        filename_new=filename_shift_lw+'_perturb_'+name+'.dat'
        fmt="{:>10.6f}\t"+"\t{:>11.7f}"*(3*n_mod)+"\n"
        with open(filename_new,'w') as f:
            f.write("# ----------------------------------------------------------------- \n")
            f.write("# len (2pi/Angstrom), SSCHA freq (cm-1), shift (cm-1) , HWHM (cm-1) \n")            
            f.write("# ----------------------------------------------------------------- \n")            
            for iq,leng in enumerate(x_length):
                 out=np.concatenate((wq[iq,:],res_pert[iq,:,0]-wq[iq,:], res_pert[iq,:,1]))
                 f.write(fmt.format(leng,*out))                     
        
        # ================================================
        # freq sorted, hwhm && Lorentzian spectral func
        # ================================================       
        
        if self_consist:
            
            wq_shifted=res[:,:,0]
            hwhm=res[:,:,1]

            sortidx=np.argsort(wq_shifted,axis=1)

            wq_shifted_sorted=np.take_along_axis(wq_shifted, sortidx, 1)
            hwhm_sorted=np.take_along_axis(hwhm, sortidx, 1)
            #wq_shifted_sorted_plus= wq_shifted_sorted+hwhm_sorted
            #wq_shifted_sorted_minus= wq_shifted_sorted-hwhm_sorted        
            #
            # freq, hwhm
            #
            filename_new=filename_freq_dyn+'_self-consist_'+name+'.dat'
            fmt="{:>10.6f}\t"+"\t{:>11.7f}"*(2*n_mod)+"\n"
            with open(filename_new,'w') as f:
                f.write("# ------------------------------------------------------------ \n")
                f.write("# len (2pi/Angstrom), SSCHA+shift (sorted) (cm-1), HWHM (cm-1) \n")
                f.write("# ------------------------------------------------------------ \n")                
                for iq,leng in enumerate(x_length):
                    #out=np.concatenate((wq_shifted_sorted[iq,:],
                                        #wq_shifted_sorted_plus[iq,:],
                                        #wq_shifted_sorted_minus[iq,:]))
                    out=np.concatenate((wq_shifted_sorted[iq,:],
                                        hwhm_sorted[iq,:]))                    
                    f.write(fmt.format(leng,*out))     
            # 
            # Lorentzian spectral func 
            #
            filename_new=filename_sp+'_lorentz_self-consist_'+name+'.dat'
            fmt="{:>10.6f}\t"+"\t{:>11.3f}"+"\t{:>11.7f}"*(n_mod+1)+"\n"
            with open(filename_new,'w') as f:
                f.write("# ---------------------------------------------------------------------------------------------------------\n")
                f.write("# len (2pi/Angstrom), energy (cm-1), spectral function (1/cm-1), spectral function mode components (1/cm-1)\n")                
                f.write("# ---------------------------------------------------------------------------------------------------------\n")                
                for iq,leng in enumerate(x_length):
                    Lor_spectralf=np.zeros((ne,n_mod),dtype=np.float64)       
                    for ifreq in range(n_mod):
                        Lor_spectralf[:,ifreq]=Lorentz(energies,
                                                    wq_shifted_sorted[iq,ifreq],
                                                    hwhm_sorted[iq,ifreq]+smear_id[ism])
                    for ie, ene in enumerate(energies):
                        out=Lor_spectralf[ie,:]
                        f.write(fmt.format(leng,ene,np.sum(out),*out))  
                    f.write("\n")    
        #
        wq_shifted=res_os[:,:,0]
        hwhm=res_os[:,:,1]
        
        sortidx=np.argsort(wq_shifted,axis=1)

        wq_shifted_sorted=np.take_along_axis(wq_shifted, sortidx, 1)
        hwhm_sorted=np.take_along_axis(hwhm, sortidx, 1)
        #wq_shifted_sorted_plus= wq_shifted_sorted+hwhm_sorted
        #wq_shifted_sorted_minus= wq_shifted_sorted-hwhm_sorted        
        #
        # freq, freq +/- hwhm
        #
        filename_new=filename_freq_dyn+'_one_shot_'+name+'.dat'
        fmt="{:>10.6f}\t"+"\t{:>11.7f}"*(2*n_mod)+"\n"
        with open(filename_new,'w') as f:
            f.write("# ------------------------------------------------------------ \n")
            f.write("# len (2pi/Angstrom), SSCHA+shift (sorted) (cm-1), HWHM (cm-1) \n")
            f.write("# ------------------------------------------------------------ \n")        
            for iq,leng in enumerate(x_length):
                 #out=np.concatenate((wq_shifted_sorted[iq,:],
                                     #wq_shifted_sorted_plus[iq,:],
                                     #wq_shifted_sorted_minus[iq,:]))
                 out=np.concatenate((wq_shifted_sorted[iq,:],
                                     hwhm_sorted[iq,:]))                 
                 f.write(fmt.format(leng,*out))     
        # 
        # Lorentzian spectral func 
        #
        filename_new=filename_sp+'_lorentz_one_shot_'+name+'.dat'
        fmt="{:>10.6f}\t"+"\t{:>11.3f}"+"\t{:>11.7f}"*(n_mod+1)+"\n"
        with open(filename_new,'w') as f:
            f.write("# --------------------------------------------------------------------------------------------------------- \n")
            f.write("# len (2pi/Angstrom), energy (cm-1), spectral function (1/cm-1), spectral function mode components (1/cm-1) \n")                
            f.write("# --------------------------------------------------------------------------------------------------------- \n")
            for iq,leng in enumerate(x_length):
                Lor_spectralf=np.zeros((ne,n_mod),dtype=np.float64)       
                for ifreq in range(n_mod):
                    Lor_spectralf[:,ifreq]=Lorentz(energies,
                                                  wq_shifted_sorted[iq,ifreq],
                                                  hwhm_sorted[iq,ifreq]+smear_id[ism])
                for ie, ene in enumerate(energies):
                 out=Lor_spectralf[ie,:]
                 f.write(fmt.format(leng,ene,np.sum(out),*out))   
                f.write("\n") 
        #                           
        wq_shifted=res_pert[:,:,0]
        hwhm=res_pert[:,:,1]
        
        sortidx=np.argsort(wq_shifted,axis=1)

        wq_shifted_sorted=np.take_along_axis(wq_shifted, sortidx, 1)
        hwhm_sorted=np.take_along_axis(hwhm, sortidx, 1)
        #wq_shifted_sorted_plus= wq_shifted_sorted+hwhm_sorted
        #wq_shifted_sorted_minus= wq_shifted_sorted-hwhm_sorted        
        #
        # freq, hwhm
        #
        filename_new=filename_freq_dyn+'_perturb_'+name+'.dat'
        fmt="{:>10.6f}\t"+"\t{:>11.7f}"*(2*n_mod)+"\n"
        with open(filename_new,'w') as f:
            f.write("# ------------------------------------------------------------ \n")
            f.write("# len (2pi/Angstrom), SSCHA+shift (sorted) (cm-1), HWHM (cm-1) \n")
            f.write("# ------------------------------------------------------------ \n")                    
            for iq,leng in enumerate(x_length):
                 #out=np.concatenate((wq_shifted_sorted[iq,:],
                                     #wq_shifted_sorted_plus[iq,:],
                                     #wq_shifted_sorted_minus[iq,:]))
                 out=np.concatenate((wq_shifted_sorted[iq,:],
                                     hwhm_sorted[iq,:]))
                 f.write(fmt.format(leng,*out))     
        # 
        # Lorentzian spectral func 
        #
        filename_new=filename_sp+'_lorentz_perturb_'+name+'.dat'
        fmt="{:>10.6f}\t"+"\t{:>11.3f}"+"\t{:>11.7f}"*(n_mod+1)+"\n"
        with open(filename_new,'w') as f:
            f.write("# --------------------------------------------------------------------------------------------------------- \n")
            f.write("# len (2pi/Angstrom), energy (cm-1), spectral function (1/cm-1), spectral function mode components (1/cm-1) \n")                
            f.write("# --------------------------------------------------------------------------------------------------------- \n")            
            for iq,leng in enumerate(x_length):
                Lor_spectralf=np.zeros((ne,n_mod),dtype=np.float64)       
                for ifreq in range(n_mod):
                    Lor_spectralf[:,ifreq]=Lorentz(energies,
                                                  wq_shifted_sorted[iq,ifreq],
                                                  hwhm_sorted[iq,ifreq]+smear_id[ism])
                for ie, ene in enumerate(energies):
                 out=Lor_spectralf[ie,:]
                 f.write(fmt.format(leng,ene,np.sum(out),*out))   
                f.write("\n") 
        #
        
 
 
 
 
 
# ===== PERTURBATIVE CORRECTION TO SSCHA FREQUENCY (SHIFT and LINEWIDTH) =====================             
 
def get_perturb_dynamic_selfnrg(tensor2, tensor3, 
                            k_grid, q, 
                            smear,
                            T,
                            verbose= False):
        
    structure = tensor2.unitcell_structure
    
    # Get the integration points 
    k_points = CC.symmetries.GetQGrid(structure.unit_cell, k_grid)
    
        
    # Get the phi2 in q
    phi2_q = tensor2.Interpolate(q, asr = False)

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
    if not (w2_q >= 0.0).all():
        print('q= ',q, '    (2pi/A)')
        print('w(q)= ',np.sign(w2_q)*np.sqrt(np.abs(w2_q))*CC.Units.RY_TO_CM,'  (cm-1)')
        print('Cannot continue with SSCHA negative frequencies')
        exit()
    w_q=np.sqrt(w2_q)   
    
    def compute_k(k):
        # phi3 in q, k, -q - k
        t1 = time.time()        
        phi3=tensor3.Interpolate(k,-q-k, asr = False)
        t2 = time.time()
        # phi2 in k
        phi2_k = tensor2.Interpolate(k, asr = False) 

        # phi2 in -q-k
        phi2_mq_mk = tensor2.Interpolate(-q -k, asr = False)

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
        if not (w2_k >= 0.0).all():
            print('k= ',k, '    (2pi/A)')
            print('w(k)= ',np.sign(w2_k)*np.sqrt(np.abs(w2_k))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_k=np.sqrt(w2_k)

        if is_mq_mk_gamma:
            w2_mq_mk[0:3]=0.0
        if not (w2_mq_mk >= 0.0).all():
            print('-q-k= ',-q-k, '    (2pi/A)')
            print('w(-q-k)= ',np.sign(w2_mq_mk)*np.sqrt(np.abs(w2_mq_mk))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_mq_mk=np.sqrt(w2_mq_mk)
        
        # Dividing the phi3 by the sqare root of masses
        d3 = np.einsum("abc, a, b, c -> abc", phi3, 1/np.sqrt(m), 1/np.sqrt(m), 1/np.sqrt(m))

        # d3 in mode components
        # d3_pols = np.einsum("abc, ai, bj, ck -> ijk", d3, pols_mq, pols_k, pols_q_mk)
        d3_pols = np.einsum("abc, ai -> ibc", d3, pols_q)
        d3_pols = np.einsum("abc, bi -> aic", d3_pols, pols_k)
        d3_pols = np.einsum("abc, ci -> abi", d3_pols, pols_mq_mk)
        
        t4 = time.time()
        
        
        nsm=smear.shape[0]
        n_mod=3*structure.N_atoms
        # Fortran duty ====
        
        selfnrg  = thirdorder.third_order_bubble.compute_perturb_selfnrg(smear,T,
                                                            np.array([w_q,w_k,w_mq_mk]).T,
                                                            np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                            d3_pols,nsm,n_mod)                

        t5 = time.time()
        
        if verbose:
            print("Time to interpolate the third order: {} s".format(t2 - t1))
            print("Time to interpolate the second order: {} s".format(t3 - t2))
            print("Time to transform the tensors: {} s".format(t4 - t3))
            print("Time to compute the bubble: {} s".format(t5 - t4))
        
        return selfnrg    
    
    CC.Settings.SetupParallel()

    selfnrg =CC.Settings.GoParallel(compute_k, k_points, reduce_op = "+")

    # divide by the N_k factor
    selfnrg /= len(k_points) # (n_mod,nsigma)
    
    w_q_ext=w_q[...,None]
        
    shift=np.divide(selfnrg.real, 2*w_q_ext, out=np.zeros_like(selfnrg.real), where=w_q_ext!=0)
    hwhm=np.divide(-selfnrg.imag, 2*w_q_ext, out=np.zeros_like(selfnrg.imag), where=w_q_ext!=0)

    return w_q, shift,hwhm
 
 
 
def get_perturb_dynamic_correction_along_path(dyn, tensor3, 
                                           k_grid,                                              
                                           sm1, sm0,
                                           nsm=1,
                                           q_path=[0.0,0.0,0.0],
                                           q_path_file=None,
                                           print_path = True,
                                           T=0, 
                                           filename_shift_lw  = 'v2_freq_shift_hwhm',                                           
                                           filename_freq_dyn = 'freq_dynamic',                                           
                                           d3_scale_factor=None,
                                           tensor2= None):                                           


    """
    The frequency shift (with respect to the SSCHA frequency) and linewidth are computed with the perturbative
    formula with respect to the SSCHA frequency.
    
    Parameters
    ----------
    
        dyn : Phonons()
            The harmonic / SSCHA dynamical matrix
        tensor3 : Tensor3()
            The third order force constant matrix
        k_grid : list(len = 3)
            The integration grid on the Brillouin zone        
        sm0, sm1 : float      
              Minimum and maximum value of the smearing (cm-1) to compute the self-energy   

        Optional
        --------

        nsm : integer
              Number of smearings to consider         
              (default = 1)
        T : float
            The temperature 
            (default: 0 K)
        q_path : list of triplets
                 Path of the q-points of the Brillouin Zone, in 2pi/Anstrom units,
                 where the caculation is performed 
                 (defualt: [0.0,0.0,0.0])            
        q_path_file : string
                      Name of the file where the q_path can be read. 
                      Format: column of triples, q points in 2pi/Angstrom
                      If the name of the file is present, it has the
                      priority over the q_path value 
                      (default: None)
        print_path : logical
                     If True (and the path is composed of more then one q-point), 
                     a file 'path_len.dat' is printed.
                     Format: column of 4 values, coordinates of 
                     the q-point and path length (in 2pi/Angstrom) .
                     (default: True)  
        filename_shift_lw : string      
                            filename_shift_lw_[id_smear]_[smear].dat
                            is the file where
                            len (2pi/Angstrom), SSCHA freq (cm-1), shift (cm-1) , HWHM (cm-1)
                            are printed.
                            (default: "v2_freq_shift_hwhm")
        filename_freq_dyn :  string                   
                             filename_freq_dyn_[id_smear]_[smear].dat
                             is the file where
                             len (2pi/Angstrom), freq (cm-1) (sorted in ascending order), HWHM (cm-1)
                             are printed. 
                             (default: "freq_dynamic")                                       
        d3_scale_factor : float 
                          If present, the 3rd order FC is multiplied by this factor
                          (e.g. it can be used to make tests about the perturbative limit)
                          (default: None)
        tensor2 : ndarray( size = (3*nat, 3*nat), dtype = np.float)
                  If present, this 2nd order FC overwrites the one 
                  obtained from dyn.
                  (default: None)  
        
    """



    print(" ") 
    print(" ===========================================" ) 
    print("   Bubble perturbative dynamic correction   " ) 
    print(" ===========================================" )
    print(" ") 
    print(" T= {:>5.1f} K".format(T))
    print(" k grid= {} x {} x {} ".format(*tuple(k_grid))) 
    print(" ")
    print(" Smearing values: ")
    for sm in np.linspace(sm0,sm1,nsm):
        print("     sm= {:>6.2f} cm-1".format(sm))  
    print(" ") 
    print(" ===========================================" ) 
    print(" " ) 
    
    if ( tensor2 == None ):
        
        # Prepare the tensor2
        tensor2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
        tensor2.SetupFromPhonons(dyn)
        tensor2.Center()     

    # Scale the FC3 ===========================================================================
    if  d3_scale_factor != None :
            print(" ")
            print("d3 scaling : d3 -> d3 x {:>7.3f}".format(d3_scale_factor))
            print(" ")
            tensor3.tensor=tensor3.tensor*d3_scale_factor
    #  ================================== q-PATH ===============================================   
    if  q_path_file == None:
        q_path=np.array(q_path)
    else:
        print(" ")
        print(" q_path read from "+q_path_file)
        print(" ")        
        q_path=np.loadtxt(q_path_file)
    if len(q_path.shape) == 1 : q_path=np.expand_dims(q_path,axis=0)    
    # Get the length of the q path
    x_length = np.zeros(len(q_path))        
    q_tot = np.sqrt(np.sum(np.diff(np.array(q_path), axis = 0)**2, axis = 1))
    x_length[1:] = q_tot
    x_length=np.cumsum(x_length)
    x_length_exp=np.expand_dims(x_length,axis=0) 
    # print the path q-points and length
    if print_path and (q_path.shape[0] > 1) :
        fmt_txt=['%11.7f\t','%11.7f\t','%11.7f\t\t','%10.6f\t']
        result=np.hstack((q_path,x_length_exp.T))        
        np.savetxt('path_len.dat',result,fmt=fmt_txt)
        print(" ")
        print(" Path printed in path_len.dat ")
        print(" ")        
    #  ======================= Smearing ==========================================    
    # smearing in input is in cm-1
    # converto to Ry
    # list of smearing
    #
    if nsm == 1 : 
        sm1=sm0
    smear=np.linspace(sm0,sm1,nsm)/CC.Units.RY_TO_CM
    #  ======================== Calculation ==========================================        
    n_mod=3*dyn.structure.N_atoms
    shift     = np.zeros( (len(q_path), n_mod, nsm), dtype = np.float64 ) # q-point,mode,smear
    hwhm      = np.zeros( (len(q_path), n_mod, nsm), dtype = np.float64 ) # q-point,mode,smear
    wq        = np.zeros( (len(q_path), n_mod), dtype = np.float64 )      # q-point,mode
    #
    for iq, q in enumerate(q_path):        
        wq[iq,:],shift[iq,:,:], hwhm[iq,:,:]  = get_perturb_dynamic_selfnrg(tensor2, tensor3,
                                                   k_grid, np.array(q),
                                                   smear, T, 
                                                   verbose=False )            
    
    # print results
    wq*=CC.Units.RY_TO_CM
    shift*=CC.Units.RY_TO_CM
    hwhm*=CC.Units.RY_TO_CM
    #
    #==================== SORTING ===============================
    wq_shifted=wq[...,None]+shift
        
    sortidx=np.argsort(wq_shifted,axis=1)
    
    wq_shifted_sorted=np.take_along_axis(wq_shifted, sortidx, 1)
    hwhm_sorted=np.take_along_axis(hwhm, sortidx, 1)
    #=============== Print Results ===============================    
    smear*=CC.Units.RY_TO_CM
    #
    for  ism in range(nsm):
        #
        name="{:6.2f}".format(smear[ism]).strip()
        #
        # v2 freq, corresponding  shift & hwhm
        #
        filename_new=filename_shift_lw+'_'+name+'.dat'
        fmt="{:>10.6f}\t"+"\t{:>11.7f}"*(3*n_mod)+"\n"
        with open(filename_new,'w') as f:
            f.write("# --------------------------------------------------------------------- \n")
            f.write("# len (2pi/Angstrom), sscha freq (cm-1), freq shift (cm-1), hwhm (cm-1) \n")
            f.write("# --------------------------------------------------------------------- \n")            
            for iq,leng in enumerate(x_length):
                 out=np.concatenate((wq[iq,:],shift[iq,:,ism], 
                                     hwhm[iq,:,ism]))
                 f.write(fmt.format(leng,*out))                                 
        #
        name="{:6.1f}".format(smear[ism]).strip()
        #
        # shifted freq sorted, corresponding hwhm 
        #
        filename_new=filename_freq_dyn+'_'+name+'.dat'
        fmt="{:>10.6f}\t"+"\t{:>11.7f}"*(2*n_mod)+"\n"
        with open(filename_new,'w') as f:
            f.write("# ----------------------------------------------------------------- \n")
            f.write("# len (2pi/Angstrom), sscha+shift freq (sorted) (cm-1), hwhm (cm-1) \n")            
            f.write("# ----------------------------------------------------------------- \n")
            for iq,leng in enumerate(x_length):
                 out=np.concatenate((wq_shifted_sorted[iq,:,ism],
                                     hwhm_sorted[iq,:,ism]))
                 f.write(fmt.format(leng,*out))     

    print(" ")
    print(" Results printed in "+filename_shift_lw+'_'+'[smear].dat')
    print(" ")      
    print(" ")
    print(" Results printed in "+filename_freq_dyn+'_'+'[smear].dat')
    print(" ")      

 # ================================================================================== 
