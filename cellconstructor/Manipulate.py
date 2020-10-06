# -*- coding: utf-8 -*-

"""
This module contains the methods that requre to call the classes defined
into this module.
"""

from __future__ import print_function
from __future__ import division 
import time
import os
import six

import numpy as np
import scipy, scipy.optimize
import warnings
from functools import partial


from cellconstructor.Structure import Structure
from cellconstructor.Phonons import Phonons
import cellconstructor.symmetries as symmetries
import cellconstructor.Methods as Methods
import cellconstructor.Settings as Settings
import cellconstructor.Units as Units

import symph

# Check if parallelization is available
# TODO: Replace mpi4py parallelization with a generic wrapper
try:
    from mpi4py import MPI
    __MPI__ = True
except:
    __MPI__ = False
    


def GetQ_vectors(structures, dynmat, u_disps = None):
    """
    Q VECTOR
    ========

    The q vector is a vector of the displacement respect to the
    polarization basis.
    It is computed as

    .. math ::

        q_\\mu = \\sum_\\alpha \\sqrt{M_\\alpha} u_\\alpha e_\\mu^{\\alpha}

    where :math:`M_\\alpha` is the atomic mass, :math:`u_\\alpha` the 
    displacement of the structure respect to the equilibrium and 
    :math:`e_\\mu^\\alpha` is the :math:`\\mu`-th polarization vector
    in the supercell.

    Parameters
    ----------
        structures : list
            The list of structures to compute q
        dynmat : Phonons()
            The dynamical matrix from which the polarization vectors and
            the reference structure are computed. This must be in the supercell
        u_disps : ndarray(size=(N_structures, 3*nat_sc), dtype = np.float64)
            The displacements with respect to the average positions. 
            This is not necessary (if None it is computed), but can speedup a
            lot the calculation if provided.
    
    Results
    -------
        q_vectors : ndarray (M x 3N)
            A fortran array of the q_vectors for each of the M structures
            dtype = numpy.float64
    """

    # Prepare the polarization vectors
    w, pols = dynmat.DyagDinQ(0)

    # Prepare the masses
    nat = dynmat.structure.N_atoms
    _m_ = np.zeros(3 * nat)
    for i in range(nat):
        _m_[3 *i : 3*i + 3] = dynmat.structure.get_masses_array()[i]

    # Delete the translations
    pols = pols[ :, ~ Methods.get_translations(pols, dynmat.structure.get_masses_array()) ]

    q_vects = np.zeros((len(structures), len(pols[0,:])), dtype = np.float64)
    
    _ms_ = np.sqrt(_m_)
    
    
    for i, struc in enumerate(structures):
        # Get the displacements
        if u_disps is None:
            u_disp =  struc.get_displacement(dynmat.structure).reshape(3 * nat)
        else:
            u_disp = u_disps[i, :]
        q_vects[i, :] = np.einsum("i, ij, i", _ms_, pols, u_disp)

    return q_vects


def LoadXYZTrajectory(fname, max_frames = -1, unit_cell = None):
    """
    Load the XYZ file containing an animation
    =========================================

    This function returns a list of structures containing the frames
    of the animation in the xyz file from fname.

    Parameters:
    -----------
        - fname : string
            Path to the xyz animation file
        - max_frames : int, default = -1
            Final number of the frame to be loaded. If negative read all the frames
        - unit_cell : ndarray 3x3, default None 
            Unit cell of the structure, if None the structure will not have a cell
    
    Result:
    -------
        - animation : list(Structure)
             A python list of the structures representing the frames
    """

    if not os.path.exists(fname):
        raise IOError("Error, the given file %s does not exist" % fname)
    
    # Get how many frames there are
    if max_frames < 0:
        ftmp = open(fname, "r")
        content = ftmp.readlines()
        ftmp.close()

        max_frames = 0
        total_length = len(content)
        current_index = 0
        nat = 0
        while current_index < total_length:
            #print nat, current_index, max_frames, content[current_index]
            nat = int(content[current_index])
            current_index += nat + 2
            max_frames += 1

    # Load the video
    animation = []
    for frame in range(max_frames):
        tmp = Structure()
        tmp.read_xyz(fname, frame_id = frame)
        if unit_cell is not None:
            tmp.has_unit_cell = True
            tmp.unit_cell = unit_cell

        animation.append(tmp)

        
    return animation

def SaveXYZTrajectory(fname, atoms, comments = []):
    """
    Save the animation on a XYZ file 
    ================================

    This function save on a file a list of structures containing the frames
    of the animation in the xyz file from fname.

    Parameters:
    -----------
        - fname : string
            Path to the xyz animation file
        - atoms : list
            A list of the Structure() object to be written
        - comments: list, optional
            A list of comments for each frame.
    
    """

    # Get how many frames there are
    for i, atms in enumerate(atoms):
        overwrite = False
        if i == 0:
            overwrite = True
        
        comment = "%d" % i
        if len(comments) -1 >= i:
            comment = comments[i]
        atms.save_xyz(fname, comment, overwrite)
            

def GenerateXYZVideoOfVibrations(dynmat, filename, mode_id, amplitude, dt, N_t, supercell=(1,1,1)):
    """
    XYZ VIDEO
    =========
    
    This function save in the filename the XYZ video of the vibration along the chosen mode.
    
    
    NOTE: this functionality is supported only at gamma.
    
    Parameters
    ----------
        filename : str
            Path of the filename in which you want to save the video. It is written in the xyz
            format, so it is recommanded to use the .xyz extension.
        mode_id : int
            The number of the mode. Modes are numbered by their frequencies increasing, starting
            from imaginary (unstable) ones (if any).
        amplitude : float
            The amplitude in Angstrom of the vibration per atom. 
        dt : float
            The time step between two different steps in femtoseconds.
        N_t : int
            The total number of frames.
        supercell : list of 3 ints
            The dimension of the supercell to be shown
        
    """
    # Define the conversion between the frequency in Ry and femptoseconds
    Ry_To_fs = 0.303990284048
    
    # Get polarization vectors and frequencies
    ws, polvects = dynmat.DyagDinQ(0)
    
    # Extract the good one
    w = ws[mode_id]
    polv = polvects[:, mode_id]
    
    # Get the basis structure
    basis = dynmat.structure.generate_supercell(supercell)
    
    # Reproduce the polarization vectors along the number of supercells
    polv = np.tile(polv, np.prod(supercell))
    
    video_list = []
    times = []
    for i in range(N_t):
        # Get the current time step
        t = dt * i
        frame = basis.copy()
        # Move the atoms
        for j in range(basis.N_atoms):
            frame.coords[j, :] = basis.coords[j, :] + np.real(polv[3*j : 3*j + 3]) * amplitude *  np.sin(2 * np.pi * w * t / Ry_To_fs)
        
        # Add the structure to the video list
        video_list.append(frame)
        times.append(" t = %.4f" % t)
    
    # Save the video
    SaveXYZTrajectory(filename, video_list, times)

def QHA_FreeEnergy(ph1, ph2, T, N_points = 2, return_interpolated_dyn = False):
    """
    QUASI HARMONIC APPROXIMATION
    ============================
    
    This function computes the quasi harmonic approximation interpolating the
    dynamical matrices between two different given.
    It will return as an output a matrix of free energies, at several N_points - 1 in between
    the two phonon matrices given and at each temperature specified. The free energy is computed
    then diagonalizing all the dynamical matrices in between by using the equation:
    
    .. math::
        
        F(T) = \\sum_{\\mu} \\frac{\\hbar \\omega_\\mu}{2} + k_b T \\ln \\left(1 - e^{-\\frac{\\hbar\\omega}{k_b T}}\\right)
    
    
    The interpolation is done on the dynamical matrix itself, not on the frequencies, as in this way
    it is possible to avoid the problems that wuold have occurred if two parameters where exchanged.
    
    
    Parameters
    ----------
        - ph1 : Phonons
            This is the first dynamical matrix through which the interpolation is performed.
        - ph2 : Phonons
            This is the second dynamical matrix. Must share the same number of atoms and q points with
            ph1, otherwise a ValueError exception is raised.
        - T : ndarray (float)
            Numpy array of the temperatures on which you wan to compute the Free energy. (Kelvin)
        - N_points : int, default = 2
            The number of points for the interpolation. If 2 only the original ph1 and ph2 will used.
            Note that any number lower then 2 will rise a ValueError exception.
        - return_interpolated_dyn : bool, default = False
            If true the interpolated dynamical matrix will be returned as well
            
    Results
    -------
        - free_energy : ndarray ( (N_points)x len(T)  )
            The matrix containing the free energy on all the points in between the interpolation at
            the temperature given as the input array. It is in Ry and in the unit cell
            
        - interp_dyns : list (Phonons()), len(N_points) [Only if asked]
             Only if return_interpolated_dyn is True, also the interpolated dyns are returned
    """
    
    # Define the boltzmann factor [Ry/K]
    k_b = 8.6173303e-5 * 0.073498618
    
    # Check if the two phonons are of the correct type
    if not (type(ph1) == type(Phonons()) and type(ph2) == type(Phonons())):
        raise ValueError("Error, both the first two argument must be phonons.")
        
    # Check if their structure is the same
    if not (ph1.structure.N_atoms == ph1.structure.N_atoms):
        raise ValueError("Error, the two phonon class given must belong to a structure with the same number of atoms")
    
    if N_points < 2:
        raise ValueError("Error, the number of points for the interpolation must be at least 2")
        
        
    # Perform the interpolation
    phs = [ph1]
    for i in range(1, N_points -1):
        x = i / float(N_points-1)
        
        new_ph = ph1.Copy()
        for j in range(len(ph1.dynmats)):
            # Dynmat interpolation
            new_ph.dynmats[j] = ph1.dynmats[j] + x * (ph2.dynmats[j] - ph1.dynmats[j])
            
            
        # Atomic position interpolation
        new_ph.structure.coords = ph1.structure.coords + x * (ph2.structure.coords - ph1.structure.coords)
        
        # Unit cell interpolation
        if new_ph.structure.has_unit_cell:
            new_ph.structure.unit_cell = ph1.structure.unit_cell + x*(ph2.structure.unit_cell - 
                                                                      ph1.structure.unit_cell)
        
        phs.append(new_ph)
    
    phs.append(ph2)
    
    # Now perform the free energy calculation
    free_energy = np.zeros(( N_points,len(T) ))
    N_total_q = len(ph1.dynmats)
    Tg = T[T>0] # Select positive temperatures
    for i, current_ph in enumerate(phs):
        for iq in range(N_total_q):
            # Get the frequencies
            freqs, pols = current_ph.DyagDinQ(iq)
            freqs = np.sort(freqs)
            
            # If the frequency is at gamma, correct them
            if np.sqrt(np.sum(current_ph.q_tot[iq]**2)) < 1e-6:
                # GAMMA POINT, apply translations
                freqs[:3] = 0
            
            # Check for negative frequencies
            if np.sum( (freqs < 0).astype(int) ) >= 1:
                print ("WARNING: NEGATIVE FREQUENCIES FOUND")
                print ("        ",   np.sum( (freqs < 0).astype(int) ))
                
                
            # Add the free energy
            _Tg_ = np.tile(Tg, (np.sum((freqs>0).astype(int)), 1)).transpose()
            _freqs_ = np.tile(freqs[freqs>0], (len(Tg),1))
            free_energy[i, T>0] += np.sum(_freqs_/2 + k_b * _Tg_ * np.log(1 - np.exp(-_freqs_ / (k_b * _Tg_))), axis = 1)
            free_energy[i, T==0] += np.sum(freqs / 2)
            free_energy[i, T < 0] = np.nan
    
    # Divide by the supercell dimension
    free_energy /= N_total_q 
    
    # Return the free energy
    if return_interpolated_dyn:
        return free_energy, phs
    return free_energy

def TransformStructure(dyn1, dyn2, T, structures, mode_exchange = None, mode_sign = None):
    """
    TRANSFORM STRUCTURE
    ===================
    
    This function transforms a set of randomly generated structures from the matrix dyn1 to 
    one generated accordingly to the dynamical matrix dyn2, at the temperature T.
    
    The transformation take place using the matrix:
        
    .. math::
        
        T_{ba} = \\sqrt\\frac{M_a}{M_b} \\sum_\\mu \\sqrt{\\left(\\frac{1 + 2n_\\mu^{(1)}}{1 + 2n_\\mu^{(0)}}\\right)\\frac{\\omega_\\mu^{(0)}}{\\omega_\\mu^{(1)}}} {e_\mu^b}^{(1)}{e_{\mu}^a}^{(0)}

    Where :math:`n_\\mu` are the bosonic occupation number of the :math:`\\mu` mode, (0) and (1) refers respectively
    to the starting and ending dynamical matrices and :math:`e_\\mu^a` is the a-th cartesian coordinate
    of the polarization vector associated to the :math:`\\mu` mode.
    
    The mode_exchange parameters can be used to associate the modes of the dyn1 matrix to the corresponding mdoe of the dyn2.
    They are by default ordered with increasing phonon frequencies.
    The translational modes are neglected by the summation.
    
    NOTE: for now only Gamma space dynamical matrices are supported.
    
    Parameters
    ----------
        dyn1 : Phonons.Phonons
            The dynamical matrix used to generate the structure
        dyn2 : Phonons.Phonons
            The dynamical matrix of the target structure
        T : float
            The temperature of the transformation
        structure : list of ndarray or Structure.Structur() 
            The original structures to be transformed. 
            If the type is the numpy ndarray, then they are interpreted
            as displacement respect to the average positions defined by the
            structure of the dynamical matrices. In this case an ndarray (or a list of ndarray)
            is given in output.
        mode_exchange : list (3*N_atoms, int), optional
            If present the modes order of dyn1 is exchanged when transforming the 
            structure to dyn2. This is usefull to optimize the transformation along
            many possible degenerate systems. This is applied before the mode_sign, if any
        mode_sign : list, optional
            If different from none, is a list of length equal 3*N_atoms, containing
            only -1 and 1. The polarization vector of the first dynmat will be changed
            of sign if requested. This is applied after the mode_exchange
        
    Returns
    -------
        Structure.Structure
            The transformed structure, or a list of structures, depending on the input.
    """
    
    # Conversion between Kelvin to Ry
    K_to_Ry=6.336857346553283e-06
    
    # Check if the matrix are comèpatible
    if not dyn1.CheckCompatibility(dyn2):
        raise ValueError("Error, incompatible dynamical matrices given.")
    
    # Check if the two dynamical matrix are Gamma.
    if dyn1.nqirr != 1 or dyn2.nqirr != 1:
        raise ValueError("Error, for now only Gamma point matrix are supported.")
        
    if T < 0:
        raise ValueError("Error, the given temperature must be positive.")
        
    if type(structures) != list:
        raise ValueError("Error, the passed structures must be a list")
    
    # Get the number of atoms and modes
    nat = dyn1.structure.N_atoms
    nmodes = nat * 3
    
    # Check the mode sign
    if mode_sign is not None:
        if len(mode_sign) != nmodes:
            raise ValueError("Error, mode_sign parameter must be of the same dimension than the number of modes")
        
        if np.sum( (np.abs(mode_sign) == 1).astype(int)) != len(mode_sign):
            raise ValueError("Error, mode_sign can contain only 1 or -1.")
    else:
        mode_sign = np.ones(nmodes)
            
            
    # Check that mode_exchange is correct if present
    if mode_exchange is not None:
        if type(mode_exchange) != np.ndarray:
            raise ValueError("Error, mode_exchange must be a ndarray type.")
        
        check = np.sum( (mode_exchange < 0).astype(int)) + np.sum( (mode_exchange >= nmodes).astype(int))
        if check > 0:
            raise ValueError("Error, mode_exchange can shuffle only between 0 and %d." % nmodes)
            
        for i in range(nmodes):
            if not i in mode_exchange:
                raise ValueError("Error, mode_exchange must contain all the index between 0 and %d." % (nmodes-1))
    else:
        # Initialize mode exchange
        mode_exchange = np.arange(nmodes)
    
    # Compute the T matrix, first of all dyagonalize the two dinamical matrix
    w0, pol0 = dyn1.DyagDinQ(0)
    w1, pol1 = dyn2.DyagDinQ(0)
    
    # Exchange modes according to mode exchange
    w0 = w0[mode_exchange]
    pol0 = pol0[:, mode_exchange]
    
    # Change the sign of the polarization vector if required
    pol0[:, mode_sign == -1] *= -1
    
    # Conver the polarization vectors into real one (we are in gamma)
    pol0 = np.real(pol0)
    pol1 = np.real(pol1)
    
    # Get the bosonic occupation numbers
    n0 = np.zeros(np.shape(w0))
    n1 = np.zeros(np.shape(w1))
    if T == 0:
        n0 = 0
        n1 = 0
    else:
        n0 = 1 / (np.exp(w0/(K_to_Ry * T)) -1)
        n1 = 1 / (np.exp(w1/(K_to_Ry * T)) -1)
    
    # An auxiliary array used to compute T
    factor = np.sqrt( (1. + 2*n1)/(1.+ 2*n0) * w0/w1 )
    
    # Sum over all the modes.
    # Ein summ will sum the equal indices, in this case the modes
    T = np.zeros( (nmodes, nmodes) )
    T = np.einsum("i, ji, ki" , factor, pol1, pol0)
    
    masses1 = np.zeros(nmodes)
    masses2 = np.zeros(nmodes)
    for i in range(nat):
        masses1[3*i : 3*i + 3] = dyn1.structure.masses[dyn1.structure.atoms[i]]
        masses2[3*i : 3*i + 3] = dyn2.structure.masses[dyn2.structure.atoms[i]]
        
    # Get the masses structure
    _m1_ = np.tile(masses1, (nmodes, 1))
    _m2_ = np.tile(masses2, (nmodes, 1)).transpose()
    
    # Renormalize with masses
    T *= np.sqrt(_m1_ / _m2_)

    
    # Now lets compute the structures
    u_shape = np.shape(dyn1.structure.coords)
    new_structures = []
    for struct in structures:
        # Check if the element is the displacement
        disp = np.zeros(u_shape)
        
        if type(struct) == np.ndarray:
            disp = struct
        else:
            # Extract the displacement from the structure
            for i in range(dyn1.structure.N_atoms):
                v_origin = dyn1.structure.coords[i,:]
                v1 = struct.coords[i,:] - v_origin
                
                # Translate the origin in the middle of the cell
                for k in range(3):
                    v1 += dyn1.structure.unit_cell[k,:] * .5
                
                # Put the displacement now into the unit cell to get the correct one
                disp[i,:] = Methods.put_into_cell(dyn1.structure.unit_cell, v1)
                
                # Translate the origin back to zero
                for k in range(3):
                    disp[i,:] -= dyn1.structure.unit_cell[k,:] * .5
                
        
        # Get the new vector of displacement
        tmp = np.einsum("ij, j", T, disp.reshape(np.prod(u_shape)))
        new_disp = tmp.reshape(u_shape)
        
        if type(struct) == np.ndarray:
            new_structures.append(new_disp)
        else:
            # Prepare the new structure
            new_struct = dyn2.structure.copy()
            new_struct.coords += new_disp
            # Decomment to have the new structure fixed into the unit cell
            #new_struct.fix_coords_in_unit_cell()
            
            new_structures.append(new_struct)
    
    return new_structures
    
    
def GetScalarProductPolVects(dyn1, dyn2, mode_exchange = None, mode_sign = None):
    """
    This subroutines computes the scalar product between all the
    polarization vectors of the dynamical matrix. It is very usefull to
    check if the associated mode is almost the same.
    This subroutine checks also if the dynamical matrices are compatible
    
    Also this subroutine works only at GAMMA
    
    Parameters
    ----------
        dyn1 : Phonons()
            The dynamical matrix 1
        dyn2 : Phonons()
            The dynamical matrix 2
        mode_exchange : ndarray (int)
            The array that shuffle the modes of the first matrix according to
            its data. This transformation is applied before mode_sign
        mode_sign : ndarray(int)
            An array made only by int equal to 1 and -1. It changes the sign
            of the corresponding polarization vector of the first dynamical matrix.
            This transformation is applied after mode_exchange
    
    Results
    -------
        ndarray ( 3*n_atoms )
            The scalar products between the polarization vectors.
    """
    
    if dyn1.nqirr != 1:
        raise ValueError("Error, this subroutine works only in the Gamma point")
    
    if not dyn1.CheckCompatibility(dyn2):
        raise ValueError("Error, dyn1 is not compatible with dyn2")
    
    nmodes = dyn1.structure.N_atoms * 3 
    
    # Check the mode sign
    if mode_sign is not None:
        if len(mode_sign) != nmodes:
            raise ValueError("Error, mode_sign parameter must be of the same dimension than the number of modes")
        
        if np.sum( (np.abs(mode_sign) == 1).astype(int)) != len(mode_sign):
            raise ValueError("Error, mode_sign can contain only 1 or -1.")
    else:
        mode_sign = np.ones(nmodes)
            
            
    # Check that mode_exchange is correct if present
    if mode_exchange is not None:
        if type(mode_exchange) != np.ndarray:
            raise ValueError("Error, mode_exchange must be a ndarray type.")
        
        check = np.sum( (mode_exchange < 0).astype(int)) + np.sum( (mode_exchange >= nmodes).astype(int))
        if check > 0:
            raise ValueError("Error, mode_exchange can shuffle only between 0 and %d." % nmodes)
            
        for i in range(nmodes):
            if not i in mode_exchange:
                raise ValueError("Error, mode_exchange must contain all the index between 0 and %d." % (nmodes-1))
    else:
        # Initialize mode exchange
        mode_exchange = np.arange(nmodes)
    
    w1, pol1 = dyn1.DyagDinQ(0)
    w2, pol2 = dyn2.DyagDinQ(0)
    
    # Since we are at gamma take only real part of the polarization vectors
    pol1 = np.real(pol1)
    pol2 = np.real(pol2)
    
    # perform the transformation if required
    pol1 = pol1[:, mode_exchange]
    pol1[:, mode_sign == -1] *= -1
    
    results = np.zeros( nmodes )
    
    for i in range(nmodes):
        results[i] = pol1[:, i].dot(pol2[:, i])
    
    return results

def ChooseParamForTransformStructure(dyn1, dyn2, small_thr = 0.8, n_ref = 100, n_max = 10000):
    """
    This subroutine is ment for automatically check the best values for mode_exchange
    and mode_sign to pass to the TransfromStructure. This is needed if you want to
    maximize correlation between the structures, because the transformation between two
    dynamical matrix is not uniquily defined.
    The algorithm checks exchange the mode and sign to algin at best the polarization vectors.
    
    At each step a random moovement is aptented. The first vector to moove is piked with
    a probability equal to
    
    .. math::
        
        P_0(\\mu) \\propto 1 - \\left|\\left< e_\\mu^{(0)} | e_\\mu^{(1)}\\right>\\right|
        
    Then the second mode is chosen as
    
    .. math::
        
        P(\\nu | \\mu) \\propto P_0(\\nu) \cdot \\frac{1}{|\\Delta\\omega| + \\eta}
        
    Where :math:`\\eta` is a smearing factor to avoid explosion in degenerate modes.
    The exchange between the two modes is accepted if the total score is increased.
    The total score is measured as
    
    .. math::
        
        S = \\sum_\\mu \\left|\\left< e_\\mu^{(0)} | e_\\mu^{(1)}\\right>\\right|
        
    
    
    NOTE: Works only at Gamma
    
    
    Parameters
    ----------
        dyn1 : Phonons.Phonons
            The starting dynamical matrix
        dyn2 : Phonons.Phonons
            The final dynamical matrix
        small_thr : float, optional
            If the scalar product for each mode is bigger than this value, then
            the optimization is considered to be converged
        n_ref : int, optional
            A positive integer. Even if small_thr is not converged. After n_ref
            refused mooves, the system is considered to be converged.
        n_max : int, optional
            The total number of move, after which the method is stopped even if convergence has not been achieved.
            
    Results
    -------
        mode_exchange : list
            The mode_exchange list to be feeded into TransformStructure
        mode_sign : list
            The mode_sign list to be feeded into TransformStructure
    """
    eta_smearing = 1e-5
    
    # Check that the two dyn are compatible
    if not dyn1.CheckCompatibility(dyn2):
        raise ValueError("Error, the two dynamical matrices seem not to be compatible.")
    
    # Get pol and frequencies
    w1, pol1 = dyn1.DyagDinQ(0)
    w2, pol2 = dyn2.DyagDinQ(0)
    
    # Count how much time the system can get negative answers
    refuses_counts = 0
    iterations= 0
    nmodes = dyn1.structure.N_atoms * 3
    
    # The two starting
    mode_exchange = np.arange(nmodes)
    mode_sign = np.ones(nmodes)
    
    # Perform the optimization
    while refuses_counts < n_ref:
        iterations += 1
        
        # Get the scalar product
        sp = GetScalarProductPolVects(dyn1, dyn2, mode_exchange, mode_sign)
        _w_ = w1[mode_exchange]
        
        # Neglect translations
        sp = sp[3:]
        _w_ = _w_[3:]
        
        # Check if convergence has been archived
        if np.min(np.abs(sp)) > small_thr:
            break
        
        # Get the probability P0
        p0 = 1.0 - np.abs(sp)
        p0 /= np.sum(p0) # Normalization
        
        # get the primitive
        F0 = np.array([np.sum(p0[:i+1]) for i in range(len(p0))])
        
        # Pick randomly
        x0 = np.random.rand()
        i0 = np.arange(len(F0))[x0 < F0][0]
        
        # Get the difference of energy
        delta_w = np.abs(_w_ - _w_[i0])
        
        # Define the probability of the second one
        p1 = p0  / (delta_w +  eta_smearing)
        
        # Avoid picking two times the same
        p1[i0] = 0
        
        # Normalize the probability and integrate it
        p1 /= np.sum(p1)
        F1 = np.array([np.sum(p1[:i+1]) for i in range(len(p1))])
        
        # Pick the second index for the exchange
        x1 = np.random.rand()
        i1 = np.arange(len(F1))[x1 < F1][0]
        
        # Add back the translations
        i0 += 3
        i1 += 3
        
        # Perform the exchange between the modes
        new_mode_exchange = np.copy(mode_exchange)
        tmp = new_mode_exchange[i0]
        new_mode_exchange[i0] = new_mode_exchange[i1]
        new_mode_exchange[i1] = tmp
        
        #Check the new scalar product if to accept the move
        new_sp = GetScalarProductPolVects(dyn1, dyn2, new_mode_exchange, mode_sign)
        if np.sum( np.abs(sp) ) < np.sum( np.abs(new_sp) ):
            # Accept the move
            mode_exchange = np.copy(new_mode_exchange)
            #print "Accepted: old = ", np.sum( np.abs(sp) ), " new = ", np.sum( np.abs(new_sp))
        else:
            # Refuse the move
            refuses_counts += 1
        
        # Break the cycle
        if iterations > n_max:
            warnings.warn("Warning: convergence not reached after %d iterations." % n_max, RuntimeWarning)
            break
            
        
    # Now the mode have been ordered, change the sign accordingly
    sp = GetScalarProductPolVects(dyn1, dyn2, mode_exchange, mode_sign)
    mode_sign[sp < 0] = -1

    # Return the final value
    return mode_exchange, mode_sign
        

def PlotRamanSpectra(w_axis, T, sigma, dyn, pol1=None, pol2=None):
    """
    PLOT HARMONIC RAMAN
    ===================
    
    
    This function computes the Raman spectrum of the dynamical matrix in the harmonic approximation.
    Note: it requires that the dynamical matrix to have a Raman spectrum.
    
    If Pol1 and pol2 are none the light is chose unpolarized
    
    
    Parameters
    ----------
        w_axis : ndarray
            The axis of the Raman shift (cm-1)
        T : float
            Temperature
        sigma : float
            The beam frequency standard deviation
        dyn : Phonons.Phonons()
            The dynamical matrix
        pol1 : ndarray 3
            The incident polarization vector
        pol2 : ndarray 3
            The scattered polarization vector
            
    Results
    -------
        I(w) : ndarray
            The intensity for each value of the w_axis in input
    """
    RyToCm = 109691.40235

    
    
    # Define the scattered intensity
    def scatter_gauss(w, I):
        return I/np.sqrt(2 * np.pi * sigma**2) * np.exp( - (w - w_axis)**2 / (2 * sigma**2))
        

    # Dyagonalize the dynamical matrix
    w, pols = dyn.DyagDinQ(0)
    
    # Discard the translations
    trans = Methods.get_translations(pols, dyn.structure.get_masses_array())
    
    # Convert in cm-1
    w *= RyToCm
    
    
    I_w = np.zeros(len(w_axis))
    
    # Check if the light is unpolarized
    if pol1 is None and pol2 is None:
        N_rep = 50
    else: 
        N_rep = 1
    
    for i_rep in range(N_rep):
        
        if N_rep > 1:
            pol1 = np.random.normal(size = 3)
            pol1 /= np.sqrt(pol1.dot(pol1))
            pol2 = np.random.normal(size = 3)
            pol2 /= np.sqrt(pol1.dot(pol1))
        
        I_new = dyn.GetRamanResponce(pol1, pol2, T)
        #I_new *= w#w**2 / RyToCm**2
        #I_new[trans] = 0
        #print "N_ACTIVE:", I_new > np.max(I_new) / 100
        
        for i in range(len(I_new)):
            I_w += scatter_gauss(w[i], I_new[i])
        
    return I_w / N_rep
    

def apply_symmetry_on_fc(structure, fc_matrix, symmetry):
    """ 
    APPLY SYMMETRY ON FC
    ====================
    
    This functio apply the given symmetry on the force constant matrix.
    The original structure must satisfy the symmetry, and it is used to
    get the atoms transformed one into the other.
    
    The application of the symmetries follow the following rule.
    
    The symmetry check is performed by comparing the two force constant matrix within the given threshold.
        
    .. math::
        
        \\Phi_{s(a)s(b)}^{ij} = \\sum_{h,k = 1}^3 S_{ik} S_{jh} \\Phi_{ab}^{kh}
        
        \\Phi = S \\Phi S^\\dagger
    
    where :math:`s(a)` is the atom in which the :math:`a` atom is mapped by the symmetry.
    
    Note: this work only in the gamma point, no phase for q != 0 are supported.
    Look at the documentation of the ApplySymmetry in the Phonon package
    
    Parameters
    ----------
        structure : CC.Structure()
            The structure satisfying the symmetry, it is used to get the atoms -> S(atoms) mapping
        fc_matrix : ndarray (3N x 3N)
            The force constant matrix to be applyed the symmetry on
        symmetry : ndarray (3x4)
            The symmetry operation as a 3x3 rotational. The last column is the fractional translation
            if any.
            
    Results
    -------
        new_fc : ndarray (3N x 3N)
            The result of the application of the symmetry on the original dynamical matrix.
    """
    
    # Create a dummy phonons variable
    ph = Phonons(structure, nqirr = 1)
    ph.dynmats = [ fc_matrix ]
     
    # Call the application of the phonon
    fc_out = ph.ApplySymmetry(symmetry)
     
    return fc_out


def MeasureProtonTransfer(structures, list_mol, verbose = False):
    """
    MEASURE THE PROTON TRANSFER
    ===========================
    
    This subroutine measures the proton transfer distribution from a
    list of structures. It requires the user to indentify the indices
    for the giving and the accemtping molecule.
    
    The proton transfer coordinate is defined as


    .. math::
        
        \\nu = d(X Y) - d(Y Z)
        
    Then the proton transfer ratio is defined as:

    .. math::
        
        \\int_0^\\infty P(\\nu) d\\nu
        
    That is the overall probability of finding the Proton that belongs to the molecule
    X closer to the molecule Z. This function returns a list of the :math:`\\nu`
    coordinates, then the user can plot an histogram or measure the proton transfer
    ratio as the simple ration of :math:`\\nu > 0` over the total.
    
    NOTE: This subroutine will exploit MPI if available.
    NOTE: UNTESTED
    
    Parameters
    ----------
        structures : list of Structure()
            The ensemble used to measure the proton transfer
        list_mols : list of (X,Y, Z)
            A list of truples. Here X is the index of the donor ion, Y 
            is the proton involved in the proton transfer and Z is the 
            acceptor
        verbose : bool
            If true (default is False) it prints some debugging stuff (like the
            number of processors thorugh which the process is spanned)
    Results
    -------
        proton_transfers : list of floats
            list of the :math:`\\nu` proton transfer coordinates.
    """
    
    
     # Initialize MPI if any
    if __MPI__:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if verbose:
            print ("Parallel ambient found: %d processes." % size)
    else:
        if verbose:
            print ("No parallelization found.")
        rank = 0
        size = 1
        
    # Setup the number of total coordinates
    n_coord_per_conf = len(list_mol)
    
    # Check if the structures can be divided in a perfect pool
    n_struct_partial = len(structures) // size
    
    # Setup the number of initial coordinate
    n_coords_tot = n_coord_per_conf * n_struct_partial
    
    coord_part = np.zeros(n_coords_tot, dtype = np.float64)
    
    # Start the parallel analysis
    j = 0
    for i in range(n_struct_partial):
        index = rank + size * i
        
        struct = structures[index]
        for pt in list_mol:
            v1 = struct.get_min_dist(pt[0], pt[1])
            v2 = struct.get_min_dist(pt[1], pt[2])
            coord_part[j] = v1 - v2
            j += 1
            
    # All gather into the main
    if __MPI__:
        tot_coords = np.zeros(n_coord_per_conf * len(structures), dtype = np.float64)
        comm.Allgather([coord_part, MPI.DOUBLE], [tot_coords, MPI.DOUBLE])
        
        if verbose:
            print ("On rank %d tot = " % rank, tot_coords)
    else:
        tot_coords = coord_part
    
    # Complete the remaining structure
    j *= size
    for i in range(n_struct_partial * size, len(structures)):
        struct = structures[i]
        
        for pt in list_mol:
            v1 = struct.get_min_dist(pt[0], pt[1])
            v2 = struct.get_min_dist(pt[1], pt[2])
            tot_coords[j] = v1 - v2
            j += 1
    
    return tot_coords
        

def BondPolarizabilityModel(structure, bonded_atoms, distance, threshold, alpha_l, alpha_p, alpha1_l, alpha1_p):
    """
    BOND POLARIZABILITY MODEL
    =========================

    This method uses an empirical model to obtain the raman tensor of a structure.
    It is based on the assumption that all the polarizability of the system is encoded in a molecular bond.

    NOTE: UNTESTED

    Parameters
    ----------
        structure : Structure.Structure()
            The molecular structure on which you want to build the polarizability
        bonded_atoms : list of two strings
            The atomic species that form a bond
        distance : float
            The average distance of a bond
        threshold : float
            The threshold on the atomic distance to identify two atoms that are bounded
        alpha_l : float
            The longitudinal polarization of the bond
        alpha_p : float
            The perpendicular polarization of the bond
        alpha1_l : float
            The derivative of the longitudinal polarization with respect to the bond length.
        alpha1_p : float
            The derivative of the perpendicular polarization with respect to the bond length.


    Results
    -------
        raman_tensor : ndarray(size = (3,3, 3*nat), dtype = np.float64)
            The raman tensor for the structure as computed by the bond polarizability model 
    """

    # Identify the bonds in the structure
    mols, indices = structure.GetBiatomicMolecules(bonded_atoms, distance, threshold, True)
    I = np.eye(3, dtype = np.float64)

    raman_tensor = np.zeros((3,3, 3*structure.N_atoms), dtype = np.float64)
    for i, mol in enumerate(mols):
        at_1 = indices[i][0]
        at_2 = indices[i][1]

        r_bond = mol.coords[0,:] - mol.coords[1,:]
        r_mod = np.sqrt(r_bond.dot(r_bond))
        r_bond /= r_mod

        Rk = np.zeros((structure.N_atoms, 3), dtype = np.float64)
        Rk[at_1, :] = r_bond 
        Rk[at_2, :] = -r_bond

        IRk =  np.einsum("ij,k", I, Rk.ravel()) #delta_ij R_k
        RRR = np.einsum("i, j, k", r_bond, r_bond, Rk.ravel())
        dRdK = -np.einsum("j, k", r_bond, Rk.ravel())
        for j in range(3):
            dRdK[j, 3*at_1 + j] += 1
            dRdK[j, 3*at_2 + j] -= 1
        dRdK /= r_mod

        RdRK2 = np.einsum("i, jk->ijk", r_bond, dRdK) + np.einsum("j, ik->ijk", r_bond, dRdK)
        
        raman_tensor += (2*alpha1_p + alpha1_l) /3 *IRk
        raman_tensor += (alpha1_l - alpha1_p) * (RRR - IRk/3)
        raman_tensor += (alpha_l - alpha_p) * RdRK2

    return raman_tensor


def RotationTranslationStretching(structure, molecules, indices, vector):
    """
    PROJECT IN ROT TRANS STRETCH
    ============================

    Project the given atomic vector into the molecular rotation, translations and 
    stretching. This is an usefull subroutine to analyze the vibrational modes
    of molecular crystals.

    Parameters
    ----------
        structure : Structure
            The atomic structure.
        molecules : list of Structure
            A list of structures that identifies the molecules.
        indices : list of truples of int
            A list containing the indices of the atoms in the structure that
            belong to the respective molecule.
        vector: ndarray(size = (N_atoms * 3), dtype = float64)
            The polarization vector that you want to analyze.
    
    Results
    -------
        rotations : float
            The fraction of the polarization projected on the molecular rotations
        translations : float
            The fraction of the polarization projected on the molecular translations
        stretching : float
            The fraction of the polarization projected on the molecular stretching
    """
    n_mols = len(molecules)
    nat = structure.N_atoms
    projector_trans = np.zeros( (3*nat, 3*nat), dtype = np.complex128)
    projector_stretch = np.zeros( (3*nat, 3*nat), dtype = np.complex128)
    projector_rot = np.zeros( (3*nat, 3*nat), dtype = np.complex128)

    for i in range(n_mols):
        mol_i = indices[i][0]
        mol_j = indices[i][1]
        
        # Create the translation for the current molecule
        v1 = np.zeros( (nat, 3), dtype = np.complex128)
        v2 = np.zeros( (nat, 3), dtype = np.complex128)
        v3 = np.zeros( (nat, 3), dtype = np.complex128)
        
        v1[mol_i,:] = np.array([1,0,0])
        v2[mol_i,:] = np.array([0,1,0])
        v3[mol_i,:] = np.array([0,0,1])
        v1[mol_j,:] = np.array([1,0,0])
        v2[mol_j,:] = np.array([0,1,0])
        v3[mol_j,:] = np.array([0,0,1])
        
        projector_trans += np.outer(v1.ravel(), v1.ravel()) / np.sqrt(2* n_mols)
        projector_trans += np.outer(v2.ravel(), v2.ravel()) / np.sqrt(2* n_mols)
        projector_trans += np.outer(v3.ravel(), v3.ravel()) / np.sqrt(2* n_mols)
        
        # Create the stretching
        # Get the vector that joints the two molecules
        v_joint = molecules[i].coords[1,:] - molecules[i].coords[0,:]
        v_joint /= np.sqrt(v_joint.dot(v_joint))
        
        v = np.zeros((nat, 3), dtype = np.complex128)
        v[mol_i,:] = v_joint
        v[mol_j,:] = -v_joint
        projector_stretch += np.outer(v.ravel(), v.ravel()) / np.sqrt(2* n_mols)
        
        # Now create the rotations
        v_aux = np.random.uniform(-1, 1, size=3)
        # Prject out the stretching direction
        v_aux -= v_aux.dot(v_joint)
        v_aux /= np.sqrt(v_aux.dot(v_aux))
        v_aux2 = np.cross(v_aux, v_joint)
        
        v1 = np.zeros( (nat, 3), dtype = np.complex128)
        v2 = np.zeros( (nat, 3), dtype = np.complex128)
        
        v1[mol_i, :] = v_aux
        v1[mol_j, :] = -v_aux
        v2[mol_i, :] = v_aux2
        v2[mol_j, :] = -v_aux2
        
        projector_rot += np.outer(v1.ravel(), v1.ravel()) / np.sqrt(2 *n_mols)
        projector_rot += np.outer(v2.ravel(), v2.ravel()) / np.sqrt(2* n_mols)


    # Check which modes belong to any group
    I_trans = np.sum(projector_trans.dot(vector)**2)    
    I_rot = np.sum(projector_rot.dot(vector)**2)    
    I_stretch = np.sum(projector_stretch.dot(vector)**2) 

    return I_rot, I_trans, I_stretch   


def AlignStructures(source, target, verbose = False):
    """
    ALIGN STRUCTURES
    ================

    This method finds the best alignment between the source and the target.
    The source structure coordinates are then translated and the periodical images
    are chosen so to match as closely as possible the target structure
    
    NOTE: This subroutine will rise an exception if the structures are not compatible.

    Parameters
    ----------
        source : Structure()
            The structure to be aligned
        target : Structure()
            The target structure of the alignment
        verbose : bool
            If true prints info during the minimization.
            Good for debugging.
    Results
    -------
        align_cost : float
            The sum of all the residual distances between atoms (after alignment)
    """

    # Check alignment
    assert source.N_atoms == target.N_atoms, "Error, the two structures must share the same number of atoms"


    # Check the atomic types
    # Get the unique values of the atomic types of the source
    source_types = list(set(source.atoms))

    for s_type in source_types:
        count_source = source.atoms.count(s_type)
        count_target = target.atoms.count(s_type)

        assert count_source == count_target, "Error, type {} different between source and target".format(s_type)


    # Align the cells
    max_consecutive_rejects = 50
    min_theta_step = 1e-8
    rejected = 0
    new_cell = source.unit_cell.copy()
    cost_function = np.sum( (new_cell - target.unit_cell)**2)
    theta_step = 0.05
    
    if verbose:
        t1 = time.time()
        print("Starting the cell optimization...")
    while theta_step > min_theta_step:
        # Pick a random direction
        direction = np.random.randint(0, 3)

        # Pick a random angle
        theta = np.random.normal(scale = theta_step)

        # Create the rotation matrix
        U_mat = np.eye(3, dtype = np.double)
        x_rot_i = (direction + 1) %3
        y_rot_i = (direction + 2) %3
        U_mat[x_rot_i, x_rot_i] = np.cos(theta)
        U_mat[y_rot_i, y_rot_i] = np.cos(theta)
        U_mat[x_rot_i, y_rot_i] = -np.sin(theta)
        U_mat[y_rot_i, x_rot_i] = np.sin(theta)

        # Rotate the cell
        test_cell = new_cell.dot(U_mat.T)

        # Look if the rotation improved the similarity
        new_cost =  np.sum( (test_cell - target.unit_cell)**2)
        #if verbose:
        #    print("REJ:{} | cost = {} | old cost = {} | New cell: {}".format(rejected, new_cost, cost_function, test_cell))
        if new_cost < cost_function:
            rejected = 0
            cost_function = new_cost
            new_cell = test_cell
        else:       
            rejected += 1

        if rejected > max_consecutive_rejects:
            rejected = 0
            theta_step /= 2

    # Align the two structure according to the rotations between the cells
    if verbose:
        t2 = time.time()
        print("Time elapsed to rotate the cell: {} s".format(t2-t1))
    source.change_unit_cell(new_cell)

    # Start the alignment of the translations
    def align_cost(trans):
        trial_struct = source.copy()
        trial_struct.coords += np.tile(np.array(trans), (source.N_atoms, 1))

        # Get the distances between atoms
        IRT, distances = trial_struct.get_equivalent_atoms(target, True)

        # Use a loss function that does not penalize strong outcomers
        loss = np.sum(1 - 1 / np.cosh(np.array(distances)))
        return loss

    # Perform the minimization
    if verbose:
        print ("Optimizing the translations...")
        t1 = time.time()
    res = scipy.optimize.minimize(align_cost, [0,0,0])

    # Shift the source
    if verbose:
        t2 = time.time()
        print ("Time elapsed to optimize the translations:", t2 - t1, " s")
    source.coords += np.tile(res.x, (source.N_atoms, 1))
    IRT = source.get_equivalent_atoms(target)

    # Exchange the order of the atoms to match the one of the target function
    source.coords = source.coords[IRT, :]
    source.atoms = [source.atoms[i] for i in IRT]

    # Fix the atoms in the unit cell
    source.fix_coords_in_unit_cell()


def GetSecondOrderDipoleMoment(original_dyn, structures, effective_charges, T, symmetrize = True):
    r"""
    GET THE SECOND ORDER DIPOLE MOMENT
    ==================================

    This method computes the second order dipole moment.
    It is the average second derivative of the dipole moment with respect to the atomic displacements.

    .. math::

        \frac{\partial M_x}{\partial R_a \partial R_b}

    It can be used to compute the two phonon IR response.

    Note: both the structures and the effective charges must be defined on the same supercell as the original_dyn

    The derivative of the dipole moment is computed using the average ensemble rule:

    .. math::

        \frac{\partial^2 M_x}{\partial R_a \partial R_b} = - \sum_q \Upsilon_{aq} \left< u_q\frac{\partial M_x}{\partial R_b}\right>

    If the original dynamical matrix has an effective charge, then it is removed from any effective charge, to better sample the linear dependence. 

    Parameters
    ----------
        - original_dyn : CC.Phonons.Phonons()
            The dynamical matrix of the equilibrium system.
        - structures : list 
            A list of CC.Structure.Structure() of the displaced structure with respect to original_dyn on which the effective charges are computed. 
            Alternatively, you can pass a list of strings that must point to the .scf files of the structures.
        - effective_charges : list
            A list of the effective charge tensor. Alternatively you may provide a list of strings that must point to the output of the ph.x package from Quantum ESPRESSO,
            where the effective charges are printed.
        - T : float
            The temperature (in K)
        - symmetrize : bool
            If True the tensor is simmetrized. This requires the
            symmetrization in real space, therefore spglib must be available.

    Results
    -------
        - dM_dRdR : ndarray (size = (3*nat_sc, 3*nat_sc, 3))
            The second derivative of the dipole moment. nat_sc are the atoms in the supercell.
            The first two components are the cartesian indices of the displacements, 
            while the last is the dipole vector

    """

    N_config = len(structures)

    # Check that the effective charges have the same number of configurations
    __ERR_MSG__ = """
    Error a differen number of effective charges and structures.
    N_structures  : {}
    N_eff_charges : {}
    """.format(N_config, len(effective_charges))
    assert N_config == len(effective_charges), __ERR_MSG__

    # Read the structures
    new_structures = []
    if isinstance(structures[0], six.string_types):
        # Here we replace the string loading
        for i, sname in enumerate(structures):
            if not os.path.exists(sname):
                raise IOError("Error, file {} does not exist".format(sname))

            struct = Structure()
            struct.read_scf(sname)
            new_structures.append(struct)
    else:
        new_structures = structures
        
    # Read the effective charges
    new_eff_charges = []
    if isinstance(effective_charges[0], six.string_types):
        # Here we replace the string loading
        for i, sname in enumerate(effective_charges):
            if not os.path.exists(sname):
                raise IOError("Error, file {} does not exist".format(sname))

            aux_dyn = original_dyn.GenerateSupercellDyn(original_dyn.GetSupercell())
            aux_dyn.ReadInfoFromESPRESSO(sname)
            new_eff_charges.append(aux_dyn.effective_charges)
    else:
        new_eff_charges = effective_charges

    # Check the consistency of the atoms
    nat_sc = original_dyn.structure.N_atoms * np.prod(original_dyn.GetSupercell())
    
    # Get the displacements from both position and effective charges
    u_disps = np.zeros((N_config, nat_sc*3), dtype = np.double) 
    eff_cgs = np.zeros((3, N_config, nat_sc*3), dtype = np.double)
    
    super_struct, itau = original_dyn.structure.generate_supercell(original_dyn.GetSupercell(), get_itau = True)
    ref_coords = super_struct.coords.ravel()

    # Get the central effective charges in the supercell
    ef_cg_new = np.zeros((3, nat_sc*3), dtype = np.double)
    if not original_dyn.effective_charges is None:
        for j in range(nat_sc):
            ef_cg_new[:, 3*j:3*(j+1)] = original_dyn.effective_charges[itau[j], :, :]

    # Prepare the array for the fast processing
    for i in range(N_config):
        coords = new_structures[i].coords.ravel()
        u_disps[i, :] = coords - ref_coords

        # Transpose the effective charges so to have the electric field as first index
        ef_new = np.einsum("abc -> bac", new_eff_charges[i])
        eff_cgs[:, i, :] = ef_new.reshape((3, 3 * nat_sc))
        eff_cgs[:, i,:] -= ef_cg_new

        u_norm = np.sqrt(u_disps[i,:].dot(u_disps[i,:]))
        #eff_project = eff_cgs[0, i, :].dot(u_disps[i,:]) / u_norm
        #print("{:d}) u_disp: {:.4f} | eff_charge (along u): {:.4f}".format(i+1, u_norm, eff_project))

    # Get the derivative of the effective charges
    super_dyn = original_dyn.GenerateSupercellDyn(original_dyn.GetSupercell())
    upsilon = super_dyn.GetUpsilonMatrix(T)

    # Get the <uZ_eff>_abj; a = cartesian, b = cartesian, j = electric field
    uZ_eff = np.einsum("ia, jib->abj", u_disps * Units.A_TO_BOHR, eff_cgs) / N_config


    # Now we can compute the average of the effective charge derivative
    # It has cartesian, cartesian, electric_field components (3N, 3N, 3)
    # dZ_dR = Y <u Z_eff>
    dZ_dR = np.einsum("aq, qbc->abc", upsilon, uZ_eff)
    #upsilon.dot(uZ_eff) WRONG, dot sum with the second last component

    # Perform the symmetrization
    if symmetrize:
        qe_sym = symmetries.QE_Symmetry(super_dyn.structure)
        qe_sym.SetupFromSPGLIB()
        qe_sym.ApplySymmetryToSecondOrderEffCharge(dZ_dR)
    else:
        # Apply only the hermitianity
        dZ_dR += np.einsum("abc->bac", dZ_dR)
        dZ_dR /= 2

    return dZ_dR

def GetIRSpectrum(dyn, w_array, smearing):
    """
    GET THE IR SPECTRUM
    ===================

    This method get a ready to plot harmonic IR spectrum.
    The effective charge must be included in the dynamical matrix

    Parameters
    ----------
        - dyn : Phonons()
            The dynamical matrix on which to compute the IR.
            It must contain valid effective charges.
        - w_array : ndarray
            The values of the frequencies at which to compute the IR.
            in [Ry]
        - smearing : float
            The value of the smearing for the plot (in Ry).


    Results
    -------
        - ir_spectrum: ndarray( size = (len(w)))
            The ir spectrum at each frequency. 
    """

    # Check if the effective charges are defined
    if dyn.effective_charges is None:
        raise ValueError("Error, effective charges must be initialized to compute the IR")

    g_propagator = dyn.get_phonon_propagator(w_array, smearing, only_gamma = True)

    # Get the masses
    m = np.tile(dyn.structure.get_masses_array(), (3,1)).T.ravel()

    # Divide the propagator by the mass square root
    new_g = np.einsum("abw, a, b-> abw", g_propagator, 1 / np.sqrt(m), 1/np.sqrt(m))

    # We move the electric field polarization on the first index
    eff_charge = np.einsum("abc -> bac", dyn.effective_charges)

    # Lets unite the last two coordinates (atomic index and cartesian coordinate)
    eff_charge = eff_charge.reshape((3, 3 * dyn.structure.N_atoms))

    # Now we can multiply the effective charge times the propagator
    ir_signal = np.einsum("ia, ib, abw->w", eff_charge, eff_charge, -np.imag(new_g))
    return ir_signal


def GetTwoPhononIRFromSecondOrderDypole(original_dyn, dM_dRdR, T, w_array, smearing, use_fortran = True, verbose = False):
    r"""
    GET THE TWO PHONON IR RESPONSE FUNCTION
    =======================================

    This method computes the two phonon IR response function for an harmonic dynamical matrix.
    The two phonon IR is due to quadratic terms in the dipole moment (linear part of the effective charge)

    The IR intensity due to the two phonon structures is

    .. math:: 

        I(\omega) = \frac 14 \sum_{x = 1}^3 \sum_{abcd} \frac{\partial^2 M_x}{\partial R_a \partial R_b} \frac{\partial^2 M_x}{\partial R_c\partial R_d} \frac{-\Im G_{abcd}(\omega)}{\sqrt{m_am_bm_cm_d}}
    
    where the $G_{abcd}(\omega)$ is the two phonon propagator

    .. math::

        G_{abcd}(z) = \sum_{\mu\nu} \frac{e_\mu^a e_\nu^be_\mu^ce_\nu^d}{2\omega_\mu\omega_\nu}\left[\frac{(\omega_\mu + \omega_\nu)(n_\mu + n_\nu + 1)}{(\omega_\mu + \omega_\nu)^2 - z^2} - \frac{(\omega_\mu - \omega_\nu)(n_\mu - n_\nu)}{(\omega_\mu - \omega_\nu)^2 - z^2}\right]


    Parameters
    ----------
        - original_dyn : CC.Phonons.Phonons()
            The dynamical matrix of the equilibrium system.
        - structures : list 
            A list of CC.Structure.Structure() of the displaced structure with respect to original_dyn on which the effective charges are computed. 
            Alternatively, you can pass a list of strings that must point to the .scf files of the structures.
        - effective_charges : list
            A list of the effective charge tensor. Alternatively you may provide a list of strings that must point to the output of the ph.x package from Quantum ESPRESSO,
            where the effective charges are printed.
        - T : float
            The temperature (in K)
        - w_array : ndarray
            A real valued array of the frequencies for the IR signal. The energy must be in [Ry]
        - smearing : float
            The value of the 0^+ in the phonon propagator. This allows the response to have a non vanishing imaginary part
        - use_fortran : bool
            If true the fortran library is used to perform the calculation
            This is very convenient, as it speed up the calculation
            and avoids storing massive amount of memory. 
            Use it to false only for testing purpouses or for very small system
            with many freqiencies (in that case the python implemetation could be faster)
        - verbose : bool
            If true prints the timing on output.
    Results
    -------
        - ir_2_ph : ndarray
            The 2-phonons IR intensity at each w_array frequency. 

    """

    # Generate the dynamical matrix in the supercell
    super_dyn = original_dyn.GenerateSupercellDyn(original_dyn.GetSupercell())

    # Get frequencies and polarization vectors
    w_freqs, pol_vec = original_dyn.DiagonalizeSupercell()

    # Get the translations
    trans_mask = Methods.get_translations(pol_vec, super_dyn.structure.get_masses_array())

    # Remove the translations from w and the polarization vectors
    w_freqs = w_freqs[~trans_mask]
    pol_vec = pol_vec[:, ~trans_mask]

    # Get an array of the masses for each 3nat_sc coordinate
    m = np.tile(super_dyn.structure.get_masses_array(), (3,1)).T.ravel()

    # get e_mu/sqrt(m)
    enew = np.einsum("ab, a -> ba", pol_vec, 1/np.sqrt(m))

    # Convert the dipole moment in polarization basis
    t1 = time.time()
    dM_dRdp = np.einsum("abj, mb->amj", dM_dRdR, enew)
    dM_dpdp = np.einsum("abj, ma->mbj", dM_dRdp, enew)

    t2 = time.time()
    if verbose:
        print("Time to transform the dipole in the polarization basis: {}s".format(t2 -t1))

    # Now dM_dpdp has (3nat_sc - 3, 3nat_sc - 3, 3) coordinates, 
    # The frist two are polarization basis

    # We can get the 2 phonon propagator
    if use_fortran == False:
        # Use the python. Good for testing but require a massive amount of memory
        # For large cells
        G_munu = super_dyn.get_two_phonon_propagator(w_array, T, smearing)

        # Get the IR response
        IR = np.einsum("abi, abi, abw->w", dM_dpdp, dM_dpdp, G_munu) / 4
    else:
        # Use the fortran accelerated library.
        # They are fast and not memory intensive
        IR = np.zeros(len(w_array), dtype = np.complex128)
        for i in range(3):
            gf_pol = symph.contract_two_ph_propagator(w_array, w_freqs, T, smearing, dM_dpdp[:,:,i])
            IR += gf_pol / 4

    t3 = time.time()
    if verbose:
        print("Time to conpute the IR response: {} s".format(t3-t2))

    # Get the IR intensity by tracing on the electric field 
    # And selecting the imaginary part of the green function.
    return -np.imag(IR)


def GetTwoPhononIR(original_dyn, structures, effective_charges, T, w_array, smearing, *args, **kwargs):
    r"""
    GET THE TWO PHONON IR RESPONSE FUNCTION
    =======================================

    This method computes the two phonon IR response function for an harmonic dynamical matrix.
    The two phonon IR is due to quadratic terms in the dipole moment (linear part of the effective charge)

    The IR intensity due to the two phonon structures is

    .. math:: 

        I(\omega) = \frac 14 \sum_{x = 1}^3 \sum_{abcd} \frac{\partial^2 M_x}{\partial R_a \partial R_b} \frac{\partial^2 M_x}{\partial R_c\partial R_d} \frac{-\Im G_{abcd}(\omega)}{\sqrt{m_am_bm_cm_d}}
    
    where the $G_{abcd}(\omega)$ is the two phonon propagator

    .. math::

        G_{abcd}(z) = \sum_{\mu\nu} \frac{e_\mu^a e_\nu^be_\mu^ce_\nu^d}{2\omega_\mu\omega_\nu}\left[\frac{(\omega_\mu + \omega_\nu)(n_\mu + n_\nu + 1)}{(\omega_\mu + \omega_\nu)^2 - z^2} - \frac{(\omega_\mu - \omega_\nu)(n_\mu - n_\nu)}{(\omega_\mu - \omega_\nu)^2 - z^2}\right]


    NOTE: This function calls the GetTwoPhononIRFromSecondOrderDypole

    So refer to that one for documentation in extra parameters

    Parameters
    ----------
        - original_dyn : CC.Phonons.Phonons()
            The dynamical matrix of the equilibrium system.
        - structures : list 
            A list of CC.Structure.Structure() of the displaced structure with respect to original_dyn on which the effective charges are computed. 
            Alternatively, you can pass a list of strings that must point to the .scf files of the structures.
        - effective_charges : list
            A list of the effective charge tensor. Alternatively you may provide a list of strings that must point to the output of the ph.x package from Quantum ESPRESSO,
            where the effective charges are printed.
        - T : float
            The temperature (in K)
        - w_array : ndarray
            A real valued array of the frequencies for the IR signal. The energy must be in [Ry]
        - smearing : float
            The value of the 0^+ in the phonon propagator. This allows the response to have a non vanishing imaginary part
    Results
    -------
        - ir_2_ph : ndarray
            The 2-phonons IR intensity at each w_array frequency. 

    """

    # Get the second order dipole moment (3nat_sc, 3nat_sc, 3)
    dM_dRdR = GetSecondOrderDipoleMoment(original_dyn, structures, effective_charges, T)

    return GetTwoPhononIRFromSecondOrderDypole(original_dyn, dM_dRdR, T, w_array, smearing, *args, **kwargs)

