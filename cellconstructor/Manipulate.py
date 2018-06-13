# -*- coding: utf-8 -*-
"""
This module contains the methods that requre to call the classes defined
into this module.
"""
import os
import numpy as np
from Structure import Structure
from Phonons import Phonons
import Methods

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
        ftmp = file(fname, "r")
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
            the temperature given as the input array.
            
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
                print "WARNING: NEGATIVE FREQUENCIES FOUND"
                print "        ",   np.sum( (freqs < 0).astype(int) )
                
                
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
            many possible degenerate systems. This is applied after the mode_sign, if any
        mode_sign : list, optional
            If different from none, is a list of length equal 3*N_atoms, containing
            only -1 and 1. The polarization vector of the first dynmat will be changed
            of sign if requested. This is applied before the mode_exchange
        
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
    
    # Change the sign of the polarization vector if required
    pol0[:, mode_sign == -1] *= -1
    
    # Exchange modes according to mode exchange
    w0 = w0[mode_exchange]
    pol0 = pol0[:, mode_exchange]
    
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
                    v1 += dyn1.struct.unit_cell[k,:] * .5
                
                # Put the displacement now into the unit cell to get the correct one
                disp[i,:] = Methods.put_into_cell(dyn1.structure.unit_cell, v1)
                
        
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
            its data. This transformation is applied after mode_sign
        mode_sign : ndarray(int)
            An array made only by int equal to 1 and -1. It changes the sign
            of the corresponding polarization vector of the first dynamical matrix.
            This transformation is applied before mode_exchange
    
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
    
    # perform the transformation if required
    pol1[:, mode_sign == -1] *= -1
    pol1 = pol1[:, mode_exchange]
    
    results = np.zeros( nmodes )
    
    for i in range(nmodes):
        results[i] = pol1[:, i].dot(pol2[:, i])
    
    return results