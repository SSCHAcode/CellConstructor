# -*- coding: utf-8 -*-
"""
This module contains the methods that requre to call the classes defined
into this module.
"""
import os
import numpy as np
from Structure import Structure
from Phonons import Phonons

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