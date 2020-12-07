# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:45:50 2018

@author: pione
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from numpy import *
import numpy as np
import sys, os

import warnings

#from . import Structure
 
BOHR_TO_ANGSTROM = 0.529177249
__EPSILON__ = 1e-6


__all__ = ["covariant_coordinates", "from_dynmat_to_spectrum", 
           "put_into_cell", "get_minimal_orthorombic_cell", 
           "write_dynmat_in_qe_format", "get_gr", "cell2abc_alphabetagamma", 
           "DistanceBetweenStructures"]


def covariant_coordinates(basis, vectors):
    """
    Covariant Coordinates
    =====================
    
    This method returns the covariant coordinates of the given vector in the chosen basis.
    Covariant coordinates are the coordinates expressed as:
        .. math::
            
            \\vec v = \\sum_i \\alpha_i \\vec e_i
            
            
    where :math:`\\vec e_i` are the basis vectors. Note: the :math:`\\alpha_i` are not the
    projection of the vector :math:`\\vec v` on :math:`\\vec e_i` if the basis is not orthogonal.
    
    
    Parameters
    ----------
        - basis : ndarray(size = (N,N))
            The basis. each :math:`\\vec e_i` is a row.
        - vector : ndarray(size = (N_vectors, N))
            The vectors expressed in cartesian coordinates.
            It coould be just one ndarray(size=N)
            
    Results
    -------
        - cov_vector : Nx float
            The :math:`\\alpha_i` values.
            
    """

    M, N = np.shape(basis)
    
    metric_tensor = np.zeros((N,N))

    metric_tensor = basis.dot(basis.T)
    # for i in range(0, N):
    #     for j in range(i, N):
    #         metric_tensor[i, j] = metric_tensor[j,i] = basis[i,:].dot(basis[j, :])

    imt = np.linalg.inv(metric_tensor)
    
    contra_vect = vectors.dot(basis.T)
    return contra_vect.dot(imt)

def cryst_to_cart(unit_cell, cryst_vectors):
    """
    Convert a vector from crystalline to cartesian. 
    Many vectors counld be pased toghether, in that case the last axis must be the one with the vector.

    Parameters
    ----------
        unit_cell : ndarray((3,3))
            The unit cell vectors. 
            The i-th cell vector is unit_cell[i, :]
        cryst_vectors : ndarray((N_vectors, 3)) or ndarray(3)
            The vector(s) in crystalline coordinates that you want to
            transform in cartesian coordinates

    Results
    -------
        cart_vectors : ndarray((N_vectors, 3)) or ndarray(3)
            The vector(s) in cartesian coordinates
    """

    return cryst_vectors.dot(unit_cell)

def cart_to_cryst(unit_cell, cart_vectors):
    """
    Convert a vector from cartesian to crystalline. 
    Many vectors counld be pased toghether, in that case the last axis must be the one with the vector.

    Parameters
    ----------
        unit_cell : ndarray((3,3))
            The unit cell vectors. 
            The i-th cell vector is unit_cell[i, :]
        cart_vectors : ndarray((N_vectors, 3)) or ndarray(3)
            The vector(s) in cartesian coordinates that you want to
            transform in crystalline coordinates

    Results
    -------
        cryst_vectors : ndarray((N_vectors, 3)) or ndarray(3)
            The vector(s) in crystalline coordinates
    """
    return covariant_coordinates(unit_cell, cart_vectors)
    
def get_equivalent_vectors(unit_cell, vectors, target, index = None):
    """
    This function returns an array mask of the vectors that are
    equivalent to the target vector.

    Parameters
    ----------
        - unit_cell : ndarray (size = (3,3))
            unit_cell[i, :] is the i-th lattice vector
        - vectors : ndarray(size = (n_vects, 3))
            The vectors to be compared to the target
        - target : ndarray(size = 3)
            The target vector
    
    Returns
    -------
        - eq_mask : ndarray(size = n_vects, dtype = bool)
            A mask that is True if the vectors[i, :] is equivalent
            to target.
    """

    # Get the inverse metric tensor
    M, N = np.shape(unit_cell)
    
    metric_tensor = np.zeros((N,N))
    for i in range(0, N):
        for j in range(i, N):
            metric_tensor[i, j] = metric_tensor[j,i] = unit_cell[i,:].dot(unit_cell[j, :])

    imt = np.linalg.inv(metric_tensor)

    # Transform matrix
    # This matrix transforms from cartesian to crystal coordinates
    transform = imt.dot(unit_cell)

    # Get the crystal coordinates for each vector
    crystal_vectors = vectors.dot(transform.T)

    # Get the crystal coordinates for the target
    crystal_target = transform.dot(target)


    # For each crystal vector, subtract the target
    crystal_vectors -= crystal_target

    # Get the mask of those vectors whose components are integers
    mask_on_coords = np.abs((crystal_vectors - np.floor(crystal_vectors + .5))) < 1e-5

    # Here we count, for each vector, how many coordinates are not integers in crystal units
    # Then we select only those whose count is 0 (all crystal coordinats are integers)
    mask_equal = np.sum(mask_on_coords.astype(int), axis = 1) == 3

    return mask_equal


def get_min_dist_into_cell(unit_cell, v1, v2):
    """
    This function obtain the minimum distance between two vector, considering the given unit cell
    
    
    Parameters
    ----------
        unit_cell : ndarray 3x3
            The unit cell
        v1 : ndarray 3
            Vector 1
        v2 : ndarray 3
            Vector 2
            
    Results
    -------
        float
            The minimum distance between the two fector inside the given unit cell
    """
    
    
    # Get the covariant components
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = unit_cell[i,:].dot(unit_cell[j, :])

    imt = np.linalg.inv(metric_tensor)
    
    # Get contravariant components
    contra_vect = np.zeros(3)
    for i in range(3):
        contra_vect[i] = v1.dot(unit_cell[i, :]) 

    # Invert the metric tensor and obtain the covariant coordinates
    covect1 = imt.dot(contra_vect)
    
    contra_vect = np.zeros(3)
    for i in range(3):
        contra_vect[i] = v2.dot(unit_cell[i, :]) 

    # Invert the metric tensor and obtain the covariant coordinates
    covect2 = imt.dot(contra_vect)

    covect_distance = covect1 - covect2

    # Bring the distance as close as possible to zero
    covect_distance -= (covect_distance + np.sign(covect_distance)*.5).astype(int)

    # Compute the distance using the metric tensor
    return np.sqrt(covect_distance.dot(metric_tensor.dot(covect_distance)))

def identify_vector(unit_cell, vector_list, target, epsil = 1e-8):
    """
    Identify whichone in vector_list is equivalent to the target (given the supercell)    
    If no vector is identified, then raise a warning and return None.

    This function is much more efficient than calling get_min_dist_into_cell in a loop.
    
    Parameters
    ----------
        unit_cell : ndarray 3x3
            The unit cell
        vector_list : ndarray (N_vectors, 3)
            The array of vector you want to compare
        target : ndarray 3
            The target vector
            
    Results
    -------
        int
            the index of the vector_list that is equivalent to target (withint the cell)
    """
    
    # Get the covariant components
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = unit_cell[i,:].dot(unit_cell[j, :])

    imt = np.linalg.inv(metric_tensor)

    N_vectors = vector_list.shape[0]
    
    # Get contravariant components
    crystal_list = vector_list.dot(unit_cell.T)
    crystal_list = crystal_list.dot(imt)

    crystal_target = target.dot(unit_cell.T)
    crystal_target = imt.dot(crystal_target)

    crystal_distance = crystal_list - np.tile(crystal_target, (N_vectors, 1))

    # Bring the distance as close as possible to zero
    crystal_distance -= (crystal_distance + np.sign(crystal_distance)*.5).astype(int)

    # Compute the distances using the metric tensor
    distances = np.einsum("ai, ai -> a", crystal_distance.dot(metric_tensor), crystal_distance)

    min_index = np.argmin(distances)
    dist = distances[min_index]

    if dist > epsil:
        warnings.warn("Warning in identify_vector, no equivalent atoms found within a threshold of {}".format(epsil))
        return None

    return min_index



def get_reciprocal_vectors(unit_cell):
    """
    GET THE RECIPROCAL LATTICE VECTORS
    ==================================
    
    Gives back the reciprocal lattice vectors given the
    unit cell.
    
    P.S.
    The output is in rad / alat^-1 
    where alat is the unit of measurement of the unit_cell.
    
    Parameters
    ----------
        unit_cell : ndarray( size = (3,3), dtype = np.float64)
            The unit cell, rows are the vectors.
    
    Results
    -------
        reciprocal_vectors : ndarray(size = (3,3), dtype = np.float64)
            The reciprocal lattice vectors 
    """
    
    reciprocal_vectors = np.zeros( (3,3), dtype = np.float64)
    reciprocal_vectors[:,:] = np.transpose(np.linalg.inv(unit_cell))
    return reciprocal_vectors

    
def from_dynmat_to_spectrum(dynmat, struct):
    """
    This method takes as input the dynamical matrix and the atomic structure of the system and
    returns the spectrum.

    Parameters
    ----------
       - dynmat : float, 3*N_atoms x 3*N_atoms
            Numpy array that contains the real-space dynamical matrix (Hartree).
       - struct : Structure
            The structure of the system. The masses must be initialized.

    Results
    -------
       - Frequencies : float, 3*N_atoms
            Numpy array containing the frequencies in cm-1
    """

    n_atoms = struct.N_atoms
    
    # Construct the matrix to be diagonalized
    new_phi = np.zeros(np.shape(dynmat))
    for i in range(n_atoms):
        M_i = struct.masses[struct.atoms[i]]
        for j in range(n_atoms):
            M_j = struct.masses[struct.atoms[j]]
            new_phi[3*i : 3*(i+1), 3*j : 3*(j+1)] = dynmat[3*i : 3*(i+1), 3*j : 3*(j+1)] / (M_i * M_j)

    
    # Diagonalize the matrix
    eigval, eigvect = np.linalg.eig(new_phi)
    eigval *= 220000. # conversion to cm-1

    return np.sort(eigval)
        

        
def put_into_cell(cell, vector):
    """
    This function take the given vector and gives as output the corresponding
    one inside the specified cell.

    Parameters
    ----------
        - cell : double, 3x3 matrix
              The unit cell, a 3x3 matrix whose rows specifies the cell vectors
        - vector : double, 3 elements ndarray
              The vector to be shifted into the unit cell.

    Results
    -------
        - new_vector : double, 3 elements ndarray
              The corresponding vector into the unit cell
    """

    # Check if the system has unit cell
    if np.linalg.det(cell) == 0:
        ERROR_MSG = """
    Error, the structure has no unit cell (or vectors are linearly dependent).
    """
        raise ValueError(ERROR_MSG)

    # Put the vector inside the unit cell
    # To do this, just obtain the covariant vector coordinates.

    # Get the metric tensor
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = cell[i,:].dot(cell[j, :])

    # Get contravariant components
    contra_vect = np.zeros(3)
    for i in range(3):
        contra_vect[i] = vector.dot(cell[i, :]) 

    # Invert the metric tensor and obta
    covect = np.linalg.inv(metric_tensor).dot(contra_vect)

    # print ""
    # print "Translating into the unit cell:"
    # print "MT:"
    # print metric_tensor
    # print "IMT:"
    # print linalg.inv(metric_tensor)
    # print "vector:", vector
    # print "contra variant:", contra_vect
    # print "covariant:", covect
    
    for i in range(3):
        covect[i] = covect[i] - int(covect[i])
        if covect[i] < 0:
            covect[i] += 1

    
    
    # Go back
    final_vect = np.zeros(3)
    for i in range(3):
        final_vect += covect[i] * cell[i,:]
        
    # print "covariant new:", covect
    # print "final:", final_vect
    # print ""
    
    return final_vect
    
    
def get_minimal_orthorombic_cell(euclidean_cell, ita=36):
    """
    This function, given an euclidean cell with 90 90 90 angles, returns the minimal
    cell. The minimal cell will not have 90 90 90 angles. 
    
    Parameters
    ----------
        - euclidean_cell : matrix 3x3, double precision
              The rows of this matrix are the unit cell vectors in euclidean cell.
        - ita : integer
              The group class in ITA standard (36 = Cmc21)
    
    Results
    -------
        - minimal_cell : matrix 3x3, double precision
              The rows of this matrix are the new minimal unit cell vectors.
    """


    # Take the last vector and project into the last-one.

    minimal_cell = euclidean_cell.copy()

    if (ita == 36):
        last_vector = .5 * euclidean_cell[1,:] + .5 * euclidean_cell[0,:]
        minimal_cell[1,:] = last_vector
    else:
        raise ValueError("Error on input, ITA = %d not yet implemented." % ita)
    
    return minimal_cell


# -------
# Compute the g(r)
def get_gr(structures, type1, type2, r_max, dr):
    """
Radial distribution function
============================

Computes the radial distribution function for the system. The
:math:`g_{AB}(r)` is defined as

.. math::


   g_{AB}(r) = \\frac{\\rho_{AB}^{(2)}(r)}{\\rho_A(r) \\rho_B(r)}


where :math:`A` and :math:`B` are two different types

Parameters
----------
    - structures : list type(Structure)
        A list of atomic structures on which compute the :math:`g_{AB}(r)`
    - type1 : character
        The character specifying the :math:`A` atomic type.
    - type2 : character
        The character specifying the :math:`B` atomic type.
    - r_max : float
        The maximum cutoff value of :math:`r`
    - dr : float
        The bin value of the distribution.

Results
-------
    - g_r : ndarray.shape() = (r/dr + 1, 2)
         The :math:`g(r)` distribution, in the first column the :math:`r` value
         in the second column the corresponding value of g(r)
    """

    # Get the r axis
    N_bin = int( r_max / float(dr)) + 1
    r_min = linspace(0, r_max, N_bin) 

    real_dr = mean(diff(r_min))
    #print("REAL DR:", real_dr)
    real_r = r_min + real_dr * .5

    # Define the counting array
    N_r = zeros(N_bin)

    # Count how many atoms are in each shell
    for i, struct in enumerate(structures):
        # Cycle for all atoms
        for first in range(0, struct.N_atoms- 1):
            other_type = ""
            if struct.atoms[first] == type1:
                other_type = type2
            elif struct.atoms[first] == type2:
                other_type = type1
            else:
                continue

            for second in range(first + 1, struct.N_atoms):
                if struct.atoms[second] == other_type:
                    # Get the distance vector between the two atoms
                    r_vec = struct.coords[first, :] - struct.coords[second,:]
                    #print("nat: {}, indexes: {}, {}".format(struct.N_atoms, first, second))
                    #print("r_vec:", r_vec)
                    if struct.has_unit_cell:
                        r_vec = get_closest_vector(struct.unit_cell, r_vec) 
                    #r = struct.get_min_dist(first, second)
                    r = np.sqrt(r_vec.dot(r_vec)) # The modulus
                    index_pos = int( r / real_dr)
                    if index_pos < N_bin:
                        N_r[index_pos] += 1

                

            
    # Now get the g(r) from N_r
    N_tot = sum(N_r)
    V = 4 * np.pi * r_max**3 / 3.
    rho = N_tot / V
    g_r = N_r / (4 * np.pi * real_r * real_dr * rho)

    # Get the final data and return
    data = np.zeros((N_bin, 2))
    data[:, 0] = real_r
    data[:, 1] = g_r
    return data
    
    
    
def cell2abc_alphabetagamma(unit_cell):
    """
This methods return a list of 6 elements. The first three are the three lengths a,b,c of the cell, while the other three
are the angles alpha (between b and c), beta (between a and c) and gamma(between a and b).

Parameters
----------
    - unit_cell : 3x3 ndarray (a float dtype)
         The unit cell in which the lattice vectors are the rows.

Results
-------
    - cell : 6 length ndarray (size = 6, dtype = type(unit_cell))
         The array containing the a,b,c length followed by alpha,beta and gamma (in degrees)
    """

    cell = np.zeros(6, dtype = unit_cell.dtype)

    # Get a,b,c
    for i in range(3):
        cell[i] = np.sqrt(unit_cell[i, :].dot(unit_cell[i, :]))

    # Get alpha beta gamma
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        cosangle = unit_cell[j,:].dot(unit_cell[k, :]) / (cell[j] * cell[k])
        
        cell[i + 3] = np.arccos(cosangle) * 180 / np.pi 

    return cell



def DistanceBetweenStructures(strc1, strc2, ApplyTrans=True, ApplyRot=False, Ordered = True):
    """
This method computes the distance between two structures.
It is usefull to check the similarity between two structure.

Note:
Ordered = False is not yet implemented

Parameters
----------
   - strc1 : type(Structure)
      The first structure. It commutes with the strc2.
   - strc2 : type(Structure)
      The second structure.
   - ApplyTrans: bool, default = False
      If true both the structures are shifted in a common origin (The first atom).
      This works only if the atoms are ordered to match properly.
   - ApplyRot : bool, default = False
      If true the structure are rotated to reduce the rotational freedom.
   - Ordered: bool, default = True
      If true the order in which the atoms appears is supposed to match in the two structures.


Results
-------
    - Similarities: float
        Similarity between the two provided structures
    """


    if not Ordered:
        raise ValueError("Error, Ordered = False not yet implemented. Sorry.")

    if strc1.N_atoms != strc2.N_atoms:
        print( "Strc1 has ", strc1.N_atoms, " atoms")
        print( "Strc2 has ", strc2.N_atoms, " atoms")
        raise ValueError("Error, the number of atoms are not the same in the given structures.")

    nat = strc1.N_atoms
    coord1 = np.zeros(nat*3)
    coord2 = np.zeros(nat*3)
    
    if ApplyTrans:
        # Shift both the strcture so that the first atom is in the origin.
        if Ordered:
            for i in range(nat):
                coord1[3*i : 3*i + 3] = strc1.coords[i,:] - strc1.coords[0,:]
                coord2[3*i : 3*i + 3] = strc2.coords[i,:] - strc2.coords[0,:]
    else:
       for i in range(nat):
           coord1[3*i : 3*i + 3] = strc1.coords[i,:]
           coord2[3*i : 3*i + 3] = strc2.coords[i,:]


    # Compute the distance between the two coordinates
    return np.sqrt(sum((coord1 - coord2)**2))


def get_unit_cell_from_ibrav(ibrav, celldm):
    """
    OBTAIN THE UNIT CELL WITH QUANTUM ESPRESSO IBRAV
    ================================================
    
    This subroutine reads the quantum espresso variables ibrav and celldm
    and built the unit cell according to them.
    
    NOTE: not all the version are still supported, they will be 
    added as the developing of the code will go on.
    
    
    Look at quantum espresso pw.x input for a clear explanation
    on how they works.
    Note the unit of QE are bohr, so we expect the celldm[0] to be
    written in bohr. However the output cell will be in angstrom.
    
    Parameters
    ----------
        ibrav : int
            This is the ibrav number identification of the cell type.
            Note if it is a float, it will be rounded to the closest integer.
            For example, 1 means simple cubic, 13 is the base-centered monoclinic ...
        
        celldm : ndarray (float, 6)
            It contains a list of 6 floats that defines the cell axis length and 
            angles. Their precise meaning can be found on quantum-espresso documentation.
            We refer at 6.2.1 version.
    
    Results
    -------
        unit_cell : ndarray (3x3)
            The unit cell in angstrom. the i-th cell vector is unit_cell[i,:]
            
    """
    # Avoid trivial problems if the ibrav is a float
    ibrav = int(ibrav + .5) 
    
    #SUPPORTED_IBRAV = [13]
    
    # Check if the ibrav is in the supported ibrav
    #if not ibrav in SUPPORTED_IBRAV:
    #    raise ValueError("Error, the specified ibrav %d is not supported." % ibrav)
        
    
    # Check if celldm is of the correct length
    if len(celldm) != 6:
        raise ValueError("Error, celldm shoud be an ndarray of size 6")
    
    # Get the cell
    unit_cell = np.zeros((3,3))
    if ibrav == 1:
        # Simple cubic
        a = celldm[0] * BOHR_TO_ANGSTROM
        unit_cell[0,:] = np.array([1, 0, 0]) * a
        unit_cell[1,:] = np.array([0, 1, 0]) * a 
        unit_cell[2,:] = np.array([0, 0, 1]) * a
    elif ibrav == 2:
        # Cubic fcc
        a = celldm[0] * BOHR_TO_ANGSTROM
        unit_cell[0,:] = np.array([-1, 0, 1]) * a / 2
        unit_cell[1,:] = np.array([0, 1, 1]) * a /2
        unit_cell[2,:] = np.array([-1, 1, 0]) * a/2
    elif ibrav == 3:
        # Cubic bcc
        a = celldm[0] * BOHR_TO_ANGSTROM
        unit_cell[0,:] = np.array([1, 1, 1]) * a / 2
        unit_cell[1,:] = np.array([-1, 1, 1]) * a /2
        unit_cell[2,:] = np.array([-1, -1, 1]) * a/2
    elif ibrav == -3:
        # Cubic bcc other kind
        a = celldm[0] * BOHR_TO_ANGSTROM
        unit_cell[0,:] = np.array([-1, 1, 1]) * a / 2
        unit_cell[1,:] = np.array([1, -1, 1]) * a /2
        unit_cell[2,:] = np.array([1, 1, -1]) * a/2
    elif ibrav == 4:
        # Hexagonal
        a = celldm[0] * BOHR_TO_ANGSTROM
        c = celldm[2] * a
        
        unit_cell[0, :] = np.array([1, 0, 0]) * a
        unit_cell[1, :] = np.array([-0.5, np.sqrt(3)/2, 0]) * a
        unit_cell[2, :] = np.array([0, 0, 1]) * c
    elif ibrav == 5:
        a = celldm[0] * BOHR_TO_ANGSTROM
        c = celldm[3]

        tx = np.sqrt( (1 - c) / 2.)
        ty = np.sqrt( (1-c) / 6.)
        tz = np.sqrt( (1+2*c)/3.)

        unit_cell[0, :] = np.array([tx, -ty, tz]) * a
        unit_cell[1, :] = np.array([0, 2*ty, tz]) * a
        unit_cell[2, :] = np.array([-tx, -ty, tz]) * a

    elif ibrav == 6:
        a = celldm[0] * BOHR_TO_ANGSTROM
        c = celldm[2] * a

        unit_cell[0, :] = a * np.array([1, 0, 0])
        unit_cell[1, :] = a * np.array([0, 1, 0])
        unit_cell[2, :] = c * np.array([0, 0, 1])

    elif ibrav == 7:
        # Tetragonal I
        a = celldm[0] * BOHR_TO_ANGSTROM
        c = celldm[2] * a
        
        unit_cell[0, :] = np.array([a*0.5, -a*0.5, c*0.5])
        unit_cell[1, :] = np.array([a*0.5, a*0.5, c*0.5])
        unit_cell[2, :] = np.array([-a*0.5, -a*0.5, c*0.5])

    elif ibrav == 8:
        # Orthorombic
        a = celldm[0] * BOHR_TO_ANGSTROM
        b = celldm[1] * a
        c = celldm[2] * a

        unit_cell[0, :] = a * np.array([1,0,0])
        unit_cell[1, :] = b * np.array([0,1,0])
        unit_cell[2, :] = c * np.array([0,0,1])

    elif ibrav == 9:
        # Orthorombinc base centered
        a = celldm[0] * BOHR_TO_ANGSTROM
        b = celldm[1] * a
        c = celldm[2] * a
        
        unit_cell[0, :] = np.array([a/2, b/2, 0])
        unit_cell[1, :] = np.array([-a/2, b/2, 0])
        unit_cell[2, :] = np.array([0, 0, c])
    elif ibrav == -9:
        # Orthorombinc base centered (the same but change the first two vectors)
        a = celldm[0] * BOHR_TO_ANGSTROM
        b = celldm[1] * a
        c = celldm[2] * a
        
        unit_cell[0, :] = np.array([-a/2, b/2, 0])
        unit_cell[1, :] = np.array([a/2, b/2, 0])
        unit_cell[2, :] = np.array([0, 0, c])
    elif ibrav == 91:
        #     Orthorhombic one-face base-centered A-type
        #                                     celldm(2)=b/a
        #                                     celldm(3)=c/a
        #       v1 = (a, 0, 0),  v2 = (0,b/2,-c/2),  v3 = (0,b/2,c/2)
        a = celldm[0] * BOHR_TO_ANGSTROM
        b = celldm[1] * a
        c = celldm[2] * a
        unit_cell[0,:] = np.array([a,0,0])
        unit_cell[1,:] = np.array([0, b/2, -c/2])
        unit_cell[2,:] = np.array([0, b/2, c/2])
    elif ibrav == 13:
        # Monoclinic base-centered
        
        # Create cell
        a = celldm[0] * BOHR_TO_ANGSTROM
        b = a * celldm[1]
        c = a * celldm[2]
        cos_ab = celldm[3]
        sin_ab = np.sqrt(1 - cos_ab**2)
        
        unit_cell[0,:] = np.array( [a/2., 0, -c/2.])
        unit_cell[1,:] = np.array( [b * cos_ab, b*sin_ab, 0])
        unit_cell[2,:] = np.array( [a/2., 0, c/2.])
    else:
        raise ValueError("Error, the specified ibrav %d is not supported." % ibrav)

    return unit_cell

def is_inside(index, indices):
    """
    Returns whether idex is inside a couple of indices.
    Usefull to check if something is inside or not something like
    parenthesys or quotes
    """
    if len(indices) == 0:
        return False
    
    a = np.array(indices, dtype = int)
    
    new_a = (index > a).astype(int)
    result = np.sum(new_a)
    
    if result % 2 == 0:
        return False
    return True

def read_namelist(line_list):
    """
    READ NAMELIST
    =============
    
    
    This function will read the quantum espresso namelist format from a list of lines.
    The info are returned in a dictionary:
        
        &control
            type_cal = "wrong"
            ecutrho = 140
        &end
    
    will be converted in a python dictionary
        dict = {"control" : {"type_cal" : "wrong", "ecutrho" : 140}}
    
    Then the dictionary is returned. Comments must start as in fortran with the '!'
    
    NOTE: Fotran is not case sensitive, therefore all the keys are converted in lower case
    
    
    
    Parameters
    ----------
        line_list : list or string (path)
            A list of lines read in a file. They should be the row output of f.readlines() function
            where f is a file obtained as f = open("something", "r"). You can also directly pass
            a string to path
    
    Returns
    -------
        dict :
            The dictionary of the namelist
    """
    
    if isinstance(line_list, str):
        if not os.path.exists(line_list):
            raise IOError("Error, file %s not found." % line_list)
        
        # Read the file
        fread = open(line_list, "r")
        line_list = fread.readlines()
        fread.close()
    
    
    inside_namespace = False
    current_namespace = ""
    namespace = {}
    total_dict = {}
    
    # Start reading
    for line in line_list:
        # Avoid case sensitivity turning everithing in lower case
        #line = line.lower()
        # Get thee string content and avoid parsing that
        quotes_indices = []
        last_found = 0
        while True:
            last_found = line.find('"', last_found + 1)
            if last_found != -1:
                quotes_indices.append(last_found)
            else:
                break
        
        
            
        
        # Delete the line after the comment
        # If it is not inside double quotes
        if not is_inside(line.find("!"), quotes_indices):
            # Delete the comment
            line = line[:line.find("!")]
                
        # Clear the line of tailoring white spaces
        line = line.strip()
        
        # Skip if the line is white
        if len(line) == 0:
            continue
        
        # Check if the line begins with an "&" sign
        if line[0] == "&":
            # Check if we are closing an existing namespace
            if line[1:].lower() == "end":
                if not inside_namespace:
                    raise IOError("Error, trying to close a namespace without having open it.")
                
                total_dict[current_namespace] = namespace.copy()
                current_namespace = ""
                inside_namespace = False
                namespace.clear()
                
                continue
            
            # New namelist ---
            
            # Check if we already are inside a namespace
            if inside_namespace:
                raise IOError("Error, the namespace %s has not been closed." % current_namespace)
            
            current_namespace = line[1:].lower()
            inside_namespace = True
            
            # Check if the namespace has a valid name
            if len(current_namespace) == 0:
                raise IOError("Error, non valid name for a namespace")
        else:
            # Check if the old namespace closure is used
            if line[0] == "/":
                if not inside_namespace:
                    raise IOError("Error, trying to close a namespace without having open it.")
                
                total_dict[current_namespace] = namespace.copy()
                current_namespace = ""
                inside_namespace = False
                namespace.clear()
                continue
            
            # First of all split for quotes
            value = None
            new_list_trial = line.split('"')
            if len(new_list_trial) == 3:
                value = '"' + new_list_trial[1] + '"'
            else:                
                new_list_trial = line.split("'")
                if len(new_list_trial) == 3:
                    value = '"' + new_list_trial[1] + '"'
            
            # Get the name of the variable
            new_list = line.split("=")
            
            if len(new_list) != 2 and value is None:
                raise IOError("Error, I do not understand the line %s" % line)
            elif len(new_list) < 2:
                raise IOError("Error, I do not understand the line %s" % line)
                
            variable = new_list[0].strip().lower()
            if value is None:
                value = new_list[1].strip()
            
            # Remove ending comma and otehr tailoring space
            if value[-1] == ",":
                value = value[:-1].strip()
            
            
            # Convert fortran bool
            if value.lower() == ".true.":
                value = True
            elif value.lower() == ".false.":
                value = False
            elif '"' == value[0]: # Get a string content
                # If it is a string cancel the " or ' or ,
                value = value.replace("\"", "")
            elif "'" == value[0]:
                value = value.replace("'", "")
            elif value.count(" ") >= 1:
                value = [float(item) for item in value.split()]
            else:
                # Check if it is a number
                try:
                    value = float(value.lower().replace("d", "e"))
                except:
                    pass
            if inside_namespace:
                namespace[variable] = value
            else:
                total_dict[variable] = value

    # The file has been analyzed
    if inside_namespace:
        raise IOError("Error, file endend before %s was closed" % current_namespace)
    
    return total_dict
            
        
        
        
        
def write_namelist(total_dict):
    """
    WRITE ESPRESSO NAMELIST
    =======================
    
    Given a particular dictionary this subroutine will transform it into an espresso dictionary
    
    Parameters
    ----------
        total_dict : dict
            A dictionary of the namespaces in the namelist
    
    Results
    -------
        list
            A list of lines that can be written into a file
    """
        
    
    lines = []
    for key in total_dict.keys():
        if type(total_dict[key]) == dict:
            # Namelist
            lines.append("&%s\n" % key)
            for key2 in total_dict[key].keys():
                value = total_dict[key][key2]
                valuestr = ""
                if type(value) == list:
                    valuestr = " ".join(value)
                elif type(value) == "ciao":
                    valuestr = "\"%s\"" % value
                else:
                    valuestr = str(value)
            
                line = "\t%s = %s\n" % (key2, valuestr)
                lines.append(line)
                
            lines.append("&end\n")
        else:
            value = total_dict[key]
            valuestr = ""
            if type(value) == list:
                valuestr = " ".join(value)
            else:
                valuestr = str(value)
        
            line = "\t%s = %s\n" % (key, valuestr)
            lines.append(line)
                
            
    return lines
                
        
    
def get_translations(pols, masses):
    """
    GET TRANSLATIONS
    ================

    This subroutine analyzes the polarization vectors of a dynamical matrix to recognize the translations.
    It is usefull to carefully remove the translations from the frequencies where w -> 0 gives an error in an equation.

    Parmaeters
    ----------
        pols : ndarray 2 rank
            The polarization vectors as they came out from DyagDinQ(0) method from Phonons.
        masses : ndarray (size nat)
            The mass of each atom. 

    Returns
    -------
        is_translation_mask : ndarray(3 * N_atoms)
             A bool array of True if the i-th polarization vectors correspond to a translation, false otherwise.



    Example
    -------

    In this example starting from the frequencies, the translations are removed (let dyn to be Phonons()):
    
    >>> w, pols = dyn.DyagDinQ(0)
    >>> t_mask = get_translations(pols)
    >>> w_without_trans = w[ ~t_mask ]
    
    The same, of course, can be applied to polarization vectors:

    >>> pols = pols[ :, ~t_mask ]

    The ~ caracter is used to get the bit not operation over the t_mask array (to mark False the translational modes and True all the others) 
    """
    
    # Check if the masses array is good
    if len(masses) * 3 != np.size(pols[:,0]):
        raise ValueError("Error, the size of the two array masses and pols are not compatible.")


    # Get the number of atoms and the number of polarization vectors
    n_atoms = len(pols[:, 0]) // 3
    n_pols = len(pols[0,:])

    # Prepare a mask filled with false
    is_translation = np.zeros( n_pols).astype (bool)

    for i in range(n_pols):
        # Check if the polarization vector is oriented in the same way for each atom
        thr_val = 0
        for j in range(n_atoms):
            thr_val += np.sum( np.abs(pols[3 * j : 3 * j + 3, i]/np.sqrt(masses[j]) - 
                                      pols[:3, i] / np.sqrt(masses[0]))**2)

        thr_val = np.sqrt(thr_val)
            
        if thr_val < __EPSILON__ :
            is_translation[i] = True

    return is_translation


            
def convert_matrix_cart_cryst(matrix, unit_cell, cryst_to_cart = False):
    """
    This methods convert the 3x3 matrix into crystalline coordinates using the metric tensor defined by the unit_cell.

    This method is a specular implementation of Quantum ESPRESSO. 
    This subroutine transforms matrices obtained as open product between vectors.
    If you want to transform a linear operator, then use
    convert_matrix_cart_cryst2

    For example, use this to transform a dynamical matrix, 
    use convert_matrix_cart_cryst2 to transform a symmetry operation.
    
    .. math::
        
        g_{\\alpha\\beta} = \\left< \\vec v_\\alpha | \\vec v_\\beta\\right>
        
        F_{\\alpha\\beta} = g_{\\alpha\\gamma} g_{\\beta\\delta} F^{\\gamma\\delta}
        
        F^{\\alpha\\beta} = g^{\\alpha\\gamma} g^{\\beta\\delta} F_{\\gamma\\delta}
        
        F_{\\alpha\\beta} = \\frac{\\partial E}{\\partial u^\\alpha \\partial u^\\beta}
        
        F^{\\alpha\\beta} = \\frac{\\partial E}{\\partial u_\\alpha \\partial u_\\beta}
    
        
    Parameters
    ----------
        matrix : ndarray 3x3
            The matrix to be converted
        unit_cell : ndarray 3x3
            The cell containing the vectors defining the metric (the change between crystalline and cartesian coordinates)
        cryst_to_cart : bool, optional
            If False (default) the matrix is assumed in cartesian coordinates and converted to crystalline. If True
            otherwise.
            
    Results
    -------
        new_matrix : ndarray(3x3)
            The converted matrix into the desidered format
    """
    
    
    # Get the metric tensor from the unit_cell
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = unit_cell[i,:].dot(unit_cell[j, :])

    # Choose which conversion perform
    comp_matrix = np.einsum("ij, jk", np.linalg.inv(metric_tensor), unit_cell) 
    comp_matrix_inv = np.linalg.inv(comp_matrix)
        
    if cryst_to_cart:
        return comp_matrix.T.dot( np.dot(matrix, comp_matrix))

    return comp_matrix_inv.T.dot( np.dot(matrix, comp_matrix_inv))
    
def convert_matrix_cart_cryst2(matrix, unit_cell, cryst_to_cart = False):
    """
    This methods convert the 3x3 matrix into crystalline coordinates using the metric tensor defined by the unit_cell.

    This perform the exact transform. 
    With this method you get a matrix that performs the transformation directly in the other space. 
    If I have a matrix that transforms vectors in crystalline coordinates, then with this I get the same operator between
    vectors in cartesian space.
    
    This subroutine transforms operators, while the previous one transforms matrices (obtained as open product between vectors)
    
    For example, use convert_matrix_cart_cryst to transform a dynamical matrix, 
    use convert_matrix_cart_cryst2 to transform a symmetry operation.
    
        
    Parameters
    ----------
        matrix : ndarray 3x3
            The matrix to be converted
        unit_cell : ndarray 3x3
            The cell containing the vectors defining the metric (the change between crystalline and cartesian coordinates)
        cryst_to_cart : bool, optional
            If False (default) the matrix is assumed in cartesian coordinates and converted to crystalline. If True
            otherwise.
            
    Results
    -------
        new_matrix : ndarray(3x3)
            The converted matrix into the desidered format
    """
    
    
    # Get the metric tensor from the unit_cell
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = unit_cell[i,:].dot(unit_cell[j, :])

    # Choose which conversion perform
    comp_matrix = np.einsum("ij, jk", np.linalg.inv(metric_tensor), unit_cell) 
    comp_matrix_inv = np.linalg.inv(comp_matrix)
        
    if cryst_to_cart:
        return comp_matrix_inv.dot( np.dot(matrix, comp_matrix))

    return comp_matrix.dot( np.dot(matrix, comp_matrix_inv))
        
        
def convert_3tensor_to_cryst(tensor, unit_cell, cryst_to_cart = False):
    """
    Convert to crystal coordinates
    ==============================

    This subroutine converts the 3 rank tensor to crystal coordinates
    and vice versa.

    Paramters
    ---------
        tensor : ndarray( size = (3,3,3))
            The 3rank tensor to be converted
        unit_cell : ndarray
            The unit cell of the structure
        cryst_to_cart : bool
            If true, reverse convert crystal to cartesian.
    """
    
    # Get the metric tensor from the unit_cell
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = unit_cell[i,:].dot(unit_cell[j, :])

    comp_matrix = np.einsum("ij, jk", np.linalg.inv(metric_tensor), unit_cell) 
    if not cryst_to_cart:
        comp_matrix = np.linalg.inv(comp_matrix)

    return np.einsum("ia, jb, kc, ijk -> abc", comp_matrix, comp_matrix, comp_matrix, tensor)
        
def convert_fc(fc_matrix, unit_cell, cryst_to_cart = False):
    """
    This method converts the force constant matrix from cartesian to crystal and the opposite.
    Check the method convert_matrix_cart_cryst to see more details.
    
    Parameters
    ----------
        fc_matrix : ndarray (3 nat x 3 nat)
            The original force constant matrix
        unit_cell : ndarray (3x3)
            The unit cell of the system
        cryst_to_cart : bool, optional, default False
            If true convert from crystal to cartesian, the opposite otherwise.
            
    Results
    -------
        new_fc_matrix : ndarray (shape(fc_matrix))
            The new force constant matrix after the conversion.
    """
    
    # Check if the fc_matrix is a good candidate
    if np.shape(fc_matrix)[0] != np.shape(fc_matrix)[1]:
        raise ValueError("Error, the force constant matrix must be a square array")
    
    if len(np.shape(fc_matrix)) != 2:
        raise ValueError("Error, the fc_matrix must be a matrix.")
    
    n_indices = np.shape(fc_matrix)[0]
    
    if n_indices %3 != 0:
        raise ValueError("Error, the size of the force constant matrix must be a multiple of 3")
        
    # Get the number of atoms
    nat = n_indices / 3
    
    # Prepare the output matrix
    new_fc_matrix = np.zeros(np.shape(fc_matrix))
    for na in range(nat):
        for nb in range(na, nat):
            # Convert the single matrix
            in_mat = fc_matrix[3 * na : 3*(na+1), 3*nb : 3*(nb+1)] 
            out_mat = convert_matrix_cart_cryst(in_mat, unit_cell, cryst_to_cart)
            new_fc_matrix[3 * na : 3*(na+1), 3*nb : 3*(nb+1)] = out_mat
            
            # Apply hermitianity
            if na != nb:
                new_fc_matrix[3 * nb : 3*(nb+1), 3*na : 3*(na+1)] = np.conjugate(out_mat.transpose())
            
    return new_fc_matrix
        

def get_directed_nn(structure, atom_id, direction):
    """
    This function get the closest neighbour along a direction of the atom.
    It returns the index of the atom in the structure that is the closest to atom_id in the
    specified direction (a vector identifying the direction).

    The algorithm selects a cone around the atom oriented to the direction, and selects
    the first atom in the structure inside the cone (the one with lowest projection on the cone axis)
    
    The method returns -1 if no near neighbour along the specified direction is detected
    """

    nat = structure.N_atoms
    new_coords = structure.coords - structure.coords[atom_id, :]



    # Center the atoms if the unit cell is present.
    if structure.has_unit_cell:
        # Put the atom at the center of the cell
        new_coords[:,:] += 0.5 * np.sum(structure.unit_cell, axis = 0)
        for i in range(nat):
            new_coords[i, :] = put_into_cell(structure.unit_cell, new_coords[i, :])
        
        # Shift back
        new_coords -= new_coords[atom_id, :]



    
    # Pop the atoms not in the cone
    versor_cone = direction / np.sqrt(direction.dot(direction))

    # Project the coordinate on the versor
    p_cone = new_coords.dot(versor_cone)
    #print p_cone


    # Now discard atoms not inside the cone
    good_atoms = []
    good_d = []
    for i in range(nat):
        if p_cone[i] > 0:
            r_vect = new_coords[i, :] - p_cone[i] * versor_cone
            r = np.sqrt(r_vect.dot(r_vect))
            #print "%d) p = %.4e, r = %.4e, r_vect = " % (i, p_cone[i], r), r_vect
            if r < p_cone[i]:
                good_atoms.append(i)
                good_d.append(r**2 + p_cone[i]**2)
    
    #print good_atoms

    if len(good_atoms) == 0:
        return -1

    # Get the minimum value of p_cone among the good atoms
    i_best = np.argmin(good_d)

    return good_atoms[i_best]

def get_closest_vector(unit_cell, v_dist):
    """
    This subroutine computes the periodic replica of v_dist
    that minimizes its modulus.

    Parameters
    ----------
        - unit_cell : ndarray(size = (3,3))
            The unit cell vector, unit_cell[i,:] is the i-th cell vector
        - v_dist : ndarray(size = 3)
            The distance vector that you want to optimize

    Returns
    -------
        - new_v_dist: ndarray(size = 3)
            The replica of v_dist that has the minimum modulus
    """

    # Define the metric tensor
    g = unit_cell.dot(unit_cell.T)
    alphas = covariant_coordinates(unit_cell, v_dist)
    
    # Define the minimum function
    def min_f(n):
        tmp = g.dot(2 * alphas + n)
        return n.dot(tmp)
    
    # Get the starting guess for the vector
    n_start = -np.floor(alphas + .5)
    n_min = n_start.copy()
    tot_min = min_f(n_min)

    # Get the supercell shift that minimizes the vector
    for n_x in [-1,0,1]:
        for n_y in [-1,0,1]:
            for n_z in [-1,0,1]:
                n_v = n_start + np.array([n_x, n_y, n_z])
                new_min = min_f(n_v)
                if new_min < tot_min:
                    tot_min = new_min 
                    n_min = n_v
    
    # Get the new vector
    new_v = v_dist + n_min.dot(unit_cell)
    return new_v

def three_to_one_len(v,v_min,v_len):
    """
    This subroutine converts a triplet index v,
    of length v_len and starting from v_min, 
    into a single progressive index starting from 0
    """    
    res=(v[0]-v_min[0])*v_len[1]*v_len[2]+(v[1]-v_min[1])*v_len[2]+(v[2]-v_min[2])
    return int(res)

def three_to_one(v,v_min,v_max):
    """
    This subroutine converts a triplet index v,
    going from v_min to v_max, into a single progressive
    index starting from 0
    """    
    v_len=np.array(v_max)-np.array(v_min)+np.ones(3,dtype=int)
    res=(v[0]-v_min[0])*v_len[1]*v_len[2]+(v[1]-v_min[1])*v_len[2]+(v[2]-v_min[2])
    return int(res)

def one_to_three_len(J,v_min,v_len):
    """
    This subroutine converts a single progressive
    index J starting from 0, to a triplet of progressive
    indexes starting from v_min, of length v_len
    """
    x              =   J // (v_len[2] * v_len[1])              + v_min[0]
    y              = ( J %  (v_len[2] * v_len[1])) // v_len[2] + v_min[1]
    z              =   J %   v_len[2]                          + v_min[2]   
    return np.array([x,y,z],dtype=int) 

def one_to_three(J,v_min,v_max):
    """
    This subroutine coonverts a single progressive
    index J starting from 0, to a triplet of progressive
    indexes going from v_min to v_max
    """        
    v_len=np.array(v_max)-np.array(v_min)+np.ones(3,dtype=int)    
    x              =   J // (v_len[2] * v_len[1])              + v_min[0]
    y              = ( J %  (v_len[2] * v_len[1])) // v_len[2] + v_min[1]
    z              =   J %   v_len[2]                          + v_min[2]   
    return np.array([x,y,z],dtype=int) 



def is_gamma(unit_cell, q):
    """
    Defines if the q point in cartesian (A^-1) is gamma or not 
    given the unit cell
    
    Parameters
    ----------
        unit_cell : ndarray (size =3,3)
            The unit cell of the structure
        q : ndarray(size=3)
            The q point that you want to check
    
    Results
    -------
        is_gamma : bool
    """
    
    bg = get_reciprocal_vectors(unit_cell)
    new_q = get_closest_vector(bg, q)
        
    return (np.abs(new_q) < 1e-6).all()

    
def save_qe(dyn,q,dynq,freqs, pol_vects,fname):
        """
        SAVE THE DYNMAT
        ===============
        
        This subroutine saves the dynamical matrix in the quantum espresso file format.
        The dynmat is the force constant matrix in Ry units.
        
        .. math::
            
            \\Phi_{ab} = \\sum_\\mu \\omega_\\mu^2 e_\\mu^a e_\\mu^b \\sqrt{M_a M_b}
            
        Where :math:`\\Phi_{ab}` is the force constant matrix between the a-b atoms (also cartesian
        indices), :math:`\\omega_\\mu` is the phonon frequency and :math:`e_\\mu` is the
        polarization vector.
        """
        
        A_TO_BOHR = 1.889725989
        RyToCm=109737.37595
        RyToTHz=3289.84377
        

        # Open the file
        fp = open(fname, "w")
        fp.write("Dynamical matrix file\n")
        
        # Get the different number of types
        types = []
        n_atoms = dyn.structure.N_atoms
        for i in range(n_atoms):
            if not dyn.structure.atoms[i] in types:
                types.append(dyn.structure.atoms[i])
        n_types = len(types)

        # Assign an integer for each atomic species
        itau = {}
        for i in range(n_types):
            itau[types[i]] = i +1
        
        # Write the comment line
        fp.write("File generated with CellConstructor\n")
        fp.write("%d %d %d %22.16f %22.16f %22.16f %22.16f %22.16f %22.16f\n" %
                 (n_types, n_atoms, 0, dyn.alat * A_TO_BOHR, 0, 0, 0, 0, 0) )
        
        # Write the basis vector
        fp.write("Basis vectors\n")
        # Get the unit cell
        for i in range(3):
            fp.write(" ".join("%22.16f" % x for x in dyn.structure.unit_cell[i,:] / dyn.alat) + "\n")
        
        # Set the atom types and masses
        for i in range(n_types):
            fp.write("\t{:d}  '{:<3s}'  {:>24.16f}\n".format(i +1, types[i], dyn.structure.masses[types[i]]))
        
        # Setup the atomic structure
        for i in range(n_atoms):
            # Convert the coordinates in alat
            coords = dyn.structure.coords[i,:] / dyn.alat
            fp.write("%5d %5d %22.16f %22.16f %22.16f\n" %
                     (i +1, itau[dyn.structure.atoms[i]], 
                      coords[0], coords[1], coords[2]))
        
        # Here the dynamical matrix starts
        fp.write("\n")
        fp.write("     Dynamical Matrix in cartesian axes\n")
        fp.write("\n")
        fp.write("     q = (    {:11.9f}   {:11.9f}   {:11.9f} )\n".format(q[0] * dyn.alat ,
                                                                           q[1] * dyn.alat, q[2] * dyn.alat ))
        fp.write("\n")
        
        # Now print the dynamical matrix
        for i in range(n_atoms):
            for j in range(n_atoms):
                # Write the atoms
                fp.write("%5d%5d\n" % (i + 1, j + 1))
                for x in range(3):
                    line = "%23.16f%23.16f   %23.16f%23.16f   %23.16f%23.16f" % \
                           ( np.real(dynq[3*i + x, 3*j]), 
                             np.imag(dynq[3*i + x, 3*j]),
                             np.real(dynq[3*i + x, 3*j+1]),
                             np.imag(dynq[3*i+x, 3*j+1]),
                             np.real(dynq[3*i + x, 3*j+2]),
                             np.imag(dynq[3*i+x, 3*j+2]) )
        
                    fp.write(line +  "\n")
        
        # Print the diagnoalization of the matrix
        fp.write("\n")
        fp.write("     Diagonalizing the dynamical matrix\n")
        fp.write("\n")
        fp.write("     q = (    {:11.9f}   {:11.9f}   {:11.9f} )\n".format(q[0] *dyn.alat ,
                                                                           q[1] *dyn.alat, q[2] *dyn.alat))
        fp.write("\n")
        fp.write("*" * 75 + "\n")
        
        nmodes = len(freqs)
        for mu in range(nmodes):
            # Print the frequency
            fp.write("%7s (%5d) = %14.8f [THz] = %14.8f [cm-1]\n" %
                     ("freq", mu+1, freqs[mu] * RyToTHz, freqs[mu] * RyToCm))
            
            # Print the polarization vectors
            for i in range(n_atoms):
                fp.write("( %10.6f%10.6f %10.6f%10.6f %10.6f%10.6f )\n" %
                         (np.real(pol_vects[3*i, mu]), np.imag(pol_vects[3*i,mu]),
                          np.real(pol_vects[3*i+1, mu]), np.imag(pol_vects[3*i+1,mu]),
                          np.real(pol_vects[3*i+2, mu]), np.imag(pol_vects[3*i+1,mu])))
        fp.write("*" * 75 + "\n")
        fp.close()    