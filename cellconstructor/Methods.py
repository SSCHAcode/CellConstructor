#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:45:50 2018

@author: pione
"""

from numpy import *
import numpy as np
import ase
from ase.visualize import view
import sys, os
import Structure
 

import symmetries as SYM

__all__ = ["covariant_coordinates", "from_dynmat_to_spectrum", 
           "put_into_cell", "get_minimal_orthorombic_cell", 
           "write_dynmat_in_qe_format", "get_gr", "cell2abc_alphabetagamma", 
           "DistanceBetweenStructures"]


def covariant_coordinates(basis, vector):
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
        - basis : 3x3 matrix
            The basis. each :math:`\\vec e_i` is a row.
        - vector : 3x float
            The vector expressed in cartesian coordinates.
            
    Results
    -------
        - cov_vector : 3x float
            The :math:`\\alpha_i` values.
            
    """
    
    metric_tensor = np.zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = basis[i,:].dot(basis[j, :])

    imt = np.linalg.inv(metric_tensor)
    contra_vect = basis.dot(vector)
    return imt.dot(contra_vect)
    
    
    
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
    new_phi = zeros(shape(dynmat))
    for i in range(n_atoms):
        M_i = struct.masses[struct.atoms[i]]
        for j in range(n_atoms):
            M_j = struct.masses[struct.atoms[j]]
            new_phi[3*i : 3*(i+1), 3*j : 3*(j+1)] = dynmat[3*i : 3*(i+1), 3*j : 3*(j+1)] / (M_i * M_j)

    
    # Diagonalize the matrix
    eigval, eigvect = linalg.eig(new_phi)
    eigval *= 220000. # conversion to cm-1

    return sort(eigval)
        

        
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

    # Put the vector inside the unit cell
    # To do this, just obtain the covariant vector coordinates.

    # Get the metric tensor
    metric_tensor = zeros((3,3))
    for i in range(0, 3):
        for j in range(i, 3):
            metric_tensor[i, j] = metric_tensor[j,i] = cell[i,:].dot(cell[j, :])

    # Get contravariant components
    contra_vect = zeros(3)
    for i in range(3):
        contra_vect[i] = vector.dot(cell[i, :]) 

    # Invert the metric tensor and obta
    covect = linalg.inv(metric_tensor).dot(contra_vect)

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
    final_vect = zeros(3)
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
        raise InputError("Error on input, ITA = %d not yet implemented." % ita)
    
    return minimal_cell

    
def write_dynmat_in_qe_format(struct, dyn_mat, filename):
    """
    Write the given dynamical matrix in the quantum espresso format.
    The system is considered to be a whole unit cell, so only the gamma matrix will be generated
    
    Parameters
    ----------
        - struct : Structure
             The structure of the system with the dynamical matrix

        - dyn_mat : float, 3*n_atoms x 3*n_atoms
             The dynamical matrix in Hartree units.

        - filename : string
             The path in which the quantum espresso dynamical matrix will be written.

    """
    
    fp = file(filename, "w")
    fp.write("Dynamical matrix file\n")

    # Get the different number of types
    types = []
    n_atoms = struct.N_atoms
    for i in range(n_atoms):
        if not struct.atoms[i] in types:
            types.append(struct.atoms[i])
    n_types = len(types)

    # Assign an integer for each atomic species
    itau = {}
    for i in range(n_types):
        itau[types[i]] = i
    
    fp.write("File generated with the build_unit_cell.py by Lorenzo Monacelli\n")
    fp.write("%d %d %d %.8f %.8f %.8f %.8f %.8f %.8f\n" %
             (n_types, n_atoms, 0, 1, 0, 0, 0, 0, 0) )

    fp.write("Basis vectors\n")
    # Get the unit cell
    for i in range(3):
        fp.write(" ".join("%12.8f" % x for x in struct.unit_cell[i,:]) + "\n")

    # Set the atom types and masses
    for i in range(n_types):
        fp.write("\t%d  '%s '  %.8f\n" % (i +1, types[i], struct.masses[types[i]]))

    # Setup the atomic structure
    for i in range(n_atoms):
        fp.write("%5d %5d %15.10f %15.10f %15.10f\n" %
                 (i +1, itau[struct.atoms[i]], struct.coords[i, 0], struct.coords[i,1], struct.coords[i,2]))

    # Here the dynamical matrix starts
    fp.write("\n")
    fp.write("     Dynamical Matrix in cartesian axes\n")
    fp.write("\n")
    fp.write("     q = (    %.9f   %.9f   %.9f )\n" % (0,0,0)) # Gamma point
    fp.write("\n")

    # Now print the dynamical matrix
    for i in range(n_atoms):
        for j in range(n_atoms):
            # Write the atoms
            fp.write("%5d%5d\n" % (i + 1, j + 1))
            for x in range(3):
                line = "%12.8f %12.8f   %12.8f %12.8f   %12.8f %12.8f" % \
                       ( real(dyn_mat[3*i + x, 3*j]), imag(dyn_mat[3*i + x, 3*j]),
                         real(dyn_mat[3*i + x, 3*j+1]), imag(dyn_mat[3*i+x, 3*j+1]),
                         real(dyn_mat[3*i + x, 3*j+2]), imag(dyn_mat[3*i+x, 3*j+2]) )

                fp.write(line +  "\n")

    fp.close()


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
                    r = struct.get_min_dist(first, second)
                    index_pos = int( r / real_dr)
                    if index_pos < N_bin:
                        N_r[index_pos] += 1

                

            
    # Now get the g(r) from N_r
    N_tot = sum(N_r)
    V = 4 * pi * r_max**3 / 3.
    rho = N_tot / V
    g_r = N_r / (4 * pi * real_r * real_dr * rho)

    # Get the final data and return
    data = zeros((N_bin, 2))
    data[:, 0] = real_r
    data[:, 1] = g_r
    return data
    
    
    
def cell2abc_alphabetagamma(unit_cell):
    """
This methods return a list of 6 elements. The first three are the three lengths a,b,c of the cell, while the other three
are the angles alpha (between b and c), beta (between a and c) and gamma(between a and b).

Parameters
----------
    - unit_cell : 3x3 ndarray (float)
         The unit cell in which the lattice vectors are the rows.

Results
-------
    - cell : 6 length ndarray (3x float, 3x int)
         The array containing the a,b,c length followed by alpha,beta and gamma (in degrees)
    """

    cell = np.zeros(6)

    # Get a,b,c
    for i in range(3):
        cell[i] = np.sqrt(unit_cell[i, :].dot(unit_cell[i, :]))

    # Get alpha beta gamma
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        cosangle = unit_cell[j,:].dot(unit_cell[k, :]) / (cell[j] * cell[k])
        
        cell[i + 3] = int(np.arccos(cosangle) * 180 / pi + .5)

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
        print "Strc1 has ", strc1.N_atoms, " atoms"
        print "Strc2 has ", strc2.N_atoms, " atoms"
        raise ValueError("Error, the number of atoms are not the same in the given structures.")

    nat = strc1.N_atoms
    coord1 = zeros(nat*3)
    coord2 = zeros(nat*3)
    
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
    return sqrt(sum((coord1 - coord2)**2))

