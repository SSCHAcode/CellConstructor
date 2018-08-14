#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:10:21 2017

@author: darth-vader
"""
import os
import numpy as np
import Methods

# Load the fortran symmetry QE module
import symph




CURRENT_PATH = os.path.realpath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_PATH)
__EPSILON__ = 1e-5

class QE_Symmetry:
    def __init__(self, structure):
        """
        Quantum ESPRESSO Symmetry class
        ===============================
        
        This class contains all the info about Quantum ESPRESSO symmetry data.
        It is used to wrap symmetries into the quantum espresso fortran subroutines.
        
        Starting from a set of symmetry operation and the structure of the system, it
        builds all the QE symmetry operations.
        
        Parameters
        ----------
            structure : CC.Structure.Structure()
                The structure to which the symmetries refer to.
                
        """
        
        self.structure = structure
        
        nat = structure.N_atoms
        
        # Define the quantum espresso symmetry variables in optimized way to work with Fortran90
        self.QE_nat = np.intc( nat )
        self.QE_s = np.zeros( (3, 3, 48) , dtype = np.intc, order = "F")
        self.QE_irt = np.zeros( (48, nat), dtype = np.intc, order = "F")
        self.QE_invs = np.zeros( (48), dtype = np.intc, order = "F")
        self.QE_rtau = np.zeros( (3, 48, nat), dtype = np.float64, order = "F")
        
        self.QE_minus_q = np.bool( False )
        self.QE_irotmq = np.intc(0)
        self.QE_nsymq = np.intc( 0 )
        
    
    def InitFromSymmetries(self, symmetries, q_point):
        """
        This function initialize the QE symmetries from the symmetries expressed in the
        Cellconstructor format, i.e. a list of numpy array 3x4 where the last column is 
        the fractional translation.
        
        TODO: add the q_point preparation by limitng the symmetries only to 
              those that satisfies the specified q_point
        """
        
        nsym = len(symmetries)
        
        self.QE_nsymq = np.intc(nsym)
        
        
        for i, sym in enumerate(symmetries):
            self.QE_s[:,:, i] = sym[:, :3]
            
            # Get the atoms correspondence
            aux_atoms = self.structure.copy()
            aux_atoms.apply_symmetry(sym, delete_original = True)
            aux_atoms.fix_coords_in_unit_cell()
            eq_atoms = self.structure.get_equivalent_atoms(aux_atoms)
            
            self.QE_irt[i, :] = eq_atoms + 1
            
            # Get the inverse symmetry
            inv_sym = np.transpose(sym[:, :3])
            for k, other_sym in enumerate(symmetries):
                if np.sum( (inv_sym - other_sym)**2) < __EPSILON__:
                    break
            
            self.QE_invs[i] = k + 1
            
            # Setup the position after the symmetry application
            for k in range(self.QE_nat):
                self.QE_rtau[:, i, k] = aux_atoms.coords[k, :].astype(np.float64)
        
        
        # Get the reciprocal lattice vectors
        b_vectors = self.structure.get_reciprocal_vectors()
        
        # Get the minus_q operation
        self.QE_minusq = False
        if np.sum( (Methods.put_into_cell(b_vectors, -q_point) - q_point)**2) < __EPSILON__:
            # q != -q
            # Get the q vectors in crystal coordinates
            q = Methods.coovariant_coordinates(b_vectors, q_point)
            for k, sym in enumerate(self.QE_s):
                new_q = self.QE_s.dot(q)
                if np.sum( (Methods.put_into_cell(b_vectors, -q_point) - new_q)**2) < __EPSILON__:
                    break
                
            self.QE_minusq = True
            self.QE_irotmq = k + 1
                
                
    def SymmetrizeDynQ(self, dyn_matrix, q_point):
        """
        DYNAMICAL MATRIX SYMMETRIZATION
        ===============================
        
        Use the Quantum ESPRESSO fortran code to symmetrize the dynamical matrix
        at the given q point.
        
        NOTE: the symmetries must be already initialized.
        
        Parameters
        ----------
            dyn_matrix : ndarray (3nat x 3nat)
                The dynamical matrix associated to the specific q point (cartesian coordinates)
            q_point : ndarray 3
                The q point related to the dyn_matrix.
        
        The input dynamical matrix will be modified by the current code.
        """
        
        # TODO: implement hermitianity to speedup the conversion
        
        #Prepare the array to be passed to the fortran code
        QE_dyn = np.zeros( (3, 3, self.QE_nat, self.QE_nat), dtype = np.complex128, order = "F")
        
        # Get the crystal coordinates for the matrix
        for na in range(self.QE_nat):
            for nb in range(self.QE_nat):
                fc = dyn_matrix[3 * na : 3* na + 3, 3*nb: 3 * nb + 3]
                QE_dyn[:, :, na, nb] = Methods.convert_matrix_cart_cryst(fc, self.structure.unit_cell, False)
        
        # Prepare the xq variable
        xq = np.zeros(3, dtype = np.float64)
        xq[:] = q_point
        
        # USE THE QE library to perform the symmetrization
        symph.symdynph_gq_new( xq, QE_dyn, self.QE_s, self.QE_invs, self.QE_rtau, 
                              self.QE_irt, self.QE_irotmq, self.QE_minus_q, self.QE_nsymq, self.QE_nat)
        
        # Return to cartesian coordinates
        for na in range(self.QE_nat):
            for nb in range(self.QE_nat):
                fc = QE_dyn[:, :, na, nb] 
                dyn_matrix[3 * na : 3* na + 3, 3*nb: 3 * nb + 3] = Methods.convert_matrix_cart_cryst(fc, self.structure.unit_cell, True)
                


def get_symmetries_from_ita(ita, red=False):
    """
    This function returns a matrix containing the symmetries from the given ITA code of the Group.
    The corresponding ITA/group label can be found on the Bilbao Crystallographic Server.
    
    Parameters
    ----------
        - ita : int
             The ITA code that identifies the group symmetry.
        - red : bool (default = False)
            If red is True then load the symmetries only in the smallest unit cell (orthorombic)
    Results
    -------
        - symmetries : list
            A list of 3 rows x 4 columns matrices (ndarray), containing the symmetry operations 
            of the chosen group.
    """
    
    if ita <= 0:
        raise ValueError("Error, ITA group %d is not valid." % ita)
      
    filename="%s/SymData/%d.dat" % (CURRENT_DIR, ita)
    if red:
        filename="%s/SymData/%d_red.dat" % (CURRENT_DIR, ita)

    
    if not os.path.exists(filename):
        print "Error, ITA group not yet implemented."
        print "You can download the symmetries for this group from the Bilbao Crystallographic Server"
        print "And just add the %d.dat file into the SymData folder of the current program." % ita
        print "It should take less than five minutes."
        
        raise ValueError("Error, ITA group  %d not yet implemented. Check stdout on how to solve this problem." % ita)
    
    fp = open(filename, "r")
    
    # Get the number of symemtries
    n_sym = int(fp.readline().strip())
    fp.close()
    
    symdata = np.loadtxt(filename, skiprows = 1)
    symmetries = []

    for i in range(n_sym):
        symmetries.append(symdata[3*i:3*(i+1), :])
    
    return symmetries


def GetSymmetriesFromSPGLIB(spglib_sym, regolarize = True):
    """
    CONVERT THE SYMMETRIES
    ======================
    
    This module comvert the symmetry fynction from the spglib format.
    
    
    Parameters
    ----------
        spglib_sym : dict
            Result of spglib.get_symmetry( ... ) function
        regolarize : bool, optional
            If True it rewrites the translation to be exact. Usefull if you want to
            constrain the symmetry exactly
        
    Returns
    -------
        symmetries : list
            A list of 4x3 matrices containing the symmetry operation
    """
    
    # Check if the type is correct
    if not spglib_sym.has_key("translations"):
        raise ValueError("Error, your symmetry dict has no 'translations' key.")
        
    if not spglib_sym.has_key("rotations"):
        raise ValueError("Error, your symmetry dict has no 'rotations' key.")
    
    # Get the number of symmetries
    out_sym = []
    n_sym = np.shape(spglib_sym["translations"])[0]
    
    translations = spglib_sym["translations"]
    rotations = spglib_sym["rotations"]
    
    for i in range(n_sym):
        # Create the symmetry
        sym = np.zeros((3,4))
        sym[:,:3] = rotations[i, :, :]
        sym[:, 3] = translations[i,:]
        
        # Edit the translation
        if regolarize:
            sym[:, 3] *= 2
            sym[:, 3] = np.floor(sym[:, 3] + .5)
            sym[:, 3] *= .5
            sym[:, 3] = sym[:,3] % 1
        
        out_sym.append(sym)
    
    return out_sym

