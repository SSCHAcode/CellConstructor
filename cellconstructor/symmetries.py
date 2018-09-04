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
    def __init__(self, structure, threshold = 1e-5):
        """
        Quantum ESPRESSO Symmetry class
        ===============================
        
        This class contains all the info about Quantum ESPRESSO symmetry data.
        It is used to wrap symmetries into the quantum espresso fortran subroutines.
        
        Starting from a set of symmetry operation and the structure of the system, it
        builds all the QE symmetry operations.
        
        NOTE:
            Use always the provided methods to change the self variables, as they are
            handled to properly cast the fortran types and array alignment.
        
        Parameters
        ----------
            structure : CC.Structure.Structure()
                The structure to which the symmetries refer to.
            threshold : float
                The threshold of the symmetry operation.
                
        """
        
        if not structure.has_unit_cell:
            raise ValueError("Error, symmetry operation can be initialize only if the structure has a unit cell")
        
        self.structure = structure
        self.threshold = np.float64(threshold)
        
        # Setup the threshold 
        symph.symm_base.set_accep_threshold(self.threshold)
        
        nat = structure.N_atoms
        
        # Define the quantum espresso symmetry variables in optimized way to work with Fortran90
        self.QE_nat = np.intc( nat )
        self.QE_s = np.zeros( (3, 3, 48) , dtype = np.intc, order = "F")
        self.QE_irt = np.zeros( (48, nat), dtype = np.intc, order = "F")
        self.QE_invs = np.zeros( (48), dtype = np.intc, order = "F")
        self.QE_rtau = np.zeros( (3, 48, nat), dtype = np.float64, order = "F")
        self.QE_ft = np.zeros( (3, 48), dtype = np.float64, order = "F")
        
        
        self.QE_minus_q = np.bool( False )
        self.QE_irotmq = np.intc(0)
        self.QE_nsymq = np.intc( 0 )
        
        # Prepare the QE structure
        self.QE_tau = np.zeros((3, nat), dtype = np.float64, order = "F")
        self.QE_ityp = np.zeros(nat, dtype = np.intc)
        
        symbs = {}
        counter = 1
        for i in range(nat):
            # Rank the atom number
            atm = structure.atoms[i]
            if not atm in symbs.keys():
                symbs[atm] = counter
                counter += 1
            
            self.QE_ityp[i] = symbs[atm]
            for j in range(3):
                self.QE_tau[j, i] = structure.coords[i, j]
                
            
        self.QE_at = np.zeros( (3,3), dtype = np.float64, order = "F")
        self.QE_bg = np.zeros( (3,3), dtype = np.float64, order = "F")
        
        bg = structure.get_reciprocal_vectors()
        for i in range(3):
            for j in range(3):
                self.QE_at[i,j] = structure.unit_cell[j,i]   
                self.QE_bg[i,j] = bg[j,i] / (2* np.pi) 

        
    def ChangeThreshold(self, threshold):
        self.threshold = np.float64(threshold)
        symph.symm_base.set_accep_threshold(self.threshold)
        
        
    def ImposeSumRule(self, force_constant, asr = "crystal", axis = 1, zeu = None):
        """
        QE SUM RULE
        ===========
        
        This subroutine imposes on the given force constant matrix the acustic sum rule
        
        Parameters
        ----------
            force_constnat : 3xnat , 3xnat
                The force constant matrix, it is overwritten with the new one
                after the sum rule has been applied.
            asr : string, optional, default = 'crystal'
                One of 'simple', 'crystal', 'one-dim' or 'zero-dim'. For a detailed
                explanation look at the Quantum ESPRESSO documentation.
            axis : int, optional
                If asr = 'one-dim' you must set the rotational axis: 1 for x, 2 for
                y and 3 for z. Ohterwise it is unused.
            zeu : ndarray (N_atoms, 3, 3), optional
                If different from None, it is the effective charge array. 
                As the force_constant, it is updated.
        
        """
        
        QE_fc = np.zeros( (3, 3, self.QE_nat, self.QE_nat), order ="F", dtype = np.complex128)
        
        # Fill the effective charges if required
        if zeu is not None:
            # Convert in the correct indexing and use the fortran order
            f_zeu = np.einsum("ijk -> kji", zeu, order = "F", dtype = np.float64)
        else:    
            f_zeu = np.zeros( (3, 3, self.QE_nat), order = "F", dtype = np.float64)
            
        # Prepare the force constant
        for na in range(self.QE_nat):
            for nb in range(self.QE_nat):
                QE_fc[:, :, na, nb] = force_constant[3 * na : 3* na + 3, 3*nb: 3 * nb + 3]
#        
#        print "ASR:", asr
#        print "AXIS:", axis
#        print "NAT:", self.QE_nat
#        print "TAU SHAPE:", np.shape(self.QE_tau)
#        print "QE_FC SHAPE:", np.shape(self.QE_fc)
        
        # Call the qe ASR subroutine
        symph.set_asr(asr, axis, self.QE_tau, QE_fc, f_zeu)
        
        # Copy the new value on output
        for na in range(self.QE_nat):
            if zeu is not None:
                zeu[na, :,:] = f_zeu[:,:, na]
            
            for nb in range(self.QE_nat):
                force_constant[3 * na : 3* na + 3, 3*nb: 3 * nb + 3] = QE_fc[:,:, na, nb]
        
    
    
        
    def SetupQPoint(self, q_point = np.zeros(3), verbose = False):
        """
        Get symmetries of the small group of q
        
        Setup the symmetries in the small group of Q.
        
        Parameters
        ----------
            q_point : ndarray
                The q vector in reciprocal space (NOT in crystal axes)
            verbose : bool
                If true the number of symmetries found for the bravais lattice, 
                the crystal and the small group of q are written in stdout
        """
        # Convert the q point in Fortran
        if len(q_point) != 3:
            raise ValueError("Error, the q point must be a 3d vector")
        
        aq = np.ones(3, dtype = np.float64) * Methods.covariant_coordinates(self.QE_bg.transpose(), q_point)
        
        # Setup the bravais lattice
        symph.symm_base.set_at_bg(self.QE_at, self.QE_bg)
        
        # Prepare the symmetries
        symph.symm_base.set_sym_bl()
        
        if verbose:
            print "Symmetries of the bravais lattice:", symph.symm_base.nrot
        
        
        # Now copy all the work initialized on the symmetries inside python
        self.QE_s = np.copy(symph.symm_base.s)
        self.QE_ft = np.copy(symph.symm_base.ft)
        self.QE_nsymq =  symph.symm_base.nrot
        
        # Prepare a dummy variable for magnetic spin
        m_loc = np.zeros( (3, self.QE_nat), dtype = np.float64, order = "F")
        
        # Find the symmetries of the crystal
        #print "TAU:", np.shape(self.QE_tau)
        symph.symm_base.find_sym(self.QE_tau, self.QE_ityp, 6, 6, 6, False, m_loc)
        #print "IRT NOW:", np.shape(symph.symm_base.irt)
        
        if verbose:
            print "Symmetries of the crystal:", symph.symm_base.nsym
        
        
        
        # Now copy all the work initialized on the symmetries inside python
        self.QE_s = np.copy(symph.symm_base.s)
        self.QE_ft = np.copy(symph.symm_base.ft)
        
        
        # Prepare the symmetries of the small group of q
        syms = np.zeros( (48), dtype = np.intc)
        
        # Initialize to true the symmetry of the crystal
        syms[:symph.symm_base.nsym] = np.intc(1)
        
        self.QE_minus_q = symph.symm_base.smallg_q(aq, 0, syms)
        self.QE_nsymq = symph.symm_base.copy_sym(symph.symm_base.nsym, syms)
        
        # Recompute the inverses
        symph.symm_base.inverse_s()
        
        if verbose:
            print "Symmetries of the small group of q:", self.QE_nsymq
        
        # Assign symmetries
        self.QE_s = np.copy(symph.symm_base.s)
        self.QE_invs = np.copy(symph.symm_base.invs)
        self.QE_ft = np.copy(symph.symm_base.ft)
        self.QE_irt = np.copy(symph.symm_base.irt)

        #print np.shape(self.QE_irt)
        
        # Compute the additional shift caused by fractional translations
        self.QE_rtau = symph.sgam_ph_new(self.QE_at, self.QE_bg, symph.symm_base.nsym, self.QE_s, 
                                         self.QE_irt, self.QE_tau, self.QE_nat)
        
        
        # If minus q check which is the symmetry
        if self.QE_minus_q:
            syms = self.GetSymmetries()
            
            # Fix in the Same BZ
            for i in range(3):
                aq[i] = aq[i] - int(aq[i])
                if aq[i] < 0:
                    aq[i] += 1
                    
            for k, sym in enumerate(syms):
                new_q = sym[:,:3].dot(aq)
                # Fix in the Same BZ
                for i in range(3):
                    new_q[i] = new_q[i] - int(new_q[i])
                    if new_q[i] < 0:
                        new_q[i] += 1
                
                if np.sum( (new_q - aq)**2) < __EPSILON__:
                    self.QE_irotmq = k + 1
                    break
                
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
            self.QE_s[:,:, i] = np.transpose(sym[:, :3])
            
            # Get the atoms correspondence
            aux_atoms = self.structure.copy()
            aux_atoms.apply_symmetry(sym, delete_original = True)
            aux_atoms.fix_coords_in_unit_cell()
            eq_atoms = np.array(self.structure.get_equivalent_atoms(aux_atoms), dtype = np.intc)
            
            self.QE_irt[i, :] = eq_atoms + 1
            
            # Get the inverse symmetry
            inv_sym = np.linalg.inv(sym[:, :3])
            for k, other_sym in enumerate(symmetries):
                if np.sum( (inv_sym - other_sym[:, :3])**2) < __EPSILON__:
                    break
            
            self.QE_invs[i] = k + 1
            
            # Setup the position after the symmetry application
            for k in range(self.QE_nat):
                self.QE_rtau[:, i, k] = aux_atoms.coords[k, :].astype(np.float64)
        
        
        # Get the reciprocal lattice vectors
        b_vectors = self.structure.get_reciprocal_vectors()
        
        # Get the minus_q operation
        self.QE_minusq = False
        
        # q != -q
        # Get the q vectors in crystal coordinates
        q = Methods.covariant_coordinates(b_vectors, q_point)
        for k, sym in enumerate(self.QE_s):
            new_q = self.QE_s[:,:, k].dot(q)
            if np.sum( (Methods.put_into_cell(b_vectors, -q_point) - new_q)**2) < __EPSILON__:
                self.QE_minus_q = True
                self.QE_irotmq = k + 1
                break
                
    def GetSymmetries(self):
        """
        GET SYMMETRIES FROM QE
        ======================
        
        This method returns the symmetries in the CellConstructor format from
        the ones elaborated here.
        
        Results
        -------
            list :
                List of 3x4 ndarray representing all the symmetry operations
        """
        
        syms = []
        for i in range(self.QE_nsymq):
            s_rot = np.zeros( (3, 4))
            s_rot[:, :3] = np.transpose(self.QE_s[:, :, i])
            s_rot[:, 3] = self.QE_ft[:, i]
            
            syms.append(s_rot)
        
        return syms
            
    
    def SymmetrizeVector(self, vector):
        """
        SYMMETRIZE A VECTOR
        ===================
        
        This is the easier symmetrization of a generic vector.
        Note, fractional translation and generic translations are not imposed.
        This is because this simmetrization acts on displacements.
        
        Parameters
        ----------
            vector : ndarray(natoms, 3)
                This is the vector to be symmetrized, it will be overwritten
                with the symmetrized version
        """
        
        # Prepare the real vector
        tmp_vector = np.zeros( (3, self.QE_nat), dtype = np.float64, order = "F")
        
        for i in range(self.QE_nat):
            tmp_vector[0, i] = vector[i,0]
            tmp_vector[1, i] = vector[i,1]
            tmp_vector[2,i] = vector[i,2]
        
        symph.symvector(self.QE_nsymq, self.QE_irt, self.QE_s, self.QE_at, self.QE_bg,
                        tmp_vector, self.QE_nat)
        
        
        for i in range(self.QE_nat):
            vector[i, :] = tmp_vector[:,i]
        
                
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
        xq = np.ones(3, dtype = np.float64)
        xq *= q_point
        
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

