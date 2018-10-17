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
        self.QE_nsym = np.intc(0)
        
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
                
    def ApplyQStar(self, fcq, q_point_group):
        """
        APPLY THE Q STAR SYMMETRY
        =========================
        
        Given the fc matrix at each q in the star, it applies the symmetries in between them.
        
        Parameters
        ----------
            - fcq : ndarray(nq, 3xnat, 3xnat) 
                The dynamical matrices for each q point in the star
            - q_point_group : ndarray(nq, 3)
                The q vectors that belongs to the same star
        """
        
        nq = np.shape(q_point_group)[0]
        final_fc = np.zeros(np.shape(fcq), dtype = np.complex128)
        
        # Setup all the symmetries
        self.SetupQPoint()
        
        new_dyn = np.zeros( (3 * self.QE_nat, 3*self.QE_nat), dtype = np.complex128, order = "F")
        
        dyn_star = np.zeros( (nq, 3, 3, self.QE_nat, self.QE_nat), dtype = np.complex128, order = "F")
        
        
        for i in range(nq):
            # Get the q points order
            nq_new, sxq, isq, imq = symph.star_q(q_point_group[i,:], self.QE_at, self.QE_bg, 
                                                 self.QE_nsymq, self.QE_s, self.QE_invs, 0)
            
            # Check if the q star is correct
            if nq_new != nq:
                print "Reciprocal lattice vectors:"
                print self.QE_bg.transpose() 
                print "Passed q star:"
                print q_point_group
                print "QE q star:"
                print sxq[:, :nq_new].transpose()
                raise ValueError("Error, the passed q star does not match the one computed by QE")
#            
#            # Print the star 
#            print "q point:", q_point_group[i,:]
#            print "Point in the stars:", nq_new
#            print "Star of q:"
#            print sxq[:, :nq_new].transpose()
#            
#            print "NEW_DYN:", np.shape(new_dyn)
#            print "AT:", np.shape(self.QE_at)
#            print "BG:", np.shape(self.QE_bg)
#            print "N SYM:", self.QE_nsymq
#            print "S:", np.shape(self.QE_s)
#            print "QE_INVS:", np.shape(self.QE_invs)
#            print "IRT:", np.shape(self.QE_irt)
#            print "RTAU:", np.shape(self.QE_rtau)
#            print "NQ_NEW:", nq_new
#            print "SXQ:", np.shape(sxq)
#            print "ISQ:", np.shape(isq)
#            print "IMQ:", imq
#            print "NAT:", self.QE_nat
            
            new_dyn[:,:] = fcq[i,:,:]
            #print "new dyn ready"
        
            # Get the new matrix
            dyn_star = symph.q2qstar_out(new_dyn, self.QE_at, self.QE_bg, self.QE_nsymq, 
                              self.QE_s, self.QE_invs, self.QE_irt, self.QE_rtau,
                              nq_new, sxq, isq, imq, self.QE_nat)
            #print "Fake"
            
            # Now to perform the match bring the star in the same BZ as the q point
            # This facilitate the comparison between q points
            current_q = q_point_group.copy()
            #print "Fake2"
#            for xq in range(nq):
#                tmp = Methods.put_into_cell(self.QE_bg, sxq[:, xq])
#                sxq[:, xq] = tmp
#                current_q[xq,:] = Methods.put_into_cell(self.QE_bg, current_q [xq,:])
#            
            # Print the order of the q star
            sorting_q = np.arange(nq)
            for xq in range(nq):
                count = 0 # Debug (avoid no or more than one identification)
                for yq in range(nq_new):
                    if Methods.get_min_dist_into_cell(self.QE_bg, sxq[:, yq], current_q[xq,:]) < __EPSILON__: 
                        sorting_q[xq] = yq
                        count += 1
                
                if count != 1:
                    print "Original star:"
                    print q_point_group
                    print "Reshaped star:"
                    print current_q
                    print "Reciprocal lattice vectors:"
                    print self.QE_bg.transpose() 
                    print "STAR:"
                    print sxq[:, :nq_new].transpose()    
                    pta = current_q[xq,:]
                    raise ValueError("Error, the vector (%.3f, %.3f, %.3f) has %d identification in the star" % (pta[0], pta[1], pta[2],
                                                                                                                 count))
            #print "Sorting array:"
            #print sorting_q
                    
                        
            # Copy the matrix in the new one
            for xq in range(nq):
                for xat in range(self.QE_nat):
                    for yat in range(self.QE_nat):
                        final_fc[xq, 3*xat: 3*xat + 3, 3*yat : 3*yat + 3] += dyn_star[sorting_q[xq], :,:, xat, yat] 
            
        
        # Now divide the matrix per the xq value
        final_fc /= nq
            
        # Overwrite the matrix
        fcq[:,:,:] = final_fc
        
        
    def SymmetrizeFCQ(self, fcq, q_stars, verbose = False, asr = "simple"):
        """
        Use the current structure to impose symmetries on a complete dynamical matrix
        in q space. Also the simple sum rule at Gamma is imposed
        
        Parameters
        ----------
            - fcq : ndarray(nq, 3xnat, 3xnat)
                The q space force constant matrix to be symmetrized (it will be overwritten)
            - q_stars : list of list of q points
                The list of q points divided by stars, the fcq must follow the order
                of the q points in the q_stars array
        """
        
        nqirr = len(q_stars)
        nq = np.sum([len(x) for x in q_stars])
        
        # Get the q_points vector
        q_points = np.zeros( (nq, 3), dtype = np.float64)
        sigma = 0
        for i in range(nqirr):
            for q_vec in q_stars[i]:
                q_points[sigma, :] = q_vec
                sigma += 1
        
        if nq != np.shape(fcq)[0]:
            raise ValueError("Error, the force constant number of q point %d does not match with the %d given q_points" % (np.shape(fcq)[0], nq))
            
        
        for iq in range(nq):
            # Prepare the symmetrization
            if verbose:
                print "Symmetries in q = ", q_points[iq, :]
            self.SetupQPoint(q_points[iq,:], verbose)
            
            # Proceed with the sum rule if we are at Gamma
            if asr == "simple" or asr == "custom":
                if np.sqrt(np.sum(q_points[iq,:]**2)) < __EPSILON__:
                    self.ImposeSumRule(fcq[iq,:,:], asr)
            elif asr == "crystal":
                self.ImposeSumRule(fcq[iq, :,:], asr = asr)
            elif asr == "no":
                pass
            else:
                raise ValueError("Error, only 'simple', 'crystal', 'custom' or 'no' asr are supported, given %s" % asr)
            
            # Symmetrize the matrix
            self.SymmetrizeDynQ(fcq[iq, :,:], q_points[iq,:])
        
        
        # For each star perform the symmetrization over that star
        q0_index = 0
        for i in range(nqirr):
            q_len = len(q_stars[i])
            self.ApplyQStar(fcq[q0_index : q0_index + q_len, :,:], np.array(q_stars[i]))
            q0_index += q_len

        
    def ChangeThreshold(self, threshold):
        """
        Change the symmetry threshold sensibility
        """
        self.threshold = np.float64(threshold)
        symph.symm_base.set_accep_threshold(self.threshold)
        
        
    def ImposeSumRule(self, force_constant, asr = "simple", axis = 1, zeu = None):
        """
        QE SUM RULE
        ===========
        
        This subroutine imposes on the given force constant matrix the acustic sum rule
        
        Parameters
        ----------
            force_constnat : 3xnat , 3xnat
                The force constant matrix, it is overwritten with the new one
                after the sum rule has been applied.
            asr : string, optional, default = 'custom'
                One of 'custom', 'simple', 'crystal', 'one-dim' or 'zero-dim'. For a detailed
                explanation look at the Quantum ESPRESSO documentation.
                The custom one, default, is implemented in python as CustomASR.
                No ASR is imposed on the effective charges in this case.
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
        if asr != "custom":
            for na in range(self.QE_nat):
                for nb in range(self.QE_nat):
                    QE_fc[:, :, na, nb] = force_constant[3 * na : 3* na + 3, 3*nb: 3 * nb + 3]
    #        
#        print "ASR:", asr
#        print "AXIS:", axis
#        print "NAT:", self.QE_nat
#        print "TAU SHAPE:", np.shape(self.QE_tau)
#        print "QE_FC SHAPE:", np.shape(self.QE_fc)
        
                
            symph.set_asr(asr, axis, self.QE_tau, QE_fc, f_zeu)
            
            # Copy the new value on output
            for na in range(self.QE_nat):
                if zeu is not None:
                    zeu[na, :,:] = f_zeu[:,:, na]
                
                for nb in range(self.QE_nat):
                    force_constant[3 * na : 3* na + 3, 3*nb: 3 * nb + 3] = QE_fc[:,:, na, nb]
        else:
            CustomASR(force_constant)
        
    
    
        
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
        self.QE_nsym =  symph.symm_base.nrot
        
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

def CustomASR(fc_matrix):
    """
    APPLY THE SUM RULE
    ==================
    
    This function applies a particular sum rule. It projects out the translations
    exactly.
    
    Parameters
    ----------
        fc_matrix : ndarray(3nat x 3nat)
            The force constant matrix. The sum rule is applied on that.
    """
    
    shape = np.shape(fc_matrix)
    if shape[0] != shape[1]:
        raise ValueError("Error, the provided matrix is not square: (%d, %d)" % (shape[0], shape[1]))
    
    nat = np.shape(fc_matrix)[0] / 3
    if nat*3 != shape[0]:
        raise ValueError("Error, the matrix must have a dimension divisible by 3: %d" % shape[0])
    
    
    dtype = type(fc_matrix[0,0])
    
    trans = np.eye(3*nat, dtype = dtype)
    for i in range(3):
        v1 = np.zeros(nat*3, dtype = dtype)
        v1[3*np.arange(nat) + i] = 1
        v1 /= np.sqrt(v1.dot(v1))
        
        trans -= np.outer(v1, v1)
        
    #print trans

    fc_matrix[:,:] = trans.dot(fc_matrix.dot(trans))
    

def ApplySymmetryToVector(symmetry, vector):
    """
    Apply the symmetry to the given vector. Translations are neglected.
    """
    
    # TODO:
    pass
