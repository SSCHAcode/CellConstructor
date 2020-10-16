#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:10:21 2017

@author: darth-vader
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import time
import os
import numpy as np

import scipy
import scipy.linalg 

import cellconstructor.Methods as Methods
from cellconstructor.Units import *

# Load the fortran symmetry QE module
import symph

# Load the LinAlgebra module in C
from cc_linalg import GramSchmidt

import warnings


__SPGLIB__ = True
try:
    import spglib
except:
    __SPGLIB__ = False




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
            # Convert in bohr
            for j in range(3):
                self.QE_tau[j, i] = structure.coords[i, j]
                
            
        self.QE_at = np.zeros( (3,3), dtype = np.float64, order = "F")
        self.QE_bg = np.zeros( (3,3), dtype = np.float64, order = "F")
        
        bg = structure.get_reciprocal_vectors()
        for i in range(3):
            for j in range(3):
                self.QE_at[i,j] = structure.unit_cell[j,i]
                self.QE_bg[i,j] = bg[j,i] / (2* np.pi) 

        # Here we define the quantities required to symmetrize the supercells
        self.QE_at_sc = self.QE_at.copy()
        self.QE_bg_sc = self.QE_bg.copy()
        self.QE_translation_nr = 1 # The supercell total dimension (Nx * Ny * Nz)
        self.QE_translations = [] # The translations in crystal axes

        # After the translation, which vector is transformed in which one?
        # This info is stored here as ndarray( size = (N_atoms, N_trans), dtype = np.intc, order = "F")
        self.QE_translations_irt = [] 
    
    def ForceSymmetry(self, structure):
        """ 
        FORCE SYMMETRY
        ==============
        
        Force the symmetries found at a given threshold to
        be satisfied also in a lower threshold.
        
        This use the irt trick
        """
        if self.QE_nsymq == 0:
            raise ValueError("Error, initialize the symmetries with SetupQPoint.")
        
        coords = np.zeros( (3, structure.N_atoms), order = "F", dtype = np.float64)
        coords[:,:] = structure.coords.transpose()
        
        # Transform in crystal coordinates
        symph.cryst_to_cart(coords, self.QE_bg, -1)
        
        new_coords = np.zeros( (3, structure.N_atoms), order = "F", dtype = np.float64)
        for s_i in range(self.QE_nsymq):
            for i in range(structure.N_atoms):
                new_coords[:, self.QE_irt[s_i, i]-1 ] += self.QE_s[:,:,s_i].dot(coords[:,i])
                new_coords[:, self.QE_irt[s_i, i]-1 ] += self.QE_ft[:, s_i]
        
        new_coords /= self.QE_nsymq
        
        # Transform back into cartesian coordinates
        symph.cryst_to_cart(new_coords, self.QE_at, 1)
        
        # Save in the structure
        structure.coords[:,:] = new_coords.transpose()
        
    def PrintSymmetries(self):
        """
        This method just prints the symmetries on stdout.
        """

        print()
        print("Number of symmetries: {}".format(self.QE_nsym))
        syms = self.GetSymmetries()
        for i in range(self.QE_nsym):
            print("  Symmetry {}".format(i+1))
            for j in range(3):
                print("  {:3.0f}{:3.0f}{:3.0f} | {:6.3f}".format(*syms[i][j,:]))
            print()

    def GetUniqueRotations(self):
        """
        This subroutine returns an alternative symmetries 
        that contains only unique rotations (without fractional translations).
        This is usefull if the peculiar cell is a supercell 
        and the symmetrization was performed with SPGLIB

        Returns
        -------
            QE_s : ndarray(size = (3,3,48), dtype = np.intc)
                The symmetries
            QE_invs : ndarray(size = 48, dtype = np.intc)
                The index of the inverse symmetry
            QE_nsym : int
                The number of symmetries
        """

        QE_s = np.zeros( self.QE_s.shape, dtype = np.intc, order = "F")
        QE_invs = np.zeros(self.QE_invs.shape, dtype = np.intc, order = "F")
        QE_nsym = 0


        for i in range(self.QE_nsym):
            # Check if the same rotation was already added
            skip = False
            for j in range(QE_nsym):
                # Check if the rotation occurred
                if (QE_s[:,:,j] == self.QE_s[:,:,i]).all():
                    skip = True 
                    break 
            
            if not skip:
                # We did not find another equal rotation
                # Lets add this one
                QE_s[:,:, QE_nsym] = self.QE_s[:,:,i]
                QE_nsym += 1
        
        # Get the inverse
        QE_invs[:] = get_invs(QE_s, QE_nsym)

        return QE_s, QE_invs, QE_nsym






            
    def SetupQStar(self, q_tot, supergroup = False):
        """
        DIVIDE THE Q POINTS IN STARS
        ============================
        
        This method divides the given q point list into the star.
        Remember, you need to pass the whole number of q points
        
        Parameters
        ----------
            q_tot : list
                List of q vectors to be divided into stars
            supergroup : bool
                If true then assume we have initialized a supercell bigger
        Results
        -------
            q_stars : list of lists
                The list of q_star (list of q point in the same star).
            sort_mask : ndarray(size=len(q_tot), dtype = int)
                a mask to sort the q points in order to match the
                same order than the q_star
        """
        
        # Setup the symmetries
        #self.SetupQPoint()
        
        # Lets copy the q list (we are going to pop items from it)
        q_list = q_tot[:]
        q_stars = []
        
        count_qstar = 0
        count_q = 0
        q_indices = np.zeros( len(q_tot), dtype = int)
        while len(q_list) > 0:
            q = q_list[0]
            # Get the star of the current q point
            _q_ = np.array(q, dtype = np.float64) # Fortran explicit conversion
        
            nq_new, sxq, isq, imq = symph.star_q(_q_, self.QE_at, self.QE_bg, 
                                                 self.QE_nsym, self.QE_s, self.QE_invs, 0)
        
            # print ("START WITH Q:", q)
            # print ("FOUND STAR:")
            # for jq in range(nq_new):
            #     print (sxq[:, jq])
            # print ()
            
            # print ("TELL ME THE BG:")
            # print (self.QE_bg.transpose())

            # print("Manual star:")
            # for k in range(self.QE_nsym):
            #     trial_q = q.dot(self.QE_s[:,:, k])
            #     distance_q = Methods.get_min_dist_into_cell(self.QE_bg.T, trial_q, q)
            #     distance_mq =  Methods.get_min_dist_into_cell(self.QE_bg.T, trial_q, -q)
            #     print("trial_q : {} | DQ: {:.4f} | DMQ: {:.4f}".format(trial_q, distance_q, distance_mq ))
            
            # Prepare the star
            q_star = [sxq[:, k] for k in range(nq_new)]

            # If imq is not zero (we do not have -q in the star) then add the -q for each in the star
            if imq == 0:
                old_q_star = q_star[:]
                min_dist = 1
                
                for q in old_q_star:
                    q_star.append(-q)

                    

            q_stars.append(q_star)
            
            # Pop out the q_star from the q_list
            for jq, q_instar in enumerate(q_star):
                # Look for the q point in the star and pop them
                #print("q_instar:", q_instar)
                q_dist = [Methods.get_min_dist_into_cell(self.QE_bg.transpose(), 
                                                         np.array(q_instar), q_point) for q_point in q_list]
                
                pop_index = np.argmin(q_dist)            
                q_list.pop(pop_index)
                
                # Use the same trick to identify the q point
                q_dist = [Methods.get_min_dist_into_cell(self.QE_bg.transpose(), 
                                                         np.array(q_instar), q_point) for q_point in q_tot]
                
                q_index = np.argmin(q_dist)
                #print (q_indices, count_q, q_index)
                q_indices[count_q] = q_index
                
                count_q += 1
            
            
        return q_stars, q_indices


    def ApplySymmetryToTensor3(self, v3, initialize_symmetries = True):
        """
        SYMMETRIZE A RANK-3 TENSOR
        ==========================

        This subroutines uses the current symmetries to symmetrize
        a rank-3 tensor. 
        This tensor must be in the supercell space.

        The v3 argument will be overwritten.

        NOTE: The symmetries must be initialized in the supercell using spglib
        

        Parameters
        ----------
            v3 : ndarray( size=(3*nat, 3*nat, 3*nat), dtype = np.double, order = "F")
                The 3-rank tensor to be symmetrized.
                It will be overwritten with the new symmetric one.
                It is suggested to specify the order of the array to "F", as this will prevent
                the parser to copy the matrix when doing the symmetrization in Fortran.
            initialize_symmetries : bool
                If True the symmetries will be initialized using spglib. Otherwise
                the already present symmetries will be use. Use it False at your own risk! 
                (It can crash with seg fault if symmetries are not properly initialized)
        """
        if initialize_symmetries:
            self.SetupFromSPGLIB()

        # Apply the permutation symmetry
        symph.permute_v3(v3)

        # Apply the translational symmetries
        symph.trans_v3(v3, self.QE_translations_irt)

        # Apply all the symmetries at gamma
        symph.sym_v3(v3, self.QE_at, self.QE_s, self.QE_irt, self.QE_nsymq)

    def ApplySymmetryToEffCharge(self, eff_charges):
        """
        SYMMETRIZE EFFECTIVE CHARGES
        ============================

        This subroutine applies the symmetries to the effective charges.

        As always, the eff_charges will be modified by this subroutine.

        Parameters
        ----------
            - eff_charges : ndarray (size = (nat, 3, 3))
                The effective charges tensor. 
                The first dimension is the index of the atom in the primitive cell
                the second index is the electric field.
                The third index is the cartesian axis.
        """
    
        nat, cart1, cart2 = np.shape(eff_charges)

        assert cart1 == cart2 
        assert cart1 == 3
        assert nat == self.QE_nat, "Error, the structure and effective charges are not compatible"


        # Apply the sum rule
        tot_sum = np.sum(eff_charges, axis = 0)
        eff_charges -= np.tile(tot_sum, (nat, 1)).reshape((nat, 3,3 )) / nat

        new_eff_charges = np.zeros((nat, cart1, cart2), dtype = np.double)

        # Get the effective charges in crystal components
        for i in range(nat):
            eff_charges[i, :, :] = Methods.convert_matrix_cart_cryst(eff_charges[i, :, :], self.QE_at.T)

        # Apply translations
        if self.QE_translation_nr > 1:
            for i in range(self.QE_translation_nr):
                irt = self.QE_translations_irt[:, i] - 1
                for j in range(nat):
                    new_mat = eff_charges[irt[j], :, :]
                    new_eff_charges[j, :, :] += new_mat

            eff_charges[:,:,:] = new_eff_charges / self.QE_translation_nr
            new_eff_charges[:,:,:] = 0.

        # Apply rotations
        for i in range(self.QE_nsym):
            irt = self.QE_irt[i, :] - 1

            for j in range(nat):
                new_mat = self.QE_s[:,:, i].dot( eff_charges[irt[j], :, :].dot(self.QE_s[:,:,i].T))
                new_eff_charges[j, :, :] += new_mat
        new_eff_charges /= self.QE_nsym

        # Convert back into cartesian
        for i in range(nat):
            eff_charges[i, :, :] = Methods.convert_matrix_cart_cryst(new_eff_charges[i, :, :], self.QE_at.T, True)

    def ApplySymmetryToRamanTensor(self, raman_tensor):
        """
        SYMMETRIZE RAMAN TENSOR
        ============================

        This subroutine applies the symmetries to the raman tensor

        As always, the raman_tensor will be modified by this subroutine.

        Parameters
        ----------
            - raman_tensor : ndarray (size = (3, 3, 3*nat))
                The raman tensor. The first two indices indicate
                the polarization of the incoming/outcoming field, while the last one
                is the atomic/cartesian coordinate
        """
    
        pol1, pol2, at_cart = np.shape(raman_tensor)

        assert pol1 == pol2 
        assert pol2 == 3
        assert at_cart == 3*self.QE_nat, "Error, the structure and effective charges are not compatible"

        # Apply the permutation on the electric fields
        raman_tensor += np.einsum("abc->bac", raman_tensor)
        raman_tensor /= 2

        # Apply the sum rule
        # The sum over all the atom for each cartesian coordinate should be zero.
        rt_reshaped = raman_tensor.reshape((3,3,self.QE_nat, 3))

        # Sum over all the atomic indices
        tot_sum = np.sum(rt_reshaped, axis = 2)

        # Rebuild the shift to the tensor of the correct shape
        shift = np.tile(tot_sum, (self.QE_nat, 1, 1, 1))

        # Place the number of atoms at the correct position
        # From the first to the third
        shift = np.einsum("abcd->bcad", shift)
        
        # Now we apply the sum rule
        rt_reshaped -= shift / self.QE_nat
        new_tensor = np.zeros(np.shape(rt_reshaped), dtype = np.double)

        # Get the raman tensor in crystal components
        for i in range(self.QE_nat):
            rt_reshaped[:,:, i, :] = Methods.convert_3tensor_to_cryst(rt_reshaped[:,:, i, :], self.QE_at.T)

        # Apply translations
        if self.QE_translation_nr > 1:
            for i in range(self.QE_translation_nr):
                irt = self.QE_translations_irt[:, i] - 1
                for j in range(self.QE_nat):
                    new_mat = rt_reshaped[:,:, irt[j], :]
                    new_tensor += new_mat

            rt_reshaped = new_tensor / self.QE_translation_nr
            new_tensor[:,:,:,:] = 0.

        # Apply rotations
        for i in range(self.QE_nsym):
            irt = self.QE_irt[i, :] - 1

            for j in range(self.QE_nat):
                # Apply the symmetry to the 3 order tensor
                new_mat = np.einsum("ai, bj, ck, ijk", self.QE_s[:,:,i], self.QE_s[:,:,i], self.QE_s[:,:,i], rt_reshaped[:,:, irt[j], :])
                #new_mat = self.QE_s[:,:, i].dot( eff_charges[irt[j], :, :].dot(self.QE_s[:,:,i].T))
                new_tensor[:,:,j,:] += new_mat

        new_tensor /= self.QE_nsym

        # Convert back into cartesian
        for i in range(self.QE_nat):
            rt_reshaped[:, :, i, :] = Methods.convert_3tensor_to_cryst(new_tensor[:,:,i,:], self.QE_at.T, True)

        # Compress again the notation
        raman_tensor[:,:,:] = rt_reshaped.reshape((3,3, 3*self.QE_nat))


    def ApplySymmetryToSecondOrderEffCharge(self, dM_drdr, apply_asr = True):
        """
        SYMMETRIZE TWO PHONON EFFECTIVE CHARGES
        =======================================

        This subroutine applies simmetries to the two phonon
        effective charges.

        Note, to symmetrize this tensor, symmetries must be imposed 
        on the supercell.

        Parameters
        ----------
            dM_drdr : ndarray (size = (3 nat_sc, 3nat_sc, 3))
                The derivative of effective charges.
            apply_asr : bool
                If True the sum rule is applied. 
                The sum rule is the 'custom' one where translations are projected
                out from the space for each polarization components.
        """

        nat3, nat3_, cart = np.shape(dM_drdr)

        assert nat3 == nat3_, "Error on the shape of the argument"
        assert nat3 == 3 * self.QE_nat, "Wrong number of atoms (Symmetries must be setup in the supercell)"
        assert cart == 3

        nat = int(nat3 / 3)
        
        # Apply hermitianity
        #print("Original:")
        #print(dM_drdr[:,:,0])

        dM_drdr += np.einsum("abc->bac", dM_drdr)
        dM_drdr /= 2

        # Apply the Sum Rule
        if apply_asr:
            for pol in range(3):
                CustomASR(dM_drdr[:,:,pol])

        #print("After the sum rule:")
        #print(dM_drdr[:,:,0])

        # Convert in crystal coordinates
        for i in range(nat):
            for j in range(nat):
                dM_drdr[3*i : 3*i + 3, 3*j: 3*j+3, :] = Methods.convert_3tensor_to_cryst(dM_drdr[3*i:3*i+3, 3*j:3*j+3,:], self.QE_at.T)


        #print("Crystal:")
        #print(dM_drdr[:,:,0])


        # Apply translations
        new_dM = np.zeros(np.shape(dM_drdr), dtype = np.double)
        if self.QE_translation_nr > 1:
            for i in range(self.QE_translation_nr):
                irt = self.QE_translations_irt[:, i] - 1
                for jat in range(nat):
                    for kat in range(nat):
                        new_mat = dM_drdr[3*irt[jat]: 3*irt[jat]+3, 3*irt[kat]:3*irt[kat] + 3,:]
                        new_dM[3*jat: 3*jat+3, 3*kat:3*kat+3, :] += new_mat

            dM_drdr[:,:,:] = new_dM / self.QE_translation_nr
            new_dM[:,:,:] = 0

        
        #print("After transl:")
        #print(dM_drdr[:,:,0])

        #self.PrintSymmetries()

        # Apply rotations
        for i in range(self.QE_nsym):
            irt = self.QE_irt[i, :] - 1

            #print("")
            #print("--------------------")
            #print("symmetry: {:d}, irt: {}".format(i+1, irt +1))

            #prova = np.zeros(np.shape(new_dM))

            for jat in range(nat):
                for kat in range(nat):
                    new_mat = dM_drdr[3*irt[jat]: 3*irt[jat]+3, 3*irt[kat]:3*irt[kat] + 3,:]
                    # Apply the symmetries

                    new_mat = np.einsum("ck, ijk->ijc", self.QE_s[:,:,i], new_mat)
                    new_mat = np.einsum("bj, ijc->ibc", self.QE_s[:,:,i], new_mat)
                    new_mat = np.einsum("ai, ibc->abc", self.QE_s[:,:,i], new_mat)
                    #prova[3*jat:3*jat+3, 3*kat:3*kat+3,:] = new_mat
                    new_dM[3*jat:3*jat+3, 3*kat:3*kat+3,:] += new_mat
        
            #print(np.einsum("abc->cab", prova))
            #print("--------------------")
        dM_drdr[:,:,:] = new_dM / self.QE_nsym



        # Convert in crystal coordinates
        for i in range(nat):
            for j in range(nat):
                dM_drdr[3*i : 3*i + 3, 3*j: 3*j+3, :] = Methods.convert_3tensor_to_cryst(dM_drdr[3*i:3*i+3, 3*j:3*j+3,:], self.QE_at.T, True)

        

    def ApplySymmetryToTensor4(self, v4, initialize_symmetries = True):
        """
        SYMMETRIZE A RANK-4 TENSOR
        ==========================

        This subroutines uses the current symmetries to symmetrize
        a rank-4 tensor. 
        This tensor must be in the supercell space.

        The v4 argument will be overwritten.

        NOTE: The symmetries must be initialized in the supercell using spglib
        

        Parameters
        ----------
            v4 : ndarray( size=(3*nat, 3*nat, 3*nat, 3*nat), dtype = np.double, order = "F")
                The 4-rank tensor to be symmetrized.
                It will be overwritten with the new symmetric one.
                It is suggested to specify the order of the array to "F", as this will prevent
                the parser to copy the matrix when doing the symmetrization in Fortran.
            initialize_symmetries : bool
                If True the symmetries will be initialized using spglib. Otherwise
                the already present symmetries will be use. Use it False at your own risk! 
                (It can crash with seg fault if symmetries are not properly initialized)
        """
        if initialize_symmetries:
            self.SetupFromSPGLIB()

        # Apply the permutation symmetry
        symph.permute_v4(v4)

        # Apply the translational symmetries
        symph.trans_v4(v4, self.QE_translations_irt)

        # Apply all the symmetries at gamma
        symph.sym_v4(v4, self.QE_at, self.QE_s, self.QE_irt, self.QE_nsymq)

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
            

            #print "Found nq:", nq_new 
            #print "IMQ?", imq

            # Check if the q star is correct
            if nq_new != nq and imq != 0:
                print ("Reciprocal lattice vectors:")
                print (self.QE_bg.transpose() )
                print ("Passed q star:")
                print (q_point_group)
                print ("QE q star:")
                print (sxq[:, :nq_new].transpose())
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
                              nq_new, sxq, isq, imq, nq, self.QE_nat)
            #print "Fake"
            
            #print "XQ:", q_point_group[i, :], "NQ_NEW:", nq_new

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
                for yq in range(nq):
                    real_y = yq
                    dot_f = 1
                    if imq == 0 and yq >= nq_new:
                        real_y -= nq_new
                        dot_f = -1
                    if Methods.get_min_dist_into_cell(self.QE_bg.transpose(), dot_f* sxq[:, real_y], current_q[xq,:]) < __EPSILON__: 
                        sorting_q[xq] = yq
                        count += 1
                
                if count != 1:
                    print ("Original star:")
                    print (q_point_group)
                    print ("Reshaped star:")
                    print (current_q)
                    print ("Reciprocal lattice vectors:")
                    print (self.QE_bg.transpose() )
                    print ("STAR:")
                    print (sxq[:, :nq_new].transpose()    )
                    pta = (current_q[xq,:])
                    print ("Distances of xq in the QE star:")
                    for yq in range(nq_new):
                        print ("%.4f %.4f %.4f  => " % (sxq[0, yq], sxq[1, yq], sxq[2, yq]), Methods.get_min_dist_into_cell(self.QE_bg.transpose(), sxq[:, yq], current_q[xq,:]))
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
        
    def ApplySymmetryToMatrix(self, matrix, err = None):
        """
        Apply the symmetries to the 3x3 matrix.
        It can be a stress tensor, a dielectric tensor and so on.

        Parameters
        ----------
            matrix : a 3x3 matrix
                The matrix to which you want to apply the symmetrization.
                The matrix is overwritten with the output.
        """

        # Setup the symmetries in the Gamma point
        #self.SetupQPoint()

        # Perform the symmetrization
        mat_f = np.array(matrix, order = "F", dtype = np.float64)
    
        symph.symmatrix(mat_f, self.QE_s, self.QE_nsymq, self.QE_at, self.QE_bg)

        # To compute the error we count which element
        # of the stress tensor are summed togheter to obtain any element.
        # Then we propagate the error only on these.
        if err is not None:
            err_new = err.copy()
            for i in range(3):
                for j in range(3):
                    work = np.zeros( (3,3), dtype = np.float64, order = "F")
                    work[i,j] = np.float64(1)

                    # Apply the symmetry
                    symph.symmatrix(work, self.QE_s, self.QE_nsymq, self.QE_at, self.QE_bg)
                    mask = (np.abs(work) > __EPSILON__)
                    naverage = np.sum( mask.astype(int))

                    if naverage == 0:
                        err_new[i,j] = 0
                    else:
                        err_new[i,j] = np.sqrt(np.sum( err[mask]**2)) / naverage
            err[:,:] = err_new
        matrix[:,:] = mat_f

        
        
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
                print ("Symmetries in q = ", q_points[iq, :])
            t1 = time.time()
            self.SetupQPoint(q_points[iq,:], verbose)
            t2 = time.time()
            if verbose:
                print (" [SYMMETRIZEFCQ] Time to setup the q point %d" % iq, t2-t1, "s")
            
            # Proceed with the sum rule if we are at Gamma
            
            if asr == "simple" or asr == "custom":
                if np.sqrt(np.sum(q_points[iq,:]**2)) < __EPSILON__:
                    if verbose:
                        print ("q_point:", q_points[iq,:])
                        print ("Applying sum rule")
                    self.ImposeSumRule(fcq[iq,:,:], asr)
            elif asr == "crystal":
                self.ImposeSumRule(fcq[iq, :,:], asr = asr)
            elif asr == "no":
                pass
            else:
                raise ValueError("Error, only 'simple', 'crystal', 'custom' or 'no' asr are supported, given %s" % asr)
            
            t1 = time.time()
            if verbose:
                print (" [SYMMETRIZEFCQ] Time to apply the sum rule:", t1-t2, "s")
            
            # # Symmetrize the matrix
            if verbose:
                old_fcq = fcq[iq, :,:].copy()
                w_old = np.linalg.eigvals(fcq[iq, :, :])
                print ("FREQ BEFORE SYM:", w_old )
            self.SymmetrizeDynQ(fcq[iq, :,:], q_points[iq,:])
            t2 = time.time()
            if verbose:
                print (" [SYMMETRIZEFCQ] Time to symmetrize the %d dynamical matrix:" % iq, t2 -t1, "s" )
                print (" [SYMMETRIZEFCQ] Difference before the symmetrization:", np.sqrt(np.sum(np.abs(old_fcq - fcq[iq, :,:])**2)))
                w_new = np.linalg.eigvals(fcq[iq, :, :])
                print ("FREQ AFTER SYM:", w_new)

        # For each star perform the symmetrization over that star
        q0_index = 0
        for i in range(nqirr):
            q_len = len(q_stars[i])
            t1 = time.time()
            if verbose:
                print ("Applying the q star symmetrization on:")
                print (np.array(q_stars[i]))
            self.ApplyQStar(fcq[q0_index : q0_index + q_len, :,:], np.array(q_stars[i]))
            t2 = time.time()
            if verbose:
                print (" [SYMMETRIZEFCQ] Time to apply the star q_irr = %d:" % i, t2 - t1, "s")
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
        
        aq = np.zeros(3, dtype = np.float64)
        aq[:] = Methods.covariant_coordinates(self.QE_bg.transpose(), q_point)
        
        # Setup the bravais lattice
        symph.symm_base.set_at_bg(self.QE_at, self.QE_bg)
        
        # Prepare the symmetries
        symph.symm_base.set_sym_bl()
        
        if verbose:
            print ("Symmetries of the bravais lattice:", symph.symm_base.nrot)
        
        
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
            print ("Symmetries of the crystal:", symph.symm_base.nsym)
        
        
        
        # Now copy all the work initialized on the symmetries inside python
        self.QE_s = np.copy(symph.symm_base.s)
        self.QE_ft = np.copy(symph.symm_base.ft)
        
        
        # Prepare the symmetries of the small group of q
        syms = np.zeros( (48), dtype = np.intc)
        
        # Initialize to true the symmetry of the crystal
        syms[:symph.symm_base.nsym] = np.intc(1)
        
        self.QE_minus_q = symph.symm_base.smallg_q(aq, 0, syms)
        self.QE_nsymq = symph.symm_base.copy_sym(symph.symm_base.nsym, syms)
        self.QE_nsym = symph.symm_base.nsym
        
        
        # Recompute the inverses
        symph.symm_base.inverse_s()
        
        if verbose:
            print ("Symmetries of the small group of q:", self.QE_nsymq)
        
        # Assign symmetries
        self.QE_s = np.copy(symph.symm_base.s)
        self.QE_invs = np.copy(symph.symm_base.invs)
        self.QE_ft = np.copy(symph.symm_base.ft)
        self.QE_irt = np.copy(symph.symm_base.irt)

        #print np.shape(self.QE_irt)
        
        # Compute the additional shift caused by fractional translations
        self.QE_rtau = symph.sgam_ph_new(self.QE_at, self.QE_bg, symph.symm_base.nsym, self.QE_s, 
                                         self.QE_irt, self.QE_tau, self.QE_nat)
        
        lgamma = 0
        if np.sqrt(np.sum(q_point**2)) > 0.0001:
            lgamma = 1
        
#        self.QE_irotmq = symph.set_irotmq(q_point, self.QE_s, self.QE_nsymq,
#                                          self.QE_nsym, self.QE_minus_q, 
#                                          self.QE_bg, self.QE_at, lgamma)
        # If minus q check which is the symmetry
#        
        #syms = self.GetSymmetries()
        self.QE_irotmq = 0
        if self.QE_minus_q:
            # Fix in the Same BZ
            #aq = aq - np.floor(aq)
            
                
            #print "VECTOR AQ:", aq
            
            # Get the first symmetry: 
            for k in range(self.QE_nsym):
                # Skip the identity
                #if k == 0:
                #    continue
                
                # Position feels the symmetries with S (fortran S is transposed)
                # While q vector feels the symmetries with S^t (so no .T required for fortran matrix)
                new_q = self.QE_s[:,:, k].dot(aq)
                # Compare new_q with aq
                dmin = Methods.get_min_dist_into_cell(np.eye(3), -new_q, aq)
                #print "Applying %d sym we transform " % (k+1), aq, "into", new_q, "dmin:", dmin
                #print "Vector in cart: ", q_point, "We used symmetry:" 
                #print self.QE_s[:, :, k]
                #print ""
                #dmin = np.sqrt(np.sum( ((new_q + aq) % 1)**2))
#            
#                print "Symmetry number ", k+1
#                print sym[:, :3]
#                print "q cryst:", aq
#                print "new_q_cryst:", new_q
#            
                #print "SYM NUMBER %d, NEWQ:" % (k+1), new_q
                #print "Distance:", dmin
                if  dmin < __EPSILON__:
                    #print "CORRECT FOR IROTMQ"
                    self.QE_irotmq = k + 1
                    break
            if self.QE_irotmq == 0:
                print ("Error, the fortran code tells me there is S so that Sq = -q + G")
                print ("But I did not find such a symmetry!")
                raise ValueError("Error in the symmetrization. See stdout")
                       
    def SetupFromSPGLIB(self):
        """
        USE SPGLIB TO SETUP THE SYMMETRIZATION
        ======================================

        This function uses spglib to find symmetries, recognize the supercell
        and setup all the variables to perform the symmetrization inside the supercell.

        NOTE: If spglib cannot be imported, an ImportError will be raised
        """
        if not __SPGLIB__:
            raise ImportError("Error, this function works only if spglib is available")

        # Get the symmetries
        spg_syms = spglib.get_symmetry(self.structure.get_ase_atoms(), symprec = self.threshold)
        symmetries = GetSymmetriesFromSPGLIB(spg_syms, regolarize= False)

        trans_irt = 0
        self.QE_s[:,:,:] = 0


        # Check how many point group symmetries do we have
        n_syms = 0
        for i, sym in enumerate(symmetries):
            # Extract the rotation and the fractional translation
            rot = sym[:,:3]

            # Check if the rotation is equal to the first one
            if np.sum( (rot - symmetries[0][:,:3])**2 ) < 0.1 and n_syms == 0 and i > 0:
                # We got all the rotations
                n_syms = i 
                break
                
            # Extract the point group
            if n_syms == 0:
                self.QE_s[:,:, i] = rot.T

                # Get the IRT (Atoms mapping using symmetries)
                irt = GetIRT(self.structure, sym)
                self.QE_irt[i, :] = irt + 1 #Py to Fort

        
        if n_syms == 0:
            n_syms = len(symmetries)
        
        # From the point group symmetries, get the supercell
        n_supercell = len(symmetries) // n_syms
        self.QE_translation_nr = n_supercell
        self.QE_nsymq = n_syms
        self.QE_nsym = n_syms

        self.QE_translations_irt = np.zeros( (self.structure.N_atoms, n_supercell), dtype = np.intc, order = "F")
        self.QE_translations = np.zeros( (3, n_supercell), dtype = np.double, order = "F")

        # Now extract the translations
        for i in range(n_supercell):
            sym = symmetries[i * n_syms]
            # Check if the symmetries are correctly setup

            I = np.eye(3)
            ERROR_MSG="""
            Error, symmetries are not correctly ordered.
            They must always start with the identity.

            N_syms = {}; N = {}; SYM = {}
            """.format(n_syms,i*n_syms, sym)
            assert np.sum( (I - sym[:,:3])**2) < 0.5, ERROR_MSG

            # Get the irt for the translation (and the translation)
            irt = GetIRT(self.structure, sym)
            self.QE_translations_irt[:, i] = irt + 1
            self.QE_translations[:, i] = sym[:,3]

        # For each symmetry operation, assign the inverse
        self.QE_invs[:] = get_invs(self.QE_s, self.QE_nsym)
        
                
            
    def ApplyTranslationsToVector(self, vector):
        """
        This subroutine applies the translations to the given vector.
        To be used only if the structure is a supercell structure
        and the symmetries have been initialized with SPGLIB

        Parameters
        ----------
            vector : size (nat, 3)
                A vector that must be symmetrized. It will be overwritten.
        """

        nat = self.QE_nat 

        assert vector.shape[0] == nat 
        assert vector.shape[1] == 3

        # Ignore if no translations are presents
        if self.QE_translation_nr <= 1:
            return

        sum_all = np.zeros((nat, 3), dtype =  type(vector[0,0]))

        for i in range(self.QE_translation_nr):
            n_supercell = np.shape(self.QE_translations_irt)[1]

            sum_all += vector[self.QE_translations_irt[:, i] - 1, :]
        sum_all /= self.QE_translation_nr
        vector[:,:] = sum_all



                
    def InitFromSymmetries(self, symmetries, q_point = np.array([0,0,0])):
        """
        This function initialize the QE symmetries from the symmetries expressed in the
        Cellconstructor format, i.e. a list of numpy array 3x4 where the last column is 
        the fractional translation.
        
        TODO: add the q_point preparation by limitng the symmetries only to 
              those that satisfies the specified q_point
        """
        
        nsym = len(symmetries)
        
        self.QE_nsymq = np.intc(nsym)
        self.QE_nsym = self.QE_nsymq
        
        
        for i, sym in enumerate(symmetries):
            self.QE_s[:,:, i] = np.transpose(sym[:, :3])
            
            # Get the atoms correspondence
            eq_atoms = GetIRT(self.structure, sym)
            
            self.QE_irt[i, :] = eq_atoms + 1
            
            # Get the inverse symmetry
            inv_sym = np.linalg.inv(sym[:, :3])
            for k, other_sym in enumerate(symmetries):
                if np.sum( (inv_sym - other_sym[:, :3])**2) < __EPSILON__:
                    break
            
            self.QE_invs[i] = k + 1
            
            # Setup the position after the symmetry application
            for k in range(self.QE_nat):
                self.QE_rtau[:, i, k] = self.structure.coords[eq_atoms[k], :].astype(np.float64)
        
        
        # Get the reciprocal lattice vectors
        b_vectors = self.structure.get_reciprocal_vectors()
        
        # Get the minus_q operation
        self.QE_minusq = False

        # NOTE: HERE THERE COULD BE A BUG
        
        # q != -q
        # Get the q vectors in crystal coordinates
        q = Methods.covariant_coordinates(b_vectors, q_point)
        for k, sym in enumerate(self.QE_s):
            new_q = self.QE_s[:,:, k].dot(q)
            if np.sum( (Methods.put_into_cell(b_vectors, -q_point) - new_q)**2) < __EPSILON__:
                self.QE_minus_q = True
                self.QE_irotmq = k + 1
                break
                
    def GetSymmetries(self, get_irt=False):
        """
        GET SYMMETRIES FROM QE
        ======================
        
        This method returns the symmetries in the CellConstructor format from
        the ones elaborated here.
        
        
        Parameters
        ----------
            get_irt : bool
                If true (default false) also the irt are returned. 
                They are the corrispondance between atoms for each symmetry operation.
        Results
        -------
            list :
                List of 3x4 ndarray representing all the symmetry operations
            irt : ndarray(size=(nsym, nat), dtype = int), optional
                Returned only if get_irt = True. 
                It is the corrispondance between atoms after the symmetry operation is applied.
                irt[x, y] is the atom mapped into y by the x symmetry. 
        """
        
        syms = []
        for i in range(self.QE_nsym):
            s_rot = np.zeros( (3, 4))
            s_rot[:, :3] = np.transpose(self.QE_s[:, :, i])
            s_rot[:, 3] = self.QE_ft[:, i]
            
            syms.append(s_rot)
        
        if not get_irt:
            return syms
        return syms, self.QE_irt[:self.QE_nsym, :].copy() - 1
            
        
        
    
    def SymmetrizeVector(self, vector):
        """
        SYMMETRIZE A VECTOR
        ===================
        
        This is the easier symmetrization of a generic vector.
        Note, fractional translation and generic translations are not imposed.
        This is because this simmetrization acts on displacements and forces.
        
        Parameters
        ----------
            vector : ndarray(natoms, 3)
                This is the vector to be symmetrized, it will be overwritten
                with the symmetrized version
        """

        # Apply Translations if any
        self.ApplyTranslationsToVector(vector)
        
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
        #xq = np.ones(3, dtype = np.float64)
        xq = np.array(q_point, dtype = np.float64)
        # print "XQ:", xq
        # print "XQ_CRYST:", Methods.covariant_coordinates(self.QE_bg.T, xq)
        # print "NSYMQ:", self.QE_nsymq, "NSYM:", self.QE_nsym
        # print "QE SYM:"
        # print np.einsum("abc->cba", self.QE_s[:, :, :self.QE_nsymq])
        # print "Other syms:"
        # print np.einsum("abc->cba", self.QE_s[:, :, self.QE_nsymq: self.QE_nsym])
        # print "QE INVS:"
        # print self.QE_invs[:self.QE_nsymq]
        # #print "QE RTAU:"
        # #print np.einsum("abc->bca", self.QE_rtau[:, :self.QE_nsymq, :])
        # print "IROTMQ:",  self.QE_irotmq
        # print "MINUS Q:", self.QE_minus_q
        # print "IRT:"
        # print self.QE_irt[:self.QE_nsymq, :]
        # print "NAT:", self.QE_nat

        # Inibhit minus q
        #self.QE_minus_q = 0
        
        
        # USE THE QE library to perform the symmetrization
        symph.symdynph_gq_new( xq, QE_dyn, self.QE_s, self.QE_invs, self.QE_rtau, 
                              self.QE_irt, self.QE_irotmq, self.QE_minus_q, self.QE_nsymq, self.QE_nat)
        
        # Return to cartesian coordinates
        for na in range(self.QE_nat):
            for nb in range(self.QE_nat):
                fc = QE_dyn[:, :, na, nb] 
                dyn_matrix[3 * na : 3* na + 3, 3*nb: 3 * nb + 3] = Methods.convert_matrix_cart_cryst(fc, self.structure.unit_cell, True)
                
    def GetQStar(self, q_vector):
        """
        GET THE Q STAR
        ==============

        Given a vector in q space, get the whole star.
        We use the quantum espresso subrouitine.

        Parameters
        ----------
            q_vector : ndarray(size= 3, dtype = np.float64)
                The q vector

        Results
        -------
            q_star : ndarray(size = (nq_star, 3), dtype = np.float64)
                The complete q star
        """
        self.SetupQPoint()
        nq_new, sxq, isq, imq = symph.star_q(q_vector, self.QE_at, self.QE_bg,
            self.QE_nsymq, self.QE_s, self.QE_invs, 0)
        
        #print ("STAR IMQ:", imq)
        if imq != 0:
            total_star = np.zeros( (nq_new, 3), dtype = np.float64)
        else:
            total_star = np.zeros( (2*nq_new, 3), dtype = np.float64)

        total_star[:nq_new, :] = sxq[:, :nq_new].transpose()

        if imq == 0:
            total_star[nq_new:, :] = -sxq[:, :nq_new].transpose()

        return total_star

    def SelectIrreducibleQ(self, q_vectors):
        """
        GET ONLY THE IRREDUCIBLE Q POINTS
        =================================

        This methods selects only the irreducible q points
        given a list of total q points for the structure.

        Parameters
        ----------
            q_vectors : list of q points
                The list of q points to be polished fromt he irreducible
        
        Results
        -------
            q_irr : list of q points
                The q_vectors without the copies by symmetry of the dynamical matrix.
        """

        qs = np.array(q_vectors)
        nq = np.shape(qs)[0]

        q_irr = [qs[x, :].copy() for x in range(nq)]
        for i in range(nq):
            if i >= len(q_irr):
                break
            
            q_stars = self.GetQStar(q_irr[i])
            n_star = np.shape(q_stars)[0]

            # Look if the list contains point in the star
            for j in range(n_star):
                q_in_star = q_stars[j,:]
                # Go reverse, in this way if we pop an element we do not have to worry about indices
                for k in range(len(q_irr)-1, i, -1):
                    if Methods.get_min_dist_into_cell(self.QE_bg.transpose(), q_in_star, q_irr[k]) < __EPSILON__:
                        q_irr.pop(k) # Delete the k element
        
        return q_irr

    def GetQIrr(self, supercell):
        """
        GET THE LIST OF IRREDUCIBLE Q POINTS
        ====================================

        This method returns a list of irreducible q points given the supercell size.

        Parameters
        ----------
            supercell : (X, Y, Z)  where XYZ are int
                The supercell size along each unit cell vector.
        
        Returns
        -------
            q_irr_list : list of q vectors
                The list of irreducible q points in the brilluin zone.
        """

        # Get all the q points
        q_points = GetQGrid(self.QE_at.T, supercell)

        # Delete the irreducible ones
        q_irr = self.SelectIrreducibleQ(q_points)

        return q_irr

    def ApplySymmetriesToV2(self, v2, apply_translations = True):
        """
        APPLY THE SYMMETRIES TO A 2-RANK TENSOR
        =======================================

        This subroutines applies the symmetries to a 2-rank
        tensor. Usefull to work with supercells.

        Parameters
        ----------
            v2 : ndarray (size = (3*nat, 3*nat), dtype = np.double)
                The 2-rank tensor to be symmetrized.
                It is directly modified
            apply_translation : bool
                If false pure translations are neglected.
        """

        # Apply the Permutation symmetry
        v2[:,:] = 0.5 * (v2 + v2.T)

        # First lets recall that the fortran subroutines
        # Takes the input as (3,3,nat,nat)
        new_v2 = np.zeros( (3,3, self.QE_nat, self.QE_nat), dtype = np.double, order ="F")
        for i in range(self.QE_nat):
            for j in range(self.QE_nat):
                new_v2[:, :, i, j] = v2[3*i : 3*(i+1), 3*j : 3*(j+1)]

        # Apply the translations
        if apply_translations:
            # Check that the translations have been setted up
            assert len(np.shape(self.QE_translations_irt)) == 2, "Error, symmetries not setted up to work in the supercell"
            symph.trans_v2(new_v2, self.QE_translations_irt)
        
        # Apply the symmetrization
        symph.sym_v2(new_v2, self.QE_at, self.QE_bg, self.QE_s, self.QE_irt, self.QE_nsym, self.QE_nat)

        # Return back
        for i in range(self.QE_nat):
            for j in range(self.QE_nat):
                v2[3*i : 3*(i+1), 3*j : 3*(j+1)] = new_v2[:, :, i, j]
        


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
        print ("Error, ITA group not yet implemented.")
        print ("You can download the symmetries for this group from the Bilbao Crystallographic Server")
        print ("And just add the %d.dat file into the SymData folder of the current program." % ita)
        print ("It should take less than five minutes.")
        
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


def GetSymmetriesFromSPGLIB(spglib_sym, regolarize = False):
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
    if not "translations" in spglib_sym:
        raise ValueError("Error, your symmetry dict has no 'translations' key.")
        
    if not "rotations" in spglib_sym:
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
    
    nat = np.shape(fc_matrix)[0] // 3
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
    

def ExcludeRotations(fc_matrix, structure):
    """
    APPLY THE ROTATION SUM RULE
    ===========================

    We exclude the rotations from the force constant matrix.

    Parameters
    ----------
       fc_matrix : ndarray(3*nat, 3*nat)
           The force constant matrix
       structure : Structure()
           The structure that is identified by the force constant matrix
   
    """

    nat = structure.N_atoms
    dtype = type(fc_matrix[0,0])
    
    # Get the center of the structure
    r_cm = np.sum(structure.coords, axis = 0) / nat
    r = structure.coords - r_cm
    
    v_rots = np.zeros((3, 3*nat), dtype = dtype)
    projector = np.eye(3*nat, dtype = dtype)
    counter = 0
    for i in range(3):
        for j in range(i+1,3):
            v = np.zeros(3*nat, dtype = dtype)
            v_i = r[:, j] 
            v_j = -r[:, i]

            v[3*np.arange(nat) + i] = v_i
            v[3*np.arange(nat) + j] = v_j

            
            # orthonormalize
            for k in range(counter):
                v -= v_rots[k, :].dot(v) * v_rots[k, :]

            # Normalize
            norm = np.sqrt(v.dot(v))
            v /= norm

            v_rots[counter, :] = v
            projector -= np.outer(v,v)
            counter += 1

    

    fc_matrix[:,:] = projector.dot(fc_matrix.dot(projector))
        

def GetIRT(structure, symmetry):
    """
    GET IRT
    =======
    
    Get the irt array. It is the array of the atom index that the symmetry operation
    swaps.
    
    the y-th element of the array (irt[y]) is the index of the original structure, while
    y is the index of the equivalent atom after the symmetry is applied.

    Parameters
    ----------
        structure: Structure.Structure()
            The unit cell structure
        symmetry: list of 3x4 matrices
            symmetries with frac translations
    
    """
    
    
    new_struct = structure.copy()
    new_struct.fix_coords_in_unit_cell()
    n_struct_2 = new_struct.copy()

    new_struct.apply_symmetry(symmetry, True)
    irt = np.array(new_struct.get_equivalent_atoms(n_struct_2), dtype =np.intc)
    return irt

def ApplySymmetryToVector(symmetry, vector, unit_cell, irt):
    """
    APPLY SYMMETRY
    ==============
    
    Apply the symmetry to the given vector of displacements.
    Translations are neglected.
    
    .. math::
        
        \\vec {v'}[irt] = S \\vec v
        
    
    Parameters
    ----------
        symmetry: ndarray(size = (3,4))
            The symmetry operation (crystalline coordinates)
        vector: ndarray(size = (nat, 3))
            The vector to which apply the symmetry.
            In cartesian coordinates
        unit_cell : ndarray( size = (3,3))
            The unit cell in which the structure is defined
        irt : ndarray(nat, dtype = int)
            The index of how the symmetry exchanges the atom.
        
    """
    
    # Get the vector in crystalline coordinate
    nat, dumb = np.shape(vector)
    work = np.zeros( (nat, 3))
    sym = symmetry[:, :3]
    
    for i in range(nat):
        # Pass to crystalline coordinates
        v1 = Methods.covariant_coordinates(unit_cell, vector[i, :])
        # Apply the symmetry
        w1 = sym.dot(v1)
        # Return in cartesian coordinates
        work[irt[i], :] = np.einsum("ab,a", unit_cell, w1)
    
    return work


def PrepareISOTROPYFindSymInput(structure, path_to_file = "findsym.in",
                                title = "Prepared with Cellconstructor",
                                latticeTolerance = 1e-5, atomicPositionTolerance = 0.001):
    """
    Prepare a FIND SYM input file
    =============================
    
    This method can be used to prepare a suitable input file for the ISOTROPY findsym program.
    
    Parameters
    ----------
        path_to_file : string
            A valid path to write the findsym input.
        title : string, optional
            The title of the job
    """
    
    lines = GetISOTROPYFindSymInput(structure, title, latticeTolerance, atomicPositionTolerance)
    
    fp = open(path_to_file, "w")
    fp.writelines(lines)
    fp.close()
    
    
def GetISOTROPYFindSymInput(structure, title = "Prepared with Cellconstructor",
                            latticeTolerance = 1e-5, atomicPositionTolerance = 0.001):
    """
    As the method PrepareISOTROPYFindSymInput, but the input is returned as a list of string (lines).
    
    """
    # Check if the structure has a unit cell
    if not structure.has_unit_cell:
        raise ValueError("Error, the given structure has not a valid unit cell.")
    
    # Prepare the standard input
    lines = []
    lines.append("!useKeyWords\n")
    lines.append("!title\n")
    lines.append(title + "\n")
    lines.append("!latticeTolerance\n")
    lines.append("%.8f\n" % latticeTolerance)
    lines.append("!atomicPositionTolerance\n")
    lines.append("%.8f\n" % atomicPositionTolerance)
    lines.append("!latticeBasisVectors\n")
    for i in range(3):
        lines.append("%16.8f %16.8f %16.8f\n" % (structure.unit_cell[i, 0],
                                                 structure.unit_cell[i, 1],
                                                 structure.unit_cell[i, 2]))
    
    lines.append("!atomCount\n")
    lines.append("%d\n" % structure.N_atoms)
    lines.append("!atomType\n")
    lines.append(" ".join(structure.atoms) + "\n")
    lines.append("!atomPosition\n")
    for i in range(structure.N_atoms):
        # Get the crystal coordinate
        new_vect = Methods.covariant_coordinates(structure.unit_cell, structure.coords[i, :])
        lines.append("%16.8f %16.8f %16.8f\n" % (new_vect[0],
                                                 new_vect[1],
                                                 new_vect[2]))
        
    return lines
    

def GetQGrid(unit_cell, supercell_size):
    """
    GET THE Q GRID
    ==============
    
    This method gives back a list of q points given the
    reciprocal lattice vectors and the supercell size.
    
    Parameters
    ----------
        unit_cell : ndarray(size=(3,3), dtype = np.float64)
            The unit cell, rows are the vectors
        supercell_size : ndarray(size=3, dtype = int)
            The dimension of the supercell along each unit cell vector.
    
    Returns
    -------
        q_list : list
            The list of q points, of type ndarray(size = 3, dtype = np.float64)
            
    """
    bg = Methods.get_reciprocal_vectors(unit_cell)

    n_vects = int(np.prod(supercell_size))
    q_final = np.zeros((3, n_vects), dtype = np.double, order = "F")
    q_final[:,:] = symph.get_q_grid(bg.T, supercell_size, n_vects)

    # Get the list of the closest vectors
    q_list = [Methods.get_closest_vector(bg, q_final[:, i]) for i in range(n_vects)]
    return q_list
    
def GetQGrid_old(unit_cell, supercell_size):
    """
    GET THE Q GRID
    ==============
    
    This method gives back a list of q points given the
    reciprocal lattice vectors and the supercell size.
    
    Parameters
    ----------
        unit_cell : ndarray(size=(3,3), dtype = np.float64)
            The unit cell, rows are the vectors
        supercell_size : ndarray(size=3, dtype = int)
            The dimension of the supercell along each unit cell vector.
    
    Returns
    -------
        q_list : list
            The list of q points, of type ndarray(size = 3, dtype = np.float64)
            
    """
    
    q_list = []
    # Get the recirpocal lattice vectors
    bg = Methods.get_reciprocal_vectors(unit_cell)
    
    # Get the supercell
    supercell = np.tile(supercell_size, (3, 1)).transpose() * unit_cell
    
    # Get the lattice vectors of the supercell
    bg_s = Methods.get_reciprocal_vectors(supercell)
    
    #print "SUPERCELL:", supercell_size
    
    for ix in range(supercell_size[0]):
        for iy in range(supercell_size[1]):
            for iz in range(supercell_size[2]):
                n_s = np.array( [ix, iy, iz], dtype = np.float64)
                q_vect = n_s.dot(bg_s)
                #q_vect = Methods.get_closest_vector(bg, q_vect)

                # Check if q is in the listcount = 0
                count = 0
                for q in q_list:
                    if Methods.get_min_dist_into_cell(bg, -q_vect, q) < __EPSILON__:
                        count += 1
                        break
                if count > 0:
                    continue

                # Add the q point
                q_list.append(q_vect)
                
                # Check if -q and q are different
                if Methods.get_min_dist_into_cell(bg, -q_vect, q_vect) > __EPSILON__:
                   q_list.append(-q_vect)
                

    
    return q_list
        


def CheckSupercellQ(unit_cell, supercell_size, q_list):
    """
    CHECK THE Q POINTS
    ==================
    
    This subroutine checks that the given q points of a dynamical matrix
    matches the desidered supercell. 
    It is usefull to spot bugs like the wrong definitions of alat units, 
    or error not spotted just by the number of q points (confusion between 1,2,2 or 2,1,2 supercell).
    
    Parameters
    ----------
        unit_cell : ndarray(size=(3,3), dtype = np.float64)
            The unit cell, rows are the vectors
        supercell_size : ndarray(size=3, dtype = int)
            The dimension of the supercell along each unit cell vector.
        q_list : list of vectors
            The total q point list.
    Returns
    -------
        is_ok : bool
            True => No error
            False => Error
    """
    # Get the q point list for the given supercell
    correct_q = GetQGrid(unit_cell, supercell_size)
    
    # Get the reciprocal lattice vectors
    bg = Methods.get_reciprocal_vectors(unit_cell)
    
    # Check if the vectors are equivalent or not
    for iq, q in enumerate(q_list):
        for jq, qnew in enumerate(correct_q):
            if Methods.get_min_dist_into_cell(bg, q, qnew) < __EPSILON__:
                correct_q.pop(jq)
                break
    
    if len(correct_q) > 0:
        print ("[CHECK SUPERCELL]")
        print (" MISSING Q ARE ")
        print ("\n".join([" q =%16.8f%16.8f%16.8f " % (q[0], q[1], q[2]) for q in correct_q]))
        return False
    return True    

def GetNewQFromUnitCell(old_cell, new_cell, old_qs):
    """
    GET NEW Q POINTS AFTER A CELL STRAIN
    ====================================
    
    This method returns the new q points after the unit cell is changed.
    Remember, when changing the cell to mantain the same kind (cubic, orthorombic, hexagonal...)
    otherwise the star identification will fail.
    
    The q point are passed (and returned) in cartesian coordinates.
    
    Parameters
    ----------
        structure : Structure.Structure()
            The structure to be changed (with the old unit celll)
        new_cell : ndarray(size=(3,3), dtype = np.float64)
            The new unit cell.
        old_qs : list of ndarray(size=3, dtype = np.float64)
            The list of q points to be converted
    
    Returns
    -------
        new_qs : list of ndarray(size=3, dtype = np.float64)
            The list of the new q points adapted in the new cell.
    """
    
    bg = Methods.get_reciprocal_vectors(old_cell) #/ (2 * np.pi)
    new_bg = Methods.get_reciprocal_vectors(new_cell)# / (2 * np.pi)
    
    new_qs = []
    for iq, q in enumerate(old_qs):
        # Get the q point in crystal coordinates
        new_qprime = Methods.covariant_coordinates(bg, q)
        
        # Convert the crystal coordinates in the new reciprocal lattice vectors
        new_q = np.einsum("ji, j", new_bg, new_qprime)
        new_qs.append(new_q)
    
    return new_qs

def GetSupercellFromQlist(q_list, unit_cell):
    """
    GET THE SUPERCELL FROM THE LIST OF Q POINTS
    ===========================================

    This method returns the supercell size from the list of q points
    and the unit cell of the structure.

    Parameters
    ----------
        q_list : list 
            List of the q points in cartesian coordinates
        unit_cell : ndarray(3,3)
            Unit cell of the structure (rows are the unit cell vectors)
    
    Results
    -------
        supercell_size : list of 3 integers
            The supercell dimension along each unit cell vector.
    """

    # Get the bravais lattice
    bg = Methods.get_reciprocal_vectors(unit_cell) 

    # Convert the q points in crystalline units
    supercell = [1,1,1]

    for q in q_list:
        qprime = Methods.covariant_coordinates(bg, q)
        qprime -= np.floor(qprime)
        qprime[np.abs(qprime) < __EPSILON__] = 1

        rmax = 1/np.abs(qprime)
        for j in range(3):
            if supercell[j] < int(rmax[j] + .5):
                supercell[j] = int(rmax[j] + .5)
    
    return supercell


# def GetSymmetriesOnModes(symmetries, structure, pol_vects):
#         """
#         GET SYMMETRIES ON MODES
#         =======================

#         This methods returns a set of symmetry matrices that explains how polarization vectors interacts between them
#         through any symmetry operation.

#         Parameters
#         ----------
#             symmetries : list 
#                The list of 3x4 matrices representing the symmetries.
#             structure : Structure.Structure()
#                The structure (supercell) to allow the symmetry to correctly identify the atoms that transforms one
#                in each other.
#             pol_vects : ndarray(size = (n_dim, n_modes))
#                The array of the polarization vectors (must be real)


#         Results
#         -------
#             pol_symmetries : ndarray( size=(n_sym, n_modes, n_modes))
#                The symmetry operation between the modes. This allow to identify which mode
#                will be degenerate, and which will not interact.
#         """

#         # Get the vector of the displacement in the polarization
#         m = np.tile(structure.get_masses_array(), (3,1)).T.ravel()
#         disp_v = np.einsum("im,i->mi", pol_vects, np.sqrt(m))

#         n_dim, n_modes = np.shape(pol_vects)

#         n_sym = len(symmetries)
#         nat = structure.N_atoms
        
#         # For each symmetry operation apply the
#         pol_symmetries = np.zeros((n_sym, n_modes, n_modes), dtype = np.float64)
#         for i, sym_mat in enumerate(symmetries):
#             irt = GetIRT(structure, sym_mat)
            
#             for j in range(n_modes):
#                 # Apply the i-th symmetry to the j-th mode
#                 new_vector = ApplySymmetryToVector(sym_mat, disp_v[j, :].reshape((nat, 3)), structure.unit_cell, irt).ravel()
#                 new_coords = Methods.covariant_coordinates(disp_v, new_vector)
#                 pol_symmetries[i, j, :] = new_coords

#         return pol_symmetries 


def GetSymmetriesOnModes(symmetries, structure, pol_vects):
        """
        GET SYMMETRIES ON MODES
        =======================

        This methods returns a set of symmetry matrices that explains how polarization vectors interacts between them
        through any symmetry operation.

        Parameters
        ----------
            symmetries : list 
               The list of 3x4 matrices representing the symmetries.
            structure : Structure.Structure()
               The structure (supercell) to allow the symmetry to correctly identify the atoms that transforms one
               in each other.
            pol_vects : ndarray(size = (n_dim, n_modes))
               The array of the polarization vectors (must be real)


        Results
        -------
            pol_symmetries : ndarray( size=(n_sym, n_modes, n_modes))
               The symmetry operation between the modes. This allow to identify which mode
               will be degenerate, and which will not interact.
        """

        # Get the vector of the displacement in the polarization
        m = np.tile(structure.get_masses_array(), (3,1)).T.ravel()
        disp_v = np.einsum("im,i->mi", pol_vects, 1 / np.sqrt(m))
        underdisp_v = np.einsum("im,i->mi", pol_vects, np.sqrt(m))

        n_dim, n_modes = np.shape(pol_vects)

        n_sym = len(symmetries)
        nat = structure.N_atoms
        
        # For each symmetry operation apply the
        pol_symmetries = np.zeros((n_sym, n_modes, n_modes), dtype = np.float64)
        for i, sym_mat in enumerate(symmetries):
            irt = GetIRT(structure, sym_mat)
            
            for j in range(n_modes):
                # Apply the i-th symmetry to the j-th mode
                new_vector = ApplySymmetryToVector(sym_mat, disp_v[j, :].reshape((nat, 3)), structure.unit_cell, irt).ravel()
                pol_symmetries[i, :, j] = underdisp_v.dot(new_vector.ravel())

        return pol_symmetries
        

def get_degeneracies(w):
    """
    GET THE SUBSPACES OF DEGENERACIES
    =================================

    From the given frequencies, for each mode returns a list of the indices of the modes of degeneracies.

    Parameters
    ----------
        w : ndarray(n_modes)   
            Frequencies
    
    Results
    -------
        deg_list : list of lists
            A list that contains, for each mode, the list of the modes (indices) that are degenerate with the latter one
    """


    n_modes = len(w)

    ret_list = []
    for i in range(n_modes):
        deg_list = np.arange(n_modes)[np.abs(w - w[i]) < 1e-8]
        ret_list.append(deg_list)
    return ret_list

def get_diagonal_symmetry_polarization_vectors(pol_sc, w, pol_symmetries):
    """
    GET THE POLARIZATION VECTORS THAT DIAGONALIZES THE SYMMETRIES
    =============================================================

    This function is very usefull to have a complex basis in which the application of symmetries
    is trivial.

    In this basis, each symmetry is diagonal.
    Indeed this forces the polarization vectors to be complex in the most general case.

    NOTE: To be tested, do not use for production run
    It seems to be impossible to correctly decompose simmetries when we have multiple rotations.

    If the symmetries are not unitary, an exception will be raised.

    Parameters
    ----------
        pol_sc : ndarray(3*nat, n_modes)
            The polarizaiton vectors in the supercell (obtained by DiagonalizeSupercell of the Phonon class)
        w : ndarray(n_modes)
            The frequency for each polarization vectors
        pol_symmetries : ndarray(N_sym, n_modes, n_modes)
            The Symmetry operator that acts on the polarization vector


    Results
    -------
        pol_vects : ndarray(3*nat, n_modes)
            The new (complex) polarization vectors that diagonalizes all the symmetries.
        syms_values : ndarray(n_modes, n_sym)
            The (complex) unitary eigenvalues of each symmetry operation along the given mode.
    """
    raise NotImplementedError("Error, this subroutine has not been implemented.")

    # First we must get the degeneracies
    deg_list = get_degeneracies(w) 

    # Now perform the diagonalization on each degeneracies
    final_vectors = np.zeros( pol_sc.shape, dtype = np.complex128)
    final_vectors[:,:] = pol_sc.copy()

    n_modes = len(w)
    n_syms = pol_symmetries.shape[0]
    skip_list = []

    syms_values = np.zeros((n_modes, n_syms), dtype = np.complex128)

    print("All modes:")
    for i in range(n_modes):
        print("Mode {} = {} cm-1  => ".format(i, w[i] * RY_TO_CM), deg_list[i])

    print()
    for i in range(n_modes):
        if i in skip_list:
            continue

        # If we have no degeneracies, we can ignore it
        if len(deg_list[i]) == 1:
            continue 

        partial_modes = np.zeros((len(deg_list[i]), len(deg_list[i])), dtype = np.complex128)
        partial_modes[:,:] = np.eye(len(deg_list[i])) # identity matrix

        mask_final = np.array([x in deg_list[i] for x in range(n_modes)])

        # If we have degeneracies, lets diagonalize all the symmetries
        for i_sym in range(n_syms):
            skip_j = []
            diagonalized = False
            np.savetxt("sym_{}.dat".format(i_sym), pol_symmetries[i_sym, :,:])

            
            # Get the symmetry matrix in the mode space (this could generate a problem with masses)
            ps = pol_symmetries[i_sym, :, :]
            sym_mat_origin = ps[np.outer(mask_final, mask_final)].reshape((len(deg_list[i]), len(deg_list[i])))    

            for j_mode in deg_list[i]:
                if j_mode in skip_j:
                    continue 

                # Get the modes that can be still degenerate by symmetries
                mode_dna = syms_values[j_mode, : i_sym]

                # Avoid a bad error if i_sym = 0
                if len(mode_dna) > 0:
                    mode_space = [x for x in deg_list[i] if np.max(np.abs(syms_values[x, :i_sym] - mode_dna)) < 1e-3]
                else:
                    mode_space = [x for x in deg_list[i]]

                # The mask for the whole symmetry and the partial_modes
                mask_all = np.array([x in mode_space for x in np.arange(n_modes)])
                mask_partial_mode = np.array([x in mode_space for x in deg_list[i]])
                n_deg_new = np.sum(mask_all.astype(int))

                if len(mode_space) == 1:
                    continue

                p_modes_new = partial_modes[:, mask_partial_mode]

                
                print()
                print("SYMMETRY_INDEX:", i_sym)
                print("SHAPE sym_mat_origin:", sym_mat_origin.shape)
                print("MODES: {} | DEG: {}".format(mode_space, deg_list[i]))
                print("SHAPE P_MODES_NEW:", p_modes_new.shape)
                sym_mat = np.conj(p_modes_new.T).dot(sym_mat_origin.dot(p_modes_new))
                
                # Decompose in upper triangular (assures that eigenvectors are orthogonal)
                s_eigvals_mat, s_eigvects = scipy.linalg.schur(sym_mat, output = "complex")
                s_eigvals = np.diag(s_eigvals_mat)

                # Check if the s_eigvals confirm the unitary of sym_mat
                # TODO: Check if some mass must be accounted or not...
                print("SYM_MAT")
                print(sym_mat)
                print("Eigvals:")
                print(s_eigvals)
                print("Eigval_mat:")
                print(s_eigvals_mat)
                print("Eigvects:")
                print(s_eigvects)
                assert np.max(np.abs(np.abs(s_eigvals) - 1)) < 1e-5, "Error, it seems that the {}-th matrix is not a rotation.".format(i_sym).format(sym_mat)

                # Update the polarization vectors to account this diagonalization
                partial_modes[:, mask_partial_mode] = p_modes_new.dot(s_eigvects)

                # Add the symmetry character on the new eigen modes
                for k_i, k in enumerate(mode_space):
                    syms_values[k, i_sym] = s_eigvals[k_i]

                # Now add the modes analyzed up to know to the skip
                for x in mode_space:
                    skip_j.append(x)
                
                diagonalized = True


            # Now we diagonalized the space
            # Apply the symmetries if we did not perform the diagonalization
            if not diagonalized:
                # Get the symmetrized matrix in the partial mode list:
                sym_mat = np.conj(partial_modes.T).dot(sym_mat_origin.dot(partial_modes))

                # Check that it is diagonal
                s_eigvals = np.diag(sym_mat) 
                disp = sym_mat - np.diag( s_eigvals)
                if np.max(np.abs(disp)) > 1e-4:
                    print("Matrix {}:".format(i_sym))
                    print(sym_mat)
                    raise ValueError("Error, I expect the symmetry {} to be diagonal".format(i_sym))

                syms_values[k, i_sym] = s_eigvals[k_i]

                # Add the symmetry character on the new eigen modes
                for k_i, k in enumerate(deg_list[i]):
                    syms_values[k, i_sym] = s_eigvals[k_i]
                

        # Now we solved our polarization vectors, add them to the final ones
        final_vectors[:, mask_final] = pol_sc[:, mask_final].dot(partial_modes)       

        # Do not further process the modes we used in this iteration
        for mode in deg_list[i]:
            skip_list.append(mode)


    return final_vectors, syms_values





def GetQForEachMode(pols_sc, unit_cell_structure, supercell_structure, \
    supercell_size, crystal = True):
    """
    GET THE Q VECTOR
    ================

    For each polarization mode in the supercell computes the 
    corresponding q vector.

    Indeed the polarization vector will be a have components both at q and at -q.

    If a polarization vector mixes two q an error will be raised.

    NOTE: use DiagonalizeSupercell of Phonons to avoid mixing q.


    Parameters
    ----------
        pols_sc : ndarray ( size = (3*nat_sc, n_modes), dtype = np.float64)
            The polarization vector of the supercell (real)
        unit_cell_structure : Structure()
            The structure in the unit cell
        supercell_structure: Structure()
            The structure in the super cell
        supercell_size : list of 3 int
            The supercell 
        crystal : bool
            If True, q points are returned in cristal coordinates.
            

    Results
    -------
        q_list : ndarray(size = (n_modes, 3), dtype = np.float, order = "C")
            The list of q points associated with each polarization mode.
            If crystal is true, they will be in crystal coordinates.
    """

    # Check the supercell
    n_cell = np.prod(supercell_size)

    nat = unit_cell_structure.N_atoms
    nat_sc = np.shape(pols_sc)[0] / 3
    n_modes = np.shape(pols_sc)[1] 

    ERR_MSG = """
    Error, the supercell {} is not commensurate with the polarization vector given.
    nat = {}, nat_sc = {}
    """
    assert n_cell * nat == nat_sc, ERR_MSG.format(supercell_size, nat, nat_sc)
    assert nat_sc == supercell_structure.N_atoms

    # Get the reciprocal lattice
    bg = Methods.get_reciprocal_vectors(unit_cell_structure.unit_cell) / (2 * np.pi)

    # Get the possible Q list
    q_grid = GetQGrid(unit_cell_structure.unit_cell, supercell_size)

    # Allocate the output variable
    q_list = np.zeros( (n_modes, 3), dtype = np.double, order = "C")

    # Get the correspondance between the unit cell and the super cell atoms
    itau = supercell_structure.get_itau(unit_cell_structure) - 1 #Fort2Py

    # Get the translational vectors
    R_vects = np.zeros( (nat_sc, 3), dtype = np.double)
    for i in range(nat_sc):
        R_vects[i, :] = unit_cell_structure.coords[itau[i],:] - supercell_structure.coords[i,:]
    
    R_vects = R_vects.ravel()
    __thr__ = 1e-6

    for imu in range(n_modes):
        pol_v = pols_sc[:, imu]

        nq = 0
        for q in q_grid:
            q_vec = np.tile(q, nat_sc)
            q_cos = np.cos(2*np.pi * q_vec * R_vects)
            q_cos /= np.sqrt(q_cos.dot(q_cos))
            q_sin = np.sin(2*np.pi * q_vec * R_vects)
            q_sin /= np.sqrt(q_cos.dot(q_cos))

            cos_proj = q_cos.dot(pol_v)
            sin_proj = q_sin.dot(pol_v)
            # Wrong, this select only a translational mode

            if np.abs(cos_proj**2 + sin_proj**2 -1) < __thr__:
                new_q = q
                if crystal:
                    new_q = Methods.covariant_coordinates(bg, q)
                q_list[imu, :] = new_q
                break
            elif cos_proj**2 + sin_proj**2 > __thr__:
                print (q_cos)
                ERROR_MSG = """
    Error, mixing between two |q|.
    Please provide polarization vectors that are well defined in |q|.
    This can be reached using the subroutine Phonons.Phonons.DiagonalizeSupercell.
    q = {}
    i_mode = {}

    cos_proj = {} | sin_proj = {}
    """
                raise ValueError(ERROR_MSG.format(q, imu, cos_proj, sin_proj))
            else:
                nq += 1

        
        # If we are here not q has been found
        if nq == len(q_grid):
            ERROR_MSG = """
    Error, the polarization vector {} cannot be identified!
    No q found in this supercell!
    """
            raise ValueError(ERROR_MSG.format(imu))


    return q_list

        
def ApplyTranslationsToSupercell(fc_matrix, super_cell_structure, supercell):
    """
    Impose the translational symmetry directly on the supercell
    matrix.

    Parameters
    ----------
        - fc_matrix : ndarray(size=(3*natsc, 3*natsc))
            The matrix in the supercell. In output will be
            modified
        - super_cell_structure : Structure()
            The structure of the super cell
        - supercell : (nx,ny,nz)
            The dimension of the supercell. 
    """

    natsc = super_cell_structure.N_atoms

    # Check the consistency of the passed options
    natsc3, _ = np.shape(fc_matrix)
    assert natsc == int(natsc3 / 3), "Error, wrong number of atoms in the supercell structure"
    assert natsc3 == _, "Error, the matrix passed has a wrong shape"
    assert natsc % np.prod(supercell) == 0, "Error, the given supercell is impossible with the number of atoms"

    # Fill the auxiliary matrix
    new_v2 = np.zeros( (3,3, natsc, natsc), dtype = np.double, order ="F")
    for i in range(natsc):
        for j in range(natsc):
            new_v2[:, :, i, j] = fc_matrix[3*i : 3*(i+1), 3*j : 3*(j+1)]


    # The number of translations
    n_trans = np.prod(supercell)
    trans_irt = np.zeros((natsc, n_trans), dtype = np.double, order = "F")

    # Setup the translational symmetries
    for nx in range(supercell[0]):
        for ny in range(supercell[1]):
            for nz in range(supercell[2]):
                # Build the translational symmetry
                symmat = np.zeros((3,4))
                symmat[:3,:3] = np.eye(3)
                symmat[:, 3] = np.array([nx, ny, nz], dtype = float) / np.array(supercell)


                nindex = supercell[2] * supercell[1] *nx 
                nindex += supercell[2] * ny 
                nindex += nz 

                # Get the IRT for this symmetry operation in the supercell
                trans_irt[:, nindex] = GetIRT(super_cell_structure, symmat) + 1 
                

                
    
    # Apply the translations
    symph.trans_v2(new_v2, trans_irt)

    # Return back to the fc_matrix
    for i in range(natsc):
        for j in range(natsc):
            fc_matrix[3*i : 3*(i+1), 3*j : 3*(j+1)] = new_v2[:, :, i, j]



def get_invs(QE_s, QE_nsym):
    """
    GET INVERSION SYMMETRY
    ======================

    For each symmetry operation, get an index that its inverse
    Note, the array must be in Fortran indexing (starts from 1)

    Parameters
    ----------
        QE_s : ndarray(size = (3,3,48), dtype = np.intc)
            The symmetries
        QE_nsym : int
            The number of symmetries    
    
    Results
    -------
        QE_invs : ndarray(size = 48, dtype = np.intc)
            The index of the inverse symmetry.
            In fortran indexing (1 => index 0)
    """
    QE_invs = np.zeros(48, dtype = np.intc)
    for i in range(QE_nsym):
        found = False
        for j in range(QE_nsym):
            if (QE_s[:,:,i].dot(QE_s[:,:,j]) == QE_s[:,:,0]).all():
                QE_invs[i] = j + 1 # Fortran index
                found = True
        
        if not found:
            warnings.warn("This is not a group, some features like Q star division may fail.")
            
    return QE_invs


def GetSymmetryMatrix(sym, structure, crystal = False):
    """
    GET THE SYMMETRY MATRIX
    =======================

    This subroutine converts the 3x4 symmetry matrix to a 3N x 3N matrix.
    It also transform the symmetry to be used directly in cartesian space.
    However, take care, it could be a very big matrix, so it is preverred to work with the small matrix,
    and maybe use a fortran wrapper if you want speed.

    NOTE: The passe structure must already satisfy the symmetry 

    Parameters
    ----------
        sym : ndarray(size = (3, 4))
            The symmetry and translations
        structure : CC.Structure.Structure()
            The structure on which the symmetry is applied (The structure must satisfy the symmetry already)
        crystal : bool
            If true, the symmetry is returned in crystal coordinate (default false)

    Results
    -------
        sym_mat : ndarray(size = (3*structure.N_atoms, 3*structure.N_atoms))
    """

    # Get the IRT array
    irt = GetIRT(structure, sym)

    nat = structure.N_atoms
    sym_mat = np.zeros((3 * nat, 3*nat), dtype = np.double)

    # Comvert the symmetry matrix in cartesian
    if not crystal:
        sym_cryst = Methods.convert_matrix_cart_cryst2(sym[:,:3], structure.unit_cell, cryst_to_cart = True)
    else:
        sym_cryst = sym[:,:3]

    # Correctly fill the atomic position of sym_mat
    for i in range(nat):
        i_irt = irt[i]
        sym_mat[3 * i_irt : 3*i_irt+3, 3*i : 3*i+ 3] = sym_cryst

    return sym_mat