from __future__ import print_function
from __future__ import division 
import Phonons
import Methods 
import symmetries

import numpy as np

"""
In this module we create a tensor class.
This is used to store high order tensors defined
in periodic bondary systems, interpolate and multiply between
them
"""


class GenericTensor:
    """
    This is the generic tensor class.
    Do not use this when programming.
    This should be used only to define new tensors.
    """

    def __init__(self, structure):
        """
        Define the tensors
        """
        self.structure = structure
        self.tensor = None
        self.r_vectors = None 


class Tensor2(GenericTensor):
    """
    This class defines the 2rank tensors, like force constant matrices.
    """

    def __init__(self, *args, **kwargs):
        GenericTensor.__init__(self, *args, **kwargs)
        

    def SetupFromPhonons(self, phonons):
        """
        SETUP FROM PHONONS
        ==================
        
        Setup the 2rank tensor from a phonons class

        Parameters
        ----------
            - phonons : Phonons.Phonons()
                The dynamical matrix from which you want to setup the tensor
        """

        self.structure = phonons.structure 

        nat = self.structure.N_atoms
        nat_sc = nat * np.prod(phonons.GetSupercell())

        # Prepare the tensor
        self.tensor = np.zeros( (nat, nat_sc, 3, 3), order = "C", dtype = np.double)
        self.r_vectors = np.zeros((nat, nat_sc, 3), order = "C", dtype = np.double)
        

        # Get the dynamical matrix in the supercell
        super_dyn = phonons.GenerateSupercellDyn(phonons.GetSupercell())

        for i in range(nat):
            for j in range(nat_sc):
                self.tensor[i, j, :, :] = super_dyn.dynmats[0][3*i: 3*i+3, 3*j:3*j+3]
                v_dist = super_dyn.structure.coords[i, :] - super_dyn.structure.coords[j,:]
                self.r_vectors[i, j, :] = Methods.get_closest_vector(super_dyn.structure.unit_cell, v_dist) 
        

    def GenerateSupercellTensor(self, supercell):
        """
        GENERATE SUPERCELL TENSOR
        =========================

        This function returns a tensor defined in the supercell
        filling to zero all the elemets that have a minimum distance
        greater than the one defined in the current tensor.
        This is the key to interpolate.

        The supercell atoms are defined using the generate_supercell
        method from the self.structure, so that is the link 
        between indices of the returned tensor and atoms in the supercell.

        Parameters
        ----------
            - supercell : (nx, ny, nz)
                The dimension of the supercell in which
                you want to compute the supercell tensor

        Results
        -------
            - tensor : ndarray(size = ( 3*natsc, 3*natsc))
                A tensor defined in the given supercell.
        """

        super_structure, itau = self.structure.generate_supercell(supercell, get_itau = True)

        nat_sc = super_structure.N_atoms
        new_tensor = np.zeros((3 * nat_sc, 3*nat_sc), dtype = np.double)


        nat, nat_sc_old, _ = np.shape(self.r_vectors) 
        
        for i in range(nat_sc):
            i_cell = itau[i] 
            for j in range(nat_sc):

                r_vector = super_structure.coords[i,:] - super_structure.coords[j,:]
                new_r_vector = Methods.get_closest_vector(super_structure.unit_cell, r_vector)

                # Now average all the values that 
                # share the same r vector
                dist_v = self.r_vectors[i_cell, :,:] - np.tile(new_r_vector, (nat_sc_old, 1))
                mask = [Methods.get_min_dist_into_cell(super_structure.unit_cell, dist_v[k, :], np.zeros(3)) < 1e-5 for k in range(nat_sc_old)]
                mask = np.array(mask)
                #mask = np.sqrt(np.einsum("ab, ab -> a", dist_v, dist_v)) < 1e-5


                # Apply the tensor
                n_elements1 = np.sum(mask.astype(int))
                n_elements2 = 0

                # if n_elements1 == 0:
                #     print("ZERO:")
                #     print("itau[{}] = {}".format(i, i_cell))
                #     print("r to find:", new_r_vector)
                #     print("r vectors:")
                #     for k in range(nat_sc_old):
                #         print("{}) {:12.6f} {:12.6f} {:12.6f}".format(k+1, *list(self.r_vectors[i_cell, k, :])))
                #     print()
                if n_elements1 > 0:
                    #print("Apply element {} {} | n = {}".format(i, j, n_elements1))
                    tens = np.sum(self.tensor[i_cell, mask, :, :], axis = 0) / n_elements1
                    #print(tens)
                    new_tensor[3*i: 3*i+3, 3*j:3*j+3] = tens 

                # NOTE: Here maybe a problem arising from the
                # double transpose inside the same unit cell
                # If the share a -1 with the vector then we found the transposed element
                if n_elements1 == 0:
                    dist_v2 = self.r_vectors[i_cell, :,:] + np.tile(new_r_vector, (nat_sc_old, 1))
                    mask2 = [Methods.get_min_dist_into_cell(super_structure.unit_cell, dist_v2[k, :], np.zeros(3)) < 1e-5 for k in range(nat_sc_old)]
                    mask2 = np.array(mask2)
                    n_elements2 = np.sum(mask2.astype(int))
                    if n_elements2 > 0:
                        tens = np.sum(self.tensor[i_cell, mask, :, :], axis = 0) / n_elements2
                        new_tensor[3*j:3*j+3, 3*i:3*i+3] = tens

                
                #print("Elements {}, {} | r_vector = {} | n1 = {} | n2 = {}".format(i+1, j+1, r_vector, n_elements1, n_elements2))

        return new_tensor

    def GeneratePhonons(self, supercell):
        """
        GENERATE PHONONS
        ================

        Interpolate the Tensor2 into a supercell and then
        transform back into the dynamical matrix with the correct q.

        It might be that the new dynamical matrix should be symmetrized.
        

        Parameters
        ----------
            - supercell : (nx, ny, nz)
                The supercell of the dynamical matrix
        
        Results
        -------
            - dynmat : Phonons.Phonons()
                The dynamical matrix interpolated into the new supercell.
                It is defined in the unit cell.
        """

        # Prepare the phonons for this supercell
        dynmat = Phonons.Phonons(self.structure)

        # Prepare the q_points
        dynmat.q_tot = symmetries.GetQGrid(self.structure.unit_cell, supercell)
        dynmat.q_stars = [dynmat.q_tot]
        dynmat.dynmats = []

        # Compute the force constant matrix in the supercell
        fc_matrix = self.GenerateSupercellTensor(supercell)

        super_structure = self.structure.generate_supercell(supercell)

        # Get the list of dynamical matrices for each q point
        dynq = Phonons.GetDynQFromFCSupercell(fc_matrix, np.array(dynmat.q_tot), self.structure, super_structure)

        for iq, q in enumerate(dynmat.q_tot):
            dynmat.dynmats.append(dynq[iq,:,:])

        # Adjust the q star according to symmetries
        dynmat.AdjustQStar()

        return dynmat

        