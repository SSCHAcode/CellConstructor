from __future__ import print_function
from __future__ import division 
import Phonons
import Methods 
import symmetries

import numpy as np
import scipy, scipy.signal, scipy.interpolate

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
        self.distances = None
        self.supercell = structure.unit_cell.copy()


        

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

        # Get the dynamical matrix in the supercell
        super_dyn = phonons.GenerateSupercellDyn(phonons.GetSupercell())

        # Setup from the supercell dynamical matrix
        self.SetupFromTensor(super_dyn.dynmats[0], super_dyn.structure)


    def SetupFromTensor(self, tensor, superstructure):
        """
        SETUP FROM A TENSOR
        ===================

        This module setup the tensor from a 3*natsc x 3*natsc matrix.
        You should also pass the structure in the supercell to infer
        which atom correspond to which one.

        NOTE: The first nat atoms of the superstructure must be a unit cell.

        Parameters
        ----------
            - tensor : ndarray( size=(3*nat_sc, 3*nat_sc))
                The matrix to be converted in this Tensor2.
            - superstructure : Structures.Structure()
                The structure in the supercell that define the tensor. 
                
        """

        nat = self.structure.N_atoms
        nat_sc = superstructure.N_atoms
        self.supercell = superstructure.unit_cell.copy()

        # Check if the passed structure is a good superstructure
        assert nat_sc % nat == 0, "Error, the given superstructure has a wrong number of atoms"

        natsc3, natsc3_ = np.shape(tensor)

        assert nat_sc == natsc3/3, "Error, the given tensor and superstructure are not compatible"

        assert natsc3 == natsc3_, "Error, the given tensor must be a square matrix"

        # Prepare the tensor
        self.tensor = np.zeros( (nat, nat_sc, 3, 3), order = "C", dtype = np.double)
        self.r_vectors = np.zeros((nat, nat_sc, 3), order = "C", dtype = np.double)
        self.distances = np.zeros((nat, nat_sc), dtype = np.double, order = "C")

        for i in range(nat):
            for j in range(nat_sc):
                self.tensor[i, j, :, :] = tensor[3*i: 3*i+3, 3*j:3*j+3]
                v_dist = superstructure.coords[i, :] - superstructure.coords[j,:]
                close_vect = Methods.get_closest_vector(superstructure.unit_cell, v_dist) 
                self.r_vectors[i, j, :] = close_vect

                # Store the distance corresponding to this matrix element
                self.distances[i, j] = np.sqrt(close_vect.dot(close_vect))

        self.multiplicity = np.ones( (nat, nat_sc), order= "C", dtype = np.double)


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

        # TODO: ADD THE MULTIPLICITY COUNT ON THE SUPERCELL

        super_structure, itau = self.structure.generate_supercell(supercell, get_itau = True)

        nat_sc = super_structure.N_atoms
        new_tensor = np.zeros((3 * nat_sc, 3*nat_sc), dtype = np.double)

        print("Unit cell coordinates:")
        print("\n".join(["{:3d}) {}".format(i, self.structure.coords[i, :]) for i in range(self.structure.N_atoms)]))
        print("Supercell coordinates:")
        print("\n".join(["{:3d}) {}".format(i, super_structure.coords[i, :]) for i in range(super_structure.N_atoms)]))



        nat, nat_sc_old, _ = np.shape(self.r_vectors) 
        
        for i in range(nat_sc):
            i_cell = itau[i] 
            for j in range(nat_sc):

                r_vector = super_structure.coords[i,:] - super_structure.coords[j,:]
                r_vector = Methods.get_closest_vector(super_structure.unit_cell, r_vector)

                # Now average all the values that 
                # share the same r vector 
                #dist_v = self.r_vectors[i_cell, :,:] - np.tile(new_r_vector, (nat_sc_old, 1))
                #mask = [Methods.get_min_dist_into_cell(super_structure.unit_cell, dist_v[k, :], np.zeros(3)) < 1e-5 for k in range(nat_sc_old)]
                #mask = np.array(mask)

                mask = Methods.get_equivalent_vectors(super_structure.unit_cell, self.r_vectors[i_cell, :, :], r_vector)

                if i == 4 and j == 11:
                    print("i = {}, j = {}".format(i, j))
                    print("r vector = {}".format(r_vector))
                    print("mask = {}".format(mask))

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
                    #dist_v2 = self.r_vectors[i_cell, :,:] + np.tile(new_r_vector, (nat_sc_old, 1))
                    mask2 = Methods.get_equivalent_vectors(super_structure.unit_cell, self.r_vectors[i_cell, :, :], -r_vector)
                    #mask2 = [Methods.get_min_dist_into_cell(super_structure.unit_cell, dist_v2[k, :], np.zeros(3)) < 1e-5 for k in range(nat_sc_old)]
                    #mask2 = np.array(mask2)
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


    def GetRDecay(self):
        """
        This function returns a 2d array ready to be plotted
        to show the localization of the tensor.

        It will have the distance between the indices of the tensor
        and the average mean square value of the tensor corresponding 
        to that distance.

        It is the absolute distance in a Tensor2, 
        the perimeter between the three elements in Tensor3 and so on.

        This function is general for any kind of tensor.

        Returns
        -------
            - distances : ndarray
                The list of the distances, sorted, between all the indices
                of the tensors.
            - mean_square : ndarray (size = len(distances))
                The mean squared value of the tensor over the
                corresponding distance.
        """

        tensor_magnitude = np.sqrt(np.einsum("...ab, ...ba", self.tensor, self.tensor))

        assert tensor_magnitude.shape == self.distances.shape

        # Get the unique radius
        real_r, counts = np.unique(self.distances, return_counts=True)

        # Compute the square average around it
        mean_square = np.zeros(len(real_r), dtype = np.double)
        for i,r in enumerate(real_r):
            mask = self.distances == r 
            mean_square[i] = np.sum( tensor_magnitude[mask]**2) / counts[i]
        mean_square = np.sqrt(mean_square)

        return real_r, mean_square


    def ApplyKaiserWindow(self, rmax, beta=14, N_sampling = 1000):
        """
        Apply a Kaiser-Bessel window to the signal. 
        This is the best tool to perform the interpolation.

        Each element of the tensor is multiplied by the kaiser
        function with the given parameters.
        The kaiser function is computed on the corresponding value of distance

        Parameters
        ----------
            - rmax : float
                The maximum distance on which the window is defined.
                All that is outside rmax is setted to 0
            - beta : float
                The shape of the Kaiser window.
                For beta = 0 the window is a rectangular function, 
                for beta = 14 it resample a gaussian. The higher beta, the
                narrower the window.
            - N_sampling : int
                The sampling of the kaiser window.
        """

        kaiser_data = scipy.signal.kaiser(N_sampling, beta)

        # Build the kaiser function
        r_value = np.linspace(-rmax, rmax, N_sampling)
        kaiser_function = scipy.interpolate.interp1d(r_value, kaiser_data, bounds_error=False, fill_value= 0)

        # Build the kaiser window
        kaiser_window = kaiser_function(self.distances)

        nat, nat_sc = np.shape(self.distances)

        # Apply the kaiser window on the tensor
        for i in range(nat):
            for j in range(nat_sc):
                self.tensor[i, j, :, :] *= kaiser_window[i,j]