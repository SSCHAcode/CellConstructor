from __future__ import print_function
from __future__ import division 

# Import for python2/3 compatibility
import cellconstructor.Phonons as Phonons
import cellconstructor.Methods as Methods 
import cellconstructor.symmetries as symmetries

import cellconstructor.Settings as Settings
from cellconstructor.Settings import ParallelPrint as print 


import numpy as np
import scipy, scipy.signal, scipy.interpolate

import symph
import time
import itertools
import thirdorder

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

    def __init__(self,  unitcell_structure, supercell_structure, supercell_size):
        """
        Define the tensors
        """
        self.supercell_size = supercell_size 
        self.itau = supercell_structure.get_itau(unitcell_structure) - 1
        
        self.tau=unitcell_structure.coords
        
        self.nat = unitcell_structure.N_atoms

        self.unitcell_structure = unitcell_structure
        self.supercell_structure = supercell_structure

        self.verbose = True







        

class Tensor2(GenericTensor):
    """
    This class defines the 2rank tensors, like force constant matrices.
    """

    def __init__(self, unitcell_structure, supercell_structure, supercell_size):
        GenericTensor.__init__(self, unitcell_structure, supercell_structure, supercell_size)
        
        self.n_R = np.prod(np.prod(supercell_size))
        self.x_r_vector2 = np.zeros((3, self.n_R), dtype = np.double, order = "F")
        self.r_vector2 = np.zeros((3, self.n_R), dtype = np.double, order = "F")
        self.tensor = np.zeros((self.n_R, 3*self.nat, 3*self.nat), dtype = np.double)


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
        # Get the dynamical matrix in the supercell
        super_dyn = phonons.GenerateSupercellDyn(phonons.GetSupercell())

        # Setup from the supercell dynamical matrix
        self.SetupFromTensor(super_dyn.dynmats[0])


    def SetupFromTensor(self, tensor):
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
                
        """

        nat = self.nat
        nat_sc = self.supercell_structure.N_atoms

        # Check if the passed structure is a good superstructure
        assert nat_sc % nat == 0, "Error, the given superstructure has a wrong number of atoms"

        natsc3, natsc3_ = np.shape(tensor)

        assert nat_sc == natsc3/3, "Error, the given tensor and superstructure are not compatible"

        assert natsc3 == natsc3_, "Error, the given tensor must be a square matrix"

        nat = self.unitcell_structure.N_atoms
        nat_sc = nat * self.n_R

        for i_x in range(self.supercell_size[0]):
            for i_y in range(self.supercell_size [1]):
                for i_z in range(self.supercell_size[2]):
                    i_block = self.supercell_size[1] * self.supercell_size[2] * i_x
                    i_block += self.supercell_size[2] * i_y 
                    i_block += i_z 
                    self.x_r_vector2[:, i_block] = np.array([i_x, i_y, i_z])
                    self.r_vector2[:, i_block] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2[:, i_block])

                    for na1 in range(nat):
                        for na2 in range(nat):
                            # Get the atom in the supercell corresponding to the one in the unit cell
                            na2_vect = self.unitcell_structure.coords[na2, :] + self.r_vector2[:, i_block]
                            nat2_sc = np.argmin( [np.sum( (self.supercell_structure.coords[k, :] - na2_vect)**2) for k in range(nat_sc)])
                            
                            self.tensor[i_block, 3*na1:3*na1+3, 3*na2: 3*na2+3] = tensor[3*na1 : 3*na1+3, 3*nat2_sc : 3*nat2_sc + 3]



    def Center(self, Far = 1, tol = 1.0e-5):
        """
        CENTERING
        =========

        This subroutine will center the second order force constant
        matrix. 
        For each atomic indices, it will be identified by the lowest perimeter
        between the replica of the atoms.

        This function should be called before performing the Fourier interpolation.
        """

        nq0 = self.supercell_size[0]
        nq1 = self.supercell_size[1]
        nq2 = self.supercell_size[2]
        n_sup = nq0*nq1*nq2 

        assert self.n_R == n_sup, "Error, maybe the tensor has already been centered?"
        assert self.tensor.shape[0] == n_sup, "Error, maybe the tensor has already been centered?"

        # The centering may require a lot of memory, it is safer to use only one processor
        if Settings.am_i_the_master():

            t1 = time.time()

            if self.verbose:
                print(" ")
                print(" ======================= Centering ==========================")
                print(" ")       
            


            # Analyze the centering to prepare the weights and the vectors that contains the centered elements
            weight,xR2 =thirdorder.second_order_centering.analysis(Far,
                                        nq0,nq1,nq2,tol, 
                                        self.unitcell_structure.unit_cell, self.tau, self.tensor,self.nat)

            # Prepare the blocks that contains the centered elements
            xR2_reshaped = np.reshape(xR2, (3, (2*Far+1)**3 * self.nat**2 * n_sup))
            
            # Pick only the unique vectors
            self.x_r_vector2 = np.unique(xR2_reshaped, axis = 1)
            self.n_R = self.x_r_vector2.shape[1]
            # Force the F ordering for a faster Fourier interpolation
            self.r_vector2 = np.zeros((3, self.n_R), dtype = np.double, order = "F")
            self.r_vector2[:,:] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2)

            # Now copy the centered tensor into the new tensor
            self.tensor = thirdorder.second_order_centering.center(self.tensor, weight, self.x_r_vector2, xR2, Far, self.nat, n_sup, self.n_R)

                        
            t2 = time.time() 
            if self.verbose:
                print("Time elapsed for computing the centering (2nd order): {} s".format(t2 - t1))
                print("Memory required by the centered matrix (2nd order): {} Gb".format(self.tensor.nbytes / 1024.**3))
                print()
            
        # Broadcast the centered tensor to all the processors
        self.tensor = Settings.broadcast(self.tensor)
        self.x_r_vector2 = Settings.broadcast(self.x_r_vector2)
        self.r_vector2 = Settings.broadcast(self.r_vector2)
        self.n_R = Settings.broadcast(self.n_R)

    def WriteOnFile(self, fname):
        """
        WRITE ON FILE
        =============

        Save the tensor on a file.
        This is usefull if you want to check if everything is working as you expect.

        The file format is the same as phono3py 
        In the first line there is the total number of blocks.
        Then for each block there is the lattice vector followed by the atomic index in the unit cell.
        Then there is the tensor for each cartesian coordinate
        
        As follows:

        N_blocks
        Block_index
        R_x R_y R_z
        at_1 at_2
        coord_1 coord_2 tensor_value
        coord_1 coord_2 tensor_value
        ...


        Parameters
        ----------
            fname : string
                Path to the file on which you want to save the tensor.
        """

        with open(fname, "w") as f:
            # Write the total number of blocks
            f.write("{:>5}\n".format(self.n_R * self.nat**2))

            i_block = 1
            for r_block in range(self.n_R):
                for nat1 in range(self.nat):
                    for nat2 in range(self.nat):
                        # Write the info on the current block
                        f.write("{:d}\n".format(i_block))
                        f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(*list(self.r_vector2[:, r_block])))
                        f.write("{:>6d} {:>6d}\n".format(nat1 + 1, nat2+1))
                        i_block += 1

                        # Write the tensor for the block
                        # For each cartesian coordinate
                        for x in range(3):
                            for y in range(3):
                                f.write("{:>2d} {:>2d} {:>20.10e}\n".format(x+1, y+1, self.tensor[r_block, 3*nat1 + x, 3*nat2 + y]))


    def Interpolate(self, q2, asr = True, verbose = False):
        """
        Perform the Fourier interpolation to obtain the force constant matrix at a given q
        This subroutine automatically performs the ASR

        Parameters
        ----------
            q2 : ndarray(size = 3) 
                The q vector in A^-1
            asr : bool
                If true, apply the acousitc sum rule
            verbose : bool
                Print some debugging info

        Results
        -------
            phi2 : ndarray(size = (3*nat, 3*nat), dtype = np.complex128)
                The second order force constant at q2. Atomic indices runs over the unit cell
        """

        final_fc = np.zeros((3*self.nat, 3*self.nat), dtype = np.complex128)

        for i in range(self.n_R):
            arg = 2 * np.pi * (q2.dot(self.r_vector2[:, i]))
            phase = np.exp(np.complex128(1j) * arg)
            final_fc += phase * self.tensor[i, :, :]

        # Apply the acoustic sum rule
        if asr:
            # Get the reciprocal lattice vectors
            bg = Methods.get_reciprocal_vectors(self.unitcell_structure.unit_cell) / (2*np.pi)
            nat = self.unitcell_structure.N_atoms

            # Create the ASR projector
            Q_proj = np.zeros((3*nat, 3*nat), dtype = np.double)
            for i in range(3):
                v1 = np.zeros(nat*3, dtype = np.double)
                v1[3*np.arange(nat) + i] = 1
                v1 /= np.sqrt(v1.dot(v1)) 

                Q_proj += np.outer(v1,v1) 

            # Lets pick the minimum and maximum lattice vectors
            xRmin = np.min(self.x_r_vector2, axis = 1)
            xRmax = np.max(self.x_r_vector2, axis = 1)

            # Now we obtain the total cell size that contains the ASR
            N_i = xRmax - xRmin + np.ones((3,), dtype = np.intc) 
            
            if verbose:
                print("Supercell WS:", N_i)

            # We check if they are even, in that case we add 1
            # f(q) is real only if we sum on odd cells
            for ik in range(3):
                if (N_i[ik] % 2 == 0):
                    N_i[ik] += 1
            
            if verbose:
                print("Supercell all odd:", N_i)

            
            # We compute the f(q) function
            at = self.unitcell_structure.unit_cell

            __tol__ = 1e-8
            f_q2i = np.ones(3, dtype = np.double)

            # We use mask to avoid division by 0,
            # As we know that the 0/0 limit is 1 in this case
            mask2 = np.abs(np.sin(at.dot(q2) * np.pi)) > __tol__
            f_q2i[mask2] = np.sin(N_i[mask2] * at[mask2,:].dot(q2) * np.pi) / (N_i[mask2] * np.sin(at[mask2,:].dot(q2) * np.pi))
            f_q2 = np.prod(f_q2i)

            if verbose:
                print("The fq:")
                print("{:16.8f} {:16.8f}".format(f_q2, f_q2))
                print("q1 = ", Methods.covariant_coordinates(bg * 2 * np.pi, -q2))
                print("q2 = ", Methods.covariant_coordinates(bg * 2 * np.pi, q2))

            # Now we can impose the acustic sum rule
            final_fc -= np.einsum("ai, bi-> ab", final_fc, Q_proj) * f_q2
            final_fc -= np.einsum("ib, ai-> ab", final_fc, Q_proj) * f_q2 


        return final_fc

    


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

    def GeneratePhonons(self, supercell, asr = True):
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
            - asr : bool
                If true, the ASR is imposed during the interpolation.
                This is the best way to correct the modes even close to gamma
        
        Results
        -------
            - dynmat : Phonons.Phonons()
                The dynamical matrix interpolated into the new supercell.
                It is defined in the unit cell.
        """

        # Prepare the phonons for this supercell
        dynmat = Phonons.Phonons(self.unitcell_structure)

        # Prepare the q_points
        dynmat.q_tot = symmetries.GetQGrid(self.unitcell_structure.unit_cell, supercell)
        q_vectors = [x.copy() for x in dynmat.q_tot]
        dynmat.q_stars = [q_vectors]
        dynmat.dynmats = []

        # Interpolate over the q points
        for i, q_vector in enumerate(q_vectors):
            dynq = self.Interpolate(q_vector, asr = asr)
            dynmat.dynmats.append(dynq)

        # Adjust the q star according to symmetries
        dynmat.AdjustQStar()

        return dynmat



    def GetRDecay(self):
        """
        Get a plot of the R decay.

        For each element of the block, plots the maximum intensity in the distance between the data
        """

        r_total = []
        max_intensity = []

        for i_R in range(self.n_R):
            # Get the distance for each atomic couple in the block
            for at_1 in range(self.nat):
                for at_2 in range(self.nat):
                    r_dist = self.r_vector2[:, i_R] + self.tau[at_2,:] - self.tau[at_1, :]
                    
                    tensor = self.tensor[i_R, 3*at_1 : 3*at_1 + 3, 3*at_2: 3*at_2 + 3]
                    intensity = np.sqrt(np.trace(tensor.dot(tensor.T)))

                    # Skip zero values
                    if intensity < 1e-10:
                        continue

                    r_mod = np.sqrt(r_dist.dot(r_dist))

                    if len(r_total) == 0:
                        r_total.append(r_mod)
                        max_intensity.append(intensity)
                        continue 

                    # Check if another vector with the same distance has already been found
                    distances = np.abs(r_mod - np.array(r_total))

                    if np.min(distances) < 1e-7:
                        # Compute the tensor intensity

                        index = np.argmin(distances)
                        if max_intensity[index] < intensity:
                            max_intensity[index] = intensity
                    else:
                        r_total.append(r_mod)
                        max_intensity.append(intensity)

        
        r_total = np.array(r_total)
        max_intensity = np.array(max_intensity)

        # Return the value sorted by R distance
        sort_mask = np.argsort(r_total) 
        return r_total[sort_mask], max_intensity[sort_mask]


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


                
                
                
                
                
                
# Third order force constant tensor
class Tensor3():
    """
    This class defines the 3rank tensors, like 3rd force constants.
    """

    def __init__(self, unitcell_structure, supercell_structure, supercell_size):
        #GenericTensor.__init__(self, *args, **kwargs)
        
        n_sup = np.prod(supercell_size)
        nat = unitcell_structure.N_atoms
        
        nat_sc= n_sup * nat
        
        self.n_sup = n_sup
        self.n_R = n_sup**2
        n_R = self.n_R
        self.nat = nat
        self.tensor = np.zeros( (n_R, 3*nat, 3*nat, 3*nat), dtype = np.double)
        
        self.supercell_size = supercell_size
        
        
        # Cartesian lattice vectors
        self.r_vector2 = np.zeros((3, n_R), dtype = np.double, order = "F")
        self.r_vector3 = np.zeros((3, n_R), dtype = np.double, order = "F")
        
        # Crystalline lattice vectors
        self.x_r_vector2 = np.zeros((3, n_R), dtype = np.intc, order = "F")
        self.x_r_vector3 = np.zeros((3, n_R), dtype = np.intc, order = "F")
        
        self.itau = supercell_structure.get_itau(unitcell_structure) - 1
        
        self.tau=unitcell_structure.coords
        
        
        self.unitcell_structure = unitcell_structure
        self.supercell_structure = supercell_structure

        self.verbose = True
        
        #self.lat_max=np.array([supercell_size[0]-1,supercell_size[1]-1,supercell_size[2]-1])
        #self.lat_min=np.array([0,0,0])
        
    def Setup(self, tensor):
        """
        Setup the third order force constant
        
        
        Parameters
        ----------
            unitcell_structure : Structure()
                The structure in the unit cell
            supercell_structure : Structure()
                The supercell structure on which the tensor has been computed
            supercell_size : truple
                The number of supercell along each lattice vector
            tensor : ndarray(size =(3*nat_sc, 3*nat_sc, 3*nat_sc, dtype = np.double)
                The third order tensor
        """
        
        
        n_sup = np.prod(self.supercell_size)
        nat = self.unitcell_structure.N_atoms
        supercell_size = self.supercell_size 
        
        
        nat_sc= n_sup * nat
        
        #self.n_R = n_sup**2
        n_R = self.n_R
        #self.nat = nat
        #self.tensor = np.zeros( (n_R, 3*nat, 3*nat, 3*nat), dtype = np.double)
        
        #self.supercell_size = supercell_size
        
        
        ## Cartesian lattice vectors
        #self.r_vector2 = np.zeros((3, n_R), dtype = np.double, order = "F")
        #self.r_vector3 = np.zeros((3, n_R), dtype = np.double, order = "F")
        
        ## Crystalline lattice vectors
        #self.x_r_vector2 = np.zeros((3, n_R), dtype = np.intc, order = "F")
        #self.x_r_vector3 = np.zeros((3, n_R), dtype = np.intc, order = "F")
        
        #self.itau = supercell_structure.get_itau(unitcell_structure) - 1
        
        #self.tau=unitcell_structure.coords
        
        
        #self.unitcell_structure = unitcell_structure
        supercell_structure = self.supercell_structure
        unitcell_structure = self.unitcell_structure

        for index_cell2 in range(n_sup):                
            n_cell_x2,n_cell_y2,n_cell_z2=Methods.one_to_three_len(index_cell2,v_min=[0,0,0],v_len=supercell_size)
            for index_cell3 in range(n_sup):
                n_cell_x3,n_cell_y3,n_cell_z3=Methods.one_to_three_len(index_cell3,v_min=[0,0,0],v_len=supercell_size)      
                #
                total_index_cell = index_cell3 + n_sup * index_cell2
                #
                self.x_r_vector2[:, total_index_cell] = (n_cell_x2, n_cell_y2, n_cell_z2)
                self.r_vector2[:, total_index_cell] = unitcell_structure.unit_cell.T.dot(self.x_r_vector2[:, total_index_cell])
                self.x_r_vector3[:, total_index_cell] =  n_cell_x3, n_cell_y3, n_cell_z3
                self.r_vector3[:, total_index_cell] = unitcell_structure.unit_cell.T.dot(self.x_r_vector3[:, total_index_cell])
                        
                        
                for na1 in range(nat):
                    for na2 in range(nat):
                        # Get the atom in the supercell corresponding to the one in the unit cell
                        na2_vect = unitcell_structure.coords[na2, :] + self.r_vector2[:, total_index_cell]
                        nat2_sc = np.argmin( [np.sum( (supercell_structure.coords[k, :] - na2_vect)**2) for k in range(nat_sc)])
                        
                        for na3 in range(nat):
                            # Get the atom in the supercell corresponding to the one in the unit cell
                            na3_vect = unitcell_structure.coords[na3, :] + self.r_vector3[:, total_index_cell]
                            nat3_sc = np.argmin( [np.sum( (supercell_structure.coords[k, :] - na3_vect)**2) for k in range(nat_sc)])
                            
                            self.tensor[total_index_cell, 
                                        3*na1 : 3*na1+3, 
                                        3*na2 : 3*na2+3, 
                                        3*na3 : 3*na3+3] = tensor[3*na1 : 3*na1 +3, 
                                                                  3*nat2_sc : 3*nat2_sc + 3,
                                                                  3*nat3_sc : 3*nat3_sc + 3]
    
    
    def WriteOnFile(self, filename):
        """
        """
        
    
        with open(filename, "w") as f:
            #TODD Print header...
            f.write("{:>5}\n".format(self.n_R * self.nat**3))
            
        
            i_block = 1
            for r_block  in range(self.n_R):
                for nat1 in range(self.nat):
                    for nat2 in range(self.nat):
                        for nat3 in range(self.nat):
                            f.write("{:d}\n".format(i_block))
                            f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(*list(self.r_vector2[:, r_block])))
                            f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(*list(self.r_vector3[:, r_block])))
                            f.write("{:>6d} {:>6d} {:>6d}\n".format(nat1+1, nat2+1, nat3+1))
                            i_block += 1
                            #
                            for xyz in range(27):
                                z = xyz % 3
                                y = (xyz %9)//3
                                x = xyz // 9
                                f.write("{:>2d} {:>2d} {:>2d} {:>20.10e}\n".format(x+1,y+1,z+1, self.tensor[r_block, 3*nat1 + x, 3*nat2 + y, 3*nat3 + z]))
       
# ========================== temporary =======================================================

    def WriteOnFile_new(self,filename,file_format='Phonopy'):
        """
        """
        
        if file_format == 'Phonopy':
            
            with open(filename, "w") as f:
            
                f.write("{:>5}\n".format(self.n_R * self.nat**3))
                
            
                i_block = 1
                for r_block  in range(self.n_R):
                    for nat1 in range(self.nat):
                        for nat2 in range(self.nat):
                            for nat3 in range(self.nat):
                                f.write("{:d}\n".format(i_block))
                                f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(*list(self.r_vector2[:, r_block])))
                                f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(*list(self.r_vector3[:, r_block])))
                                f.write("{:>6d} {:>6d} {:>6d}\n".format(nat1+1, nat2+1, nat3+1))
                                i_block += 1
                                #
                                for xyz in range(27):
                                    z = xyz % 3
                                    y = (xyz %9)//3
                                    x = xyz // 9
                                    f.write("{:>2d} {:>2d} {:>2d} {:>20.10e}\n".format(x+1,y+1,z+1, self.tensor[r_block, 3*nat1 + x, 3*nat2 + y, 3*nat3 + z]))            
        
        
        elif file_format == 'D3Q':
    
            with open(filename, "w") as f:
                #TODD Print header...

                for nat1 in range(self.nat):
                    for nat2 in range(self.nat):
                        for nat3 in range(self.nat):
                            for alpha in range(3):
                                for beta in range(3):
                                    for gamma in range(3):
                                        f.write("{:>6d} {:>6d} {:>6d} {:>6d} {:>6d} {:>6d}\n".format(alpha+1,beta+1,gamma+1,nat1+1, nat2+1, nat3+1))
                                        f.write("{:>5}\n".format(self.n_R))
                                        for r_block  in range(self.n_R):
                                            f.write("{:>6d} {:>6d} {:>6d} {:>6d} {:>6d} {:>6d} {:16.8e}\n".format(self.x_r_vector2[0, r_block],self.x_r_vector2[1, r_block],self.x_r_vector2[2, r_block],self.x_r_vector3[0, r_block],self.x_r_vector3[1, r_block],self.x_r_vector3[2, r_block], self.tensor[r_block, 3*nat1 + alpha, 3*nat2 + beta, 3*nat3 + gamma]))
                                            
     
     
     
     
    def ReadFromFile(self, filename):
        f = open(filename, "r")
        lines = [l.strip() for l in f.readlines()]
        f.close()
        
        n_blocks = int(lines[0])
        
        id_block = 0
        total_lat_vec = 0
        lat_vect_2 = 0
        lat_vect_3 = 0
        nat1 = 0
        nat2 = 0
        nat3 = 0
        reading_lat2 = False
        reading_lat3 = False 
        reading_atoms = False
        for i, line in enumerate(lines):
            if i == 0:
                continue
            
            data = line.split()
            if len(data) == 0:
                continue
            
            if len(data) == 1:
                id_block = int(data[0]) - 1
                total_lat_vec = id_block // self.nat**3
                nat_id = id_block % self.nat**3
                reading_lat2 = True
                continue
            
            if reading_lat2:
                self.r_vector2[:, total_lat_vec] = [float(x) for x in data]
                self.x_r_vector2[:, total_lat_vec] = Methods.covariant_coordinates(self.unitcell_structure.unit_cell, self.r_vector2[:, total_lat_vec])
                reading_lat2 = False
                reading_lat3 = True
            elif reading_lat3:
                self.r_vector3[:, total_lat_vec] = [float(x) for x in data]
                self.x_r_vector3[:, total_lat_vec] = Methods.covariant_coordinates(self.unitcell_structure.unit_cell, self.r_vector3[:, total_lat_vec])
                reading_lat3 = False
                reading_atoms = True
            elif reading_atoms:
                nat1, nat2, nat3 = [int(x) - 1 for x in data]
                reading_atoms = False
                print("Reading the vectors: ", self.r_vector2[:, total_lat_vec], self.r_vector3[:, total_lat_vec], total_lat_vec)
            
            if len(data) == 4:
                xx, yy, zz = [int(x)-1 for x in data[:3]]
                
                self.tensor[total_lat_vec, 3*nat1+xx, 3*nat2+yy, 3*nat3+zz] = np.double(data[-1])
            
            
    def Center(self, Far=1,tol=1.0e-5):
        """
        CENTERING 
        =========

        This subrouine will center the third order force constant inside the Wigner-Seitz supercell.
        This means that for each atomic indices in the tensor, it will be identified by the lowest 
        perimiter between the replica of the atoms.
        Moreover, in case of existance of other elements not included in the original supercell with
        the same perimeter, the tensor will be equally subdivided between equivalent triplets of atoms. 

        This function should be called before performing the Fourier interpolation.
        """    

        t1 = time.time()
        
        
        if Settings.am_i_the_master():
        
            if self.verbose:
                print(" ")
                print(" ======================= Centering ==========================")
                print(" ")         

            # The supercell total size
            #
            nq0=self.supercell_size[0]
            nq1=self.supercell_size[1]
            nq2=self.supercell_size[2]
            
            n_sup = np.prod(self.supercell_size)
            tensor_reshaped = np.transpose(self.tensor.reshape((n_sup, n_sup, 
                                                                3*self.nat, 3*self.nat, 3*self.nat)),axes=[2,3,4,0,1])
            alat=self.unitcell_structure.unit_cell
            
            weight,xR2,xR3 =thirdorder.third_order_centering.analysis(Far,
                                        nq0,nq1,nq2,tol, 
                                        self.unitcell_structure.unit_cell, self.tau, tensor_reshaped,self.nat)  
            
            
            xR2_reshaped=np.reshape(xR2,(3,(2*Far+1)*(2*Far+1)*(2*Far+1)*n_sup*self.nat*self.nat*self.nat*n_sup*n_sup))
            xR3_reshaped=np.reshape(xR3,(3,(2*Far+1)*(2*Far+1)*(2*Far+1)*n_sup*self.nat*self.nat*self.nat*n_sup*n_sup))
            xR23 = np.unique(np.vstack((xR2_reshaped,xR3_reshaped)),axis=1)
            
            self.n_R=xR23.shape[1]
            self.x_r_vector2,self.x_r_vector3 = np.vsplit(xR23,2)
            self.r_vector2=self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2)
            self.r_vector3=self.unitcell_structure.unit_cell.T.dot(self.x_r_vector3)
    
            centered=thirdorder.third_order_centering.center(tensor_reshaped,weight,
                                                             self.x_r_vector2,xR2,self.x_r_vector3,xR3,
                                                             Far,self.nat,n_sup,self.n_R)
            

            
            t2 = time.time() 
            
            self.tensor = np.transpose(centered, axes=[3,0,1,2])
  
            if self.verbose:               
                print("Time elapsed for computing the centering: {} s".format( t2 - t1)) 
                #print("Memory required for the not centered tensor {} Gb".format(self.tensor.nbytes / 1024.**3))
                print("Memory required for the centered tensor: {} Gb".format(centered.nbytes / 1024.**3))
                #print("Memory required after removing zero elements: {} Gb".format(self.tensor.nbytes / 1024.**3))
                print(" ")
                print(" ============================================================")

           
        
        self.tensor = Settings.broadcast(self.tensor)
        self.x_r_vector2 = Settings.broadcast(self.x_r_vector2)
        self.x_r_vector3 = Settings.broadcast(self.x_r_vector3)
        self.r_vector2 = Settings.broadcast(self.r_vector2)
        self.r_vector3 = Settings.broadcast(self.r_vector3)
        self.n_R = Settings.broadcast(self.n_R)
        self.n_sup = Settings.broadcast(self.n_sup)

    def ApplySumRule(self) :
        
        t1 = time.time()
        
        if Settings.am_i_the_master():
        
            if self.verbose:
                print(" ")
                print(" ======================= Imposing ASR ==========================")
                print(" ")         
            

            
            xR_list=np.unique(self.x_r_vector3, axis = 1)
            totnum_R=xR_list.shape[1]
          
            xRmin=np.min(xR_list,1) 
            xRmax=np.max(xR_list,1) 
            xRlen=xRmax-xRmin+np.ones((3,),dtype=int)
            n_Rnew=np.prod(xRlen)**2          
          
            if n_Rnew != self.n_R: 
                if self.verbose:
                    print("*********")
                    print("Enlarging")
                    print("*********")                    
                # reassignement
                xr2_old=self.x_r_vector2
                xr3_old=self.x_r_vector3
                #
                self.x_r_vector2=np.zeros((3,n_Rnew),dtype=np.int)
                self.x_r_vector3=np.zeros((3,n_Rnew),dtype=np.int)
                self.r_vector2=np.zeros((3,n_Rnew))
                self.r_vector3=np.zeros((3,n_Rnew))                
                for index_cell2 in range(np.prod(xRlen)):                
                    n_cell_x2,n_cell_y2,n_cell_z2=Methods.one_to_three_len(index_cell2,v_min=xRmin,v_len=xRlen)
                    for index_cell3 in range(np.prod(xRlen)):
                        n_cell_x3,n_cell_y3,n_cell_z3=Methods.one_to_three_len(index_cell3,v_min=xRmin,v_len=xRlen)      
                        #
                        total_index_cell = index_cell3 + np.prod(xRlen) * index_cell2
                        #
                        self.x_r_vector2[:, total_index_cell] = (n_cell_x2, n_cell_y2, n_cell_z2)
                        self.r_vector2[:, total_index_cell] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2[:, total_index_cell])
                        self.x_r_vector3[:, total_index_cell] =  n_cell_x3, n_cell_y3, n_cell_z3
                        self.r_vector3[:, total_index_cell] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector3[:, total_index_cell])                        
                ###                
                tensor_trnsp = np.transpose(self.tensor,axes=[3,2,1,0])
                tensor=thirdorder.third_order_asr.enlarge(tensor_trnsp,
                                                          self.x_r_vector2,self.x_r_vector3,
                                                          xr2_old,xr3_old,
                                                          self.nat,n_Rnew,self.n_R)
                      
            xR_list=np.unique(self.x_r_vector3, axis = 1)
            totnum_R=xR_list.shape[1]
          
            xRmin=np.min(xR_list,1) 
            xRmax=np.max(xR_list,1) 
            xRlen=xRmax-xRmin+np.ones((3,),dtype=int)
            self.n_R=np.prod(xRlen)**2          
           
          
            xRdiff_list=np.unique(self.x_r_vector2-self.x_r_vector3, axis = 1)
            totnum_Rdiff=xRdiff_list.shape[1]


            #phi_asr=thirdorder.third_order_asr.impose_asr(tensor,xR_list,xRdiff_list,self.x_r_vector2,self.x_r_vector3,totnum_Rdiff,self.n_R,totnum_R,self.nat)
            phi_asr=thirdorder.third_order_asr.impose_asr(tensor,xRlen,xR_list,xRdiff_list,
                                                          self.x_r_vector2,self.x_r_vector3,
                                                          totnum_Rdiff,self.n_R,totnum_R,self.nat)

            phi_asr=np.transpose(phi_asr, axes=[3,2,1,0])  

           # Select the element different from zero ########################
            tensor_block_nonzero = np.sum(phi_asr**2, axis = (1,2,3)) > 1e-8 
            
            self.tensor = phi_asr[tensor_block_nonzero, :, :, :]
            self.x_r_vector2 = self.x_r_vector2[:, tensor_block_nonzero]
            self.x_r_vector3 = self.x_r_vector3[:, tensor_block_nonzero]
            self.r_vector2 = self.r_vector2[:, tensor_block_nonzero]
            self.r_vector3 = self.r_vector3[:, tensor_block_nonzero]
            self.n_R = np.sum(tensor_block_nonzero.astype(int))
            t2 = time.time()
            if self.verbose:               
                print("Time elapsed for imposing the ASR: {} s".format( t2 - t1)) 
                print("Memory required for the ASR centered tensor: {} Gb".format(phi_asr.nbytes / 1024.**3))
                print("Memory required after removing zero elements: {} Gb".format(self.tensor.nbytes / 1024.**3))
                print(" ")
                print(" ============================================================")


 
        self.tensor = Settings.broadcast(self.tensor)
        self.x_r_vector2 = Settings.broadcast(self.x_r_vector2)
        self.x_r_vector3 = Settings.broadcast(self.x_r_vector3)
        self.r_vector2 = Settings.broadcast(self.r_vector2)
        self.r_vector3 = Settings.broadcast(self.r_vector3)
        self.n_R = Settings.broadcast(self.n_R)
        self.n_sup = Settings.broadcast(self.n_sup)



    def Center_and_ApplySumRule(self,Far=1,tol=1.0e-5) :
        
        t1 = time.time()
        
        if Settings.am_i_the_master():


            if self.verbose:
                print(" ")
                print(" ======================= Centering ==========================")
                print(" ")         

            # The supercell total size
            #
            nq0=self.supercell_size[0]
            nq1=self.supercell_size[1]
            nq2=self.supercell_size[2]
            
            n_sup = np.prod(self.supercell_size)
            tensor_reshaped = np.transpose(self.tensor.reshape((n_sup, n_sup, 
                                                                3*self.nat, 3*self.nat, 3*self.nat)),axes=[2,3,4,0,1])
            alat=self.unitcell_structure.unit_cell
            
            weight,xR2,xR3 =thirdorder.third_order_centering.analysis(Far,
                                        nq0,nq1,nq2,tol, 
                                        self.unitcell_structure.unit_cell, self.tau, tensor_reshaped,self.nat)  
            
            
            xR2_reshaped=np.reshape(xR2,(3,(2*Far+1)*(2*Far+1)*(2*Far+1)*n_sup*self.nat*self.nat*self.nat*n_sup*n_sup))
            xR3_reshaped=np.reshape(xR3,(3,(2*Far+1)*(2*Far+1)*(2*Far+1)*n_sup*self.nat*self.nat*self.nat*n_sup*n_sup))
            xR23 = np.unique(np.vstack((xR2_reshaped,xR3_reshaped)),axis=1)
            
            self.n_R=xR23.shape[1]
            self.x_r_vector2,self.x_r_vector3 = np.vsplit(xR23,2)
            self.r_vector2=self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2)
            self.r_vector3=self.unitcell_structure.unit_cell.T.dot(self.x_r_vector3)
    
            centered=thirdorder.third_order_centering.center(tensor_reshaped,weight,
                                                             self.x_r_vector2,xR2,self.x_r_vector3,xR3,
                                                             Far,self.nat,n_sup,self.n_R)
            
            t2 = time.time() 
            
            self.tensor = np.transpose(centered, axes=[3,0,1,2])
  
            if self.verbose:               
                print("Time elapsed for computing the centering: {} s".format( t2 - t1)) 
                print(" Memory required for the centered tensor: {} Gb".format(centered.nbytes / 1024.**3))

        
            if self.verbose:
                print(" ")
                print(" ======================= Imposing ASR ==========================")
                print(" ")         
            

            
            xR_list=np.unique(self.x_r_vector3, axis = 1)
            totnum_R=xR_list.shape[1]
          
            xRmin=np.min(xR_list,1) 
            xRmax=np.max(xR_list,1) 
            xRlen=xRmax-xRmin+np.ones((3,),dtype=int)
            n_Rnew=np.prod(xRlen)**2          
          
            if n_Rnew != self.n_R: 
                if self.verbose:
                    print("*********")
                    print("Enlarging")
                    print("*********")                    
                # reassignement
                xr2_old=self.x_r_vector2
                xr3_old=self.x_r_vector3
                #
                self.x_r_vector2=np.zeros((3,n_Rnew),dtype=int)
                self.x_r_vector3=np.zeros((3,n_Rnew),dtype=int)
                self.r_vector2=np.zeros((3,n_Rnew))
                self.r_vector3=np.zeros((3,n_Rnew))                
                for index_cell2 in range(np.prod(xRlen)):                
                    n_cell_x2,n_cell_y2,n_cell_z2=Methods.one_to_three_len(index_cell2,v_min=xRmin,v_len=xRlen)
                    for index_cell3 in range(np.prod(xRlen)):
                        n_cell_x3,n_cell_y3,n_cell_z3=Methods.one_to_three_len(index_cell3,v_min=xRmin,v_len=xRlen)      
                        #
                        total_index_cell = index_cell3 + np.prod(xRlen) * index_cell2
                        #
                        self.x_r_vector2[:, total_index_cell] = (n_cell_x2, n_cell_y2, n_cell_z2)
                        self.r_vector2[:, total_index_cell] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2[:, total_index_cell])
                        self.x_r_vector3[:, total_index_cell] =  n_cell_x3, n_cell_y3, n_cell_z3
                        self.r_vector3[:, total_index_cell] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector3[:, total_index_cell])                        
                ###                
                tensor_trnsp = np.transpose(self.tensor,axes=[3,2,1,0])
                tensor=thirdorder.third_order_asr.enlarge(tensor_trnsp,
                                                          self.x_r_vector2,self.x_r_vector3,
                                                          xr2_old,xr3_old,
                                                          self.nat,n_Rnew,self.n_R)
                      
            xR_list=np.unique(self.x_r_vector3, axis = 1)
            totnum_R=xR_list.shape[1]
          
            xRmin=np.min(xR_list,1) 
            xRmax=np.max(xR_list,1) 
            xRlen=xRmax-xRmin+np.ones((3,),dtype=int)
            self.n_R=np.prod(xRlen)**2          
           
          
            xRdiff_list=np.unique(self.x_r_vector2-self.x_r_vector3, axis = 1)
            totnum_Rdiff=xRdiff_list.shape[1]


            #phi_asr=thirdorder.third_order_asr.impose_asr(tensor,xR_list,xRdiff_list,self.x_r_vector2,self.x_r_vector3,totnum_Rdiff,self.n_R,totnum_R,self.nat)
            phi_asr=thirdorder.third_order_asr.impose_asr(tensor,xRlen,xR_list,xRdiff_list,
                                                          self.x_r_vector2,self.x_r_vector3,
                                                          totnum_Rdiff,self.n_R,totnum_R,self.nat)

            phi_asr=np.transpose(phi_asr, axes=[3,2,1,0])  

           # Select the element different from zero
            tensor_block_nonzero = np.sum(phi_asr**2, axis = (1,2,3)) > 1e-8
            
            self.tensor = phi_asr[tensor_block_nonzero, :, :, :]
            self.x_r_vector2 = self.x_r_vector2[:, tensor_block_nonzero]
            self.x_r_vector3 = self.x_r_vector3[:, tensor_block_nonzero]
            self.r_vector2 = self.r_vector2[:, tensor_block_nonzero]
            self.r_vector3 = self.r_vector3[:, tensor_block_nonzero]
            self.n_R = np.sum(tensor_block_nonzero.astype(int))
 
            t3 = time.time() 
            if self.verbose:               
                print(" Time elapsed for imposing the ASR: {} s".format( t3 - t2)) 
                print(" Memory required for the ASR centered tensor: {} Gb".format(phi_asr.nbytes / 1024.**3))
                print(" Memory required after removing zero elements: {} Gb".format(self.tensor.nbytes / 1024.**3))
                print(" ")
                print(" ============================================================")
 
        self.tensor = Settings.broadcast(self.tensor)
        self.x_r_vector2 = Settings.broadcast(self.x_r_vector2)
        self.x_r_vector3 = Settings.broadcast(self.x_r_vector3)
        self.r_vector2 = Settings.broadcast(self.r_vector2)
        self.r_vector3 = Settings.broadcast(self.r_vector3)
        self.n_R = Settings.broadcast(self.n_R)
        self.n_sup = Settings.broadcast(self.n_sup)

    def Interpolate(self, q2, q3, asr = True, verbose = False):
        """
        Interpolate the third order to the q2 and q3 points
        
        Parameters
        ----------
            q2, q3 : ndarray(3)
                The q points
            asr : bool
                If true, the Acoustic sum rule is applied directly in q space
            verbose : bool
                If true print debugging info
        
        Results
        -------
            Phi3 : ndarray(size = (3*nat, 3*nat, 3*nat))
                The third order force constant in the defined q points.
                atomic indices runs over the unit cell
        """
        
        final_fc = np.zeros((3*self.nat, 3*self.nat, 3*self.nat), 
                            dtype = np.complex128)
        for i in range(self.n_R):
            arg = 2  * np.pi * (q2.dot(self.r_vector2[:, i]) + 
                                q3.dot(self.r_vector3[:, i]))
            
            phase =  np.complex128(np.exp(-1j * arg))
            final_fc += phase * self.tensor[i, :, :, :]

        # Apply the acoustic sum rule if necessary
        if asr:
            # Get the reciprocal lattice vectors
            bg = Methods.get_reciprocal_vectors(self.unitcell_structure.unit_cell) / (2* np.pi)

            nat = self.unitcell_structure.N_atoms

            # Create the projector on the orthonormal space to the ASR
            Q_proj = np.zeros((3*nat, 3*nat), dtype = np.double)
            for i in range(3):
                v1 = np.zeros(nat*3, dtype = np.double)
                v1[3*np.arange(nat) + i] = 1
                v1 /= np.sqrt(v1.dot(v1))
                
                Q_proj += np.outer(v1, v1)
            
            # Get the N_i in the centered cell
            # First we get the list of vectors in crystal coordinates
            xR_list=np.unique(self.x_r_vector3, axis = 1)
          
            # We pick the minimum and maximum values of the lattice vectors
            # in crystal coordinates
            xRmin=np.min(xR_list, axis = 1)
            xRmax=np.max(xR_list, axis = 1)

            # Now we can obtain the dimension of the cell along each direction.
            N_i = xRmax-xRmin+np.ones((3,),dtype=int)

            if verbose:
                print("Centered supercell: ", N_i)

            # We check if they are even, in that case we add 1
            # f(q) is real only if we sum on odd cells
            for ik in range(3):
                if (N_i[ik] % 2 == 0):
                    N_i[ik] += 1
            
            if verbose:
                print("Supercell all odd:", N_i)

            
            # We compute the f(q) function
            at = self.unitcell_structure.unit_cell

            __tol__ = 1e-8
            f_q3i = np.ones(3, dtype = np.double)
            f_q2i = np.ones(3, dtype = np.double)
            f_q1i = np.ones(3, dtype = np.double)

            # We use mask to avoid division by 0,
            # As we know that the 0/0 limit is 1 in this case

            mask3 = np.abs(np.sin(at.dot(q3) * np.pi)) > __tol__ 
            f_q3i[mask3] = np.sin(N_i[mask3] * at[mask3,:].dot(q3) * np.pi) / (N_i[mask3] * np.sin(at[mask3,:].dot(q3) * np.pi))
            f_q3 = np.prod(f_q3i)

            mask2 = np.abs(np.sin(at.dot(q2) * np.pi)) > __tol__
            f_q2i[mask2] = np.sin(N_i[mask2] * at[mask2,:].dot(q2) * np.pi) / (N_i[mask2] * np.sin(at[mask2,:].dot(q2) * np.pi))
            f_q2 = np.prod(f_q2i)

            q1 = -q2 - q3
            mask1 = np.abs(np.sin(at.dot(q1) * np.pi)) > __tol__
            f_q1i[mask1] = np.sin(N_i[mask1] * at[mask1,:].dot(q1) * np.pi) / (N_i[mask1] * np.sin(at[mask1,:].dot(q1) * np.pi))
            f_q1 = np.prod(f_q1i)

            if verbose:
                print("The fq factors:")
                print("{:16.8f} {:16.8f} {:16.8f}".format(f_q1, f_q2, f_q3))
                print("q1 = ", Methods.covariant_coordinates(bg * 2 * np.pi, q1))
                print("q2 = ", Methods.covariant_coordinates(bg * 2 * np.pi, q2))
                print("q3 = ", Methods.covariant_coordinates(bg * 2 * np.pi, q3))

            # Now we can impose the acustic sum rule
            final_fc -= np.einsum("abi, ci-> abc", final_fc, Q_proj) * f_q3 
            final_fc -= np.einsum("aic, bi-> abc", final_fc, Q_proj) * f_q2
            final_fc -= np.einsum("ibc, ai-> abc", final_fc, Q_proj) * f_q1 



        
        return final_fc


# ============================================================================================================================================ 
# Tentative Sparse  ========================================================================================================================== 
# ============================================================================================================================================   
    
    def WriteOnFile_sparse(self, filename):
        """
        """
        
    
        with open(filename, "w") as f:
            #TODD Print header...
            f.write("{:>5}\n".format(self.n_R_sparse))
            
        
            for i_block  in range(self.n_R_sparse):
                            f.write("{:d}\n".format(i_block+1))
                            f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(*list(self.r_vector2_sparse[:, i_block])))
                            f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(*list(self.r_vector3_sparse[:, i_block])))
                                                        
                            nat=self.atom_sparse[:,i_block]                            
                            f.write("{:>6d} {:>6d} {:>6d}\n".format(nat[0]+1,nat[1]+1,nat[2]+1))
                        
                            #
                            for xyz in range(27):
                                z = xyz % 3
                                y = (xyz %9)//3
                                x = xyz // 9
                                f.write("{:>2d} {:>2d} {:>2d} {:>20.10e}\n".format(x+1,y+1,z+1, self.tensor[self.r_blocks_sparse_list[i_block], 3*nat[0] + x, 3*nat[1] + y, 3*nat[2] + z]))
       
    def Center_sparse(self, Far=1,tol=1.0e-5):
        """
        CENTERING 
        =========

        This subrouine will center the third order force constant inside the Wigner-Seitz supercell.
        This means that for each atomic indices in the tensor, it will be identified by the lowest 
        perimiter between the replica of the atoms.
        Moreover, in case of existance of other elements not included in the original supercell with
        the same perimeter, the tensor will be equally subdivided between equivalent triplets of atoms. 

        This function should be called before performing the Fourier interpolation.
        """    

        t1 = time.time()
        
        
        if Settings.am_i_the_master():
        
            if self.verbose:
                print(" ")
                print(" ======================= Centering ==========================")
                print(" ")         

            # The supercell total size
            #
            nq0=self.supercell_size[0]
            nq1=self.supercell_size[1]
            nq2=self.supercell_size[2]
            
            n_sup = np.prod(self.supercell_size)
            tensor_reshaped = np.transpose(self.tensor.reshape((n_sup, n_sup, 
                                                                3*self.nat, 3*self.nat, 3*self.nat)),axes=[2,3,4,0,1])
            alat=self.unitcell_structure.unit_cell
            
            weight,xR2,xR3 =thirdorder.third_order_centering.analysis(Far,
                                        nq0,nq1,nq2,tol, 
                                        self.unitcell_structure.unit_cell, self.tau, tensor_reshaped,self.nat)  
            
            
            xR2_reshaped=np.reshape(xR2,(3,(2*Far+1)*(2*Far+1)*(2*Far+1)*n_sup*self.nat*self.nat*self.nat*n_sup*n_sup))
            xR3_reshaped=np.reshape(xR3,(3,(2*Far+1)*(2*Far+1)*(2*Far+1)*n_sup*self.nat*self.nat*self.nat*n_sup*n_sup))
            xR23 = np.unique(np.vstack((xR2_reshaped,xR3_reshaped)),axis=1)
            
            self.n_R=xR23.shape[1]
            self.x_r_vector2,self.x_r_vector3 = np.vsplit(xR23,2)
            self.r_vector2=self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2)
            self.r_vector3=self.unitcell_structure.unit_cell.T.dot(self.x_r_vector3)
    
            centered,self.n_R_sparse,x_r_vector2_sparse,x_r_vector3_sparse,atom_sparse,r_blocks_sparse_list=thirdorder.third_order_centering.center_sparse(tensor_reshaped,weight,
                                self.x_r_vector2,xR2,self.x_r_vector3,xR3,
                                Far,self.nat,n_sup,self.n_R)
            

             
            self.x_r_vector2_sparse=x_r_vector2_sparse[:,0:self.n_R_sparse] 
            self.x_r_vector3_sparse=x_r_vector3_sparse[:,0:self.n_R_sparse]            
            self.atom_sparse=atom_sparse[:,0:self.n_R_sparse]         
            self.r_blocks_sparse_list=r_blocks_sparse_list[0:self.n_R_sparse]             
            self.r_vector2_sparse=self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2_sparse)
            self.r_vector3_sparse=self.unitcell_structure.unit_cell.T.dot(self.x_r_vector3_sparse)
            
            t2 = time.time() 
            
            self.tensor = np.transpose(centered, axes=[3,0,1,2])
            
            
            if self.verbose:               
                print("Time elapsed for computing the centering: {} s".format( t2 - t1)) 
                #print("Memory required for the not centered tensor {} Gb".format(self.tensor.nbytes / 1024.**3))
                print("Memory required for the centered tensor: {} Gb".format(centered.nbytes / 1024.**3))
                #print("Memory required after removing zero elements: {} Gb".format(self.tensor.nbytes / 1024.**3))
                print(" ")
                print(" ============================================================")

           
        
        self.tensor = Settings.broadcast(self.tensor)
        self.x_r_vector2 = Settings.broadcast(self.x_r_vector2)
        self.x_r_vector3 = Settings.broadcast(self.x_r_vector3)
        self.r_vector2 = Settings.broadcast(self.r_vector2)
        self.r_vector3 = Settings.broadcast(self.r_vector3)
        self.n_R = Settings.broadcast(self.n_R)
        self.n_sup = Settings.broadcast(self.n_sup)
        #
        self.x_r_vector2_sparse = Settings.broadcast(self.x_r_vector2_sparse)
        self.x_r_vector3_sparse = Settings.broadcast(self.x_r_vector3_sparse)
        self.r_vector2_sparse = Settings.broadcast(self.r_vector2_sparse)
        self.r_vector3_sparse = Settings.broadcast(self.r_vector3_sparse)
        self.n_R_sparse = Settings.broadcast(self.n_R_sparse)
        self.atom_sparse = Settings.broadcast(self.atom_sparse)
        self.r_blocks_sparse_list = Settings.broadcast(self.r_blocks_sparse_list)    
    
    
    
    
    
    
    
# ============================================================================================================================================ 
# Slower/To check/Old  ======================================================================================================================= 
# ============================================================================================================================================  


    def ApplySumRule_old(self) :
        
        t1 = time.time()
        
        if Settings.am_i_the_master():
        
            if self.verbose:
                print(" ")
                print(" ======================= Imposing ASR ==========================")
                print(" ")         
            

            
            #tensor_new = np.transpose(self.tensor.reshape((n_sup_WS, n_sup_WS, 3*self.nat, 3*self.nat, 3*self.nat)),axes=[2,3,4,0,1])
            
           
        
            
            #xR2_list, xR2_block_index, xR2_num_tot=np.unique(self.x_r_vector2, axis = 1, return_index = True, return_counts=True )
            #xR3_list, xR3_block_index, xR3_num_tot=np.unique(self.x_r_vector3, axis = 1, return_index = True, return_counts=True )            
            
            xR_list=np.unique(self.x_r_vector3, axis = 1)
            totnum_R=xR_list.shape[1]
          
            xRdiff_list=np.unique(self.x_r_vector2-self.x_r_vector3, axis = 1)
            totnum_Rdiff=xRdiff_list.shape[1]


            #xRmin=np.min(xR_list,1) 
            #xRmax=np.max(xR_list,1) 
            #xRlen=xRmax-xRmin+np.ones((3,),dtype=int)
            #n_sup=np.prod(xRlen)**2

            
            #xR2_block_index=[  xR2_num_tot[i]  for i in xR3_block_index ]
            
            #print(np.sum(xR2_list-xR3_list))
            #print(np.sum(xR2_block_index-xR3_block_index))            
            #print(np.sum(xR2_num_tot-xR3_num_tot))
                       
            tensor_new = np.transpose(self.tensor,axes=[3,2,1,0])
            phi_asr=thirdorder.third_order_asr.impose_asr(tensor_new,xR_list,xRdiff_list,self.x_r_vector2,self.x_r_vector3,totnum_Rdiff,self.n_R,totnum_R,self.nat)

            #phi_asr=thirdorder.third_order_asr.impose_asr(tensor_trnsp,xRlen,n_sup,xR_list,xRdiff_list,
                                                          #self.x_r_vector2,self.x_r_vector3,
                                                          #totnum_Rdiff,
                                                          #self.n_R,totnum_R,self.nat)

                
                
          
            self.tensor=np.transpose(phi_asr, axes=[3,2,1,0])  


            # DEBUG        
            tensor_block_nonzero = np.sum(self.tensor**2, axis = (1,2,3)) > 1e-8        
            nn_R = np.sum(tensor_block_nonzero.astype(int))                    
            print(nn_R)
            ##################



            t2 = time.time() 
            if self.verbose:               
                print(" Time elapsed for imposing the ASR: {} s".format( t2 - t1)) 
                print(" ")
                print(" ============================================================")
 
 
        self.tensor = Settings.broadcast(self.tensor)



    def Center_fast(self, Far=1,tol=1.0e-5):
        """
        CENTERING 
        =========

        This subrouine will center the third order force constant inside the Wigner-Seitz supercell.
        This means that for each atomic indices in the tensor, it will be identified by the lowest 
        perimiter between the replica of the atoms.
        Moreover, in case of existance of other elements not included in the original supercell with
        the same perimeter, the tensor will be equally subdivided between equivalent triplets of atoms. 

        This function should be called before performing the Fourier interpolation.
        
        Faster but more memory demanding
        """    

        t1 = time.time()
        
        
        if Settings.am_i_the_master():
        
            if self.verbose:
                print(" ")
                print(" ======================= Centering ==========================")
                print(" ")         

            # The supercell total size
            #
            nq0=self.supercell_size[0]
            nq1=self.supercell_size[1]
            nq2=self.supercell_size[2]
            
            n_sup = np.prod(self.supercell_size)
            tensor_reshaped = np.transpose(self.tensor.reshape((n_sup, n_sup, 3*self.nat, 3*self.nat, 3*self.nat)),axes=[2,3,4,0,1])
            alat=self.unitcell_structure.unit_cell
            
            lat_min_prev=np.array([-Far*nq0,-Far*nq1,-Far*nq2])
            lat_max_prev=np.array([nq0-1+Far*nq0,nq1-1+Far*nq1,nq2-1+Far*nq2])
            lat_len_prev=lat_max_prev-lat_min_prev+np.ones(3,dtype=int)
            n_sup_WS_prev=np.prod(lat_len_prev,dtype=int)        
               
            centered_tmp, lat_min_new, lat_max_new =thirdorder.third_order_centering.pre_center(Far,
                                        nq0,nq1,nq2,tol, 
                                        self.unitcell_structure.unit_cell, self.tau, tensor_reshaped,self.nat)  
            
            #
            # Reassignement
            #
            lat_len=lat_max_new-lat_min_new+np.ones(3,dtype=int)
            n_sup_WS=np.prod(lat_len,dtype=int)
        
            
            self.n_R=n_sup_WS**2
                 
            centered,self.x_r_vector2,self.x_r_vector3,self.r_vector2,self.r_vector3= \
                    thirdorder.third_order_centering.assign(alat,lat_min_prev,lat_max_prev,centered_tmp,lat_min_new,
                                                    lat_max_new,n_sup_WS,self.nat,n_sup_WS_prev)
                
                    
            
            
            t2 = time.time() 
            
            centered = np.transpose(centered, axes=[3,0,1,2])
                    
                    
            # Select the element different from zero
            tensor_block_nonzero = np.sum(centered**2, axis = (1,2,3)) > 1e-8
            
            self.tensor = centered[tensor_block_nonzero, :, :, :]
            self.x_r_vector2 = self.x_r_vector2[:, tensor_block_nonzero]
            self.x_r_vector3 = self.x_r_vector3[:, tensor_block_nonzero]
            self.r_vector2 = self.r_vector2[:, tensor_block_nonzero]
            self.r_vector3 = self.r_vector3[:, tensor_block_nonzero]
            self.n_R = np.sum(tensor_block_nonzero.astype(int))

            if self.verbose:               
                print("Time elapsed for computing the centering: {} s".format( t2 - t1)) 
                print("Memory required for the centered tensor: {} Gb".format(centered.nbytes / 1024.**3))
                print("Memory required after removing zero elements: {} Gb".format(self.tensor.nbytes / 1024.**3))
                print(" ")
                print(" ============================================================")

           
        
        self.tensor = Settings.broadcast(self.tensor)
        self.x_r_vector2 = Settings.broadcast(self.x_r_vector2)
        self.x_r_vector3 = Settings.broadcast(self.x_r_vector3)
        self.r_vector2 = Settings.broadcast(self.r_vector2)
        self.r_vector3 = Settings.broadcast(self.r_vector3)
        self.n_R = Settings.broadcast(self.n_R)


    def Center_fast_noreduce(self, Far=1,tol=1.0e-5):
        """
        CENTERING 
        =========

        This subrouine will center the third order force constant inside the Wigner-Seitz supercell.
        This means that for each atomic indices in the tensor, it will be identified by the lowest 
        perimiter between the replica of the atoms.
        Moreover, in case of existance of other elements not included in the original supercell with
        the same perimeter, the tensor will be equally subdivided between equivalent triplets of atoms. 

        This function should be called before performing the Fourier interpolation.
        
        Faster but more memory demanding
        """    

        t1 = time.time()
        
        
        if Settings.am_i_the_master():
        
            if self.verbose:
                print(" ")
                print(" ======================= Centering ==========================")
                print(" ")         

            # The supercell total size
            #
            nq0=self.supercell_size[0]
            nq1=self.supercell_size[1]
            nq2=self.supercell_size[2]
            
            n_sup = np.prod(self.supercell_size)
            tensor_reshaped = np.transpose(self.tensor.reshape((n_sup, n_sup, 3*self.nat, 3*self.nat, 3*self.nat)),axes=[2,3,4,0,1])
            alat=self.unitcell_structure.unit_cell
            
            lat_min_prev=np.array([-Far*nq0,-Far*nq1,-Far*nq2])
            lat_max_prev=np.array([nq0-1+Far*nq0,nq1-1+Far*nq1,nq2-1+Far*nq2])
            lat_len_prev=lat_max_prev-lat_min_prev+np.ones(3,dtype=int)
            n_sup_WS_prev=np.prod(lat_len_prev,dtype=int)        
               
            centered_tmp, lat_min_new, lat_max_new =thirdorder.third_order_centering.pre_center(Far,
                                        nq0,nq1,nq2,tol, 
                                        self.unitcell_structure.unit_cell, self.tau, tensor_reshaped,self.nat)  
            
            #
            # Reassignement
            #
            lat_len=lat_max_new-lat_min_new+np.ones(3,dtype=int)
            n_sup_WS=np.prod(lat_len,dtype=int)
        
            
            self.n_R=n_sup_WS**2
                 
            centered,self.x_r_vector2,self.x_r_vector3,self.r_vector2,self.r_vector3= \
                    thirdorder.third_order_centering.assign(alat,lat_min_prev,lat_max_prev,centered_tmp,lat_min_new,
                                                    lat_max_new,n_sup_WS,self.nat,n_sup_WS_prev)
                
                    
            
            
            t2 = time.time() 
            
            centered = np.transpose(centered, axes=[3,0,1,2])
                    
 
                    
            # Select the element different from zero
            tensor_block_nonzero = np.sum(centered**2, axis = (1,2,3)) > -10 
            
            self.tensor = centered[tensor_block_nonzero, :, :, :]
            self.x_r_vector2 = self.x_r_vector2[:, tensor_block_nonzero]
            self.x_r_vector3 = self.x_r_vector3[:, tensor_block_nonzero]
            self.r_vector2 = self.r_vector2[:, tensor_block_nonzero]
            self.r_vector3 = self.r_vector3[:, tensor_block_nonzero]
            self.n_R = np.sum(tensor_block_nonzero.astype(int))
            print(self.n_R)
            if self.verbose:               
                print("Time elapsed for computing the centering: {} s".format( t2 - t1)) 
                print("Memory required for the centered tensor: {} Gb".format(centered.nbytes / 1024.**3))
                print("Memory required after removing zero elements: {} Gb".format(self.tensor.nbytes / 1024.**3))
                print(" ")
                print(" ============================================================")

           
        
        #self.tensor = Settings.broadcast(self.tensor)
        #self.x_r_vector2 = Settings.broadcast(self.x_r_vector2)
        #self.x_r_vector3 = Settings.broadcast(self.x_r_vector3)
        #self.r_vector2 = Settings.broadcast(self.r_vector2)
        #self.r_vector3 = Settings.broadcast(self.r_vector3)
        #self.n_R = Settings.broadcast(self.n_R)        


  
    def Interpolate_fort(self, q2, q3): # OK but slower
       """
       Interpolate the third order to the q2 and q3 points
       
       Parameters
       ----------
           q2, q3 : ndarray(3)
               The q points
       
       Results
       -------
           Phi3 : ndarray(size = (3*nat, 3*nat, 3*nat))
               The third order force constant in the defined q points.
               atomic indices runs over the unit cell
       """
       tensor_new = np.zeros( shape = (3*self.nat, 3*self.nat, 3*self.nat, self.n_R), order = "F", dtype = np.complex128)
       tensor_new[:,:,:,:] = np.transpose(self.tensor,axes=[3,2,1,0])
       interpolated_fc_tmp=thirdorder.third_order_interpol.interpol(tensor_new,
                                                           self.r_vector2,self.r_vector3,
                                                           q2,q3,self.n_R,self.nat)
       
       interpolated_fc=np.transpose(interpolated_fc_tmp,axes=[2,1,0])
       #final_fc = np.zeros((3*self.nat, 3*self.nat, 3*self.nat), 
                           #dtype = np.complex128)
       #for i in range(self.n_R):
           #arg = 2  * np.pi * (q2.dot(self.r_vector2[:, i]) + 
                               #q3.dot(self.r_vector3[:, i]))
           
           #phase =  np.complex128(np.exp(1j * arg))
           
           #final_fc += phase * self.tensor[i, :, :, :]
       return interpolated_fc       
    

    def Center_py(self,Far=1,tol=1.0e-5):
        """
        CENTERING 
        =========

        This is the python routine to center the third order force constant inside the Wigner-Seitz supercell.
        This means that for each atomic indices in the tensor, it will be identified by the lowest 
        perimiter between the replica of the atoms.
        Moreover, in case of existance of other elements not included in the original supercell with
        the same perimeter, the tensor will be equally subdivided between equivalent triplets of atoms. 

        This function should be called before performing the Fourier interpolation.
        """    
        
        if self.verbose:
            print(" ")
            print(" === Centering === ")
            print(" ")        
        
        # The supercell total size
        nq0=self.supercell_size[0]
        nq1=self.supercell_size[1]
        nq2=self.supercell_size[2]

        # We prepare the tensor for the Wigner-Seitz cell (twice as big for any direction)
        WS_nsup = 2**3* np.prod(self.supercell_size)
        WS_n_R = (WS_nsup)**2

        # Allocate the vector in the WS cell
        # TODO: this could be memory expensive
        #       we could replace this tensor with an equivalent object
        #       that stores only the blocks that are actually written
        WS_r_vector2 = np.zeros((3, WS_n_R), dtype = np.double, order = "F")
        WS_r_vector3 = np.zeros((3, WS_n_R), dtype = np.double, order = "F")
        WS_xr_vector2 = np.zeros((3, WS_n_R), dtype = np.double, order = "F")
        WS_xr_vector3 = np.zeros((3, WS_n_R), dtype = np.double, order = "F")

        # Allocate the tensor in the WS cell
        WS_tensor = np.zeros((WS_n_R, 3*self.nat, 3*self.nat, 3*self.nat), dtype = np.double)

        # Here we prepare the vectors
        # Iterating for all the possible values of R2 and R3 in the cell that encloses the Wigner-Seitz one
        t1 = time.time()
        for i, (a2,b2,c2) in enumerate(itertools.product(range(-nq0, nq0), range(-nq1, nq1), range(-nq2, nq2))):
            for j, (a3,b3,c3) in enumerate(itertools.product(range(-nq0, nq0), range(-nq1, nq1), range(-nq2, nq2))):
                
                # Enclose in one index i and j
                total_index = i * WS_nsup + j

                # Get the crystal lattice
                WS_xr_vector2[:, total_index] = (a2,b2,c2)
                WS_xr_vector3[:, total_index] = (a3,b3,c3)


        # Convert all the vectors in cartesian coordinates
        WS_r_vector2[:,:] = self.unitcell_structure.unit_cell.T.dot(WS_xr_vector2) 
        WS_r_vector3[:,:] = self.unitcell_structure.unit_cell.T.dot(WS_xr_vector3)
        
        # print("WS vectors:")
        # print(WS_r_vector2.T)
        


        t2 = time.time()
        if (self.verbose):
            print("Time elapsed to prepare vectors in the WS cell: {} s".format(t2-t1))

        # Here we create the lattice images
        # And we save the important data

        # Allocate the distance between the superlattice vectors for each replica
        tot_replicas = (2*Far + 1)**3
        total_size = tot_replicas**2
        dR_12 = np.zeros( (total_size, 3))
        dR_23 = np.zeros( (total_size, 3))
        dR_13 = np.zeros( (total_size, 3))
        # Allocate the perimeter of the superlattice for each replica
        PP = np.zeros(total_size)
        # Alloca the the vector of the superlattice in crystalline units
        V2_cryst = np.zeros((total_size,3))
        V3_cryst = np.zeros((total_size,3))

        # Lets cycle over the replica  (the first index is always in the unit cell)
        # To store the variables that will be used to compute the perimeters of
        # all the replica
        t1 = time.time()
        for i, (a2,b2,c2) in enumerate(itertools.product(range(-Far, Far+1), range(-1,Far+1),range(-Far, Far+1))):
            xR_2 = np.array((a2, b2, c2))
            R_2 = xR_2.dot(self.supercell_structure.unit_cell)
            for j, (a3, b3, c3) in enumerate(itertools.product(range(-1, Far+1), range(-1,Far+1),range(-Far, Far+1))):
                xR_3 = np.array((a3, b3, c3))
                R_3 = xR_3.dot(self.supercell_structure.unit_cell)

                # Prepare an index that runs over both i and j
                total_index = tot_replicas*i + j
                #print(total_index, i, j)

                # Store the replica vector in crystal coordinates
                V2_cryst[total_index, :] = np.array((a2,b2,c2)) * np.array(self.supercell_size)
                V3_cryst[total_index, :] = np.array((a3,b3,c3)) * np.array(self.supercell_size)
                
                # Compute the distances between the replica of the indices
                dR_12[total_index, :] = xR_2
                dR_13[total_index, :] = xR_3
                dR_23[total_index, :] = xR_3 - xR_2

                # Store the perimeter of this replica triplet
                PP[total_index] = R_2.dot(R_2)
                PP[total_index]+= R_3.dot(R_3)
                PP[total_index]+= np.sum((R_3 - R_2)**2)
                
                #print("R2:", R_2, "R3:", R_3, "PP:", PP[total_index])
        t2 = time.time()

        if self.verbose:
            print("Time elapsed to prepare the perimeter in the replicas: {} s".format(t2 - t1))


        # Now we cycle over all the blocks and the atoms
        # For each triplet of atom in a block, we compute the perimeter of the all possible replica
        # Get the metric tensor of the supercell
        G = np.einsum("ab, cb->ac", self.supercell_structure.unit_cell, self.supercell_structure.unit_cell)

        # We cycle over atoms and blocks
        for iR in range(self.n_R):
            for at1, at2, at3 in itertools.product(range(self.nat), range(self.nat), range(self.nat)):
                # Get the positions of the atoms
                r1 = self.tau[at1,:]
                r2 = self.r_vector2[:, iR] + self.tau[at2,:]
                r3 = self.r_vector3[:, iR] + self.tau[at3,:]
                
                # Lets compute the perimeter without the replicas
                pp = np.sum((r1-r2)**2)
                pp+= np.sum((r2-r3)**2)
                pp+= np.sum((r1-r3)**2)
                
                # Get the crystalline vectors (in the supercell)
                x1 = Methods.cart_to_cryst(self.supercell_structure.unit_cell, r1)
                x2 = Methods.cart_to_cryst(self.supercell_structure.unit_cell, r2)
                x3 = Methods.cart_to_cryst(self.supercell_structure.unit_cell, r3)
                
                # Now we compute the quantities that do not depend on the lattice replica
                # As the current perimeter and the gg vector
                G_12 = 2*G.dot(x2-x1)
                G_23 = 2*G.dot(x3-x2)
                G_13 = 2*G.dot(x3-x1)
                
                # Now we can compute the perimeters of all the replica
                # all toghether
                P = PP[:] + pp
                P[:] += dR_12.dot(G_12)
                P[:] += dR_23.dot(G_23)
                P[:] += dR_13.dot(G_13)
                
                # if self.tensor[iR, 3*at1, 3*at2, 3*at3] > 0:
                #     #print("all the perimeters:")
                #     #print(P)
                #     print("The minimum:", np.min(P))
                #     index = np.argmin(P)
                #     print("R2 = ", self.r_vector2[:, iR], "R3 = ", self.r_vector3[:,iR])
                #     print("The replica perimeter:", PP[index])
                #     print("The standard perimeter:", pp)
                #     print("The the cross values:")
                #     print(dR_12.dot(G_12)[index], dR_13.dot(G_13)[index], dR_23.dot(G_23)[index]) 
                #     print("The replica vectors are:", "R2:", V2_cryst[index,:], "R3:", V3_cryst[index,:])
                
                # Now P is filled with the perimeters of all the replica
                # We can easily find the minimum
                P_min = np.min(P)
                
                # We can find how many they are and a mask on their positions
                min_P_mask = (np.abs(P_min - P) < 1e-6).astype(bool)
                
                # The number of minimium perimeters
                n_P = np.sum(min_P_mask.astype(int))

                # Get the replica vector for the minimum perimeters
                v2_shift = V2_cryst[min_P_mask, :]
                v3_shift = V3_cryst[min_P_mask, :]

                # Now we can compute the crystalline coordinates of the lattice in the WS cell
                r2_cryst = np.tile(self.x_r_vector2[:, iR], (n_P, 1)) + v2_shift
                r3_cryst = np.tile(self.x_r_vector3[:, iR], (n_P, 1)) + v3_shift
                verb = False
                

                # Get the block indices in the WS cell
                WS_i_R = get_ws_block_index(self.supercell_size, r2_cryst, r3_cryst, verbose = verb)
                
                # Now we fill all the element of the WS tensor 
                # with the current tensor, dividing by the number of elemets
                new_elemet = np.tile(self.tensor[iR, 3*at1:3*at1+3, 3*at2:3*at2+3, 3*at3:3*at3+3], (n_P, 1,1,1))
                new_elemet /= n_P
                WS_tensor[WS_i_R, 3*at1: 3*at1+3, 3*at2:3*at2+3, 3*at3:3*at3+3] = new_elemet


        t2 = time.time()

        if self.verbose:
            print("Time elapsed for computing the cenetering: {} s".format( t2 - t1))

        # We can update the current tensor
        self.tensor = WS_tensor
        self.r_vector2 = WS_r_vector2
        self.r_vector3 = WS_r_vector3
        self.x_r_vector2 = WS_xr_vector2
        self.x_r_vector3 = WS_xr_vector3
        self.n_R = WS_n_R

    
    def ApplySumRule_py(self):
        r"""
        Enforce the sum rule by projecting the third order force constant
        in the sum rule space defined by
        
        .. math::
        
            P = ( 1 - Q )
            
            Q_{st}^{\alpha\beta} = 1 / nat_{sc} \left( \sum_x \delta_{x\alpha}\delta{x\beta}\right)
        """
        
        # Apply the sum rule on the last index
        
        n_sup_WS = int(np.sqrt(self.n_R))
        
        for I_r2 in range(n_sup_WS):
            # Here we want to sum over r3 (indices that goes from total_index to total_index + n_sup_WS
            total_index = n_sup_WS * I_r2
            
            for s, t in itertools.product(range(3 * self.nat) , range(3*self.nat)):
                for gamma in range(3):
                    # gamma mask runs over the third atom index
                    gamma_mask = np.tile(np.arange(3) == gamma, (self.nat, 1)).ravel()
                    delta = np.sum(self.tensor[total_index : total_index + n_sup_WS, s, t, gamma_mask]) / (self.nat * self.n_R)
                    self.tensor[total_index : total_index + n_sup_WS, s, t, gamma_mask] -= delta
            
            
        # Now apply the sum rule on the second index
        for I_r3 in range(n_sup_WS):
            total_index_mask = (np.arange(self.n_R) % n_sup_WS) == I_r3
            
            for s, t in itertools.product(range(3 * self.nat) , range(3*self.nat)):
                for gamma in range(3):
                    gamma_mask = np.tile(np.arange(3) == gamma, (self.nat, 1)).ravel()
                    #print("BLOCK MASK:",total_index_mask)
                    #print("GAMMA MASK:", gamma_mask)
                    #print("S:",s , "T:", t)
                    #print(np.shape(total_index_mask), np.shape(gamma_mask))
                    #print(np.shape(self.tensor))
                    #print (self.tensor[total_index_mask, s, 0, t])
                    delta = 0
                    for r in range(self.nat):
                        delta += np.sum(self.tensor[total_index_mask, s, 3*r + gamma, t]) / (self.nat * self.n_R)
                        
                    for r in range(self.nat):
                        self.tensor[total_index_mask, s, 3*r + gamma, t] -= delta
            
        
        # Now apply the sum rule on the second index
        for total_index in range(n_sup_WS**2):
            r2 = self.r_vector2[:, total_index]
            r3 = self.r_vector3[:, total_index]
            
            for s, t in itertools.product(range(3 * self.nat) , range(3*self.nat)):
                for gamma in range(3):
                    gamma_mask = np.tile(np.arange(3) == gamma, (self.nat, 1)).ravel()
                    
                    delta = 0
                    for i_Rt in range(n_sup_WS):
                        Rt = self.r_vector2[:, i_Rt * n_sup_WS]
                        Rnew = r3 - r2 + Rt
                        Rnew = np.tile(Rnew, (n_sup_WS,1)).T 
                        
                        distances = np.sum(np.abs( Rnew - self.r_vector3[:, :n_sup_WS]), axis = 0)
                        good_vector  = np.argmin(distances)
                        if distances[good_vector] < 1e-6:
                            
                            vect_index = i_Rt * n_sup_WS + good_vector
                            delta += np.sum(self.tensor[vect_index, gamma_mask, s, t]) / (self.nat * self.n_R)
                    
                    self.tensor[total_index, gamma_mask, s, t] -= delta
                
def get_ws_block_index(supercell_size, cryst_vectors2, cryst_vectors3, verbose = False):
    """
    Get the block in the supercell that contains the WS cell, 
    given the crystalline positions of vectors, returns the indices of the block.

    The block identifies the supercell of the three atoms as
    iR <=> (0, R_2, R_3)

    This subroutines takes R_2 and R_3 in crystalline components, and computes the iR.
    Assuming that the Wigner-Seitz cell is initialized correctly

    Parameters
    ----------
        supercell_size : ndarray(size = (3), dtype = int)
            The size of the original supercell (not the Wigner-Seitz)
        cryst_vectors2 : ndarray(size= (N_vectors, 3), dtype = int)
            The crystalline coordinates of the second vector.
            It could also be a single vector.
        cryst_vectors3 : ndarray(size= (N_vectors, 3), dtype = int)
            The crystalline coordinates of the third vector.
            It could also be a single vector.
    
    Results
    -------
        block_id : ndarray(size = N_vectors, dtype = int)
            The index of the block for each crystalline vector.
    """

    # Get the composed index in the iteration of the two vectors
    # Rescale the vectors
    new_v2 = cryst_vectors2.copy() 
    new_v2 += np.array(supercell_size)

    new_v3 = cryst_vectors3.copy()
    new_v3 += np.array(supercell_size)

    # Get the total dimension of the block for each vector in the WS cell
    WS_sup = np.prod(supercell_size) * 8

    ws_z = 2 * supercell_size[2]
    ws_y = 2 * supercell_size[1]

    # Transform the two vectors in the indices of the iterator
    i2 = new_v2[:, 2] + new_v2[:,1] * ws_z
    i2 += new_v2[:,0] * ws_z * ws_y

    i3 = new_v3[:, 2] + new_v3[:,1] * ws_z 
    i3 += new_v3[:,0] * ws_z * ws_y

    # Collect everything in the block index
    WS_i_R = i2 * WS_sup + i3 

    # Convert in integer for indexing
    WS_i_R = WS_i_R.astype(int)

    if verbose:
        print("All values of i2:")
        print(i2)
        print("All values of i3:")
        print(i3)
        print("The block value:", WS_i_R)

    # Check if the block indices are correct
    assert (WS_i_R >= 0).all()
    assert (WS_i_R < WS_sup**2).all()

    return WS_i_R


