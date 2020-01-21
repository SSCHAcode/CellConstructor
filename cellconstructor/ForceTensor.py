from __future__ import print_function
from __future__ import division 

# Import for python2/3 compatibility
import cellconstructor.Phonons as Phonons
import cellconstructor.Methods as Methods 
import cellconstructor.symmetries as symmetries

import numpy as np
import scipy, scipy.signal, scipy.interpolate

import time
import itertools

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
            n_cell_x2,n_cell_y2,n_cell_z2=one_to_three_len(index_cell2,v_min=[0,0,0],v_len=supercell_size)
            for index_cell3 in range(n_sup):
                n_cell_x3,n_cell_y3,n_cell_z3=one_to_three_len(index_cell3,v_min=[0,0,0],v_len=supercell_size)      
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
            for total_lat_vec  in range(self.n_R):
                lat_vect_2 = total_lat_vec //  np.prod(self.supercell_size)
                lat_vect_3 = total_lat_vec %  np.prod(self.supercell_size)
                total_index_cell = lat_vect_3 + np.prod(self.supercell_size) * lat_vect_2
                for nat1 in range(self.nat):
                    for nat2 in range(self.nat):
                        for nat3 in range(self.nat):
                            f.write("{:d}\n".format(i_block))
                            f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(*list(self.r_vector2[:, total_lat_vec])))
                            f.write("{:16.8e} {:16.8e} {:16.8e}\n".format(*list(self.r_vector3[:, total_lat_vec])))
                            f.write("{:>6d} {:>6d} {:>6d}\n".format(nat1+1, nat2+1, nat3+1))
                            i_block += 1
                            
                            for xyz in range(27):
                                z = xyz % 3
                                y = (xyz %9)//3
                                x = xyz // 9
                                f.write("{:>2d} {:>2d} {:>2d} {:>20.10e}\n".format(x+1,y+1,z+1, self.tensor[total_index_cell, 3*nat1 + x, 3*nat2 + y, 3*nat3 + z]))
       
    
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
            
            

    def Center(self,Far=1,tol=1.0e-5):
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


    def __old_centering__(self, Far=1,tol=1.0e-5):
        """
        This is an old and slow implementation of the centering,
        check if it is doing the same thing as the new one.
        """
        print(" OLD CENTERING")

        # The supercell total size
        nq0=self.supercell_size[0]
        nq1=self.supercell_size[1]
        nq2=self.supercell_size[2]
        
        n_sup = np.prod(self.supercell_size)
        tensor_new = self.tensor.reshape((n_sup, n_sup, 3*self.nat, 3*self.nat, 3*self.nat))
        
        # weight(s,t,iii,u,jjj) weight of the triangle with vertices (s,0),(t,iii),(u,jjj)
        weight=np.zeros([self.nat,self.nat,n_sup,self.nat,n_sup],dtype=int)
        RRt=np.zeros([3,self.nat,self.nat,n_sup,self.nat,n_sup,(2*Far+1)**3])
        RRu=np.zeros([3,self.nat,self.nat,n_sup,self.nat,n_sup,(2*Far+1)**3])
        
        lat_min=np.array([np.inf,np.inf,np.inf])
        lat_max=np.array([-np.inf,-np.inf,-np.inf])
        #
        for s in range(self.nat):
            s_vec=self.tau[s,:]
            # Loop on the supercell atoms
            for t,lt,mt,nt in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                t_vec=self.tau[t,:]    
                t_lat=three_to_one_len([lt,mt,nt],[0,0,0],[nq0,nq1,nq2])
                # Loop on the supercell atoms
                for u,lu,mu,nu in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                    u_vec=self.tau[u,:] 
                    u_lat=three_to_one_len([lu,mu,nu],[0,0,0],[nq0,nq1,nq2])
                    #
                    perim_min=np.inf
                    counter=0
                    #
                    # Now we cycle moving the lattice vector to find the equivalent vectors
                    # that minimize the perimeter
                    

                    # Perimeter
                    for LLt, MMt, NNt,  in itertools.product(range(-Far,Far+1),range(-Far,Far+1),range(-Far,Far+1)):
                        xt=[lt+LLt*nq0,mt+MMt*nq1,nt+NNt*nq2]
                        SC_t_vec=self.unitcell_structure.unit_cell.T.dot(xt)+t_vec          
                        for LLu, MMu, NNu,  in itertools.product(range(-Far,Far+1),range(-Far,Far+1),range(-Far,Far+1)):
                            xu=[lu+LLu*nq0,mu+MMu*nq1,nu+NNu*nq2]
                            SC_u_vec=self.unitcell_structure.unit_cell.T.dot(xu)+u_vec                
                            #
                            #
                            perimeter=compute_perimeter(s_vec,SC_t_vec ,SC_u_vec)
                            #
                            if perimeter < (perim_min - tol) :
                                perim_min = perimeter
                                counter = 1
                                weight[s,t,t_lat,u,u_lat]=counter
                                #
                                RRt[:,s,t,t_lat,u,u_lat,counter-1]=xt        
                                RRu[:,s,t,t_lat,u,u_lat,counter-1]=xu        
                                lat_min_tmp=np.min( [ xt, xu ] , axis=0 )
                                lat_max_tmp=np.max( [ xt, xu ] , axis=0 )            
                            elif abs(perimeter - perim_min) <= tol:
                                counter += 1
                                weight[s,t,t_lat,u,u_lat]=counter
                                #
                                RRt[:,s,t,t_lat,u,u_lat,counter-1]=xt        
                                RRu[:,s,t,t_lat,u,u_lat,counter-1]=xu        
                                lat_min_tmp=np.min( [ lat_min_tmp, xt , xu ], axis=0 )
                                lat_max_tmp=np.max( [ lat_max_tmp, xt , xu ], axis=0 )                
                    #
                    #
                    lat_min=np.min( [lat_min,lat_min_tmp], axis=0)
                    lat_max=np.max( [lat_max,lat_max_tmp], axis=0)
                    
        #
        lat_len=lat_max-lat_min+np.ones(3,dtype=int)
        n_sup_WS=np.prod(lat_len,dtype=int)

        centered=np.zeros([n_sup_WS,n_sup_WS,self.nat*3,self.nat*3,self.nat*3],dtype=np.double)
        
        for s in range(self.nat):
            for t,lt,mt,nt in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                t_lat=three_to_one_len([lt,mt,nt],[0,0,0],[nq0,nq1,nq2])
                for u,lu,mu,nu in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                    u_lat=three_to_one_len([lu,mu,nu],[0,0,0],[nq0,nq1,nq2])
                    for h in range(weight[s,t,t_lat,u,u_lat]):
                        I=three_to_one(RRt[:,s,t,t_lat,u,u_lat,h],lat_min,lat_max)
                        J=three_to_one(RRu[:,s,t,t_lat,u,u_lat,h],lat_min,lat_max)    
                        #
                        for alpha,beta,gamma in itertools.product(range(3),range(3),range(3)):
                            jn1 = alpha + s*3
                            jn2 = beta  + t*3
                            jn3 = gamma + u*3
                            #
                            centered[I,J,jn1,jn2,jn3]=tensor_new[t_lat,u_lat,jn1,jn2,jn3]/weight[s,t,t_lat,u,u_lat]

        # Reassignement
        
        self.n_R=n_sup_WS**2
        
        # Cartesian lattice vectors
        self.r_vector2 = np.zeros((3, self.n_R), dtype = np.double, order = "F")
        self.r_vector3 = np.zeros((3, self.n_R), dtype = np.double, order = "F")
        
        # Crystalline lattice vectors
        self.x_r_vector2 = np.zeros((3, self.n_R), dtype = np.intc, order = "F")
        self.x_r_vector3 = np.zeros((3, self.n_R), dtype = np.intc, order = "F")
        
        self.tensor=centered.reshape((self.n_R, 3*self.nat, 3*self.nat, 3*self.nat))
        
        for index,(I,J) in enumerate(itertools.product(range(n_sup_WS),range(n_sup_WS))): 
            self.x_r_vector2[:, index] = one_to_three(I,lat_min,lat_max)
            self.r_vector2[:, index] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2[:, index])       
            self.x_r_vector3[:, index] = one_to_three(J,lat_min,lat_max)
            self.r_vector3[:, index] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector3[:, index])       


    def __old_centering_2_(self, Far=1,tol=1.0e-5):
        """
        This is an old and slow implementation of the centering,
        check if it is doing the same thing as the new one.
        """
        print(" OLD CENTERING_2")

        # The supercell total size
        nq0=self.supercell_size[0]
        nq1=self.supercell_size[1]
        nq2=self.supercell_size[2]
        
        n_sup = np.prod(self.supercell_size)
        tensor_new = self.tensor.reshape((n_sup, n_sup, 3*self.nat, 3*self.nat, 3*self.nat))
        
        # weight(s,t,iii,u,jjj) weight of the triangle with vertices (s,0),(t,iii),(u,jjj)
        weight=np.zeros([self.nat,self.nat,n_sup,self.nat,n_sup],dtype=int)
        RRt=np.zeros([3,self.nat,self.nat,n_sup,self.nat,n_sup,(2*Far+1)**3])
        RRu=np.zeros([3,self.nat,self.nat,n_sup,self.nat,n_sup,(2*Far+1)**3])
        
        lat_min=np.array([np.inf,np.inf,np.inf])
        lat_max=np.array([-np.inf,-np.inf,-np.inf])
        
        lat_min_tmp=np.array([-Far,-Far,-Far])
        lat_max_tmp=np.array([Far,Far,Far])
        
        centered_tmp=np.zeros([n_sup_WS,n_sup_WS,self.nat*3,self.nat*3,self.nat*3],dtype=np.double)
        n_sup_WS_tmp=np.prod(lat_len,dtype=int)        
        
        #
        for s in range(self.nat):
            s_vec=self.tau[s,:]
            # Loop on the supercell atoms
            for t,lt,mt,nt in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                t_vec=self.tau[t,:]    
                t_lat=three_to_one_len([lt,mt,nt],[0,0,0],[nq0,nq1,nq2])
                # Loop on the supercell atoms
                for u,lu,mu,nu in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                    u_vec=self.tau[u,:] 
                    u_lat=three_to_one_len([lu,mu,nu],[0,0,0],[nq0,nq1,nq2])
                    #
                    perim_min=np.inf
                    counter=0
                    #
                    # Now we cycle moving the lattice vector to find the equivalent vectors
                    # that minimize the perimeter
                    

                    # Perimeter
                    for LLt, MMt, NNt,  in itertools.product(range(-Far,Far+1),range(-Far,Far+1),range(-Far,Far+1)):
                        xt=[lt+LLt*nq0,mt+MMt*nq1,nt+NNt*nq2]
                        SC_t_vec=self.unitcell_structure.unit_cell.T.dot(xt)+t_vec          
                        for LLu, MMu, NNu,  in itertools.product(range(-Far,Far+1),range(-Far,Far+1),range(-Far,Far+1)):
                            xu=[lu+LLu*nq0,mu+MMu*nq1,nu+NNu*nq2]
                            SC_u_vec=self.unitcell_structure.unit_cell.T.dot(xu)+u_vec                
                            #
                            #
                            perimeter=compute_perimeter(s_vec,SC_t_vec ,SC_u_vec)
                            #
                            if perimeter < (perim_min - tol) :
                                perim_min = perimeter
                                counter = 1
                                weight[s,t,t_lat,u,u_lat]=counter
                                #
                                RRt[:,s,t,t_lat,u,u_lat,counter-1]=xt        
                                RRu[:,s,t,t_lat,u,u_lat,counter-1]=xu        
                                lat_min_tmp=np.min( [ xt, xu ] , axis=0 )
                                lat_max_tmp=np.max( [ xt, xu ] , axis=0 )            
                            elif abs(perimeter - perim_min) <= tol:
                                counter += 1
                                weight[s,t,t_lat,u,u_lat]=counter
                                #
                                RRt[:,s,t,t_lat,u,u_lat,counter-1]=xt        
                                RRu[:,s,t,t_lat,u,u_lat,counter-1]=xu        
                                lat_min_tmp=np.min( [ lat_min_tmp, xt , xu ], axis=0 )
                                lat_max_tmp=np.max( [ lat_max_tmp, xt , xu ], axis=0 )                
                    #
                    #
                    lat_min=np.min( [lat_min,lat_min_tmp], axis=0)
                    lat_max=np.max( [lat_max,lat_max_tmp], axis=0)
                    
        #
        lat_len=lat_max-lat_min+np.ones(3,dtype=int)
        n_sup_WS=np.prod(lat_len,dtype=int)

        centered=np.zeros([n_sup_WS,n_sup_WS,self.nat*3,self.nat*3,self.nat*3],dtype=np.double)
        
        for s in range(self.nat):
            for t,lt,mt,nt in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                t_lat=three_to_one_len([lt,mt,nt],[0,0,0],[nq0,nq1,nq2])
                for u,lu,mu,nu in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                    u_lat=three_to_one_len([lu,mu,nu],[0,0,0],[nq0,nq1,nq2])
                    for h in range(weight[s,t,t_lat,u,u_lat]):
                        I=three_to_one(RRt[:,s,t,t_lat,u,u_lat,h],lat_min,lat_max)
                        J=three_to_one(RRu[:,s,t,t_lat,u,u_lat,h],lat_min,lat_max)    
                        #
                        for alpha,beta,gamma in itertools.product(range(3),range(3),range(3)):
                            jn1 = alpha + s*3
                            jn2 = beta  + t*3
                            jn3 = gamma + u*3
                            #
                            centered[I,J,jn1,jn2,jn3]=tensor_new[t_lat,u_lat,jn1,jn2,jn3]/weight[s,t,t_lat,u,u_lat]

        # Reassignement
        
        self.n_R=n_sup_WS**2
        
        # Cartesian lattice vectors
        self.r_vector2 = np.zeros((3, self.n_R), dtype = np.double, order = "F")
        self.r_vector3 = np.zeros((3, self.n_R), dtype = np.double, order = "F")
        
        # Crystalline lattice vectors
        self.x_r_vector2 = np.zeros((3, self.n_R), dtype = np.intc, order = "F")
        self.x_r_vector3 = np.zeros((3, self.n_R), dtype = np.intc, order = "F")
        
        self.tensor=centered.reshape((self.n_R, 3*self.nat, 3*self.nat, 3*self.nat))
        
        for index,(I,J) in enumerate(itertools.product(range(n_sup_WS),range(n_sup_WS))): 
            self.x_r_vector2[:, index] = one_to_three(I,lat_min,lat_max)
            self.r_vector2[:, index] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector2[:, index])       
            self.x_r_vector3[:, index] = one_to_three(J,lat_min,lat_max)
            self.r_vector3[:, index] = self.unitcell_structure.unit_cell.T.dot(self.x_r_vector3[:, index])       


                       
    def Interpolate(self, q2, q3):
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
        
        final_fc = np.zeros((3*self.nat, 3*self.nat, 3*self.nat), 
                            dtype = np.complex128)
        for i in self.n_R:
            arg = 2  * np.pi * (q2.dot(self.r_vector2[:, i]) + 
                                q3.dot(self.r_vector3[:, i]))
            
            phase =  np.complex128(np.exp(1j * arg))
            
            final_fc += phase * self.tensor[i, :, :, :]
        return final_fc
            
            
            
    def ApplySumRule(self):
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
                
            



def compute_perimeter(v1,v2,v3):
    res=np.linalg.norm(v1-v2)+np.linalg.norm(v2-v3)+np.linalg.norm(v3-v1)
    return res
 
def three_to_one_len(v,v_min,v_len):
   res=(v[0]-v_min[0])*v_len[1]*v_len[2]+(v[1]-v_min[1])*v_len[2]+(v[2]-v_min[2])
   return int(res)

def three_to_one(v,v_min,v_max):
   v_len=np.array(v_max)-np.array(v_min)+np.ones(3,dtype=int)
   res=(v[0]-v_min[0])*v_len[1]*v_len[2]+(v[1]-v_min[1])*v_len[2]+(v[2]-v_min[2])
   return int(res)

def one_to_three_len(J,v_min,v_len):
   x              =   J // (v_len[2] * v_len[1])              + v_min[0]
   y              = ( J %  (v_len[2] * v_len[1])) // v_len[2] + v_min[1]
   z              =   J %   v_len[2]                          + v_min[2]   
   return np.array([x,y,z],dtype=int) 

def one_to_three(J,v_min,v_max):
   v_len=np.array(v_max)-np.array(v_min)+np.ones(3,dtype=int)    
   x              =   J // (v_len[2] * v_len[1])              + v_min[0]
   y              = ( J %  (v_len[2] * v_len[1])) // v_len[2] + v_min[1]
   z              =   J %   v_len[2]                          + v_min[2]   
   return np.array([x,y,z],dtype=int) 

    
#def one_to_three(J,v_min,v_len):
   #x              =(       J                                                               )/(v_len[2]*v_len[1])+v_min[0]
   #y              =(       J-(x-v_min[0])*v_len[1]*v_len[2]                                )/(v_len[2])         +v_min[1]
   #z              =(       J-(x-v_min[0])*v_len[1]*v_len[2]-(y-v_min[1])*v_len[2]          )                    +v_min[2]   
   #return [x,y,z]    
    


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


