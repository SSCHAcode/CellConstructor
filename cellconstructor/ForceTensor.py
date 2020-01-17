from __future__ import print_function
from __future__ import division 
import Phonons
import Methods 
import symmetries

import numpy as np
import scipy, scipy.signal, scipy.interpolate

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
        """    
        
        print(" ")
        print(" === Centering === ")
        print(" ")        
        
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
            #
            for t,lt,mt,nt in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                t_vec=self.tau[t,:]    
                t_lat=three_to_one_len([lt,mt,nt],[0,0,0],[nq0,nq1,nq2])
                for u,lu,mu,nu in itertools.product(range(self.nat),range(nq0),range(nq1),range(nq2)):
                    u_vec=self.tau[u,:] 
                    u_lat=three_to_one_len([lu,mu,nu],[0,0,0],[nq0,nq1,nq2])
                    #
                    perim_min=np.inf
                    counter=0
                    #
                    #
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
            
            \Phị_{abc} \sum_{pqr} \Phi_{pqr} P_{ap}P_{bq}P_{cr}
            
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
    
