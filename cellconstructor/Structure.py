# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:44:27 2018

@author: pione
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
try:
    __ASE__ = True
    import ase
    import ase.io
except:
    __ASE__ = False
    
import sys, os

import cellconstructor.Methods as Methods
import cellconstructor.symmetries as SYM
from cellconstructor.Units import *

import symph




__all__ = ["Structure"]
BOHR_TO_ANGSTROM=0.529177249

class Structure:    
    def __init__(self, nat=0):
        self.N_atoms=nat
        # Coordinates are always express in chartesian axis
        self.coords = np.zeros((self.N_atoms, 3), dtype = np.float64)
        self.atoms = ["H"] * nat
        self.unit_cell = np.zeros((3,3))
        self.has_unit_cell = False
        self.masses = {}
        self.ita = 0 # Symmetry group in ITA standard
        
    def get_volume(self):
        """
        Returns the volume of the unit cell
        """
        ERR_MSG = """
Error, to compute the volume the structure must have a unit cell initialized:
(i.e. the has_unit_cell attribute must be True)."""

        assert self.has_unit_cell, ERR_MSG

        return np.abs(np.linalg.det(self.unit_cell))
        
    def generate_from_ase_atoms(self, atoms, get_masses = True):
        """
        This subroutines generate the current structure
        from the ASE Atoms object
        
        Parameters
        ----------
            atoms : the ASE Atoms object
            get_masses : bool
                If true, also build the masses. 
                Note that massess are saved in Ry units (electron mass)
        
        """
        
        self.unit_cell = atoms.get_cell()
        self.has_unit_cell = True
        self.atoms = atoms.get_chemical_symbols()
        self.N_atoms = len(self.atoms)
        self.coords = atoms.positions.copy()

        if get_masses:
            self.masses = {}
            mass = atoms.get_masses()
            for i, ma in enumerate(mass):
                if not self.atoms[i] in self.masses:
                    self.masses[self.atoms[i]] = ma * ELECTRON_MASS_UMA

    
    def build_masses(self):
        """
        Use the ASE database to build the masses.
        The masses will be in [Ry] units (the electron mass)
        """

        ase_struct = self.get_ase_atoms()


        self.masses = {}
        mass = ase_struct.get_masses()
        for i, ma in enumerate(mass):
            if not self.atoms[i] in self.masses:
                self.masses[self.atoms[i]] = ma * ELECTRON_MASS_UMA
        

    def copy(self):
        """
        This method simply returns a copy of the current structure

        Results
        -------
            - aux : Structure
                A copy of the self structure.
        """

        aux = Structure()
        aux.N_atoms = self.N_atoms
        aux.coords = self.coords.copy()
        aux.atoms = [atm for atm in self.atoms]
        aux.unit_cell = self.unit_cell.copy()
        aux.has_unit_cell = self.has_unit_cell

        # Deep copy of the masses
        aux.masses = {}
        for k in self.masses.keys():
            aux.masses[k] = self.masses[k]
            
        aux.ita = self.ita
        return aux
        
    def set_masses(self, masses):
        """
        This method set up the masses of the system. It requires a dictionary containing the
        symobl and the value of the masses in a.u. (mass of the electron)

        Parameters
        ----------
            - masses : dictionary
                A dictionary containing the label and the corresponding mass in a.u. 
                Ex. masses = {'H' : 918.68, 'O' : 14582.56}
        """

        self.masses = masses
    
    def get_masses_array(self):
        """
        Convert the masses of the current structure 
        in a numpy array of size N_atoms
        
        NOTE: This method will rise an exception if the masses are not initialized well
        
        Results
        -------
            masses : ndarray (size self.N_atoms)
                The array containing the mass for each atom of the system.
        """
        
        masses = np.zeros( self.N_atoms)
        for i in range(self.N_atoms):
            masses[i] = self.masses[ self.atoms[i] ]
        
        return masses
    
    def get_atomic_types(self):
        """
        Get an array of integer, starting from 1, for each atom of the structure,
        so that two equal atoms share the same index. 
        
        This is how different types are stored in Quantum ESPRESSO and it
        is usefull for the wrapping Fortran => Python.
        
        Result
        ------
            ityp : ndarray dtype=(numpy.intc)
                The type array
        """
        
        ityp = []
        cont = 1
        for i, atm in enumerate(self.atoms):
            ityp.append(cont)
            
            if atm not in self.atoms[:i]:
                cont += 1

        # For fortran and C compatibility parse the array
        return np.array(ityp, dtype = np.intc)
        
    def read_xyz(self, filename, alat = False, epsilon = 1e-8, frame_id = 0):
        """
        This function reads the atomic position from a xyz file format.
        if the passed file contains an animation, only the frame specified by frame_id
        will be processed.

        Parameters
        ----------
            - filename : string
                  The path of the xyz file, read access is required
            - alat : bool, optional
                  If true the coordinates will be rescaled with the loaded unit cell, otherwise
                  cartesian coordinates are supposed (default False).
            - frame_id : int
                  The id of the frame to be processed in an animation.
            - epsilon : double precision, optional
                  Each value below this is considered to be zero (defalut 1e-8)
        """
        # Check if the input is consistent
        if (alat and not self.has_unit_cell):
            sys.stderr.write("ERROR, alat setted to true, but no unit cell initialized\n")
            raise ValueError("Function read_xyz, alat = True but no unit cell.")
        
        ##  Check if the unit cell can be used with alat (only for orthorombic diagonal unit cell)
        # if (alat):
        #     if (sum(self.unit_cell ** 2) - sum(diag(self.unit_cell)**2) < epsilon):
        #         sys.stderr.write("ERROR, alat compatible only with diagonal unit cell.\n")
        #         raise ErrorInUnitCell("Function read_xyz, alat parameters not combatible with cell.")


        # Open the file
        xyz = open(filename, "r")

        # Jump to the correct frame id
        for k in range(frame_id):
            njump = int(xyz.readline()) + 1
            for jl in range(njump):
                xyz.readline()
            

        self.N_atoms = int(xyz.readline())
        self.coords = np.zeros((self.N_atoms, 3))
        
        # Read the comment line
        xyz.readline()

        for i in range(self.N_atoms):
            line = xyz.readline()
            atom, x, y, z = line.split()

            self.atoms.append(atom)
            self.coords[i,0] = np.float64(x)
            self.coords[i,1] = np.float64(y)
            self.coords[i,2] = np.float64(z)

            # Rescale the coordinates with the unit cell if requested
            if alat:
                # Not shure if the dot product must be done with the transposed unit cell matrix
                self.coords[i, :] = np.dot( np.transpose(self.unit_cell), self.coords[i, :])

        # Close the xyz file
        xyz.close()

    def read_scf(self, filename, alat=1):
        """
        Read the given filename in the quantum espresso format.
        Note:
        The file must contain only the part reguarding ATOMIC POSITIONS.
        
        Parameters
        ----------
           - filename : str
               The filename containing the atomic positions
           - alat : double
               If present the system will convert both the cell and the atoms position
               by this factor. If it is also specified in the CELL_PARAMETERS line,
               the one specified in the file will be used.
        """
        # Check if the specified filename exists
        if not os.path.exists(filename):
            raise ValueError("File %s does not exist" % filename)

        # Read the input filename
        fp = open(filename, "r")

        n_atoms = 0
        #good_lines = []

        # First read
        read_cell = False
        cell_index = 0
        read_atoms = True
        cell_present = False
        
        read_crystal = False
        
        #print "ALAT:", alat
        
        #atom_index = 0
        cell = np.zeros((3,3), dtype = np.float64)
        tmp_coords = []
        for line in fp.readlines():
            line = line.strip()

            #Skipp comments
            if len(line) == 0:
                continue
            
            if line[0] == "!":
                continue

            # Split the line into the values
            values = line.split()
            if values[0] == "CELL_PARAMETERS":
                read_cell = True
                read_atoms = False
                self.has_unit_cell = True
                cell_present = True
                
                # Check if the alat value is specified here
                if "alat=" in line.lower().replace(" ", ""):
                    value_alat = np.float64(line[ line.find("=") + 1:].strip().replace(")",""))
                    alat = value_alat * BOHR_TO_ANGSTROM
                    
                continue
            if values[0] == "ATOMIC_POSITIONS":
                read_cell = False
                read_atoms = True
                if "crystal" in values[1].lower():
                    read_crystal = True
                    
                continue
            
            
            if read_cell and cell_index < 3:
                cell[cell_index, :] = [np.float64(v)*alat for v in values]
                cell_index += 1
            elif cell_index == 3:
                read_cell = False

            if read_atoms:
                self.atoms.append(values[0])
                if not read_crystal:
                    tmp_coords.append([np.float64(v)*alat for v in values[1:4]])
                else:
                    # Read the crystal coordinate without taking care of alat
                    tmp_coords.append([np.float64(v) for v in values[1:4]])

                n_atoms += 1
        fp.close()
        
            
            
        # Initialize the structure
        self.coords = np.zeros((n_atoms, 3), dtype = np.float64)
        self.N_atoms = n_atoms
        
        if cell_present:
            self.has_unit_cell = True
            self.unit_cell = cell
            
        for i, coord in enumerate(tmp_coords):
            self.coords[i,:] = np.array(coord, dtype = np.float64)
            
            # Transform the coordinates if crystal
            if read_crystal:
                if not cell_present:
                    raise ValueError("Error, read crystal coordinates but no cell given in %s" % filename)
               
                self.coords[i,:] = np.einsum("ij, i", self.unit_cell, self.coords[i,:])
        
        #print "COORDS:", self.coords
        

    def read_generic_file(self, filename):
        """
        This reader use ASE to parse the input and build the appropriate structure.
        Any ASE accepted file is welcome.
        This very simple reader uses the ase environment.
        """
        
        if not __ASE__:
            print("ASE library not found.")
            raise ImportError("Error, ASE library is required to read generic file.")
            
        atoms = ase.io.read(filename)

        # Now obtain all the information
        self.unit_cell = atoms.get_cell()
        self.has_unit_cell = True
        self.atoms = atoms.get_chemical_symbols()
        self.N_atoms = len(self.atoms)
        self.coords = atoms.positions.copy()
        

    def set_unit_cell(self, filename, delete_copies = False, rescale_coords = False):
        """
        Read the unit cell from the filename.
        The rows of the filename are the unit cell vectors!

        Parameters
        ----------
            - filename : string
                 The path of the file that contains the unit cell, in the numpy datafile format (text)
            - delete_copies : bool, optional
                 If true the delete_copies subroutine is lounched after the creation of the unit cell
                 (default False)
            - rescale_coords : bool, optional
                 If true ths system will be multiplied rows by column by the unit cell (default False)
        """
        
        # Load the unit cell
        self.unit_cell = np.loadtxt(filename)
        self.has_unit_cell = True

        if delete_copies:
            self.delete_copies(verbose = False)

        if rescale_coords:
            for i in range(self.N_atoms):
                self.coords[i,:] = self.unit_cell.dot(self.coords[i,:])
                
    def change_unit_cell(self, unit_cell):
        """
        This method change the unit cell of the structure keeping fixed the crystal coordinates.
        
        NOTE: the unit_cell argument will be copied, so if the unit_cell variable is modified, this will not
        affect the unit cell of this structure.
        
        Parameters
        ----------
            unit_cell : numpy ndarray (3x3)
                The new unit cell
        """
        if not self.has_unit_cell:
            raise ValueError("Error, the structure must already have a unit cell initialized.")
        
        # Get the crystal coordinates
        crys_coord = np.zeros(np.shape(self.coords))
        for i in range(self.N_atoms):
            crys_coord[i,:] = Methods.covariant_coordinates(self.unit_cell, self.coords[i,:])
        
        # Setup the new unit cell
        self.unit_cell = unit_cell.copy()
        
        # Modify the coordinates
        for i in range(self.N_atoms):
            self.coords[i,:] = np.einsum("ij, i", self.unit_cell, crys_coord[i,:])
        

    def export_unit_cell(self, filename):
        """
        This method save the unit cell on the given file.
        The rows will be the direct lattice vectors.
        
        Parameters
        ----------
           - filename : string
                The filename in which to save the unit cell
        
        """

        np.savetxt(filename, self.unit_cell, header = "Rows are the unit cell vectors")

    def get_reciprocal_vectors(self):
        """
        RECIPROCAL LATTICE
        ==================

        Get the vectors of the reciprocal lattice. The self structure
        must have the unit cell initialized (A NoUnitCell exception will be reised otherwise).

        Results
        -------
           - reciprocal_vectors : float ndarray 3x3
                 A matrix whose rows are the vectors of the reciprocal lattice
        """

        if not self.has_unit_cell:
            raise ValueError("Error: the specified structure has not the unit cell.")

        return Methods.get_reciprocal_vectors(self.unit_cell) * 2 * np.pi
        #return np.transpose(np.linalg.inv(self.unit_cell)) * 2 * np.pi
        
        
    def delete_copies(self, minimum_dist=1e-6, verbose=False):
        """
        This method checks if double atoms are present in the structure,
        and delete them.

        Parameters
        ----------
           - minimum_dist : double precision, optional
                the minimum distance between two atoms of the same type allowed in the structure.
           - verbose : bool (logical), optional
                if True print on stdout how many atoms have been deleted (default False)
        """

        list_pop = []
        for i in range(self.N_atoms-1):
            # Avoid to consider replica atoms already found
            if (i in list_pop): 
                continue

            # If the atom is not a replica, then found if there are its replica missing
            for j in range(i+1, self.N_atoms):
                if (self.atoms[i] != self.atoms[j]): 
                    continue
                if j in list_pop:
                    continue 
                
                # Get the axis
                v1 = self.coords[i, :]
                v2 = self.coords[j, :]

                d = np.sqrt(np.sum( (v1-v2)**2))
                if (self.has_unit_cell):
                    d = Methods.get_min_dist_into_cell(self.unit_cell, v1, v2)
                
                # # Apply the unit cell if necessary
                # distances = []
                # if (self.has_unit_cell):
                #     # For each vector in the unit cell, add a distance 
                #     shifts = [-1,0,1]
                #     for i_x, x_u in enumerate(shifts):
                #         new_x = x1 + x_u * self.unit_cell[i_x, :]
                #         for i_y, y_u in enumerate(shifts):
                #             new_y = y1 + y_u * self.unit_cell[i_y, :]
                #             for i_z, z_u in enumerate(shifts):
                #                 new_z = z1 + z_u * self.unit_cell[i_z, :]

                #                 # Add the transformed distance
                #                 distances.append( np.sqrt((x-new_x)**2 + (y - new_y)**2 + (z - new_z)**2))
                # else:
                #     # Get the first distance between atoms
                #     distances.append(np.sqrt( (x-x1)**2 + (y-y1)**2 + (z-z1)**2 ))
                                           
                        
                if (d < minimum_dist):
                    # Add the atom as a replica
                    list_pop.append(j)


        # Print how many replica have been found
        N_rep = len(list_pop)
        if verbose:
            print("Found %d replica" % N_rep)

        # Delete the replica
        #list_pop = list(set(list_pop)) # Avoid duplicate indices
        list_pop.sort(reverse=True)
        #print list_pop, self.N_atoms
        for index in list_pop:
            #print index
            del self.atoms[index]

        self.coords = np.delete(self.coords, list_pop, axis = 0)
        self.N_atoms -= N_rep
            
    def apply_symmetry(self, sym_mat, delete_original = False, thr = 1e-6):
        """
        This function apply the symmetry operation to the atoms
        of the current structure.

        Parameters
        ----------
         - sym_mat : (matrix 3x4)
               The matrix of the symemtri operation, the final column is the translation
        
         - delete_original : bool, default False
               If true only the atoms after the symmetry application are left (good to force symmetry)
         
         - thr : float, optional
               The threshold for two atoms to be considered the same in the reduction process 
               (must be smaller than the minimum distance between two generic atoms in the struct,
               but bigger than the numerical error in the wyckoff positions of the structure).
        """

        if not self.has_unit_cell:
            raise ValueError("The structure has no unit cell!")

        #self.N_atoms *= 2
        new_atoms = np.zeros( (self.N_atoms, 3))
        self.fix_coords_in_unit_cell()
        for i in range(self.N_atoms):
            # Convert the coordinates into covariant
            old_coords = Methods.covariant_coordinates(self.unit_cell, self.coords[i, :])

            # Apply the symmetry
            new_coords = sym_mat[:, :3].dot(old_coords)
            new_coords += sym_mat[:, 3]

            # Return into the cartesian coordinates
            coords = np.dot( np.transpose(self.unit_cell), new_coords)

            # Put the atoms into the unit cell
            new_atoms[i, :] = Methods.put_into_cell(self.unit_cell, coords)
                
            # Add also the atom type
            if not delete_original:
                self.atoms.append(self.atoms[i])

        # Concatenate
        if delete_original:
            self.coords = new_atoms
        else:
            self.N_atoms *= 2
            self.coords = np.concatenate( (self.coords, new_atoms), axis = 0)

            self.delete_copies(verbose = False, minimum_dist = thr)

    def check_symmetry(self, sym_mat, thr = 1e-6):
        """
        This method check if the provided matrix is actually a symmetry for the given system
        
        Parameters
        ----------
          - sym_mat: a 3 rows by 4 columns matrix (float)
               It contains the rotation matrix (the first 3x3 block) 
               and the traslation vector (the last column) of the symmetry
          - thr : float, optional
               The threshold for two atoms to be considered the same. 

        Results
        -------
          - check : bool
              It is true if the given matrix is a real symmetry of the system.
        """

        # Copy the struct
        new_struct = self.copy()

        # Apply the symmetry
        new_struct.apply_symmetry(sym_mat, delete_original=True)
        
        # Get the equivalence
        eq_atoms = new_struct.get_equivalent_atoms(self)
        
        # Exchange the atoms
        new_struct.coords[eq_atoms, :] = new_struct.coords.copy()
        
        # Fix the structure in the unit cell
        new_struct.fix_coords_in_unit_cell()
        
        # Get the displacements
        u_vect = self.get_displacement(new_struct)
        
        # Get the distance between the structures
        dist = np.sqrt(np.sum(u_vect ** 2))
        
        if dist > thr:
            return False
        return True

        
    def set_ita_group(self, group):
        """
        This function setup the ita group of the cell,
        the unit cell must be initialized and the ITA group must be
        inside the supported one.
        All the symmetries of the specified group are applied.

        Parameters
        ----------
            - group : int
                The ITA identifier of the symmetry group.
        """

        # Apply all the symmetries
        sym_mats = SYM.get_symmetries_from_ita(group)
        self.ita = group

        for mat in sym_mats:
            self.apply_symmetry(mat)
        
    def load_symmetries(self, filename, progress_bar=False, verbose = False):
        """
        This function loads the symmetries operation from a specific file
        and applies them to the system.
        The file must init with the total number of symmetries, and followed by
        N 3 rows x 4 columns matrices that represent the symmetry application.

        Parameters
        ----------
            filename : string
                The path in which the symmetries are stored, a text file.
            progress_bar : bool
                If true a progress bar on stderr is shown, usefull if the system is very large and
                this function can take a while.
        """


        # Get the number of symmetries
        symfile = open(filename)
        N_sym = int(symfile.readline().strip())
        symfile.close()

        # Get the symmetries
        symdata = np.loadtxt(filename, skiprows = 1)

        if (progress_bar): print()

        for i in range(N_sym):
            sym_mat = symdata[3*i:3*(i+1), :]

            self.apply_symmetry(sym_mat)

            if (progress_bar):
                if not verbose:
                    sys.stderr.write("\rProgress computing symmetries... %d of %d %%" % (i, N_sym) )
                else:
                    sys.stderr.write("\rSymmetry %d out of %d, %d atoms" % (i, N_sym, self.N_atoms ) )

                sys.stderr.flush()

        if (progress_bar): print()

    def impose_symmetries(self, symmetries, threshold = 1.0e-6, verbose = True):
        """
        This methods impose the list of symmetries found in the given filename.
        It solves a self-consistente equation: Sx = x. If this equation is not satisfied at precision
        of the initial_threshold the method will raise an exception.
        
        Parameters
        ----------
           - symmetries : list
                The simmetries to be imposed as a list of 3x4 ndarray matrices. The last column is the
                fractional translations
           - threshold : float
                The threshold for the self consistent equation. The algorithm stops when Sx = x is satisfied
                up to the given threshold value for all the symmetries.
           - verbose : bool
                If true the system will print on stdout info about the self-consistent threshold
        
        """

        # An array storing which symmetry operation has reached the threshold
        aux_struct = self.copy()
        

        # Start the self consistent algorithm
        running = True
        index = 0
        while running:
            old_coords = np.zeros( np.shape(self.coords))
            
            for sym in symmetries:
                aux_struct = self.copy()
                aux_struct.apply_symmetry(sym, delete_original = True)
                aux_struct.fix_coords_in_unit_cell()
                
                # Get the equivalent atoms
                eq_atoms = self.get_equivalent_atoms(aux_struct)
                
                #ase.visualize.view(self.get_ase_atoms())
                #ase.visualize.view(aux_struct.get_ase_atoms())
                
                # Order the atoms
                aux_struct.atoms = [aux_struct.atoms[item] for item in eq_atoms]
                aux_struct.coords = aux_struct.coords[eq_atoms,:]
                
                # Get the displacements
                old_coords += self.get_displacement(aux_struct)

            # Average
            old_coords /= len(symmetries)
            
            r = np.max(np.sqrt(np.sum((old_coords)**2, axis = 1)))
#            
#            if verbose:
#                print np.sqrt(np.sum((old_coords - self.coords)**2, axis = 1))
#                print "Self:"
#                print self.coords
#                print "New:"
#                print old_coords
                
            
            self.coords -= old_coords
            if r < threshold:
                running = False

            index += 1
            if (verbose):
                print("Self-consistent iteration %d -> r = %.3e | threshold = %.3e" % (index, r, threshold))
            
        if (verbose):
            print("Symmetrization reached in %d steps." % index)
        


    def get_equivalent_atoms(self, target_structure, return_distances = False):
        """
        GET EQUIVALENT ATOMS BETWEEN TWO STRUCTURES
        ===========================================
        
        
        This function returns a list of the atom index in the target structure that 
        correspond to the current structure.
        NOTE: This method assumes that the two structures are equal.
        
        
        Parameters
        ----------
            target_structure : Structure()
                This is the target structure to be used to get the equivalent atoms.
            return_distances : bool
                If True it returns also the list of the distances between the atoms
                
        Results
        -------
            list
                list of int. Each integer is the atomic index of the target_structure equivalent to the i-th element
                of the self structure.
        """
        
        # Check if the structures are compatible
        if self.N_atoms != target_structure.N_atoms:
            raise ValueError("Error, the target structure must be of the same type of the current one")
            
        for typ in self.atoms:
            if not typ in target_structure.atoms:
                raise ValueError("Error, the target structure must be of the same type of the current one")
            if self.atoms.count(typ) != target_structure.atoms.count(typ):
                raise ValueError("Error, the target structure must be of the same type of the current one")
        
        
        
        equiv_atoms = []
        effective_distances = []
        for i in range(self.N_atoms):
            i_typ = self.atoms[i]
            
            # Select the possible equivalent atoms in the target structure
            target_indices = [x for x in range(self.N_atoms) if target_structure.atoms[x] == i_typ and not (x in equiv_atoms)]
            
            # For each possible equivalent atoms get the minimum distance
            d = []
            for j in target_indices:
                d.append(Methods.get_min_dist_into_cell(self.unit_cell, self.coords[i,:], target_structure.coords[j, :]))
            
            # Pick the minimum
            j_min = target_indices[ np.argmin(d) ]
            effective_distances.append(np.min(d))
            
            # Set the equivalent atom index
            equiv_atoms.append(j_min)
        
        #print "Max distance:", np.max(effective_distances)
        if return_distances:
            return equiv_atoms, effective_distances
        return equiv_atoms


    def sort_molecules(self, distance = 1.3):
        """
        This method sorts the atom lists to have the atoms in the same molecule written subsequentially.

        Parameters
        ----------
           - distance : double precision, optional
                The distance below wich two atoms are considered to be bounded. The unit is in Argstrom.
        """

        molecules = []
        pop_indices = []
        for i in range(self.N_atoms):
            if i in pop_indices: continue

            molecule = [i]
            
            # Get the closest molecules
            for j in range(i+1, self.N_atoms):
                if j in pop_indices: continue

                if np.sqrt(np.sum( (self.coords[i,:] - self.coords[j,:])**2 )) < distance:
                    molecule.append(j)
                    pop_indices.append(j)

            molecules.append(molecule)

        # Resort the atoms
        coords = np.zeros( (self.N_atoms, 3))
        atoms = ["X"] * self.N_atoms

        cont = 0
        for mol in molecules:
            for index in mol:
                atoms[cont] = self.atoms[index]
                coords[cont, :] = self.coords[index,:]
                cont += 1
        self.atoms = atoms
        self.coords = coords
            
    def save_xyz(self, filename, comment="Generated with BUC", overwrite = True):
        """
        This function write the structure on the given filename in the xyz file format
        
        Parameters
        ----------
            filename : string
                The path of the file in which to save the structure. The user must have write access
            comment : string, optional
                This line is written in the comment line of the xyz file.
                NOTE: this string is followed by the unit cell info is present
            overwrite : bool, optional
                If true any precedent file will be erased, otherwise the structure is appended
                on the bottom of the previous one. In this way it is possible to save videos.
                
        """

        if overwrite:
            xyz = open(filename, "w")
        else:
            xyz = open(filename, "a")
            
        # Write the number of atoms
        xyz.write("%d\n" % self.N_atoms)

        # Write the comment line
        unit_cell_string = ""
        if self.has_unit_cell:
            unit_cell_string = "  cell: "
            for i in range(3):
                unit_cell_string += chr( ord('A') + i) + " ".join([str(x_val) for x_val in self.unit_cell[i,:]]) + "   "
        xyz.write("%s\n" % (comment + unit_cell_string))

        # Write the strcture
        lines = []
        for i in range(self.N_atoms):
            label = self.atoms[i]
            x, y, z = self.coords[i, :]
            
            line = " ".join([label, str(x), str(y), str(z)]) + "\n"
            lines.append(line)
        xyz.writelines(lines)
        xyz.close()

    def save_bcs(self, filename, symmetry_file = ""): # STILL NOT WORKING
        """
        Save the current structure in the Bilbao Crystallographic Server file format
        This is very usefull since the BCS website provide a conversor between 
        BCS with most of widely used crystallographic file format.

        NOTE:
        remember to specify the correct ITA group symmetry in the structure.
        You can find more about ITA on BCS website. Otherwise you must specify a file with symmetries

        Parameters
        ----------
           - filename : str
                The path of the bcs file in which you want to save the structure
           - symmetry_file : str, optional
                The path to a file containing the symmetries operations of the group space
                This is not needed if a ITA grup has been specified.
        """

        fp = open(filename, "w")
        fp.write("# Space Group ITA number\n%d\n# Lattice parameters\n" % self.ita)

        # Convert the cell into the a,b,c,alpha,beta,gamma format
        cellbcs = Methods.cell2abc_alphabetagamma(self.unit_cell)
        fp.write("%.8f %.8f %.8f %3d %3d %3d\n" % (cellbcs[0], cellbcs[1], cellbcs[2],
                                                   cellbcs[3], cellbcs[4], cellbcs[5]))

        # Get the independent atoms
        if symmetry_file != "":
            syms = []
            #TODO !!!!!! 
        syms = SYM.get_symmetries_from_ita(self.ita, True)
        
        removing_struct = self.copy()
        
        running = True
        while running:
            # Try to remove an atoms
            total_removed = 0
            for i in range(removing_struct.N_atoms):
                tmp_struct = removing_struct.copy()
                
                # Delete the atom
                tmp_struct.N_atoms -= 1
                tmp_struct.atoms.pop(i)
                np.delete(tmp_struct.coords, i, axis = 0)
                
                # Apply all the symmetries
                for ind, sym in enumerate(syms):
                    tmp_struct.apply_symmetry(sym)
                    print("atom %d, sym %d - NEW %d / %d" % (i, ind, tmp_struct.N_atoms, removing_struct.N_atoms))
                
                if tmp_struct.N_atoms == removing_struct.N_atoms:
                    total_removed += 1
                    removing_struct = tmp_struct.copy()
            
            # If no atoms can be removed, then we obtained the minimal structure
            if not total_removed:
                running = False
            
                    
        # Write the atoms
        fp.write("# Number of independent atoms\n%d\n" % removing_struct.N_atoms)
        fp.write("# [atom type] [number] [WP] [x] [y] [z]\n")
                
        for i in range(removing_struct.N_atoms):
            cvect = Methods.covariant_coordinates(self.unit_cell, removing_struct.coords[i,:])
            vect_str = " ".join(["%.8f" % item for item in cvect])
            fp.write("%2s %3d - %s\n" % (removing_struct.atoms[i], i+1, vect_str))
            
        fp.close()

    def get_xcoords(self):
        """
        Returns the crystalline coordinates
        """

        assert self.has_unit_cell

        xcoords = np.zeros(self.coords)
        for i in range(self.N_atoms):
            xcoords[i,:] = Methods.covariant_coordinates(self.unit_cell, self.coords[i,:])

        return xcoords
    def set_from_xcoords(self, xcoords):
        """
        Set the cartesian coordinates from crystalline
        """

        assert self.has_unit_cell

        for i in range(self.N_atoms):
            self.coords[i,:]  = self.unit_cell.T.dot(xcoords[i,:])


    def save_scf(self, filename, alat = 1, avoid_header=False):
        """
        This methods export the phase in the quantum espresso readable format.
        Of course, only the data reguarding the unit cell and the atomic position will be written.
        The rest of the file must be edited by  the user to start a calculation.
        
        Parameters
        ----------
            filename : string
                The name of the file that you want to save.
            alat : float, optional
                If different from 1, both the cell and the coordinates are saved in alat units.
                It must be in Angstrom.
            avoid_header : bool, optional
                If true nor the cell neither the ATOMIC_POSITION header is printed.
                Usefull for the sscha.x code.
        """

        if alat <= 0:
            raise ValueError("Error, alat must be positive [Angstrom]")

        data = []
        if self.has_unit_cell and not avoid_header:
            unit_cell = np.copy(self.unit_cell)
            if alat == 1:
                data.append("CELL_PARAMETERS angstrom\n")
            else:
                data.append("CELL_PARAMETERS alat\n")
            
            unit_cell /= alat
                
            for i in range(3):
                    data.append("%.16f  %.16f  %.16f\n" % (unit_cell[i, 0],
                                                           unit_cell[i, 1],
                                                           unit_cell[i, 2]))
            data.append("\n")
            
            
        if not avoid_header:
            if alat == 1:
                data.append("ATOMIC_POSITIONS angstrom\n")
            else:
                data.append("ATOMIC_POSITIONS alat\n")
        for i in range(self.N_atoms):
            coords = np.copy(self.coords)
            coords /= alat
            data.append("%s    %.16f  %.16f  %.16f\n" % (self.atoms[i],
                                                         coords[i, 0],
                                                         coords[i, 1],
                                                         coords[i, 2]))

        # Write
        fdata = open(filename, "w")
        fdata.writelines(data)
        fdata.close()
        
        
    def fix_coords_in_unit_cell(self):
        """
        This method fix the coordinates of the structure inside
        the unit cell. It works only if the structure has 
        predefined unit cell.
        """

        if not self.has_unit_cell:
            raise ValueError("Error, try to fix the coordinates without the unit cell")

        for i in range(self.N_atoms):
            self.coords[i,:] = Methods.put_into_cell(self.unit_cell, self.coords[i,:])

        # Delete duplicate atoms
        self.delete_copies()

    def fix_wigner_seitz(self):
        """
        Atoms will be replaced in the periodic images inside the wigner_seitz cell
        """

        assert self.has_unit_cell, "Error, the wigner_seitz is defined for periodic boundary conditions"

        for i in range(self.N_atoms):
            new_r = Methods.get_closest_vector(self.unit_cell, self.coords[i,:])
            self.coords[i, :] = new_r
        
    def get_strct_conventional_cell(self):
        """
        This methods, starting from the primitive cell, returns the same structure 
        in the conventional cell. It picks the angle that mostly differs from 90 deg,
        and transfrom the axis of the cell accordingly to obtain a bigger cell, but similar
        to an orthorombic one.
        The atoms are then replicated and correctly placed inside the new cell.
        
        If the structure does not have a unit cell, the method will raise an error.
        
        NOTE: The new structure will be returned, but this will not be modified
        
        Returns
        -------
            Structure.Structure()
                The structure with the conventional cell
        """
        
        
        if not self.has_unit_cell:
            raise ValueError("Error, the given structure does not have a valid unit cell.")
            
        # Compute the three angles
        angls = np.zeros(3)
        for i in range(3):
            nexti = (i+1)%3
            otheri = (i+2)%3
            angls[otheri] = np.arccos( np.dot(self.unit_cell[i,:], self.unit_cell[nexti,:]) / 
                 (np.sqrt(np.dot(self.unit_cell[i,:], self.unit_cell[i,:])) * 
                  np.sqrt(np.dot(self.unit_cell[nexti,:], self.unit_cell[nexti,:])))) * 180 / np.pi
        
        # Pick the angle that differ the most from 90
        otheri = np.argmax( np.abs( angls - 90))
        #print angls, otheri
        
        # Now select the two vectors between this angle
        vec1 = self.unit_cell[(otheri + 1) % 3,:].copy()
        vec2 = self.unit_cell[(otheri + 2) % 3,:].copy()
        
        # Get the new system
        vec1_prime = vec1 + vec2
        vec2_prime = vec1 - vec2
        
        # Get the new structure
        s_new = self.generate_supercell( (2,2,2) )
        s_new.unit_cell = self.unit_cell.copy()
        s_new.unit_cell[(otheri+1)%3,:] = vec1_prime
        s_new.unit_cell[(otheri+2)%3,:] = vec2_prime

        s_new.fix_coords_in_unit_cell()
        
        return s_new

    def get_ase_atoms(self):
        """
        This method returns the ase atoms structure, ready for computations.

        Results
        -------
            - atoms : ase.Atoms()
                  The ase.Atoms class containing the self structure.
        """
        
        if not __ASE__:
            print ("ASE library not found")
            raise ImportError("Error, ASE library not found")

        # Get thee atom list
        atm_list = []
        for i in range(self.N_atoms):
            atm_list.append(ase.Atom(self.atoms[i], self.coords[i,:]))

        atm = ase.Atoms(atm_list)
        
        if self.has_unit_cell:
            atm.set_cell(self.unit_cell)
            atm.pbc[:] = True

        
        return atm
    
    def get_ityp(self):
        """
        GET THE TYPE ATOMS
        ==================
        
        This is for fortran compatibility. 
        Get the ityp array for the structure. 
        Pass it + 1 to the fortran subroutine to match also the difference
        between python and fortran indices
        
        Results
        -------
            ityp : ndarray of int
                The type of the atom in integer (starting from 0)
        """
        
        if self.masses is None:
            raise ValueError("Error, to return the ityp the masses must be initialized.")
        
        ityp = np.zeros(self.N_atoms, dtype = np.intc)
        
        for i in range(self.N_atoms):
            # Rank the atom number
            
            ityp[i] = list(self.masses).index(self.atoms[i])
        
        return ityp
    
    def get_itau(self, unit_cell_structure):
        """
        GET ITAU
        ========
        
        This subroutine (called by a supercell structure), returns the array
        of the corrispondence between its atoms and those in the unit cell.s

        NOTE: The ITAU is returned in Fortran indexing, subtract by 1 if you want to use it in python
        
        Parameters
        ----------
            - unit_cell_structure : Structure()
                The structure of the unit cell used to generate this supercell structure.

        Results
        -------
            - itau : ndarray (size = nat_sc, type = int)
                For each atom in the supercell contains the index of the corrisponding
                atom in the unit_cell, starting from 1 to unit_cell_structure.N_atoms (included)
        """
        
        itau = np.zeros( self.N_atoms, dtype = np.intc)
        
        for i in range(self.N_atoms):
            
            v1 = Methods.put_into_cell(unit_cell_structure.unit_cell, self.coords[i,:])
            d = np.zeros(unit_cell_structure.N_atoms)
            for j in range(unit_cell_structure.N_atoms):
                d[j] = Methods.get_min_dist_into_cell(unit_cell_structure.unit_cell, v1, unit_cell_structure.coords[j,:])
            
            itau[i] = np.argmin(d) + 1
            
        return itau

    def get_sublattice_vectors(self, unit_cell_structure):
        """
        Get the lattice vectors that connects the atom of this supercell structure to those of
        the unit_cell structure.
        """

        itau = self.get_itau(unit_cell_structure) - 1 
        return self.coords[:,:] - unit_cell_structure.coords[itau[:], :]

    def generate_supercell(self, dim, itau = None, QE_convention = True, get_itau = False):
        """
        This method generate a supercell of specified dimension, replicating the system
        on the n-th neighbours unit cells.

        Parameters
        ----------
            - dim : list, size(3), integer
                  A list that specifies the number of cells for each dimension.
            - itau : ndarray of int, size(Natoms * supercell_size)
                  An array of integer. If it is of the correct shape and type it will be filled
                  with the correspondance of each new vector to the corresponding one in the unit cell
            - QE_convention : bool, optional
                  If true (default) the quantum espresso set_tau subroutine is used to determine
                  the order of how the atoms in the supercell are generated
            - get_itau : bool
                If true also the itau order is returned in output (python convention). 

        Results
        -------
            - supercell : Structure
                  This structure is the supercell of the system.
        """
        

        if len(dim) != 3:
            raise ValueError("ERROR, dim must have 3 integers.")

        if not self.has_unit_cell:
            raise ValueError("ERROR, the specified system has not the unit cell.")

        total_dim = np.prod(dim)

        new_N_atoms = self.N_atoms * total_dim
        new_coords = np.zeros( (new_N_atoms, 3))
        atoms = [None] * new_N_atoms # Create an empty list for the atom's label
        
        
        # Get the new data
        
        
        # Check if itau is passed
        if itau is not None:
            try:
                itau[:] = np.zeros(new_N_atoms, dtype = np.intc)
            except:
                raise ValueError("Error, itau passed to generate_supercell does not match the required shape\nRequired %d, passed %d"% (new_N_atoms, len(itau)))    

        # Start the generation of the new supercell
        if not QE_convention:
            for i_z in range(dim[2]):
                for i_y in range(dim[1]):
                    for i_x in range(dim[0]):
                        basis_index = self.N_atoms * (i_x + dim[0] * i_y + dim[0]*dim[1] * i_z)
                        for i_atm in range(self.N_atoms):
                            new_coords[basis_index + i_atm, :] = self.coords[i_atm, :] + \
                                                                 i_z * self.unit_cell[2, :] + \
                                                                 i_y * self.unit_cell[1, :] + \
                                                                 i_x * self.unit_cell[0, :]
                            atoms[i_atm + basis_index] = self.atoms[i_atm]
                            if itau is not None:
                                itau[i_atm + basis_index] = i_atm
                        
        # Define the new structure
        supercell = Structure()
        supercell.coords = new_coords
        supercell.N_atoms = new_N_atoms
        supercell.atoms = atoms
        supercell.masses = self.masses.copy()
        
        # Define the supercell
        supercell.has_unit_cell = True

        for i in range(3):
            supercell.unit_cell[i, :] = self.unit_cell[i,:] * dim[i]
            
        
        if QE_convention:
            # Prepare the variables
            tau = np.array(self.coords.transpose(), dtype = np.float64, order = "F")
            tau_sc = np.zeros((3, new_N_atoms), dtype = np.float64, order = "F")
            ityp_sc = np.zeros( new_N_atoms, dtype = np.intc)
            ityp = self.get_atomic_types()
            
            at_sc = np.array( supercell.unit_cell.transpose(), dtype = np.float64, order = "F")
            at = np.array( self.unit_cell.transpose(), dtype = np.float64, order = "F")
            
            itau = np.zeros(new_N_atoms, dtype = np.intc)
#            
#            print "AT SC:", at_sc
#            print "AT:", at
#            print "TAU SC:", tau_sc
#            print "TAU:", tau
#            
            # Fill the atom
            symph.set_tau(at_sc, at, tau_sc, tau, ityp_sc, ityp, itau, new_N_atoms, self.N_atoms)
            
            
            supercell.coords[:,:] = tau_sc.transpose()
            itau -= 1 # Fortran To Python indexing
            supercell.atoms = [self.atoms[x] for x in itau] 
            
            
        
        if get_itau:
            return supercell, itau 
        return supercell
    
    def reorder_atoms_supercell(self, reference_structure):
        """
        ORDER THE ATOMS
        ===============
        
        This subroutines order the atoms to match the same order as in the 
        generate_supercell method.
        The self structure is supposed to be a structure that belongs to a supercell 
        of the given unit_cell, then it is reordered so that each atom in any different 
        supercell are consequent and the order of the supercell matches the one
        created by generate supercell. The code will work even if the structures 
        do not match exactly the supercell generation. In this case, the closest
        unit cell atom of the correct type is used as reference.
        
        TODO: THIS DOES NOT WORK!!!!
        
        Parameters
        ----------
            - reference_structure : Structure()
                The cell and coordinates that must be used as a reference
                to reorder the atoms
                
        Results
        -------
            - itau : ndarray of int
                The shuffling array to order any array of this list
                
        """
        #raise ValueError("Subroutine still not working...")
        
        if not reference_structure.has_unit_cell:
            raise ValueError("Error, the reference structure must have a unit cell")
        
        if not self.has_unit_cell:
            raise ValueError("Error, the self structure must have a unit cell")
        
        unit_cell = reference_structure.unit_cell
        reference_coords = reference_structure.coords
        
        # Get the supercell size
        sx  = self.unit_cell[0,:].dot(unit_cell[0,:]) / unit_cell[0,:].dot(unit_cell[0,:])
        sy  = self.unit_cell[1,:].dot(unit_cell[1,:]) / unit_cell[1,:].dot(unit_cell[1,:])
        sz  = self.unit_cell[2,:].dot(unit_cell[2,:]) / unit_cell[2,:].dot(unit_cell[2,:])
        
        supercell_size = (int(sx + .5), int(sy + .5), int(sz + .5))
        
        print ("SUPERCELL:", supercell_size)
        
        # Atoms in the unit cell
        nat_uc = np.shape(reference_coords)[0]
        
        # Check if they match the given structure
        if self.N_atoms % nat_uc != 0:
            raise ValueError("Error, the number of atoms in this structure %d is not a multiple of %d (the reference structure)" % (self.N_atoms, nat_uc))
            
        # The shuffling array
        itau = np.arange(self.N_atoms)
        
        # Get cristal coordinates
        for i in range(self.N_atoms):
            cov = Methods.covariant_coordinates(unit_cell, self.coords[i,:])
        
            # Identify the cell
            i_x = int(cov[0] + .5)
            i_y = int(cov[1] + .5)
            i_z = int(cov[2] + .5)
            
            print (cov[0], cov[1], cov[2], i_x, i_y, i_z)
            
            # Get the index of the cell
            basis_index = nat_uc * (i_x + supercell_size[0] * i_y + supercell_size[0] * supercell_size[1] * i_z)
            
            # Identify the atom
            d = np.zeros(nat_uc, dtype = np.float32)
            mask_good = np.zeros(nat_uc, dtype = bool)
            for j in range(nat_uc):
                if self.atoms[i] == reference_structure.atoms[j]:
                    mask_good[j] = True
                    d[j] = Methods.get_min_dist_into_cell(unit_cell, self.coords[i,:], reference_coords[j])
                
            # Avoid to pick a wrong atom type
            d[~mask_good] = np.max(d) + 1
            
            # Avoid that two atoms point to the same position
            process = True
            while process:
                
                # Get the atom corresponding to the minimum distance
                atm_index = np.argmin(d)
                index = basis_index + atm_index
                
                #print "Chosen %d -> %d" % (i, index), "ITAU:", itau[:i]
                
                # Check if another atom already matched this one
                if index in itau[:i]:
                    d[atm_index] = np.max(d) + 1
                else:
                    process = False
                
            
            itau[i] = index
        
        # Now shuffle the current structure
        self.coords = self.coords[itau, :]
        
        # Now shuffle the atom types
        new_atoms = []
        for i in range(self.N_atoms):
            new_atoms.append(self.atoms[itau[i]])
        self.atoms = new_atoms
            
        # Return the shuffling array
        return itau
            
        

    def get_min_dist(self, index_1, index_2):
        """
        This method returns the minimum distance between atom index 1 and atom index 2.
        It uses the unit cell to correctly take into account the atoms at the edge of the unit cell.
        
        Parameters
        ----------
          - index_1 : int
              The index of the first atom in the structure
          - index_2 : int
              The index of the second atom in the structure

        Results
        -------
          - min_dist : float
              The minimum distance between the chosen atoms, eventually traslated by the unit cell.
        """

        vector1 = self.coords[index_1, :]
        vector2 = self.coords[index_2, :]
        
        if not self.has_unit_cell:
            return np.sqrt( np.sum( (vector1 - vector2)**2))

        # Get the covariant components
        cell = self.unit_cell
        metric_tensor = np.zeros((3,3))
        for i in range(0, 3):
            for j in range(i, 3):
                metric_tensor[i, j] = metric_tensor[j,i] = cell[i,:].dot(cell[j, :])

        imt = np.linalg.inv(metric_tensor)
        
        # Get contravariant components
        contra_vect = np.zeros(3)
        for i in range(3):
            contra_vect[i] = vector1.dot(cell[i, :]) 

        # Invert the metric tensor and obtain the covariant coordinates
        covect1 = imt.dot(contra_vect)
        
        contra_vect = np.zeros(3)
        for i in range(3):
            contra_vect[i] = vector2.dot(cell[i, :]) 

        # Invert the metric tensor and obtain the covariant coordinates
        covect2 = imt.dot(contra_vect)

        covect_distance = covect1 - covect2

        # Bring the distance as close as possible to zero
        covect_distance -= (covect_distance + np.sign(covect_distance)*.5).astype(int)

        # Compute the distance using the metric tensor
        return np.sqrt(covect_distance.dot(metric_tensor.dot(covect_distance)))


    def get_brillouin_zone(self, ISO_MESH=10): # NOT WORKING -----
        """
        BRILLOUIN ZONE
        ==============

        This function uses ase utilities to plot the Brillouin zone.
        TODO: Z primitive cell must be perpendicular to the others (only few reticulus)

        NOT WORKING!!!! 

        Parameters
        ----------
            - ISO_MESH : int (default 100)
                 The number of points for the volume mesh (to the 3 power)

        Results
        -------
            - BZone : array of 3D vectors
                 The points of the ISO_MESH inside the first brillouin zone
        """

        # Get the reciprocal lattice vectors
        b_vectors = self.get_reciprocal_vectors()

        b_mod = np.sum( b_vectors**2, axis = 1)

        metric_tensor = np.zeros((3,3))
        for i in range(0, 3):
            for j in range(i, 3):
                metric_tensor[i, j] = metric_tensor[j,i] = b_vectors[i,:].dot(b_vectors[j, :])
        invmt = np.linalg.inv(metric_tensor)
                
        # Uniformly fill the Reciprocal Unit Cell
        spacing = np.linspace(-.5, .5, ISO_MESH)

        vectors = []

        # Create all the surface in the contravariant coordinates
        for x in spacing:

            mask1 = (spacing - x <= 0.5) & (spacing -x >= -.5)
            mask2 = (spacing + x <= .5) & (spacing + x >= -.5)
            
            for y in spacing[mask1 & mask2]:
                # if abs(y) != 0.5 and  abs(x) != 0.5 and abs(y+x) != 0.5 and abs(y-x) != 0.5:
                #     continue
                
                for z in [-0.5, 0.5]:
                    contravect = np.array([b_mod[0] * x, b_mod[1] * y, b_mod[2] * z])
                    covect = invmt.dot(contravect)
                    vectors.append(covect)

        # TODO NOT WORKING
        pass

        return np.array(vectors)
    
    def GetBiatomicMolecules(self, atoms, distance, tollerance=0.01, return_indices = False):
        """
        GET BIATOMIC MOLECULE
        =====================
        
        This function allows one to extract from a structure all the biatomic 
        molecules that contains the two atoms specified and that are at the distance
        with a given tollerance.
        This is very usefull to compute some particular average bond length.
        
        Parameters
        ----------
            - atoms : list (char) (size = 2)
                The atomic symbols of the molecule
            - distance : float
                The average distance between the two atom in the molecule
            - tollerance : float, default 0.01
                The tollerance on the distance after which the two atoms are
                no more consider inside the same molecule.
            - return_indices : bool, default false
                If true, per each molecule is returned also the list of the
                original indices inside the structure.
            
        Results
        -------
            - Molecules : list
                List of molecules (Structure) that matches the input.
                If none is found an empty list is returned
        """
        # Check if the atoms is a 2 char list
        if len(atoms) != 2:
            raise ValueError("Error, the molecule must be biatomic")
            
        for a in atoms:
            if not a in self.atoms:
                raise ValueError("Error, the atom %s is not into this structure" % a)
        
        # Scroll all the atoms in the list that match the first type.
        molecules = []
        original_indices = []
        for index1 in range(self.N_atoms):
            atm1 = self.atoms[index1]
            if atm1 != atoms[0]:
                continue
            
            # Avoid double counting if the molecule is omonuclear
            starting_index = 0
            if atoms[0] == atoms[1]:
                starting_index = index1 + 1
            
            for index2 in range(starting_index, self.N_atoms):
                atm2 = self.atoms[index2]
                if atm2 != atoms[1]:
                    continue
                
                # Check if the distances between the two atoms matches
                d = self.get_min_dist(index1, index2)
                if d > distance - tollerance and d < distance + tollerance:
                    # Create the structure of the molecule
                    mol = Structure()
                    mol.N_atoms = 2
                    mol.atoms = atoms
                    mol.coords = np.zeros((2, 3))
                    
                    # Translate the molecule in the middle of the cell
                    if self.has_unit_cell:
                        for i in range(3):
                            mol.coords[0,:] += self.unit_cell[i,:] / 2.
                            mol.coords[1,:] += self.unit_cell[i,:] / 2.

                    mol.coords[1,:] += self.coords[index2,:] - self.coords[index1,:] 
                    
                    # If the system has a unit cell, put the second atom inside the cell
                    if self.has_unit_cell:
                        mol.coords[1,:] = Methods.put_into_cell(self.unit_cell, mol.coords[1,:])
                    
                    # Append the molecule to the structure
                    molecules.append(mol)
                    original_indices.append( (index1, index2) )
        
        if return_indices:
            return molecules, original_indices
        
        return molecules
    
    def get_displacement(self, target, dtype = np.float64):
        """
        GET THE DISPLACEMENT STRUCTURE
        ==============================
        
        This function will return an array of displacement respect to the target
        of the current structure. Note that the two structures must be compatible.
        
        
        NOTE: if any the self unit_cell will be considered, otherwise the target one.
              no unit cell is used only if neither the self nor the target have one.
        
        Parameters
        ----------
            target : Structure.Structure()
                The reference atomic positions (also this is a structure)
            dtype : type
                The type to be cast the result. By default is the double precision
        
        Results
        -------
            ndarray N_atoms x 3
                The displacements (same shape as self.coords)
        """
        
        # Check if the two structures are compatible
        if self.N_atoms != target.N_atoms:
            raise ValueError("Error, the target must share the same number of atoms")
            
        unit_cell = np.zeros((3,3))
        easy = False
        if not self.has_unit_cell:
            if not target.has_unit_cell:
                easy = True
            else:
                unit_cell = target.unit_cell
        else:
            unit_cell = self.unit_cell
        
        disp = np.zeros(np.shape(self.coords))
        disp = np.float64(self.coords - target.coords)
        if easy:
            return disp
        
        # Check that the cell is good
        for i in range(self.N_atoms):
            # Add half of the the unit cell
            for j in range(3):
                disp[i,:] += unit_cell[j,:] * .5
            
            disp[i,:] = Methods.put_into_cell(unit_cell, disp[i,:])
            
            # Remove again the half cell
            for j in range(3):
                disp[i,:] -= unit_cell[j,:] * .5
        
        return disp
            
    
    def get_angle(self, index1, index2, index3, rad = False):
        """
        GET ANGLE BETWEEN THREE ATOMS
        =============================
        
        This function evaluate the angle between three atoms located
        in the structure at the correct indices. The unit cell is centered around
        the second atom to compute correctly the structure.
        
        
        Parameters
        ----------
            indexI : int
                Index of the Ith atom. (The angle is the one between 1-2-3)
            rad : bool, optional
                If true, the angle is returned in radiants (otherwise in degrees)
        
        Return
        ------
            angle : float
                Value of the angle in degrees (unles rad is specified) between the index1-index2-index3
                atoms of the structure.
            
        """
        
        if index1 >= self.N_atoms or index2 >= self.N_atoms or index3 >= self.N_atoms:
            raise ValueError("Error, the indices must be lower than the number of atoms.")
        
        
        # Get the three vectors
        v1 = self.coords[index1,:].copy()
        v2 = self.coords[index2,:].copy()
        v3 = self.coords[index3,:].copy()
        
        # center with respect of v2
        v1 -= v2
        v2 -= v2
        v3 -= v2
        
        # Manipulate them if there is an unitcell
        if self.has_unit_cell:
            # Sum half of the cell vectors
            for i in range(3):
                v1 += self.unit_cell[i,:] * .5
                v2 += self.unit_cell[i,:] * .5
                v3 += self.unit_cell[i,:] * .5
            
            # Put the vectors in the unit cell
            v1 = Methods.put_into_cell(self.unit_cell, v1)
            v2 = Methods.put_into_cell(self.unit_cell, v2)
            v3 = Methods.put_into_cell(self.unit_cell, v3)
            
            # Center again around v2
            for i in range(3):
                v1 -= self.unit_cell[i,:] * .5
                v2 -= self.unit_cell[i,:] * .5
                v3 -= self.unit_cell[i,:] * .5
        
        # Now we can measure the angle
        angle = np.arccos(np.dot(v1, v3) / np.sqrt(np.dot(v1, v1) * np.dot(v3, v3)))
        
        # Degree conversion
        if not rad:
            angle *= 180 / np.pi
        
        return angle
                        
                                    
    
    def GetTriatomicMolecules(self, atoms, distance1, distance2, angle, thr_dist=0.01, thr_ang = 1, return_indices = False):
        """
        GET TRIATOMIC MOLECULE
        =====================
        
        This function allows one to extract from a structure all the triatomic 
        molecules that contains the atoms specified and that are at the distance and angle
        with a given tollerance.
        This is very usefull to compute some particular average bond length.
        
        The two distances are between the first-second and second-third atom, while the angle
        is between first-second-third atom.
        
        Be carefull if the atoms are equal and the distance1 and distance2 are very similar
        the algorithm can find twice the same molecules.
        
        Parameters
        ----------
            - atoms : list (char) (size = 3)
                The atomic symbols of the molecule
            - distance1 : float
                The average distance between the first two atom in the molecule
            - distance2 : float
                The average distance between the last two atom in the molecule
            - angle : float
                Angle (in degree) between the central atom and the other two.
            - thr_dist: float, default 0.01
                The tollerance on the distance after which the two atoms are
                no more consider inside the same molecule.
            - thr_angle: float, default 1
                Tollerance for the angle
            - return_indices : bool, default false
                If true, per each molecule is returned also the list of the
                original indices inside the structure.
            
        Results
        -------
            - Molecules : list
                List of molecules (Structure) that matches the input.
                If none is found an empty list is returned
        """
        # Check if the atoms is a 3 char list
        if len(atoms) != 3:
            raise ValueError("Error, the molecule must be triatomic")
            
        for a in atoms:
            if not a in self.atoms:
                raise ValueError("Error, the atom %s is not into this structure" % a)
        
        # Scroll all the atoms in the list that match the first type.
        molecules = []
        original_indices = []
        for index1 in range(self.N_atoms):
            atm1 = self.atoms[index1]
            if atm1 != atoms[0]:
                continue
            
            # Avoid double counting if the molecule is omonuclear
            starting_index = 0
            if atoms[0] == atoms[1]:
                starting_index = index1 + 1
            
            for index2 in range(starting_index, self.N_atoms):
                atm2 = self.atoms[index2]
                if atm2 != atoms[1]:
                    continue
                
                if index2 == index1:
                    continue
                
                # Check if the distances between the two atoms matches
                d = self.get_min_dist(index1, index2)
                #print "1) Selected %d %d => d = %.3f" % (index1, index2, d)
                if not (d > distance1 - thr_dist and d < distance1 + thr_dist):
                    continue
                
                # Accepted the first two atoms
                for index3 in range(0, self.N_atoms):
                    if index3 in [index1, index2]:
                        continue
                    
                    d = self.get_min_dist(index2, index3)
                    #print "2) Selected %d %d => d = %.3f" % (index2, index3, d)
                    
                    if not (d > distance2 - thr_dist and d < distance2 + thr_dist):
                        continue
                    
                    # Ok accepted for distance
                    # Check also the angle
                    ang = self.get_angle(index1, index2, index3)
                    print ("A> %d %d %d = %.3f" % (index1, index2, index3, ang))
                    
                    if not (ang > angle - thr_ang and ang < angle + thr_ang):
                        continue
                    
                    
                    # Create the structure of the molecule
                    mol = Structure()
                    mol.N_atoms = 3
                    mol.atoms = atoms
                    mol.coords = np.zeros((3, 3))
                    mol.unit_cell = self.unit_cell
                    mol.has_unit_cell = True
                    
                    # Translate the molecule in the middle of the cell
                    if self.has_unit_cell:
                        for i in range(3):
                            mol.coords[0,:] += self.unit_cell[i,:] / 2.
                            mol.coords[1,:] += self.unit_cell[i,:] / 2.
                            mol.coords[2,:] += self.unit_cell[i,:] / 2.


                    mol.coords[0,:] += self.coords[index1,:] - self.coords[index2,:] 
                    mol.coords[2,:] += self.coords[index3,:] - self.coords[index2,:] 
                    
                    print ("1-Accepted:", mol.get_min_dist(0,1), mol.get_min_dist(1,2), mol.get_angle(0, 1, 2))

                    # If the system has a unit cell, put the second atom inside the cell
                    if self.has_unit_cell:
                        for k in range(3):
                            mol.coords[k,:] = Methods.put_into_cell(self.unit_cell, mol.coords[k,:])
                    
                    print ("2-Accepted:", mol.get_min_dist(0,1), mol.get_min_dist(1,2), mol.get_angle(0, 1, 2))
                    
                    # Append the molecule to the structure
                    molecules.append(mol)
                    original_indices.append( (index1, index2, index3) )
        
        if return_indices:
            return molecules, original_indices
        
        return molecules
    
    def generate_espresso_input(self, flags):
        """
        GENERATE ESPRESSO INPUT
        =======================
        
        This subroutine will generate the input for a quantum espresso calculation
        """
        pass

    def IsolateAtoms(self, atoms_indices):
        """
        This subroutine returns a Structure() with only the atoms indices identified
        by the provided list.
        
        Parameters
        ----------
            atoms_indices : list of int
                List of the atoms that you want to isolate
        
        Returns
        -------
            new_structure : Structure()
                A structure with only the isolated atoms.
        """
        
        
        new_struct = self.copy()
        nat = len(atoms_indices)
        new_struct.N_atoms = nat
        
        new_struct.coords = np.zeros( (nat, 3), dtype = np.float64)
        new_struct.atoms = [None] * nat
        
        for i, x in enumerate(atoms_indices):
            new_struct.coords[i,:] = self.coords[x,:]
            new_struct.atoms[i] = self.atoms[x]
        
        return new_struct

    def get_inertia_tensor(self):
        """
        GET INERTIA TENSOR
        ====================

        This method get the intertial tensor of the current structure.
        Note periodic boundary conditions will be ingored, 
        so take care that the atoms are correctly centered.

        The units will be the units given for the mass dot the position^2

        Results
        -------
            I : ndarray ( size = (3,3), dtype = np.double)
                The inertia tensor
        """

        # Extract the masses
        m = self.get_masses_array()

        I = np.zeros( (3,3), dtype = np.double)
        E = np.eye(3, dtype = np.double)

        # Get the center of mass
        r_cm = np.einsum("a, ab->b", m, self.coords) / np.sum(m)

        # Get the inertia tensor
        for i in range(self.N_atoms):
            r = self.coords[i, :] - r_cm

            I += m[i] * (E * r.dot(r) - np.outer(r,r))
        
        return I

    def get_classical_rotational_free_energy(self, temperature, unit_mass = "Ry"):
        """
        ROTATIONAL FREE ENERGY
        ======================

        Get the classical free energy of a rigid rotor.

        Parameters
        ----------
            temperature : float
                Temperature in K
            unit_mass : string
                The unit of measurement of the masses.
                It can be one of:
                    - "uma" : the atomic mass unit (1/12 of the C12 nucleus)
                    - "Ry" : the rydberg mass (twice electron mass)
                    - "Ha" : the hartree mass (electron mass)
        Results
        -------
            free_energy : float
                The rotational free energy in eV
        """

        # Get the inertia tensor
        It = self.get_inertia_tensor()

        # convert the mass
        if unit_mass.lower() == "ry":
            It *= MASS_RY_TO_UMA
        elif unit_mass.lower() == "ha":
            It /= ELECTRON_MASS_UMA
        elif unit_mass.lower() == "uma":
            pass
        else:       
            ERROR_MSG = """
    Error, unkwown unit type: {}
"""
            raise ValueError(ERROR_MSG.format(unit_mass))
        

        Idiag, dumb = np.linalg.eigh(It)

        kbT = temperature* K_B
        Z = np.sqrt((2 * np.pi* kbT)**3 * np.prod(Idiag))
        free_energy = - kbT * np.log(Z)
        #free_energy = 3 * kbT * np.log(2*kbT)/2 + kbT / 2 * np.sum(np.log(Idiag))

        return free_energy
