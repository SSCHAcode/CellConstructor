#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:44:27 2018

@author: pione
"""
from numpy import *
import numpy as np
import ase
from ase.visualize import view
import sys, os


import symmetries as SYM


class Structure:    
    def __init__(self):
        self.N_atoms=0
        # Coordinates are always express in chartesian axis
        self.coords = zeros((self.N_atoms, 3))
        self.atoms = []
        self.unit_cell = zeros((3,3))
        self.has_unit_cell = False
        self.masses = {}
        self.ita = 0 # Symmetry group in ITA standard

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
        aux.masses = self.masses
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
            raise ErrorInParameters("Function read_xyz, alat = True but no unit cell.")
        
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
        self.coords = zeros((self.N_atoms, 3))
        
        # Read the comment line
        xyz.readline()

        for i in range(self.N_atoms):
            line = xyz.readline()
            atom, x, y, z = line.split()

            self.atoms.append(atom)
            self.coords[i,0] = float(x)
            self.coords[i,1] = float(y)
            self.coords[i,2] = float(z)

            # Rescale the coordinates with the unit cell if requested
            if alat:
                # Not shure if the dot product must be done with the transposed unit cell matrix
                self.coords[i, :] = dot( transpose(self.unit_cell), self.coords[i, :])

        # Close the xyz file
        xyz.close()

    def read_scf(self, filename):
        """
        Read the given filename in the quantum espresso format.
        Note:
        The file must contain only the part reguarding ATOMIC POSITIONS.
        
        Parameters
        ----------
           - filename : str
               The filename containing the atomic positions
        """
        # Check if the specified filename exists
        if not os.path.exists(filename):
            raise InputError("File %s does not exist" % filename)

        # Read the input filename
        fp = open(filename, "r")

        n_atoms = 0
        good_lines = []

        # First read
        read_cell = False
        cell_index = 0
        read_atoms = True
        cell_present = False
        atom_index = 0
        cell = zeros((3,3))
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
                continue
            if values[0] == "ATOMIC_POSITIONS":
                read_cell = False
                read_atoms = True
                continue
            
            
            if read_cell and cell_index < 3:
                cell[cell_index, :] = [float(v) for v in values]
                cell_index += 1
            elif cell_index == 3:
                read_cell = False

            if read_atoms:
                self.atoms.append(values[0])
                tmp_coords.append([float(v) for v in values[1:4]])
                n_atoms += 1
        fp.close()
            
        # Initialize the structure
        self.coords = zeros((n_atoms, 3))
        self.N_atoms = n_atoms
        for i, coord in enumerate(tmp_coords):
            self.coords[i,:] = array(coord)
        if cell_present:
            self.unit_cell = cell
        

    def read_generic_file(self, filename):
        """
        This reader use ASE to parse the input and build the appropriate structure.
        Any ASE accepted file is welcome.
        This very simple reader uses the ase environment.
        """
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
        self.unit_cell = loadtxt(filename)
        self.has_unit_cell = True

        if delete_copies:
            self.delete_copies(verbose = False)

        if rescale_coords:
            for i in range(self.N_atoms):
                self.coords[i,:] = self.unit_cell.dot(self.coords[i,:])

    def export_unit_cell(self, filename):
        """
        This method save the unit cell on the given file.
        The rows will be the direct lattice vectors.
        
        Parameters
        ----------
           - filename : string
                The filename in which to save the unit cell
        
        """

        savetxt(filename, self.unit_cell, header = "Rows are the unit cell vectors")

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
            raise NoUnitCell("Error: the specified structure has not the unit cell.")

        return transpose(linalg.inv(self.unit_cell)) * 2 * pi
        
        
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
            if (i in list_pop): continue

            # If the atom is not a replica, then found if there are its replica missing
            for j in range(i+1, self.N_atoms):
                if (self.atoms[i] != self.atoms[j]): continue
                
                # Get the axis
                x, y, z = self.coords[i, :]
                x1, y1, z1 = self.coords[j, :]

                
                # Apply the unit cell if necessary
                distances = []
                if (self.has_unit_cell):
                    # For each vector in the unit cell, add a distance 
                    shifts = [-1,0,1]
                    for i_x, x_u in enumerate(shifts):
                        new_x = x1 + x_u * self.unit_cell[i_x, :]
                        for i_y, y_u in enumerate(shifts):
                            new_y = y1 + y_u * self.unit_cell[i_y, :]
                            for i_z, z_u in enumerate(shifts):
                                new_z = z1 + z_u * self.unit_cell[i_z, :]

                                # Add the transformed distance
                                distances.append( sqrt((x-new_x)**2 + (y - new_y)**2 + (z - new_z)**2))
                else:
                    # Get the first distance between atoms
                    distances.append(sqrt( (x-x1)**2 + (y-y1)**2 + (z-z1)**2 ))
                                           
                        

                # Select from all the possible atoms in the unit cell translation the
                # one that is closer
                d0 = np.min(distances)
                
                if (d0 < minimum_dist):
                    # Add the atom as a replica
                    list_pop.append(j)


        # Print how many replica have been found
        N_rep = len(list_pop)
        if verbose:
            print "Found %d replica" % N_rep

        # Delete the replica
        list_pop = list(set(list_pop)) # Avoid duplicate indices
        list_pop.sort(reverse=True)
        #print list_pop, self.N_atoms
        for index in list_pop:
            #print index
            del self.atoms[index]

        self.coords = delete(self.coords, list_pop, axis = 0)
        self.N_atoms -= N_rep
            
    def apply_symmetry(self, sym_mat, delete_original = False, thr = 1.e-6):
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
            raise CellError("The structure has no unit cell!")

        #self.N_atoms *= 2
        new_atoms = zeros( (self.N_atoms, 3))
        for i in range(self.N_atoms):
            # Convert the coordinates into covariant
            old_coords = covariant_coordinates(self.unit_cell, self.coords[i, :])

            # Apply the symmetry
            new_coords = sym_mat[:, :3].dot(old_coords)
            new_coords += sym_mat[:, 3]

            # Return into the cartesian coordinates
            coords = dot( transpose(self.unit_cell), new_coords)

            # Put the atoms into the unit cell
            new_atoms[i, :] = put_into_cell(self.unit_cell, coords)
                
            # Add also the atom type
            if not delete_original:
                self.atoms.append(self.atoms[i])

        # Concatenate
        if delete_original:
            self.coords = new_atoms
        else:
            self.N_atoms *= 2
            self.coords = concatenate( (self.coords, new_atoms), axis = 0)

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
        new_struct.apply_symmetry(sym_mat, thr = thr)

        # Count the number of atoms, if they are the same as before, it is a symmetry
        if (new_struct.N_atoms != self.N_atoms):
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
        N_sym = len(sym_mats)

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
        symdata = loadtxt(filename, skiprows = 1)

        if (progress_bar): print ""

        for i in range(N_sym):
            sym_mat = symdata[3*i:3*(i+1), :]

            self.apply_symmetry(sym_mat)

            if (progress_bar):
                if not verbose:
                    sys.stderr.write("\rProgress computing symmetries... %d of %d %%" % (i, N_sym) )
                else:
                    sys.stderr.write("\rSymmetry %d out of %d, %d atoms" % (i, N_sym, self.N_atoms ) )

                sys.stderr.flush()

        if (progress_bar): print ""

    def impose_symmetries(self, filename, threshold = 1.0e-6, beta = 1, initial_threshold = 1.0e-2, verbose = True):
        """
        This methods impose the list of symmetries found in the given filename.
        It solves a self-consistente equation: Sx = x. If this equation is not satisfied at precision
        of the initial_threshold the method will raise an exception.
        
        Parameters
        ----------
           - filename : string
                Path to the file containing the symmetries. It must strat with the number of symmetries
                and followed with the N 3 rows x 4 columns symmetry operations.
           - threshold : float
                The threshold for the self consistent equation. The algorithm stops when Sx = x is satisfied
                up to the given threshold value for all the symmetries.
           - beta : float
                This is the mixing parameter for the self-consistent evolution. The next point is chosen as:
           .. math::

                  x_{n+1} = (1 - \\beta) x_n + \\beta S x_n

           - initial_threshold : float
                This is the required initial threshold for the algorithm. If the self consistent equation Sx = x
                is not satisfied at the begining up to the initial_threshold, the algorithm stops raising an exception.
           - progress_bar : bool
                If true the system will print on stdout info about the self-consistent threshold
        
        """
        symmetries = []
        
        # Get the number of symmetries
        symfile = open(filename)
        N_sym = int(symfile.readline().strip())
        symfile.close()
        
        # Get the symmetries
        symdata = loadtxt(filename, skiprows = 1)

        for i in range(N_sym):
            symmetries.append(symdata[3*i:3*(i+1), :])

        # An array storing which symmetry operation has reached the threshold
        aux_struct = self.copy()
        

        # Start the self consistent algorithm
        running = True
        index = 0
        while running:
            aux_struct.coords = self.coords.copy()
            
            for sym in symmetries:
                aux_struct.apply_symmetry(sym, delete_original = True)

            r = np.max(np.sqrt(np.sum((aux_struct.coords - self.coords)**2, axis = 1)))
            print np.sqrt(np.sum((aux_struct.coords - self.coords)**2, axis = 1))
            print "Self:"
            print self.coords
            print "Aux:"
            print aux_struct.coords
                
            if r > initial_threshold:
                sys.stderr.write("Error on the self consistent algorithm. Initial threshold violated.\n")
                sys.stderr.write("Check carefully if the symmetries, the unit cell and the structure are ok.\n")
                sys.stderr.write("If so try to increase the initial_threshold parameter.\n")

                raise InitialThresold("Initial threshold not satisfied by symmetry %d" % i)
            if r < threshold:
                running = False
            else:
                # Mix the coordinates for the next step
                self.coords = self.coords * (1 - beta) + beta * aux_struct.coords

            index += 1
            if (verbose):
                print "Self-consistent iteration %d -> r/threshold = %.3e" % (index, r / threshold)

        if (verbose):
            print "Symmetrization reached in %d steps." % index
        

                


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

                if sqrt(sum( (self.coords[i,:] - self.coords[j,:])**2 )) < distance:
                    molecule.append(j)
                    pop_indices.append(j)

            molecules.append(molecule)

        # Resort the atoms
        coords = zeros( (self.N_atoms, 3))
        atoms = ["X"] * self.N_atoms

        cont = 0
        for mol in molecules:
            for index in mol:
                atoms[cont] = self.atoms[index]
                coords[cont, :] = self.coords[index,:]
                cont += 1
        self.atoms = atoms
        self.coords = coords
            
    def save_xyz(self, filename, comment="Generated with BUC"):
        """
        This function write the structure on the given filename in the xyz file format
        
        Parameters
        ----------
            filename : string
                The path of the file in which to save the structure. The user must have write access
            comment : string, optional
                This line is written in the comment line of the xyz file.
                NOTE: this string is followed by the unit cell info is present
        """

        xyz = file(filename, "w")

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
        cellbcs = cell2abc_alphabetagamma(self.unit_cell)
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
                    print "atom %d, sym %d - NEW %d / %d" % (i, ind, tmp_struct.N_atoms, removing_struct.N_atoms)
                
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
            cvect = covariant_coordinates(self.unit_cell, removing_struct.coords[i,:])
            vect_str = " ".join(["%.8f" % item for item in cvect])
            fp.write("%2s %3d - %s\n" % (removing_struct.atoms[i], i+1, vect_str))
            
        fp.close()


    def save_scf(self, filename):
        """
        This methods export the phase in the quantum espresso readable format.
        Of course, only the data reguarding the unit cell and the atomic position will be written.
        The rest of the file must be edited by  the user to start a calculation.
        """

        data = []
        data.append("CELL_PARAMETERS angstrom\n")
        for i in range(3):
            data.append("%.16f  %.16f  %.16f\n" % (self.unit_cell[i, 0],
                                            self.unit_cell[i, 1],
                                            self.unit_cell[i, 2]))

        data.append("\n")
        data.append("ATOMIC_POSITIONS angstrom\n")
        for i in range(self.N_atoms):
            data.append("%s    %.16f  %.16f  %.16f\n" % (self.atoms[i],
                                                       self.coords[i, 0],
                                                       self.coords[i, 1],
                                                       self.coords[i, 2]))

        # Write
        fdata = file(filename, "w")
        fdata.writelines(data)
        fdata.close()
        
        
    def fix_coords_in_unit_cell(self):
        """
        This method fix the coordinates of the structure inside
        the unit cell. It works only if the structure has 
        predefined unit cell.
        """

        if not self.has_unit_cell:
            raise InputError("Error, try to fix the coordinates without the unit cell")

        for i in range(self.N_atoms):
            self.coords[i,:] = put_into_cell(self.unit_cell, self.coords[i,:])

        # Delete duplicate atoms
        self.delete_copies()

    def get_ase_atoms(self):
        """
        This method returns the ase atoms structure, ready for computations.

        Results
        -------
            - atoms : ase.Atoms()
                  The ase.Atoms class containing the self structure.
        """

        # Get thee atom list
        atm_list = []
        for i in range(self.N_atoms):
            atm_list.append(ase.Atom(self.atoms[i], self.coords[i,:]))

        atm = ase.Atoms(atm_list)
        atm.set_cell(self.unit_cell)
        return atm

    def generate_supercell(self, dim):
        """
        This method generate a supercell of specified dimension, replicating the system
        on the n-th neighbours unit cells.

        Parameters
        ----------
            - dim : list, size(3), integer
                  A list that specifies the number of cells for each dimension.

        Results
        -------
            - supercell : Structure
                  This structure is the supercell of the system.
        """

        if len(dim) != 3:
            raise InputError("ERROR, dim must have 3 integers.")

        if not self.has_unit_cell:
            raise InputError("ERROR, the specified system has not the unit cell.")

        total_dim = prod(dim)

        new_N_atoms = self.N_atoms * total_dim
        new_coords = zeros( (new_N_atoms, 3))
        atoms = [None] * new_N_atoms # Create an empty list for the atom's label

        # Start the generation of the new supercell
        for i_x in range(dim[2]):
            for i_y in range(dim[1]):
                for i_z in range(dim[0]):
                    basis_index = self.N_atoms * (i_z + dim[0] * i_y + dim[0]*dim[1] * i_x)
                    for i_atm in range(self.N_atoms):
                        new_coords[basis_index + i_atm, :] = self.coords[i_atm, :] + \
                                                             i_z * self.unit_cell[2, :] + \
                                                             i_y * self.unit_cell[1, :] + \
                                                             i_x * self.unit_cell[0, :]
                        atoms[i_atm + basis_index] = self.atoms[i_atm]
                        
        # Define the new structure
        supercell = Structure()
        supercell.coords = new_coords
        supercell.N_atoms = new_N_atoms
        supercell.atoms = atoms

        # Define the supercell
        supercell.has_unit_cell = True

        for i in range(3):
            supercell.unit_cell[i, :] = self.unit_cell[i,:] * dim[i]

        return supercell

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

        # Get the covariant components
        cell = self.unit_cell
        metric_tensor = zeros((3,3))
        for i in range(0, 3):
            for j in range(i, 3):
                metric_tensor[i, j] = metric_tensor[j,i] = cell[i,:].dot(cell[j, :])

        imt = linalg.inv(metric_tensor)
        
        # Get contravariant components
        contra_vect = zeros(3)
        for i in range(3):
            contra_vect[i] = vector1.dot(cell[i, :]) 

        # Invert the metric tensor and obtain the covariant coordinates
        covect1 = imt.dot(contra_vect)
        
        contra_vect = zeros(3)
        for i in range(3):
            contra_vect[i] = vector2.dot(cell[i, :]) 

        # Invert the metric tensor and obtain the covariant coordinates
        covect2 = imt.dot(contra_vect)

        covect_distance = covect1 - covect2

        # Bring the distance as close as possible to zero
        covect_distance -= (covect_distance + sign(covect_distance)*.5).astype(int)

        # Compute the distance using the metric tensor
        return sqrt(covect_distance.dot(metric_tensor.dot(covect_distance)))


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

        b_mod = sum( b_vectors**2, axis = 1)

        metric_tensor = zeros((3,3))
        for i in range(0, 3):
            for j in range(i, 3):
                metric_tensor[i, j] = metric_tensor[j,i] = b_vectors[i,:].dot(b_vectors[j, :])
        invmt = linalg.inv(metric_tensor)
                
        # Uniformly fill the Reciprocal Unit Cell
        spacing = linspace(-.5, .5, ISO_MESH)

        vectors = []

        # Create all the surface in the contravariant coordinates
        for x in spacing:

            mask1 = (spacing - x <= 0.5) & (spacing -x >= -.5)
            mask2 = (spacing + x <= .5) & (spacing + x >= -.5)
            
            for y in spacing[mask1 & mask2]:
                # if abs(y) != 0.5 and  abs(x) != 0.5 and abs(y+x) != 0.5 and abs(y-x) != 0.5:
                #     continue
                
                for z in [-0.5, 0.5]:
                    contravect = array([b_mod[0] * x, b_mod[1] * y, b_mod[2] * z])
                    covect = invmt.dot(contravect)
                    vectors.append(covect)

        # TODO NOT WORKING
        pass

        return array(vectors)