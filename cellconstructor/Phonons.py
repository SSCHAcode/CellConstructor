#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:29:32 2018

@author: pione
"""
import Structure
import numpy as np
import os

class Phonons:
    """
    Phonons
    ================
    
    
    This class contains the phonon of a given structure.
    It can be used to show and display dinamical matrices, as well as for operating 
    with them
    """
    def __init__(self, structure, nqirr = 1):
        """
        INITIALIZE PHONONS
        ==================
        
        The dynamical matrix for a given structure.
        
        Parameters
        ----------
            - structure : type(Structure)  or  type(string)
                This is the atomic structure for which you want to use the phonon calculation.
                It is needed to correctly initialize all the arrays.
                It can be both the Structure, or a filepath containing a quantum ESPRESSO
                dynamical matrix. Up to now only ibrav0 dymat are supported.
            - nqirr : type(int) , default 1
                The number of irreducible q point of the supercell on which you want 
                to compute the phonons. 
                Use 1 if you want to perform a Gamma point calculation.
                
        Results
        -------
            - Phonons : this
                It returns the Phonon class initializated.
        """
        
        # Initialize standard variables
        self.dynmats = []
        self.nqirr = nqirr
        # Q tot contains the total q points (also those belonging to the same star)
        self.q_tot = []
        
        # If this is true then the dynmat can be used
        self.initialized = False
        
        # This contains all the q points in the stars of the irreducible q point
        self.q_stars = []
        self.structure = None
        
        # Check whether the structure argument is a path or a Structure
        if (type(structure) == type("hello there!")):
            # Quantum espresso
            self.LoadFromQE(structure, nqirr)
        else:   
            # Get the structure
            self.structure = structure
            
            if structure.N_atoms <= 0:
                raise ValueError("Error, the given structure cannot be empty.")
            
            # Check that nqirr has a valid value
            if nqirr <= 0:
                raise ValueError("Error, nqirr argument must be a strictly positive number.")
            
            self.dynmats = []
            for i in nqirr:
                # Create a dynamical matrix
                self.dynmats.append(np.zeros((3 * structure.N_atoms, 3*structure.N_atoms)))
                
    def LoadFromQE(self, fildyn_prefix, nqirr=1):
        """
        This Function loads the phonons information from the quantum espresso dynamical matrix.
        the fildyn prefix is the prefix of the QE dynamical matrix, that must be followed by numbers from 1 to nqirr.
        All the dynamical matrices are loaded.
        
        NOTE:
            for now only gamma calculation are supported.
            nqirr = 1
        
        Parameters
        ----------
            - fildyn_prefix : type(string)
                Quantum ESPRESSO dynmat prefix (the files are followed by the q irreducible index)
            - nqirr : type(int), default 1
                Number of irreducible q points in the space group (supercell phonons).
                If 0 or negative an exception is raised.
        """
        
        # Check if the nqirr is correct
        if nqirr <= 0:
            raise ValueError("Error, the specified nqirr is not valid: it must be positive!")
            
        if nqirr != 1:
            raise ValueError("Error, up to now only Gamma point calculation (nqirr=1) are supported.")
            
        # Initialize the atomic structure
        self.structure = Structure.Structure()
        
        # Start processing the dynamical matrices
        for iq in range(nqirr):
            # Check if the selected matrix exists
            filepath = "%s%i" % (fildyn_prefix, iq + 1)
            if not os.path.isfile(filepath):
                raise ValueError("Error, file %s does not exist." % filepath)
            
            # Load the matrix as a regular file
            dynfile = file(filepath, "r")
            dynlines = [line.strip() for line in dynfile.readlines()]
            dynfile.close()
            
            if (iq == 0):
                # This is a gamma point file, generate the structure
                # Go to the third line
                struct_info = dynlines[2].split()
                
                # Check if the ibrav is 0
                print dynlines[2]
                print struct_info
                ibrav = int(struct_info[2])
                if ibrav != 0:
                    raise ValueError("Error, only ibrav 0 supported up to now")
                
                nat = int(struct_info[1])
                ntyp = int(struct_info[0])
                alat = float(struct_info[3])
                
                # Allocate the coordinates
                self.structure.N_atoms = nat
                self.structure.coords = np.zeros((nat, 3))
                
                # Read the atomic type
                atoms_dict = {}
                masses_dict = {}
                for atom_index in range(1, ntyp + 1):
                    atm_line = dynlines[6 + atom_index]
                    print "ATOM %d / %d :" % (atom_index, ntyp), atm_line
                    atoms_dict[atom_index] = atm_line.split("'")[1].strip()
                    
                    # Get also the atomic mass
                    masses_dict[atoms_dict[atom_index]] = float(atm_line.split("'")[-1].strip())
                    
                self.structure.set_masses(masses_dict)
                
                # Read the unit cell
                unit_cell = np.zeros((3,3))
                for i in range(3):
                    unit_cell[i, :] = np.array([float(item) for item in dynlines[4 + i].split()]) * alat
                    
                self.structure.unit_cell = unit_cell
                self.structure.has_unit_cell = True
                
                # Read the atoms
                for i in range(nat):
                    # Jump the lines up to the structure
                    line_index = 7 + ntyp + i
                    print "ATOM %d / %d" % (i, nat),  dynlines[line_index]
                    atom_info = np.array([float(item) for item in dynlines[line_index].split()])
                    print atoms_dict, atom_info[1]
                    self.structure.atoms.append(atoms_dict[int(atom_info[1])])
                    self.structure.coords[i, :] = atom_info[2:] * alat
                    
                
            # From now start reading the dynamical matrix -----------------------
            reading_dyn = True
            q_star = []
            
            # Pop the beginning of the matrix
            while reading_dyn:      
                # Pop the file until you reach the dynamical matrix
                if "Dynamical  Matrix in cartesian axes" in dynlines[0]:
                    reading_dyn = False
                dynlines.pop(0)
                
            # Get the small q point
            reading_dyn = True
            index = 0
            current_dyn = np.zeros((3*self.structure.N_atoms, 3*self.structure.N_atoms), dtype = np.complex64)    
            
            # The atom indices
            atm_i = 0
            atm_j = 0
            coordline = 0
            while reading_dyn:
                if "Diagonalizing" in dynlines[index]:
                    reading_dyn = False
                    
                if "q = " in dynlines[index]:
                    #Read the q
                    dynlines[index].replace("(", ")")
                    qpoint = np.array([float(item) for item in dynlines[index].split(')')[1].split()])
                    q_star.append(qpoint)
                elif "ynamical" in dynlines[index]:
                    # Save the dynamical matrix
                    self.dynmats.append(current_dyn.copy())
                else:
                    # Read the numbers
                    numbers_in_line = dynlines[index].split()
                    if (len(numbers_in_line) == 2):
                        # Setup which atoms are 
                        atm_i = int(numbers_in_line[0]) - 1
                        atm_j = int(numbers_in_line[1]) - 1
                        coordline = 0
                    elif(len(numbers_in_line) == 6):
                        # Read the dynmat
                        for k in range(3):
                            current_dyn[3 * atm_i + coordline, 3*atm_j + k] = float(numbers_in_line[2*k]) + 1j*float(numbers_in_line[2*k + 1])
                        coordline += 1
                
                # Advance in the reading
                index += 1
            
        self.initialized = True
        
    def DyagDinQ(self, iq):
        """
        Dyagonalize the dynamical matrix in the given q point index.
        This methods returns both frequencies and polarization vectors.
        
        Parameters
        ----------
            - iq : int
                Tbe index of the q point of the matrix to be dyagonalized.
                
        Results
        -------
            - frequencies : ndarray (float)
                The frequencies (square root of the eigenvalues divided by the masses).
            - pol_vectors : ndarray (N_modes x 3)
                The polarization vectors for the dynamical matrix
        """
        
        
        
        # First of all get correct dynamical matrix by dividing per the masses.
        real_dyn = np.zeros((3* self.structure.N_atoms, 3*self.structure.N_atoms), dtype = np.complex64)
        for i, atm_type1 in enumerate(self.structure.atoms):
            m1 = self.structure.masses[atm_type1]
            for j, atm_type2 in enumerate(self.structure.atoms):
                m2 = self.structure.masses[atm_type2]
                real_dyn[3*i : 3*i + 3, 3*j : 3*j + 3] = 1 / np.sqrt(m1 * m2)
        

        real_dyn *= self.dynmats[iq]
        
        eigvals, pol_vects = np.linalg.eig(real_dyn)
        
        f2 = np.real(eigvals)
        
        # Check for imaginary frequencies (unstabilities) and return them as negative
        frequencies = np.zeros(len(f2))
        frequencies[f2 > 0] = np.sqrt(f2[f2 > 0])
        frequencies[f2 < 0] = -np.sqrt(-f2[f2 < 0])
        
        return frequencies, pol_vects