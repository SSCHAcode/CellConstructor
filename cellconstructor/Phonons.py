#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:29:32 2018

@author: pione
"""
import Structure
import numpy as np
import os

BOHR_TO_ANGSTROM = 0.52918

class Phonons:
    """
    Phonons
    ================
    
    
    This class contains the phonon of a given structure.
    It can be used to show and display dinamical matrices, as well as for operating 
    with them
    """
    def __init__(self, structure = None, nqirr = 1, full_name = False):
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
            - full_name : bool
                If full_name is True, then the structure is loaded without appending the
                q point index. This is compatible only with nqirr = 1.
                
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
            self.LoadFromQE(structure, nqirr, full_name = full_name)
        elif (type(structure) == type(Structure.Structure())):   
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
        
                
    def LoadFromQE(self, fildyn_prefix, nqirr=1, full_name = False):
        """
        This Function loads the phonons information from the quantum espresso dynamical matrix.
        the fildyn prefix is the prefix of the QE dynamical matrix, that must be followed by numbers from 1 to nqirr.
        All the dynamical matrices are loaded.
        
        
        Parameters
        ----------
            - fildyn_prefix : type(string)
                Quantum ESPRESSO dynmat prefix (the files are followed by the q irreducible index)
            - nqirr : type(int), default 1
                Number of irreducible q points in the space group (supercell phonons).
                If 0 or negative an exception is raised.
            - full_name : bool, optional
                If it is True, then the dynamical matrix is loaded without appending the q index.
                This is compatible only with gamma point matrices.
        """
        
        # Check if the nqirr is correct
        if nqirr <= 0:
            raise ValueError("Error, the specified nqirr is not valid: it must be positive!")

        if full_name and nqirr > 1:
            raise ValueError("Error, with full_name only gamma matrices are loaded.")

        # Initialize the atomic structure
        self.structure = Structure.Structure()
        
        # Start processing the dynamical matrices
        for iq in range(nqirr):
            # Check if the selected matrix exists
            if not full_name:
                filepath = "%s%i" % (fildyn_prefix, iq + 1)
            else:
                filepath = fildyn_prefix
                
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
                ibrav = int(struct_info[2])
                if ibrav != 0:
                    raise ValueError("Error, only ibrav 0 supported up to now")
                
                nat = int(struct_info[1])
                ntyp = int(struct_info[0])
                alat = float(struct_info[3]) * BOHR_TO_ANGSTROM # We want a structure in angstrom
                
                # Allocate the coordinates
                self.structure.N_atoms = nat
                self.structure.coords = np.zeros((nat, 3))
                
                # Read the atomic type
                atoms_dict = {}
                masses_dict = {}
                for atom_index in range(1, ntyp + 1):
                    atm_line = dynlines[6 + atom_index]
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
                    atom_info = np.array([float(item) for item in dynlines[line_index].split()])
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
                    qpoint = np.array([float(item) for item in dynlines[index].replace("(", ")").split(')')[1].split()])
                    q_star.append(qpoint)
                    self.q_tot.append(qpoint)
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
                
            # Append the new stars for the irreducible q point
            self.q_stars.append(q_star)
        
        # Ok, the matrix has been initialized
        self.initialized = True
        
    def DyagDinQ(self, iq):
        """
        Dyagonalize the dynamical matrix in the given q point index.
        This methods returns both frequencies and polarization vectors.
        The frequencies and polarization are ordered. Negative frequencies are to
        be interpreted as instabilities and imaginary frequency, as for QE.
        
        They are returned 
        
        Parameters
        ----------
            - iq : int
                Tbe index of the q point of the matrix to be dyagonalized.
                
        Results
        -------
            - frequencies : ndarray (float)
                The frequencies (square root of the eigenvalues divided by the masses).
                These are in Ry units.
            - pol_vectors : ndarray (N_modes x 3)^2
                The polarization vectors for the dynamical matrix. They are returned
                in a Fortran fashon order: pol_vectors[:, i] is the i-th polarization vector.
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
        
        # Order the frequencies and the polarization vectors
        sorting_mask = np.argsort(frequencies)
        frequencies = frequencies[sorting_mask]
        pol_vects = pol_vects[:, sorting_mask]
        
        return frequencies, pol_vects
    
    def Copy(self):
        """
        Return an exact copy of itself. 
        This will implies copying all the dynamical matricies and structures inside.
        So take care if the structure is big, because it will overload the memory.
        """
        
        ret = Phonons()
        ret.structure = self.structure.copy()
        ret.q_tot = self.q_tot
        ret.nqirr = self.nqirr
        ret.initialized = self.initialized
        ret.q_stars = self.q_stars
        
        for i, dyn in enumerate(self.dynmats):
            ret.dynmats.append(dyn.copy())
        
        return ret
    
    def CheckCompatibility(self, other):
        """
        This function checks the compatibility between two dynamical matrices.
        The check includes the number of atoms and the atomic type.

        Parameters
        ----------
            - other : Phonons.Phonons()
                The other dynamical matrix to check the compatibility.
                
        Returns
        -------
            bool 
        """
        
        # First of all, check if other is a dynamical matrix:
        if type(other) != type(self):
            return False
        
        # Check if the two structures shares the same number of atoms:
        if self.structure.N_atoms != other.structure.N_atoms:
            return False
        
        # Check if they belong to the same supercell:
        if self.nqirr != other.nqirr:
            return False
        
        # Then they are compatible
        return True
    
    def GetUpsilonMatrix(self, T):
        """
        This subroutine returns the inverse of the correlation matrix.
        It is computed as following
        
        .. math::
            
            \\Upsilon_{ab} = \\sqrt{M_aM_b}\\sum_\\mu \\frac{2\\omega_\\mu}{(1 + n_\\mu)\\hbar} e_\\mu^a e_\\mu^b
            
        It is used to compute the probability of a given atomic displacement.
        The resulting matrix is a 3N x 3N one ordered as the dynamical matrix here.
        
        NOTE: only works for the gamma point.
        
        Parameters
        ----------
            T : float
                Temperature of the calculation (Kelvin)
        
        Returns
        -------
            ndarray(3N x3N)
                The inverse of the correlation matrix.
        """
        K_to_Ry=6.336857346553283e-06

        if T < 0:
            raise ValueError("Error, T must be posititive (or zero)")
        
        if self.nqirr != 1:
            raise ValueError("Error, this function yet not supports the supercells.")
        
        # We need frequencies and polarization vectors
        w, pols = self.DyagDinQ(0)
        
        # Transform the polarization vector into real one
        pols = np.real(pols)
        
        # Discard translations
        w = w[3:]
        pols = pols[:, 3:]
        
        # Get the bosonic occupation number
        nw = np.zeros(np.shape(w))
        if T == 0:
            nw = 0.
        else:
            nw =  1. / (np.exp(w/(K_to_Ry * T)) -1)
        
        # Compute the matrix
        factor = 2 * w / (1. + 2*nw)
        Upsilon = np.einsum( "i, ji, ki", factor, pols, pols)
        
        # Get the masses for the final multiplication
        mass1 = np.zeros( 3*self.structure.N_atoms)
        for i in range(self.structure.N_atoms):
            mass1[ 3*i : 3*i + 3] = np.sqrt(self.structure.masses[ self.structure.atoms[i]])
        
        _m1_ = np.tile(mass1, (3 * self.structure.N_atoms, 1))
        _m2_ = np.tile(mass1, (3 * self.structure.N_atoms, 1)).transpose()
        
        return Upsilon * _m1_ * _m2_
    
    
    def GetProbability(self, displacement, T, upsilon_matrix = None):
        """
        This function, given a particular displacement, returns the probability density
        of finding the system around that displacement. This in practical computes 
        density matrix of the system in this way
        
        .. math::
            
            \\rho(\\vec u) = \\sqrt{\\det(\\Upsilon / 2\\pi)} \\times \\exp\\left[-\\frac 12 \\sum_{ab} u_a \\Upsilon_ab u_b\\right]
            
        Where :math:`\\vec u` is the displacement, :math:`\\Upsilon` is the inverse of the covariant matrix
        computed through the method self.GetUpsilonMatrix().
        
        Parameters
        ----------
            displacement : ndarray(3xN) or ndarray(N, 3)
                The displacement on which you want to compute the probability.
                It can be both an array of dimension 3 x self.structure.N_atoms or
                a bidimensional array of structure (N_atoms, 3).
            T : float
                Temperature (Kelvin) for the calculation. It will be discarded 
                if a costum upsilon_matrix is provided.
            upsilon_matrix : ndarray (3xN)^2, optional
                If you have to compute many times this probability it can be convenient
                to compute only once the upsilon matrix, and recycle it. If it is
                None (as default) the upsilon matrix will be recomputed each time.
                
        Returns
        -------
            float
                The probability density of finding the system in the given displacement.
                
        """
        
        disp = np.zeros( 3 * self.structure.N_atoms)
        
        # Reshape the displacement
        if len(np.shape(displacement)) == 2:
            disp = displacement.reshape( len(disp))
        else:
            disp = displacement
        
        
        if upsilon_matrix is None:
            upsilon_matrix = self.GetUpsilonMatrix(self, T)
        
        # Compute the braket
        braket = np.einsum("i, ij, j", disp, upsilon_matrix, disp)
        
#        # Get the normalization
#        vals = np.linalg.eigvals(upsilon_matrix)
#        vals = vals[np.argsort(np.abs(vals))]
#        
#        vals /= 2*np.pi
#        det = np.prod(vals[3:])
        
        #print "VALS:", vals
        #print "DET : ", det
        #print "BRAKET : ", braket
        
        #norm = np.sqrt( det)
                
        return  np.exp(-braket)