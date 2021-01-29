
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:29:32 2018
@author: pione
"""

from __future__ import print_function
from __future__ import division 

import numpy as np
import os, sys
import scipy, scipy.optimize

import itertools
import cellconstructor.Structure as Structure
import cellconstructor.symmetries as symmetries
import cellconstructor.Methods as Methods
from cellconstructor.Units import *

import warnings

# Import the Fortran Code
import symph

import time


try:
    from mpi4py import MPI
    __MPI__ = True
except:
    __MPI__ = False
    
try:
    import spglib
    __SPGLIB__ = True
except:
    __SPGLIB__ = False

__EPSILON__ = 1e-5 
__EPSILON_W__ = 1e-8

class Phonons:
    """
    Phonons
    ================
    
    
    This class contains the phonon of a given structure.
    It can be used to show and display dinamical matrices, as well as for operating 
    with them
    """
    def __init__(self, structure = None, nqirr = 1, full_name = False, use_format = False, force_real = False):
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
            - force_real : bool
                If True, force the dynamical matrix allocated to be real. This is usefull to spare memory when 
                generating a dynamical matrix at Gamma in real space, that is real by construction.
                
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
        
        # Prepare additional information that can be loaded
        self.dielectric_tensor = None # (3x3 matrix)
        self.effective_charges = None # 3-rank (Natoms, pol electric field, atomic coords) = (nat, 3, 3)
        self.raman_tensor = None # 3-rank (incoming field, outcoming field, atomic coords) = (3,3, 3*nat)
        
        # This alat is read just from QE, but not used
        self.alat = 1
        
        # If this is true then the dynmat can be used
        self.initialized = False
        
        # This contains all the q points in the stars of the irreducible q point
        self.q_stars = []
        self.structure = None

        dtype = np.complex128 
        if force_real:
            dtype = np.float64
        
        # Check whether the structure argument is a path or a Structure
        if (type(structure) == type("hello there!")):
            # Quantum espresso
            self.LoadFromQE(structure, nqirr, full_name = full_name, use_format = use_format)
        elif (type(structure) == type(Structure.Structure())):   
            # Get the structure
            self.structure = structure
            
            if structure.N_atoms <= 0:
                raise ValueError("Error, the given structure cannot be empty.")
            
            # Check that nqirr has a valid value
            if nqirr <= 0:
                raise ValueError("Error, nqirr argument must be a strictly positive number.")
            
            self.dynmats = []
            for i in range(nqirr):
                # Create a dynamical matrix
                self.dynmats.append(np.zeros((3 * structure.N_atoms, 3*structure.N_atoms), dtype = dtype))
                
                # Initialize the q vectors
                self.q_stars.append([np.zeros(3, dtype = np.float64)])
                self.q_tot.append(np.zeros(3, dtype = np.float64))
        
                
    def LoadFromQE(self, fildyn_prefix, nqirr=1, full_name = False, use_format= False):
        r"""
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
            - use_format : bool
                If true, the IQ index of the dynamical matrix is replaced in the specified format, i.e.
                a standard matrix with prefix dyn (dyn1, dyn2, ...) will be dyn{} with the format notation.
                This allows the user to insert the IQ index in many formats and any position of the file name.
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
            if use_format:
                filepath = fildyn_prefix.format(iq+1)
            else:
                if not full_name:
                    filepath = "%s%i" % (fildyn_prefix, iq + 1)
                else:
                    filepath = fildyn_prefix
                    
            if not os.path.isfile(filepath):
                raise ValueError("Error, file %s does not exist." % filepath)
            
            # Load the matrix as a regular file
            dynfile = open(filepath, "r")
            dynlines = [line.strip() for line in dynfile.readlines()]
            dynfile.close()
            
            if (iq == 0):
                # This is a gamma point file, generate the structure
                # Go to the third line
                struct_info = dynlines[2].split()
                
                # Check if the ibrav is 0
                ibrav = int(struct_info[2])
                celldm = np.zeros(6)
                celldm[0] = float(struct_info[3])
                celldm[1] = float(struct_info[4])
                celldm[2] = float(struct_info[5])
                celldm[3] = float(struct_info[6])
                celldm[4] = float(struct_info[7])
                celldm[5] = float(struct_info[8])
                
#                if ibrav != 0:
#                    raise ValueError("Error, only ibrav 0 supported up to now")
                
                nat = int(struct_info[1])
                ntyp = int(struct_info[0])
                self.alat = np.float64(struct_info[3]) * BOHR_TO_ANGSTROM # We want a structure in angstrom
                
                # Allocate the coordinates
                self.structure.N_atoms = nat
                self.structure.coords = np.zeros((nat, 3))
                
                # Read the unit cell
                unit_cell = np.zeros((3,3))
                if ibrav == 0:
                    for i in range(3):
                        unit_cell[i, :] = np.array([np.float64(item) for item in dynlines[4 + i].split()]) * self.alat
                else:
                    unit_cell = Methods.get_unit_cell_from_ibrav(ibrav, celldm)
                    # Insert 4 lines to match the same number of lines as in ibrav = 0
                    dynlines.insert(3, "")
                    dynlines.insert(3, "")
                    dynlines.insert(3, "")
                    dynlines.insert(3, "")
                    
                # Read the atomic type
                atoms_dict = {}
                masses_dict = {}
                for atom_index in range(1, ntyp + 1):
                    atm_line = dynlines[6 + atom_index]
                    atoms_dict[atom_index] = atm_line.split("'")[1].strip()
                    
                    # Get also the atomic mass
                    masses_dict[atoms_dict[atom_index]] = np.float64(atm_line.split("'")[-1].strip())
                    
                self.structure.set_masses(masses_dict)
                
                
                self.structure.unit_cell = unit_cell
                self.structure.has_unit_cell = True
                
                # Read the atoms
                for i in range(nat):
                    # Jump the lines up to the structure
                    line_index = 7 + ntyp + i
                    atom_info = np.array([np.float64(item) for item in dynlines[line_index].split()])
                    self.structure.atoms.append(atoms_dict[int(atom_info[1])])
                    self.structure.coords[i, :] = atom_info[2:] * self.alat
                    
                
            # From now start reading the dynamical matrix -----------------------
            reading_dyn = True
            q_star = []
            
            # Pop the beginning of the matrix
            while reading_dyn:      
                # Pop the file until you reach the dynamical matrix
                if "cartesian axes" in dynlines[0]:
                    reading_dyn = False
                dynlines.pop(0)
                
            # Get the small q point
            reading_dyn = True
            index = -1
            current_dyn = np.zeros((3*self.structure.N_atoms, 3*self.structure.N_atoms), dtype = np.complex128)    
            
            # The atom indices
            atm_i = 0
            atm_j = 0
            coordline = 0
            
            dielectric_read = 0
            pol_read = 0
            
            # Info about what I'm reading
            reading_dielectric = False
            reading_eff_charges = False
            reading_raman = False
            
            while reading_dyn:
                # Advance in the reading
                index += 1

                if index >= len(dynlines):
                    reading_dyn = False
                    self.dynmats.append(current_dyn.copy())
                    continue
                
                # Setup what I'm reading
                if "Diagonalizing" in dynlines[index]:
                    reading_dyn = False
                    self.dynmats.append(current_dyn.copy())

                    continue
                if "Dielectric" in dynlines[index]:
                    reading_dielectric = True
                    reading_eff_charges = False
                    reading_raman = False
                    
                    # Reset the dielectric tensor
                    self.dielectric_tensor = np.zeros((3,3))
                    dielectric_read = 0
                    
                    continue
                elif "Effective" in dynlines[index]:
                    reading_dielectric = False
                    reading_eff_charges = True
                    reading_raman = False
                
                    
                    # Reset the effective charges
                    self.effective_charges = np.zeros((self.structure.N_atoms, 3, 3))
                    
                    continue
                elif  "Raman" in dynlines[index]:
                    reading_dielectric = False
                    reading_eff_charges = False
                    reading_raman = True
                    
                    # Reset the raman tensor
                    self.raman_tensor = np.zeros((3,3, 3*self.structure.N_atoms))
                    continue
                elif "q = " in dynlines[index]:
                    #Read the q
                    qpoint = np.array([float(item) for item in dynlines[index].replace("(", ")").split(')')[1].split()])
                    q_star.append(qpoint / self.alat)
                    self.q_tot.append(qpoint / self.alat)
                    reading_dielectric = False
                    reading_eff_charges = False
                    reading_raman = False
                    continue
                elif "ynamical" in dynlines[index]:
                    # Save the dynamical matrix
                    self.dynmats.append(current_dyn.copy())
                    reading_dielectric = False
                    reading_eff_charges = False
                    reading_raman = False
                    continue
                    
                
                # Read what is needed
                numbers_in_line = dynlines[index].split()
                if len(numbers_in_line) == 0:
                    continue
                
                if reading_dielectric:
                    # Reading the dielectric
                    if len(numbers_in_line) == 3:
                        self.dielectric_tensor[dielectric_read, :] = np.array([np.float64(x) for x in numbers_in_line])
                        dielectric_read += 1
                elif reading_eff_charges:
                    if numbers_in_line[0].lower() == "atom":
                        atm_i = int(numbers_in_line[2]) - 1
                        dielectric_read = 0
                    elif len(numbers_in_line) == 3:
                        self.effective_charges[atm_i, dielectric_read,:] = np.array([np.float64(x) for x in numbers_in_line])
                        dielectric_read += 1
                elif reading_raman:
                    if numbers_in_line[0].lower() == "atom":
                        atm_i = int(numbers_in_line[2]) - 1
                        pol_read = int(numbers_in_line[4]) - 1
                        dielectric_read = 0
                    elif len(numbers_in_line) == 3:
                        self.raman_tensor[dielectric_read,:, 3*atm_i + pol_read] = np.array([np.float64(x) for x in numbers_in_line])
                        dielectric_read += 1
                else:
                    # Read the numbers
                    if (len(numbers_in_line) == 2):
                        # Setup which atoms are 
                        atm_i = int(numbers_in_line[0]) - 1
                        atm_j = int(numbers_in_line[1]) - 1
                        coordline = 0
                    elif(len(numbers_in_line) == 6):
                        # Read the dynmat
                        for k in range(3):
                            current_dyn[3 * atm_i + coordline, 3*atm_j + k] = np.float64(numbers_in_line[2*k]) + 1j*np.float64(numbers_in_line[2*k + 1])
                        coordline += 1
                
                
            # Append the new stars for the irreducible q point
            self.q_stars.append(q_star)
        
        
        # Ok, the matrix has been initialized
        self.initialized = True
        
    def DyagDinQ(self, iq, force_real_at_gamma = True):
        """
        Dyagonalize the dynamical matrix in the given q point index.
        This methods returns both frequencies and polarization vectors.
        The frequencies and polarization are ordered. Negative frequencies are to
        be interpreted as instabilities and imaginary frequency, as for QE.
        
        They are returned.
        
        NOTE: The normalization is forced, as it is problematic for degenerate modes
        NOTE: if the q point is gamma, then the matrix is forced to be real
        
        Parameters
        ----------
            - iq : int
                Tbe index of the q point of the matrix to be dyagonalized.
            - force_real_at_gamma : bool, optional
                If True (default) the matrix is forced to be real during the
                dyagonalization (if q = 0). This assures to have real eigenvectors.
                This is usefull for supercells.

                
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
        real_dyn = np.zeros((3* self.structure.N_atoms, 3*self.structure.N_atoms), dtype = np.complex128)
        for i, atm_type1 in enumerate(self.structure.atoms):
            m1 = self.structure.masses[atm_type1]
            for j, atm_type2 in enumerate(self.structure.atoms):
                m2 = self.structure.masses[atm_type2]
                real_dyn[3*i : 3*i + 3, 3*j : 3*j + 3] = 1 / np.sqrt(m1 * m2)
        

        real_dyn *= self.dynmats[iq]
        
        q_vec = self.q_tot[iq]
        if np.sqrt(q_vec.dot(q_vec)) < __EPSILON__:
            eigvals, pol_vects = np.linalg.eigh(np.real(real_dyn))
        else:
            eigvals, pol_vects = np.linalg.eigh(real_dyn)
        
        f2 = eigvals
        
        # Check for imaginary frequencies (unstabilities) and return them as negative
        frequencies = np.zeros(len(f2), dtype = np.double)
        frequencies[f2 > 0] = np.sqrt(f2[f2 > 0])
        frequencies[f2 < 0] = -np.sqrt(-f2[f2 < 0])
        
        # Order the frequencies and the polarization vectors
        sorting_mask = np.argsort(frequencies)
        frequencies = frequencies[sorting_mask]
        pol_vects = pol_vects[:, sorting_mask]
        
        # Force normalization
        for i in range(3 * self.structure.N_atoms):
            # Check the normalization 
            norm = np.sqrt(pol_vects[:, i].dot(np.conj(pol_vects[:, i])))
            if abs(norm - 1) > __EPSILON__:
                sys.stderr.write("WARNING: Phonon mode %d at q point %d not normalized!\n" % (i, iq))
                print ("WARNING: Normalization of the phonon %d mode at %d q = %16.8f" % (i, iq, norm))
                
            # Check if it is an eigenvector
            not_eigen = np.sqrt(np.sum( abs(real_dyn.dot(pol_vects[:, i]) - eigvals[i] * pol_vects[:, i])**2))
                
            if not_eigen > 1e-2:
                sys.stderr.write("WARNING: Phonon mode %d at q point %d not an eigenvector!\n" % (i, iq))
                print ("WARNING: Error of the phonon %d mode eigenvector %d q = %16.8f" % (i, iq, not_eigen))
                    
            pol_vects[:, i] /= norm
        
        return frequencies, pol_vects
    
    def Copy(self):
        """
        Return an exact copy of itself. 
        This will implies copying all the dynamical matricies and structures inside.
        So take care if the structure is big, because it will overload the memory.


        """
        
        ret = Phonons()
        ret.structure = self.structure.copy()
        ret.q_tot = [x.copy() for x in self.q_tot]
        ret.nqirr = self.nqirr
        ret.initialized = self.initialized
        ret.q_stars = []
        for qstar in self.q_stars:
            ret.q_stars.append([x.copy() for x in qstar])

        ret.alat = self.alat
        
        for i, dyn in enumerate(self.dynmats):
            ret.dynmats.append(dyn.copy())

        if not self.effective_charges is None:
            ret.effective_charges = self.effective_charges.copy()
        if not self.raman_tensor is None:
            ret.raman_tensor = self.raman_tensor.copy()
        if not self.dielectric_tensor is None:
            ret.dielectric_tensor = self.dielectric_tensor.copy()
        
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
            
            \\Upsilon_{ab} = \\sqrt{M_aM_b}\\sum_\\mu \\frac{2\\omega_\\mu}{(1 + 2n_\\mu)\\hbar} e_\\mu^a e_\\mu^b
            
        It is used to compute the probability of a given atomic displacement.
        The resulting matrix is a 3N x 3N one ordered as the dynamical matrix here.
        The result is in bohr^-2, please be carefull.
        
        
        Parameters
        ----------
            T : float
                Temperature of the calculation (Kelvin)
        
        Returns
        -------
            ndarray(3N x3N), dtype = np.float64
                The inverse of the correlation matrix in the supercell.
                N is the number of atoms in the supercell
        """
        K_to_Ry=6.336857346553283e-06

        if T < 0:
            raise ValueError("Error, T must be posititive (or zero)")
#        
#        if self.nqirr != 1:
#            raise ValueError("Error, this function yet not supports the supercells.")
        
        # We need frequencies and polarization vectors
        w, pols = self.DiagonalizeSupercell() #self.DyagDinQ(iq)
        # Transform the polarization vector into real one
        #pols = np.real(pols)
        
        # Remove translations if we are at Gamma
        type_cal = np.float64#np.complex128
        
        super_struct = self.structure.generate_supercell(self.GetSupercell())
        trans_mask = Methods.get_translations(pols, super_struct.get_masses_array())

        # Exclude also other w = 0 modes
        locked_original = np.abs(w) < __EPSILON_W__
        if np.sum(locked_original.astype(int)) > np.sum(trans_mask.astype(int)):
            trans_mask = locked_original

        no_trans = ~trans_mask

        # Discard translations
        w = w[no_trans]
        pols = pols[:, no_trans]
        
        
        pols_conj = np.conj(pols)
            
        # Get the bosonic occupation number
        nw = np.zeros(np.shape(w))
        if T < __EPSILON__:
            nw = np.float64(0)
            #print "T = 0"
        else:
            nw =  1. / (np.exp(w/(K_to_Ry * T)) -1)
            #print "T > 0"
        
        # Compute the matrix
        factor = 2 * w / (1. + 2*nw)
        Upsilon = np.einsum( "i, ji, ki", factor, pols, pols_conj, dtype = type_cal)
        
        #_p1_, _p1vect_ = np.linalg.eigh(Upsilon)
        #np.savetxt("factor.dat", np.transpose([factor * RY_TO_CM / 2, _p1_[3:]* RY_TO_CM / 2]))
        
        # Get the masses for the final multiplication
        mass_sqrt = np.sqrt(np.tile(super_struct.get_masses_array(), (3,1)).T.ravel())
        
        #mass1 = np.zeros( 3*super_struct.N_atoms)
        #for i in range(self.structure.N_atoms):
        #    mass1[ 3*i : 3*i + 3] = np.sqrt(self.structure.masses[ super_struct.atoms[i]])
        
        _m1_ = np.tile(mass_sqrt, (3 * super_struct.N_atoms, 1))
        _m2_ = np.tile(mass_sqrt, (3 * super_struct.N_atoms, 1)).transpose()
        
        return Upsilon * _m1_ * _m2_
    
    
    def GetProbability(self, displacement, T, upsilon_matrix = None, normalize = True, return_braket_vals = False):
        """
        This function, given a particular displacement, returns the probability density
        of finding the system around that displacement. This in practical computes 
        density matrix of the system in this way
        
        .. math::
            
            \\rho(\\vec u) = \\sqrt{\\det(\\Upsilon / 2\\pi)} \\times \\exp\\left[-\\frac 12 \\sum_{ab} u_a \\Upsilon_{ab} u_b\\right]
            
        Where :math:`\\vec u` is the displacement, :math:`\\Upsilon` is the inverse of the covariant matrix
        computed through the method self.GetUpsilonMatrix().
        
        NOTE: I think there is an error in the implementation, in fact the Upsilon matrix is in bohr^-2 while displacements are in Angstrom.
        
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
            normalize : bool, optional
                If false (default true) the probability distribution will not be normalized.
                Useful to check if the exponential weight is the same after some manipulation
            return_braket_vals : bool, optional
                If true the value returned is only the <u | Upsilon |u> braket followed by the
                eigenvalues of the Upsilon matrix.
                
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
            upsilon_matrix = self.GetUpsilonMatrix(T)
        
        # Compute the braket
        braket = np.einsum("i, ij, j", disp, upsilon_matrix, disp)
        
        # Get the normalization
        vals = np.linalg.eigvals(upsilon_matrix)
        vals = vals[np.argsort(np.abs(vals))]
        
        vals /= 2*np.pi
        det = np.prod(vals[3:])

        if return_braket_vals:
            return braket, vals
        
        if normalize:
            return  np.sqrt(det) * np.exp(-braket)
        else:
            return  np.exp(-braket)
    
    def GetRatioProbability(self, structure, T, dyn0, T0):
        """
        IMPORTANCE SAMPLING
        ===================
        
        This method compute the ration of the probability of extracting a given structure at temperature T
        generated with dyn0 at T0 if the extraction is made with the self dynamical matrix.
        
        It is very usefull to perform importance sampling tests.
        
        .. math::
            
            w(\\vec u) = \\frac{\\rho_{D_1}(\\vec u, T)}{\\rho_{D_0}(\\vec u, T_0)}
            
        Where :math:`D_1` is the current dynamical matrix, while :math:`D_0` is the
        dynamical matrix that has been actually used to generate dyn0
        
        TODO: It seems to return wrong results
        NOTE: This subroutine seems to return fake results, please be carefull.
        
        Parameters
        ----------
            structure : Structure.Structure()
                The atomic structure generated according to dyn0 and T0 to evaluate the statistical significance ratio.
            T : float
                The target temperature
            dyn0 : Phonons.Phonons()
                The dynamical matrix used to generate the given structure.
            T0 : float
                The temperature used in the generation of the structure
        
        Results
        -------
            float
                The ratio :math:`w(\\vec u)` between the probabilities.
        """
        K_to_Ry = 6.336857346553283e-06

        if not self.CheckCompatibility(dyn0):
            raise ValueError("Error, dyn0 and the current dyn are incompatible")
        
        # Get the displacement respect the two central atomic positions
        disp1 = structure.get_displacement(self.structure)
        disp0 = structure.get_displacement(dyn0.structure)
        
        # # TODO: Improve the method with a much more reliable one
        # # In fact the ratio between them is much easier (this can be largely affected by rounding)
        # #print "disp1:", disp1
        # #print "Ratio1:",  self.GetProbability(disp1, T) , "Ratio2:",  dyn0.GetProbability(disp0, T0)

        # b1, v1 =  self.GetProbability(disp1, T, return_braket_vals = True)
        # b2, v2 =  dyn0.GetProbability(disp0, T0, return_braket_vals = True)
        # new_v = v1[3:] / v2[3:]
        # ret =  np.exp(b2- b1) * np.prod(np.sqrt(new_v))

        # #print "comparison:", ret, self.GetProbability(disp1, T) / dyn0.GetProbability(disp0, T0)


        # This should be the fastest way
        w1, pols1 = self.DyagDinQ(0)
        w0, pols0 = dyn0.DyagDinQ(0)

        # Remove translations (acustic modes in gamma)
        tmask1 = Methods.get_translations(pols1, self.structure.get_masses_array())
        tmask0 = Methods.get_translations(pols0, dyn0.structure.get_masses_array())
        

        w1 = w1[  ~tmask1 ]
        pols1 = pols1[:, ~tmask1]
        w0 = w0[~tmask0]
        pols0 = pols0[:, ~tmask0]

        #print "TMASK:", tmask0, tmask1
        
        
        _m1_ = np.zeros(self.structure.N_atoms * 3)
        _m0_ = np.zeros(dyn0.structure.N_atoms * 3)

        for i in range(self.structure.N_atoms):
            _m1_[3*i : 3*i + 3] = self.structure.masses[self.structure.atoms[i]]
            _m0_[3*i : 3*i + 3] = dyn0.structure.masses[dyn0.structure.atoms[i]]

        # Get the q values
        q1 = np.real(np.einsum("i, ij, i", np.sqrt(_m1_), pols1, disp1.reshape(3 * self.structure.N_atoms)))
        q0 = np.real(np.einsum("i, ij, i", np.sqrt(_m0_), pols0, disp0.reshape(3 * self.structure.N_atoms)))

        a1 = np.zeros(np.shape(w1))
        a0 = np.zeros(np.shape(w0))

        if T == 0:
            a1 = 1 / np.sqrt(2* w1)
        else:
            beta =  1 / (K_to_Ry*T)
            a1 = 1 / np.sqrt( np.tanh(beta*w1 / 2) *2* w1)

        if T0 == 0:
            a0 = 1 / np.sqrt(2* w0)
        else:
            beta =  1 / (K_to_Ry*T0)
            a0 = 1 / np.sqrt( np.tanh(beta*w0 / 2) *2* w0)

        weight = np.prod((a0 / a1) * np.exp(- (q1 / (a1))**2 + (q0 / (a0))**2))

        #print "COMPARISON:", ret, weight

        return weight
        
    def AdjustToNewCell(self, new_cell, symmetrize = True):
        """
        ADJUST THE DYNAMICAL MATRIX IN A NEW CELL
        =========================================
        
        This method is used, if you want to change the unit cell,
        to adjust the dynamical matrix, as the q points, in the new cell.
        
        The method forces also the symmetrization after the strain
        
        Parameters
        ----------
            new_cell : ndarray(size=(3,3), dtype=np.float64)
                The new unit cell
        """
        
        new_qs = symmetries.GetNewQFromUnitCell(self.structure.unit_cell, new_cell, self.q_tot)
        
        # Get the new structure
        self.structure.change_unit_cell(new_cell)
        
        # Get the new q points
        for iq, q in enumerate(new_qs):
            self.q_tot[iq] = q
            
        count = 0
        for iqirr in range(len(self.q_stars)):
            for iq in range(len(self.q_stars[iqirr])):
                self.q_stars[iqirr][iq] = new_qs[count]
                count += 1

        self.AdjustQStar()
        
        # Force the symmetrization in the new structure
        # NOTE: This will rise an exception if something is wrong        
        if symmetrize:
            qe_sym = symmetries.QE_Symmetry(self.structure)
            fcq = np.array(self.dynmats, dtype = np.complex128)
            qe_sym.SymmetrizeFCQ(fcq, self.q_stars)
            for iq, q in enumerate(self.q_tot):
                self.dynmats[iq] = fcq[iq, :, :]
    
    def GetStrainMatrix(self, new_cell, T,threshold=1e-5,x_start = 0.01):
        """
        STRAIN THE DYNAMICAL MATRIX
        ===========================
        
        This function strains the dynamical matrix to fit into the new cell.
        It will modify both the polarization vectors and the frequencies.
        
        The strain is performed on the covariance matrix.
        
        .. math::
            
            {\\Upsilon_{axby}^{-1}}' = \\sum_{\\alpha,\\beta = x,y,z} \\varepsilon_{x\\alpha}\\varepsilon_{y\\beta}\\Upsilon_{a\\alpha b\\beta}^{-1}
        
        Then the new :math:`\\Upsilon^{-1}` matrix is diagonalized, eigenvalues and eigenvector are built,
        and from them the new dynamical matrix is computed.
        
        NOTE: This works only at Gamma
              I think there is a bug if T != 0 in the solver. BE CAREFULL!
        
        Parameters
        ----------
            new_cell : ndarray 3x3
                The new unit cell after the strain.
            T : float
                The temperature of the strain (default 0)
            threshold : float
                The threshold for the convergence of the newton algorithm to find the 
                frequencies given the eigenvalues of the upsilon matrix.
            x_start : float
                The initial guess for the newton algorithm.
                
        Results
        -------
            dyn : Phonons.Phonons()
                A new dynamical matrix strained. Note, the current dynamical matrix will not be modified.
        """
        K_to_Ry=6.336857346553283e-06

        if T < 0:
            raise ValueError("Error, the temperature must be positive.")
        
        # Get the polarization vectors and frequencies
        w, pol_vects = self.DyagDinQ(0)
        
        n_modes = len(w) 
        
        # Strain the polarization vectors
        new_vect = np.zeros(np.shape(pol_vects))
        for i in range(3, n_modes):
            for j in range(self.structure.N_atoms):
                # Get the crystal representation of the polarization vector
                cov_coord = Methods.covariant_coordinates(self.structure.unit_cell, 
                                                          pol_vects[3*j: 3*(j+1), i])
                
                # Transform the crystal representation into the cartesian in the new cell
                new_vect[3*j: 3*(j+1), i] = np.einsum("ij, i", new_cell, cov_coord)
        
        # Now prepare the new Covariance Matrix
        factor = np.zeros(n_modes)
        if T == 0:
            factor[3:] = 1 / (2. * w[3:])
        else:
            n = 1 / (np.exp(w[3:] / (K_to_Ry * T)) - 1)
            factor[3:] = (1. + 2*n) / (2*w[3:])
        
        cmat = np.einsum("i, hi,ki", factor, new_vect, new_vect)
        
        # Diagonalize once again
        newf, new_pols = np.linalg.eig(cmat)
#        
#        # DEBUG PRINT
#        prova1 = np.sort(newf)
#        prova2 = np.sort(factor)
#        for i in range(n_modes):
#            print "New: %e | Old: %e" % (prova1[i], prova2[i])
#        
        
        # Sort the results
        sort_mask = np.argsort(newf)
        newf = newf[sort_mask]
        new_pols = new_pols[:, sort_mask]
        
        # Initialize the array of the new frequencies
        new_w = np.zeros(n_modes)
        new_w[3:] = 1. / (2 * newf[3:])
        
        
        
        # If the temperature is different from zero, we must obtain a new frequency
        # using a numerical nonlinear solver
        if T != 0:
            
            #def opt_func(w):
            #    ret = 2*w*newf - 1./( 1 - np.exp(w / (K_to_Ry * T)))
            #    if not np.shape(w):
            #        if np.abs(w) < __EPSILON__:
            #            return 0
            #    else:
            #        ret[np.abs(w) < __EPSILON__] = 0
            #    return ret

	    

	    

            #try:
            #    for k in range(len(new_w)):
            #       def new_func(x):
            #            _x_ = np.ones(np.shape(newf)) * x
            #            return opt_func(_x_)[k]
            #        if np.abs(new_w[k]) < __EPSILON__:
            #            continue
            #        new_w[k] = scipy.optimize.anderson(new_func, new_w[k], verbose = True) 
                
      	    for k in range(3,36):
                def g(w):
                    f1= 2*w*newf[k]-1/np.tanh(w*0.5/(K_to_Ry*T))
                    return f1
        
                def g_prime(w):
                    f2=2*newf[k]+0.5/(K_to_Ry*T*(np.sinh(w*0.5/(K_to_Ry*T)))**2)
                    return f2
        
                x_old=x_start
                while True : 
                    x_new=x_old-g(x_old)/g_prime(x_old)
                    if np.abs(g(x_new)) < threshold :
                        break
                    else:		
                        x_old=x_new
                new_w[k]=x_new
       
            #except ValueError:
            #    print "Error, Nan encountered during the scipy minimization (T != 0)"
            #    print "Starting w value:"
            #    print new_w
            #    print "new_f value:"
            #    print newf
            #    print "T:", T
            #    raise ValueError("Aborting, error in scipy minimization.")
                
                
                
                #
                #        print "Compare frequencies:"
#        for i in range(0,n_modes):
#            print "New: %e | Old: %e" % (new_w[i], w[i])


        # Sort once again
        sort_mask = np.argsort(new_w)
        new_w = new_w[sort_mask]
        new_pols = new_pols[:, sort_mask]


        # Now we can rebuild the dynamical matrix
        out_dyn = self.Copy()
        out_dyn.structure.change_unit_cell(new_cell)
        out_dyn.dynmats[0] = np.einsum("i, hi, ki", new_w**2, new_pols, new_pols)
        
        # Get the masses for the final multiplication
        mass1 = np.zeros( 3*self.structure.N_atoms)
        for i in range(self.structure.N_atoms):
            mass1[ 3*i : 3*i + 3] = self.structure.masses[ self.structure.atoms[i]]
        
        _m1_ = np.tile(mass1, (3 * self.structure.N_atoms, 1))
        _m2_ = np.tile(mass1, (3 * self.structure.N_atoms, 1)).transpose()
        
        out_dyn.dynmats[0] *= np.sqrt( _m1_ * _m2_ )
        
        return out_dyn
        
        
    def save_qe(self, filename, full_name = False):
        """
        SAVE THE DYNMAT
        ===============
        
        This subroutine saves the dynamical matrix in the quantum espresso file format.
        The dynmat is the force constant matrix in Ry units.
        
        .. math::
            
            \\Phi_{ab} = \\sum_\\mu \\omega_\\mu^2 e_\\mu^a e_\\mu^b \\sqrt{M_a M_b}
            
        Where :math:`\\Phi_{ab}` is the force constant matrix between the a-b atoms (also cartesian
        indices), :math:`\\omega_\\mu` is the phonon frequency and :math:`e_\\mu` is the
        polarization vector.
        
        
        Parameters
        ----------
            filename : string
                The path in which the quantum espresso dynamical matrix will be written.
            full_name : bool
                If true only the gamma matrix will be saved, and the irreducible q
                point index will not be appended. Otherwise all the file filenameIQ 
                where IQ is an integer between 0 and self.nqirr will be generated.
                filename0 will contain all the information about the Q points and the supercell.
        """
        #A_TO_BOHR = 1.889725989
        #RyToCm=109737.37595
        RyToTHz=3289.84377
        
        # Check if all the dynamical matrix must be saved, or only the 
        nqirr = self.nqirr
        if full_name:
            nqirr = 1
        
        # The following counter counts the total number of q points
        count_q = 0
        for iq in range(nqirr):
            # Prepare the file name appending the q point index
            fname = filename
            if not full_name:
                fname += str(iq+1)
            
            # Open the file
            fp = open(fname, "w")
            fp.write("Dynamical matrix file\n")
        
            # Get the different number of types
            types = []
            n_atoms = self.structure.N_atoms
            for i in range(n_atoms):
                if not self.structure.atoms[i] in types:
                    types.append(self.structure.atoms[i])
            n_types = len(types)
        
            # Assign an integer for each atomic species
            itau = {}
            for i in range(n_types):
                itau[types[i]] = i +1
            
            # Write the comment line
            fp.write("File generated with the CellConstructor by Lorenzo Monacelli\n")
            fp.write("%d %d %d %22.16f %22.16f %22.16f %22.16f %22.16f %22.16f\n" %
                     (n_types, n_atoms, 0, self.alat * A_TO_BOHR, 0, 0, 0, 0, 0) )
        
            # Write the basis vector
            fp.write("Basis vectors\n")
            # Get the unit cell
            for i in range(3):
                fp.write(" ".join("%22.16f" % x for x in self.structure.unit_cell[i,:] / self.alat) + "\n")
        
            # Set the atom types and masses
            for i in range(n_types):
                fp.write("\t%d  '%s '  %22.16f\n" % (i +1, types[i], self.structure.masses[types[i]]))
        
            # Setup the atomic structure
            for i in range(n_atoms):
                # Convert the coordinates in alat
                coords = self.structure.coords[i,:] / self.alat
                fp.write("%5d %5d %22.16f %22.16f %22.16f\n" %
                         (i +1, itau[self.structure.atoms[i]], 
                          coords[0], coords[1], coords[2]))
        
            # Iterate over all the q points in the star
            nqstar = len(self.q_stars[iq])
            q_star = self.q_stars[iq] #* self.alat
            
            # Store the first matrix index of the star
            # This will be used to dyagonalize the matrix in the end of the file
            dyag_q_index = count_q
            
            for jq in range(nqstar):
                # Here the dynamical matrix starts
                fp.write("\n")
                fp.write("     Dynamical Matrix in cartesian axes\n")
                fp.write("\n")
                fp.write("     q = (    %.12f   %.12f   %.12f )\n" % 
                         (q_star[jq][0] * self.alat , q_star[jq][1]*self.alat, q_star[jq][2]*self.alat ))
                fp.write("\n")
            
                # Now print the dynamical matrix
                for i in range(n_atoms):
                    for j in range(n_atoms):
                        # Write the atoms
                        fp.write("%5d%5d\n" % (i + 1, j + 1))
                        for x in range(3):
                            line = "%23.16f%23.16f   %23.16f%23.16f   %23.16f%23.16f" % \
                                   ( np.real(self.dynmats[count_q][3*i + x, 3*j]), np.imag(self.dynmats[count_q][3*i + x, 3*j]),
                                     np.real(self.dynmats[count_q][3*i + x, 3*j+1]), np.imag(self.dynmats[count_q][3*i+x, 3*j+1]),
                                     np.real(self.dynmats[count_q][3*i + x, 3*j+2]), np.imag(self.dynmats[count_q][3*i+x, 3*j+2]) )
            
                            fp.write(line +  "\n")
                
                # Go to the next q point
                count_q += 1

            # Here save the Dielectric tensor, the effective charges and the Raman response
            if not self.dielectric_tensor is None:
                fp.write("\n")
                fp.write("     Dielectric Tensor:\n")
                fp.write("\n")
                for i in range(3):
                    fp.write("{:24.12f} {:24.12f} {:24.12f}\n".format(*list(self.dielectric_tensor[i,:])))
                
            if not self.effective_charges is None:
                fp.write("\n")
                fp.write("     Effective Charges E-U: Z_{alpha}{s,beta}\n")
                fp.write("\n")
                for i in range(self.structure.N_atoms):
                    fp.write("    atom # {:5d}\n".format(i+1))
                    for j in range(3):
                        fp.write("{:24.12e} {:24.12e} {:24.12e}\n".format(*list(self.effective_charges[i, j, :])))
            
            if not self.raman_tensor is None:
                fp.write("\n")
                fp.write("     Raman tensor (A^2)\n")
                fp.write("\n")
                for i_atm in range(self.structure.N_atoms):
                    for j_pol in range(3):
                        fp.write("     atom # {:5d}    pol. {:2d}\n".format(i_atm+1, j_pol+1))
                        for k in range(3):
                            fp.write("{:24.12e} {:24.12e} {:24.12e}\n".format(*list(self.raman_tensor[k, :, 3*i_atm + j_pol])))
                

        
            # Print the diagnoalization of the matrix
            fp.write("\n")
            fp.write("     Diagonalizing the dynamical matrix\n")
            fp.write("\n")
            fp.write("     q = (    %.12f   %.12f   %.12f )\n" % 
                     (q_star[0][0] *self.alat , q_star[0][1] *self.alat, q_star[0][2] *self.alat))
            fp.write("\n")
            fp.write("*" * 75 + "\n")
            
            # Diagonalize the dynamical matrix
            freqs, pol_vects = self.DyagDinQ(dyag_q_index)

            # Compute the displacemets from the polarization vectors
            _m_ = self.structure.get_masses_array()
            _m_ = np.tile(_m_, (3,1)).T.ravel()

            # Compute the atomic displacements
            atomic_disp = np.einsum("ab, a -> ab", pol_vects, 1 / np.sqrt(_m_) )
            # Normalize the displacements
            atomic_disp[:,:] /= np.tile( np.sqrt(np.sum(np.abs(atomic_disp)**2, axis = 0)), (self.structure.N_atoms * 3, 1))

            nmodes = len(freqs)
            for mu in range(nmodes):
                # Print the frequency
                fp.write("%7s (%5d) = %14.8f [THz] = %14.8f [cm-1]\n" %
                         ("freq", mu+1, freqs[mu] * RyToTHz, freqs[mu] * RY_TO_CM))
                
                # Print the polarization vectors
                for i in range(n_atoms):
                    fp.write("( %10.6f%10.6f %10.6f%10.6f %10.6f%10.6f )\n" %
                             (np.real(atomic_disp[3*i, mu]), np.imag(atomic_disp[3*i,mu]),
                              np.real(atomic_disp[3*i+1, mu]), np.imag(atomic_disp[3*i+1,mu]),
                              np.real(atomic_disp[3*i+2, mu]), np.imag(atomic_disp[3*i+1,mu])))
            fp.write("*" * 75 + "\n")
            fp.close()
            
    def save_phononpy(self, supercell_size = (1,1,1)):
        """
        EXPORT THE DYN IN THE PHONONPY FORMAT
        =====================================

        This tool export the dynamical matrix into the PHONONPY plain text format.
        We save them in Ry/bohr^2, as the quantum espresso format. Please, remember
        this when using Phononpy for the conversion factors.

        It will create a file called FORCE_CONSTANTS, one called unitcell.in
        with the info on the structure

        Parameters
        ----------
            supercell_size : list of 3
                The supercell that defines the dynamical matrix, note phononpy 
                works in the supercell.


        """

        # Save it into the phononpy in the supercell
        superdyn = self.GenerateSupercellDyn(supercell_size)
        filename = "FORCE_CONSTANTS"

        nat_sc = superdyn.structure.N_atoms
        nat = self.structure.N_atoms

        # This is the text to be written
        lines = []
        lines.append("%d   %d\n" % (nat_sc, nat_sc))
        for i in range(nat_sc):
            for j in range(nat_sc):
                lines.append("%4d\t%4d\n" % (i, j))
                mat = np.real(superdyn.dynmats[0][3*i : 3*i+ 3, 3*j: 3*j+3])
                lines.append("%16.8f   %16.8f   %16.8f\n"  % (mat[0,0], mat[0,1], mat[0,2]))
                lines.append("%16.8f   %16.8f   %16.8f\n"  % (mat[1,0], mat[1,1], mat[1,2]))
                lines.append("%16.8f   %16.8f   %16.8f\n"  % (mat[2,0], mat[2,1], mat[2,2]))
        
        # Write to the file
        f = open(filename, "w")
        f.writelines(lines)
        f.close()

        # Produce the unit cell
        lines = []
        lines.append("&system\n")
        lines.append("ibrav = 0\n")
        lines.append("celldm(1) = 1.889726125836928\n")
        lines.append("nat = %d\n" % self.structure.N_atoms)
        
        typs = self.structure.masses.keys()
        lines.append("ntyp = %d\n" % len(typs))
        lines.append("&end\n")

        # Write the atomic species
        lines.append("ATOMIC_SPECIES\n")
        for i in typs:
            m = self.structure.masses[i]
            lines.append("%s %16.8f   XXX\n" % (i, m / 911.444243096))

        # Write the unit cell
        lines.append("CELL_PARAMETERS alat\n")
        for i in range(3):
            uc_v = self.structure.unit_cell[i, :] #* 1.889726125836928
            lines.append("%16.8f   %16.8f  %16.8f\n" % (uc_v[0], uc_v[1], uc_v[2]))
        
        lines.append("ATOMIC_POSITIONS crystal\n")
        for i in range(nat):
            atm = self.structure.atoms[i]
            cov_vect = Methods.covariant_coordinates(self.structure.unit_cell, self.structure.coords[i, :])
            lines.append("%s  %16.8f   %16.8f   %16.8f\n" % (atm, cov_vect[0], cov_vect[1], cov_vect[2]))
        

        f = open("unitcell.in", "w")
        f.writelines(lines)
        f.close()

            
            
    def ForcePositiveDefinite(self):
        """
        FORCE TO BE POSITIVE DEFINITE
        =============================
        
        This method force the matrix to be positive defined. 
        Usefull if you want to start with a matrix for a SCHA calculation.
        
        It will take the Dynamical matrix and rebuild it as
        
        .. math::
            
            \\Phi'_{ab} = \\sqrt{M_aM_b}\sum_{\mu} |\omega_\mu^2| e_\\mu^a e_\\mu^b 
            
        
        In this way the dynamical matrix will be always positive definite.
        """
        
        # Prepare the masses matrix
        mass1 = np.zeros( 3*self.structure.N_atoms)
        for i in range(self.structure.N_atoms):
            mass1[ 3*i : 3*i + 3] = self.structure.masses[ self.structure.atoms[i]]
        
        _m1_ = np.tile(mass1, (3 * self.structure.N_atoms, 1))
        _m2_ = np.tile(mass1, (3 * self.structure.N_atoms, 1)).transpose()
        
        for iq in range(len(self.dynmats)):
            # Diagonalize the matrix
            w, pols = self.DyagDinQ(iq)
            
            matrix = np.einsum("i, ji, ki", w**2, pols, np.conj(pols)) * np.sqrt(_m1_ * _m2_)
            self.dynmats[iq] = matrix


    def ForcePositiveDefinite_2(self):
        """
        FORCE TO BE POSITIVE DEFINITE
        =============================
        
        This method force the matrix to be positive defined. 
        Usefull if you want to start with a matrix for a SCHA calculation.
        
        It will take the Dynamical matrix and rebuild it as
        
        .. math::
            
            \\Phi'_{ab} = \\sqrt{M_aM_b}\sum_{\mu} |\omega_\mu^2| e_\\mu^a e_\\mu^b 
            
        
        In this way the dynamical matrix will be always positive definite.
        """
        
        # Prepare the masses matrix
        mass1 = np.zeros( 3*self.structure.N_atoms)
        for i in range(self.structure.N_atoms):
            mass1[ 3*i : 3*i + 3] = self.structure.masses[ self.structure.atoms[i]]
        
        _m1_ = np.tile(mass1, (3 * self.structure.N_atoms, 1))
        _m2_ = np.tile(mass1, (3 * self.structure.N_atoms, 1)).transpose()
        
        
        numq=len(self.dynmats)
        w=np.zeros((numq,3*self.structure.N_atoms), dtype = np.float64)
        pols=np.zeros((numq,3*self.structure.N_atoms,3*self.structure.N_atoms), dtype = np.complex128 )
        
        for iq in range(numq):
            # Diagonalize the matrix
            w[iq,:], pols[iq,:,:] = self.DyagDinQ(iq)
        
        
        fact=np.amin(w)
        
        if fact < 0.0 :
            w+=np.abs(fact)*0.1
        
        for iq in range(numq):        
            v=pols[iq,:,:]
            fr=w[iq,:]
            matrix = np.einsum("i, ji, ki", fr**2, v, np.conj(v)) * np.sqrt(_m1_ * _m2_)
            self.dynmats[iq] = matrix

                        
                        
    def GetRamanResponce(self, pol_in, pol_out, T = 0):
        """
        RAMAN RESPONSE
        ==============
        
        Evaluate the raman response using the Mauri-Lazzeri equation.
        This subroutine needs the Raman tensor to be defined, and computes the intensity for each mode.
        It returns a list of intensity associated to each mode.
        
        .. math::
            
            I_{\\nu} = \\left| \\sum_{xy} \\epsilon_x^{(1)} A^\\nu_{xy} \\epsilon_y^{(2)}\\right|^2 \\frac{n_\\nu + 1}}{\\omega_\\nu}
    
        Where :math:`\\epsilon` are the polarization vectors of the incoming/outcoming light, :math:`n_\\nu` is the bosonic
        occupation number associated to the :math:`\\nu` mode, and :math:`A^\\nu_{xy}` is the Raman tensor in the mode rapresentation
    
        Parameters
        ----------
            pol_in : ndarray 3
                The polarization versor of the incominc electric field
            pol_out : ndarray 3
                The polarization versor of the outcoming electric field
            T : float
                The tempearture of the calculation
        
        Results
        -------
            ndarray (nmodes)
                Intensity for each mode of the current dynamical matrix.
        """
        
        K_to_Ry=6.336857346553283e-06
        
        
        if self.raman_tensor is None:
            raise ValueError("Error, to get the raman responce the raman tensor must be defined")
        
        w, pol_vects = self.DyagDinQ(0)
        
        # Get the mass array
        _m_ = np.zeros( 3*self.structure.N_atoms)
        for i in range(self.structure.N_atoms):
            _m_[ 3*i : 3*i + 3] = self.structure.masses[ self.structure.atoms[i]]
            
        # Apply translation
        trans = Methods.get_translations(pol_vects, self.structure.get_masses_array())
        pol_vects[:, trans] = 0
        
        # The super sum
        #print np.shape(self.raman_tensor), np.shape(pol_vects), np.shape(_m_), np.shape(pol_in), np.shape(pol_out)
        I = np.einsum("ijk, kl, k, i, j", self.raman_tensor, pol_vects, 1/np.sqrt(_m_), pol_in, pol_out)
        
        # Get the bosonic occupation number
        n = np.zeros(len(w))
        if T > 0:            
            beta = 1 / (K_to_Ry*T)
            n = 1 / (np.exp(beta * w) - 1.)
        
        return np.abs(I**2) * (1. + n) / w

    def GetIRIntensities(self):
        """
        GET THE IR INTENSITIES
        ======================

        This function uses the effective charges to compute the infrared responce.

        A list of value is returned, at each index the IR intensity of the
        relative mode.
        """

        if self.effective_charges is None:
            raise ValueError("Error, I cannot compute IR intensities without effective charges")

        w, pols = self.DyagDinQ(0)
        m = self.structure.get_masses_array()


        # Get the eigendisplacement z
        nat3, nmodes = np.shape(pols)
        z = np.zeros( (nmodes, self.structure.N_atoms, 3), dtype = np.float64)
        for i in range(self.structure.N_atoms):
            z[:, i, :] = pols[3*i: 3*(i+1), :].T / np.sqrt(m[i])

        # Get the I_mu,i where mu is the mode and i is the polarization of the light
        I = np.einsum("cbd, acd->ab", self.effective_charges, z)
        # Average over polarizations
        I = np.sum( I*I, axis = 1) * 2

        return I

    def GetIRActivityVector(self):
        """
        GET THE IR VECTOR
        =================

        This vector returns the activity of the infrared mode.
        It is the matrix element to compute the responce function of the IR experiment.

        Results
        -------
            v_ir : ndarray(size = (3, 3*natoms), dtype = np.double)
                The ir activity amplitude for each polarization mode, for each polarizations of the incoming field
        """

        if self.effective_charges is None:
            raise ValueError("Error, I cannot compute IR intensities without effective charges")

        w, pols = self.DyagDinQ(0)
        m = self.structure.get_masses_array()

        # Get the eigendisplacement z
        nat3, nmodes = np.shape(pols)
        z = np.zeros( (nmodes, self.structure.N_atoms, 3), dtype = np.float64)
        for i in range(self.structure.N_atoms):
            z[:, i, :] = pols[3*i: 3*(i+1), :].T / np.sqrt(m[i])

        # Get the I_mu,i where mu is the mode and i is the polarization of the light
        v_ir = np.einsum("cbd, acd->ba", self.effective_charges, z)

        return v_ir


    def GetRamanVector(self, pol_in, pol_out):
        r"""
        GET THE RAMAN VECTOR
        ====================

        Get the Raman vector. It is the vector obtained from the Raman Tensor:

        .. math::

            v_\nu = \sum_{xy} \epsilon^{(1)}_x \epsilon_y^{(2)} A^{\nu}_{xy}

        This is defined in real space.

        Parameters
        ----------
            pol_in : ndarray(size = 3)
                Incoming polarization
            pol_out : ndarray(size = 3)
                Outcoming polarization
        Results
        -------
            vnu : ndarray(size = 3*nat)
                The raman intensity vector along each atomic displacement.
        """

        if self.raman_tensor is None:
            raise ImportError("Error, the raman tensor is not defined.")
            
        v =  np.einsum("ija, i, j", self.raman_tensor, pol_in, pol_out)        
        
        # Take out the translations from v
        t1 = np.tile(np.array([1,0,0], dtype = np.float64), (self.structure.N_atoms, 1)).ravel()
        t2 = np.tile(np.array([0,1,0], dtype = np.float64), (self.structure.N_atoms, 1)).ravel()
        t3 = np.tile(np.array([0,0,1], dtype = np.float64), (self.structure.N_atoms, 1)).ravel()
        
        v -= t1.dot(v)
        v -= t2.dot(v)
        v -= t3.dot(v)
        
        return v
    
    def GetRamanActive(self, use_spglib = False):
        """
        This simple subroutines tries to guess by symmetry analisys which mode is active or not.
        If a raman tensor is present, it will be used to test the activity, otherwise, a random one will
        be generated

        Parameters
        ----------
            use_spblib: bool
                If True the spglib library is used to initialize symmetries.
                Usefull if the phonon matrix is in a super cell.

        Results
        -------
            raman_activity_mask : ndarray(size = (3*nat), dtype = bool)
                A mask that is False or True if a mode in the unit cell is Raman-active or not.
        """

        there_is_raman_tensor = True
        if self.raman_tensor is None:
            there_is_raman_tensor = False 
            self.raman_tensor = np.zeros((3,3, 3*self.structure.N_atoms), dtype = np.double)
            self.raman_tensor[:,:,:] = np.random.uniform( size = self.raman_tensor.shape)

            # Get the symmetries
            qe_sym = symmetries.QE_Symmetry(self.structure)
            if use_spglib:
                qe_sym.SetupFromSPGLIB()
            else:
                qe_sym.SetupQPoint()
            
            # Symmetrize the effective charges
            qe_sym.ApplySymmetryToRamanTensor(self.raman_tensor)

        # Save a debugging one
        self.save_qe("Raman")

        # Simulate the Raman signal for all possible incoming and outcoming polarizations
        res = np.zeros(self.structure.N_atoms * 3, dtype = np.double)
        for i, j in itertools.product(range(3) , range(3)):
            pol_in = np.zeros(3)
            pol_in[i] = 1
            pol_out = np.zeros(3)
            pol_out[j] = 1

            res += self.GetRamanResponce(pol_in, pol_out)
        
        print("total_raman_res:", res)

        is_raman_active = res > 1e-5    

        # Delete the random raman tensor if any
        if not there_is_raman_tensor:
            self.raman_tensor = None

        return is_raman_active



    def GenerateSupercellDyn(self, supercell_size, img_thr = 1e-6):
        """
        GENERATE SUPERCEL DYN
        =====================
        
        This method returns a Phonon structure as it was computed directly in the supercell.
        
        
        NOTE: For now this neglects bohr effective charges
        
        Parameters
        ----------
            supercell_size : array int (size=3)
                the dimension of the cell on which you want to generate the new 
                Phonon
        
        Results
        -------
            dyn_supercell : Phonons()
                A Phonons class of the supercell
        
        """
        # First check if the q vectors are compatible with the supercell
        if not symmetries.CheckSupercellQ(self.structure.unit_cell, supercell_size, self.q_tot):
            print("Q points:", self.q_tot)
            print("Supercell size:", supercell_size)
            print("Unit cell:", self.structure.unit_cell)
            raise ValueError("Error, the list of q point does not match the given supercell.")

        super_struct = self.structure.generate_supercell(supercell_size)
        
        dyn_supercell = Phonons(super_struct, nqirr = 1, force_real = True)
        
        dyn_supercell.dynmats[0] = self.GetRealSpaceFC(supercell_size, img_thr = img_thr)
        
        return dyn_supercell


    def GetMatrixCFFT(self):
        """
        Generate the dynamical matrix ready for the Fast Fourier Transform.
        This is an alternative way to go in real space.
        NOTE: Use only for debug purpouses
        """

        s1, s2, s3 = self.GetSupercell() 
        nat = self.structure.N_atoms
        output_dyn = np.zeros((s1, s2, s3, 3 * nat, 3 * nat), dtype = np.complex128, order = "F")

        super_struct = self.structure.generate_supercell((s1,s2,s3))
        bg = super_struct.get_reciprocal_vectors() / (2 * np.pi)

        for iq, q in enumerate(self.q_tot):
            x_vect = Methods.covariant_coordinates(bg, q)
            x1 = int((x_vect[0] + s1) % s1 + .5)
            x2 = int((x_vect[1] + s2) % s2 + .5)
            x3 = int((x_vect[2] + s3) % s3 + .5)

            #print("Q = ", q, "| xv:", x_vect, "x = ", x1, x2,x3)

            output_dyn[x1, x2, x3, :, :] = self.dynmats[iq]

        return output_dyn            

            
    def ExtractRandomStructures(self, size=1, T=0, isolate_atoms = [], project_on_vectors = None):
        """
        EXTRACT RANDOM STRUCTURES
        =========================
        
        This method is used to extract a pool of random structures according to the current dinamical matrix.
                
        Parameters
        ----------
            size : int
                The number of structures to be generated
            T : float
                The temperature for the generation of the ensemble
            isolate_atoms : list, optional
                A list of the atom index. Only the specified atoms are present in the output structure and displaced.
                This is very usefull if you want to measure properties of a particular region of the structure.
                By default all the atoms are used.
        
        Returns
        -------
            list
                A list of Structure.Structure()
        """
        K_to_Ry=6.336857346553283e-06

            
        # Check if isolate atoms is good
        if len(isolate_atoms):
            if np.max(isolate_atoms) >= self.structure.N_atoms:
                raise ValueError("Error, index in isolate_atoms out of boundary")
            
        # Now extract the values
        ws, pol_vects = self.DiagonalizeSupercell()
        super_structure = self.structure.generate_supercell(self.GetSupercell())
        
        # Remove translations
        trans_mask = Methods.get_translations(pol_vects, super_structure.get_masses_array())

        # Exclude also other w = 0 modes
        locked_original = np.abs(ws) < __EPSILON__
        if np.sum(locked_original.astype(int)) > np.sum(trans_mask.astype(int)):
            trans_mask = locked_original

        ws = ws[~trans_mask]
        pol_vects = pol_vects[:, ~trans_mask]
        
        nat = self.structure.N_atoms * np.prod(self.GetSupercell())

        # Check that the matrix is positive definite
        if any([w < 0 for w in ws]):
            ERR_MSG = """
    Error, the current matrix is not positive definite.
           I cannot extract a random ensamble.
           If you want to skip this error,
           consider calling the method ForcePositiveDefinite() before extracting the ensemble.
        
        It could also be a consequence of a sum rule not well imposed. 
        Try to run Symmetrize() to force the sum rule.
    """

            raise ValueError(ERR_MSG)
        
        n_modes = len(ws)
        if T == 0:
            a_mu = 1 / np.sqrt(2* ws) * BOHR_TO_ANGSTROM
        else:            
            beta = 1 / (K_to_Ry*T)
            a_mu = 1 / np.sqrt( np.tanh(beta*ws / 2) *2* ws) * BOHR_TO_ANGSTROM
        
        # Prepare the random numbers
        size = int(size)
        rand = np.random.normal(size = (size, n_modes))
        
        # Get the masses for the final multiplication
        mass1 = np.tile(super_structure.get_masses_array(), (3, 1)).T.ravel()
            
        total_coords = np.einsum("ij, i, j, kj->ik", pol_vects, 1/np.sqrt(mass1), a_mu, rand)



        # Project the displacements along the selected modes
        if not project_on_vectors is None:
            check, N_proj = np.shape(project_on_vectors)
            if check != 3*nat:
                print("Expected nat: " + str(nat) + " project_on_modes nat: " + str(check/3))
                raise ValueError("Error, the input project_on_modes has a wrong shape")
            
            for confid in range(size):
                new_coords = np.zeros( nat*3, dtype = np.float64)
                for i in range(N_proj):
                    new_coords += project_on_vectors[:, i].dot(total_coords[:, confid]) * project_on_vectors[:, i]
                
                total_coords[:, confid] = new_coords
        
        # Prepare the structures
        final_structures = []
        for i in range(size):
            tmp_str = super_structure.copy()
            # Prepare the new atomic positions 
            for k in range(tmp_str.N_atoms):
                tmp_str.coords[k,:] += total_coords[3*k : 3*(k+1), i] 
            
            # Check if you must to pop some atoms:
            if len (isolate_atoms):
                
                tmp_str.N_atoms = len(isolate_atoms)
                new_coords = tmp_str.coords.copy()
                for j, x in enumerate(isolate_atoms):
                    tmp_str.coords[j,:] = new_coords[x,:]
            final_structures.append(tmp_str)
        
        
        return final_structures

    def GetHarmonicFreeEnergy(self, T, allow_imaginary_freq = False):
        """
        COMPUTE THE HARMONIC QUANTUM FREE ENERGY
        ========================================
        
        The dynamical matrix can be used to obtain the vibrational contribution
        to the Free energy.
        
        ..math:: 
            
            F(\\Phi) = \\sum_\mu \\left[\\frac{\\hbar \\omega_\\mu}{2} + kT \\ln\\left(1 + e^{-\\beta \hbar\\omega_\\mu}\\right)\\right]
            
        
        Acustic modes at Gamma are discarded from the summation. 
        An exception is raised if there are imaginary frequencies.
        
        Parameter
        ---------
            T : float
                Temperature (in K) of the system.
                
        Returns
        -------
            fe : float
                Free energy (in Ry) at the given temperature.
        """
        
        K_to_Ry=6.336857346553283e-06

        w, pols = self.DiagonalizeSupercell()
            
        # Remove translations
        tmask = Methods.get_translations(pols, self.structure.generate_supercell(self.GetSupercell()).get_masses_array())

        # Exclude also other w = 0 modes (good for rotations)
        locked_original = np.abs(w) < __EPSILON__
        if np.sum(locked_original.astype(int)) > np.sum(tmask.astype(int)):
            tmask = locked_original

        w = w[ ~tmask ]
            
        # if imaginary frequencies are allowed, put w->0
        if allow_imaginary_freq:
            w[w<0] = __EPSILON__
            
        if len(w[w < 0]) >= 1:
            raise ValueError("Error while computing the free energy, the dynamical matrix has imaginary frequencies")
        
        # Zero point energy
        free_energy = np.sum( w / 2)

        # Add also the entropy
        if T > 0:
            beta = 1 / (K_to_Ry * T)
            free_energy += np.sum( 1 / beta * np.log(1 - np.exp(-beta * w)))
                
        return free_energy
        
    def get_phonon_dos(self, w_array, smearing, exclude_acoustic = True, use_cm = False):
        r"""
        GET THE PHONON DOS
        ==================

        This function plots the phonon dos.

        Parameters
        ----------
            w_array : ndarray
                The frequencies at which you want to compute the phonon dos [in Ry] (or cm-1, see use_cm).
            smearing : float
                The smearing [in Ry] (or cm-1, see use_cm).
            exclude_acoustic : bool
                If true, the acoustic modes at gamma are excluded.
            use_cm : bool
                If true, the frequency array and the smearing is supposed to be given in cm-1 
                instead of Ry.
        
        Results
        -------
            dos : ndarray(size = (w_array), dtype = np.float64)
                The phonon density of state.
        """

        nq = len(self.q_tot)

        if use_cm:
            w_array = w_array.copy() / RY_TO_CM
            smearing /= RY_TO_CM

        dos = np.zeros(np.shape(w_array), dtype = np.float64)
        for i, q_vec in enumerate(self.q_tot):
            w, p = self.DyagDinQ(i)

            if q_vec.dot(q_vec) < __EPSILON__:
                trans = Methods.get_translations(p, self.structure.get_masses_array())
                w = w[~trans]
            
            for w0 in w:
                dos += np.exp( -(w_array - w0)**2 / (2 * smearing*smearing)) / np.sqrt(2 * np.pi * smearing*smearing)
        
        dos /= nq

        return dos

    
    def get_two_phonon_dos(self, w_array, smearing, temperature, q_index = 0, exclude_acoustic = True):
        r"""
        COMPUTE THE TWO PHONON DOS
        ==========================

        This subroutine compute the two phonon DOS of the given dynamical matrix.
        It analyzes all possible phonon-phonon scattering and decayment to
        build the two body density of states. This can be used to get an idea how much 
        each phonon can interact with the other in presence of anharmonicity just
        considering energy conservation law and Bose-Einstein statistic.

        The DOS equation is
        
        .. math ::

            \rho^{(2)}(q, \omega) = \int d^3k_1d^3k_2\sum_{\mu\nu}\left[(n_\mu + n_\nu + 1)\delta(\omega - \omega_\mu(k_1) - \omega_\nu(k_2))\delta^3(\vec k_1 + \vec k_2 - \vec q)\right.
       
            \left.  + 2 (n_\mu - n_\nu)\delta(\omega - \omega_\mu(k_1) + \omega_\nu(k_2))\delta^3(\vec q + \vec k_1 - \vec k_2)\right]


        Where the Delta function are replaced by the Lorenzian shape to consider a smearing.

        Parameters
        ----------
            w_array : ndarray
                The frequency of the dos
            smearing : float
                The smearing used to compute the DOS. 
                To converge the smearing you need to study the limit
                :math:`\lim_{\sigma\rightarrow 0} \lim_{N_q\rightarrow\infty} DOS`
            q_index : int
                The q point in which to compute the phonon DOS. 
                You must pass the index that matches the q_tot list.
            exclude_acustic : bool, default = False
                If True the acoustic modes at gamma are neglected in the DOS.
                NOTE: if you have few q points, you will not see the frequencies of the real mode in the DOS!
                
        Results
        -------
            dos : ndarray
                The array of the density of state returned. Same shape as w_array
        """
        K_to_Ry=6.336857346553283e-06


        q_vector = self.q_tot[q_index]
        bg = Methods.get_reciprocal_vectors(self.structure.unit_cell)

        nat = self.structure.N_atoms
        

        DOS = np.zeros( np.shape(w_array), dtype = np.float64)
        for k1_i, k1 in enumerate(self.q_tot):
            # Get the k vectors from the delta relations
            k2_dists = [Methods.get_min_dist_into_cell(bg, k1, q_vector - x) for x in self.q_tot]
            k2p_dists = [Methods.get_min_dist_into_cell(bg, k1, x - q_vector) for x in self.q_tot]

            k2_i = np.argmin(k2_dists)
            k2p_i = np.argmin(k2p_dists)

            k2 = self.q_tot[k2_i]
            k2p = self.q_tot[k2p_i]

            # Get the frequencies at the correct Q points
            _wmu_, _pmu_ = self.DyagDinQ(k1_i)
            _wnu_, _pnu_ = self.DyagDinQ(k2_i)
            _wnu2_, _pnu2_ = self.DyagDinQ(k2p_i)

            trans1 = Methods.get_translations(_pmu_, self.structure.get_masses_array())
            trans2 = Methods.get_translations(_pnu_, self.structure.get_masses_array())
            trans3 = Methods.get_translations(_pnu2_, self.structure.get_masses_array())

            # Sum over mu nu
            for mu in range(3*nat):
                if exclude_acoustic and trans1[mu]:
                    continue
                w_mu = _wmu_[mu]
                n_mu = 0
                if temperature > 0:
                    n_mu = 1 / (np.exp(w_mu  / (temperature * K_to_Ry)) - 1)
                for nu in range(3*nat):
                    w_nu = _wnu_[nu]
                    n_nu = 0
                    if temperature > 0:
                        n_nu = 1 / (np.exp(w_nu  / (temperature * K_to_Ry)) - 1)

                    chi1 = 0
                    if not (exclude_acoustic and trans2[nu]):
                        chi1 = 2*smearing * w_array * (w_mu +  w_nu) * (n_nu + n_mu + 1)
                        chi1 /= 4 * smearing**2*w_array**2 + ( (w_mu + w_nu)**2 - w_array**2)**2
                        chi1 /= w_mu * w_nu

                    w_nu = _wnu2_[nu]
                    if temperature > 0:
                        n_nu = 1 / (np.exp(w_nu  / (temperature * K_to_Ry)) - 1)

                    chi2 = 0            
                    if not (exclude_acoustic and trans3[nu]):
                        chi2 = 2 * smearing * w_array * (w_mu - w_nu) * (n_nu - n_mu)
                        chi2 /= 4*smearing**2 *w_array**2 + ( (w_nu - w_mu)**2 - w_array**2)**2
                        chi2 /= w_mu*w_nu

                    DOS += chi1 + chi2

        return DOS / 2 # We need a 1/2 factor


    def get_phonon_propagator(self, w_array, smearing = 1e-5, only_gamma = False):
        r"""
        GET THE SINGLE PHONON PROPAGATOR
        ================================

        This method computes the single phonon harmonic propagator.
        It is computed in the supercell

        .. math::

            G_{ab}(\omega) = \sum_{\mu}\frac{e_\mu^a e_\mu^b}{(\omega - i\eta)^2 - \omega_\mu^2}

        This is in real space

        Parameters
        ----------
            - w_array : ndarray
                The frequencies at which you want to compute the propagator.
                In [Ry]
            - smearing : float
                The :math:`\eta` value.
            - only_gamma : bool
                If True, only the phonons at gamma will be used

        Results
        -------
            - G_abw : ndarray(size = (3nat, 3nat, len(w)))
                The real space green function

        """

        if not only_gamma:
            w, pols = self.DiagonalizeSupercell()

            super_struct = self.structure.generate_supercell(self.GetSupercell())
            trans = Methods.get_translations(pols, super_struct.get_masses_array())
            nat = super_struct.N_atoms
        else:
            w, pols = self.DyagDinQ(0)
            trans = Methods.get_translations(pols, self.structure.get_masses_array())
            nat = self.structure.N_atoms
            
        G_final = np.zeros( (3*nat, 3*nat, len(w_array)), dtype = np.complex128)

        w = w[~trans]
        pols = pols[:, ~trans]

        nmodes = len(w)
        for mu in range(nmodes):
            epol = np.outer(pols[:, mu], pols[:, mu])
            freq = 1 / ((w_array + 1j*smearing)**2 - w[mu]**2)
            G_final[:,:,:] += np.einsum("ab,c ->abc", epol, freq)
            
        return G_final

    def get_two_phonon_propagator(self, w, T, smearing = 1e-5):
        r"""
        GET THE TWO PHONONS PROPAGATOR
        =========================

        This subroutine computes the two phonons propagator defined as

        .. math ::

            \chi_{\mu\nu}(z, q) = \frac{1}{\beta} \sum_{l} G_\mu(i\Omega_l, \vec k) G_\nu(z - i\Omega_l, \vec q - \vec k)

            \chi_{\mu\nu}(z, q) = \frac{\hbar}{2\omega_\mu\omega_\nu}\left[ \frac{(\omega_\nu + \omega_\mu)[1 + n_\nu + n_\mu]}{(\omega_\nu + \omega_\mu)^2 - z^2} - \frac{(\omega_\nu - \omega_\mu)[n_\nu - n_\mu]}{(\omega_\nu - \omega_\mu)^2 -z^2}\right]
        
        This is the phonon dynamical bubble.
        This is computed in the polarization basis.
        The translational modes are discarted.

        
        
        Parameters
        ----------
            w : ndarray
                The values of the dynamical frequency to compute the phonon propagator.
            T : float
                The temperature to compute the bosonic occupation numbers :math:`n_\mu`.
            semaring : float, default = 1e-5
                The smearing [Ry] to achieve a faster convergence with the k-mesh sampling.

        Result
        ------
            chi : ndarray(size=(3*nat, 3*nat), dtype = np.complex128)
                The bubble phonon propagator
        """


        K_to_Ry=6.336857346553283e-06


        # Get the frequencies at the correct Q points
        _w_, _p_ = self.DiagonalizeSupercell()

        # Get the translational vectors
        trans = Methods.get_translations(_p_, self.structure.generate_supercell(self.GetSupercell()).get_masses_array())

        _w_ = _w_[~trans]
        _p_ = _p_[~trans]

        nmodes = len(_w_)
        ChiMuNu = np.zeros( (nmodes, nmodes, len(w)), dtype = np.complex128)

        # Sum over mu nu
        for mu, w_mu in enumerate(_w_):
            n_mu = 0
            dn_dw = 0
            if T > __EPSILON__:
                n_mu = 1 / (np.exp(w_mu  / (T * K_to_Ry)) - 1)
                dn_dw = -n_mu / (T * K_to_Ry * (1 - np.exp(-w_mu  / (T * K_to_Ry))))
            for nu, w_nu in enumerate(_w_):
                n_nu = 0
                if T > __EPSILON__:
                    n_nu = 1 / (np.exp(w_nu  / (T * K_to_Ry)) - 1)

                chi1 = np.zeros(np.shape(w), dtype = np.complex128)
                chi2 = np.zeros(np.shape(w), dtype = np.complex128)
                chi1 = (w_mu +  w_nu) * (n_nu + n_mu + 1)
                chi1 /= ( (w_mu + w_nu)**2 - (w - 1j*smearing)**2 )
                chi1 /= 2*w_mu*w_nu

                #if np.abs(w) < __EPSILON__ and np.abs(w_nu - w_mu) < __EPSILON__:
                #    chi2 = - dn_dw / (2*w_mu*w_nu)
                #else:
                chi2 = (w_mu - w_nu) * (n_nu - n_mu)
                chi2 /= ( (w_nu - w_mu)**2 - (w - 1j*smearing)**2 )
                chi2 /= 2*w_mu*w_nu
                    
                if np.isnan(chi1 + chi2).any():
                    print("NaN value found in the propagator.")
                    print("NaN value error details:")
                    print("chi1: ", chi1)
                    print("chi2: ", chi2)
                    print("mu = %d, nu = %d" % (mu, nu))
                    print("w_mu = %10.4f, n_mu = %10.4f" % (w_mu * RY_TO_CM, n_mu))
                    print("w_nu = %10.4f, n_nu = %10.4f" % (w_nu * RY_TO_CM, n_nu))
                    raise ValueError("Error, the propagator is NAN, check stdout for details.")
                ChiMuNu[mu, nu, :] = chi1 + chi2

        return ChiMuNu
    
    def get_energy_forces(self, structure, vector1d = False, real_space_fc = None, super_structure = None, supercell = None,
                          displacement = None, use_unit_cell = True):
        """
        COMPUTE ENERGY AND FORCES
        =========================
        
        This subroutine computes the harmonic energy and the forces 
        for the given dynamical matrix at harmonic level.
        
        .. math::
            
            E = \frac 12 \\sum_{\\alpha\\beta} \\left(r_{\\alpha} - r^0_{\\alpha}\\right)\\Phi_{\\alpha\\beta} \\left(r_\\beta - r^0_\\beta\right)
            
            F_\\alpha = \\sum_{\\beta} \\Phi_{\\alpha\\beta} \\left(r_\\beta - r^0_\\beta\right)
            
        The energy is given in Rydberg, while the force is given in Ry/Angstrom
        
        NOTE: In this very moment it has been tested only at Gamma (unit cell)
        
        Parameters
        ----------
            structure : Structure.Structure()
                A unit cell structure in which energy and forces on atoms are computed
            vector1d : bool, optional
                If true the forces are returned in a reshaped 1d vector.
            real_space_fc : ndarray 3nat_sc x 3nat_sc, optional (default None)
                If provided the real space force constant matrix is not recomputed each time the
                method is called. Usefull if you have to repeat this calculation many times.
                You can get the real_space_fc using the method GetRealSpaceFC. 
            super_structure : Structure.Structure()
                Optional, not required. If given is the superstructure used to compute the distance from the
                target one. You can pass it to avoid regenerating it each time this subroutine is called.
                If you do not pass it, you must provide the supercell size (if different than the unit cell)
            super_cell : list of 3 items
                This is the supercell on which compute the energy and force. If none it is inferred by the dynamical matrix.
            displacement:
                The displacements from the self average position to be used. It is not
                necessary since they can be recomputed, however if provided, the calculation is faster.
                It must be in Angstrom.
                To speedup the calculations, many displacements can be provided, in the form:
                displacement.shape = (N_config, 3*nat_sc)
                where N_config are the number of configurations, nat_sc the atoms in the supercell
            use_unit_cell : bool
                If ture, do not compute the real space force constant matrix on the super cell. This is the fastest option.
                Put it to false only for debugging purpouses.
        
        Returns
        -------
            energy : float (or ndarray.shape(N_config))
                The harmonic energy (in Ry) of the structure
            force : ndarray N_atoms x 3 or N_config, nat_sc, 3)
                The harmonic forces that acts on each atoms (in Ry / A)
        """
        
        if supercell is None:
            supercell = self.GetSupercell()

        # Convert the displacement vector in bohr
        #A_TO_BOHR=np.float64(1.889725989)
        if super_structure is None: 
            super_structure = self.structure.generate_supercell(supercell)
        
        # Get the displacement vector (bohr)
        if displacement is None:
            rv = structure.get_displacement(super_structure).reshape(structure.N_atoms * 3) * A_TO_BOHR
        else:
            rv = displacement * A_TO_BOHR

        # Check how many configurations
        many_configs = False
        if len(rv.shape) > 1:
            many_configs = True
            n_configs = rv.shape[0]

        # Fast computation
        if use_unit_cell:
            w, pols = self.DiagonalizeSupercell()

            # Correctly account for not positive definite dynamical matrices
            w2 = w**2 * np.sign(w)

            m = np.tile(super_structure.get_masses_array(), (3,1)).T.ravel()
            m_sqrt = np.sqrt(m)

            epols = np.einsum("ab, a -> ab", pols, m_sqrt)
            x_mu = rv.dot(epols)

            # Check if more configurations needs to be used

            # TODO: add the possibility to pass several structures toghether
            #       to avoid computing many times the same passages
            #       This works only if the displacements are passed
            if not many_configs:
                energy = 0.5 * np.sum(x_mu**2 * w2)
                forces = - epols.dot( w2 * x_mu)
            else:
                w2_tile = np.tile(w2, (n_configs, 1))
                energy = 0.5 * np.sum(x_mu**2 * w2_tile, axis = 1)
                forces = - (w2_tile * x_mu).dot(epols.T)
        else:   
            if real_space_fc is None:
                real_space_fc = self.GetRealSpaceFC(supercell)

            if many_configs:
                raise NotImplementedError("Error, use the use_unit_cell = True if you want to compute many configurations.")
            
            # Get the energy
            energy = 0.5 * rv.dot ( np.real(real_space_fc)).dot(rv)
            
            # Get the forces (Ry/ bohr)
            forces = - real_space_fc.dot(rv) 

        nat_sc = self.structure.N_atoms * np.prod(supercell)
        
        # Translate the force in Ry / A
        forces *= A_TO_BOHR
        if not vector1d:
            if not many_configs:
                forces = forces.reshape( (nat_sc, 3))
            else:
                forces = forces.reshape( (n_configs, nat_sc, 3))
        
        return energy, forces
        
    
    def GetRealSpaceFC(self, supercell_array = (1,1,1), super_structure = None, img_thr = 1e-6):
        """
        GET THE REAL SPACE FORCE CONSTANT 
        =================================
        
        This subroutine uses the fourier transformation to get the real space force constant,
        starting from the fourer space matrix.
        
        .. math::
            
            C_{k\\alpha,k'\\beta}(0, b) = \\frac{1}{N_q} \\sum_q \\tilde C_{k\\alpha k'\\beta}(q) e^{i\\vec q \\cdot \\vec R_b}
            
        Then the translationa property is applied.
        
        .. math::
            
            C_{k\\alpha,k'\\beta}(a, b) = C_{k\\alpha,k'\\beta}(0, b-a)
            
        Here :math:`k` is the atom index in the unit cell, :math:`a` is the supercell index, :math:`\\alpha` is the
        cartesian indices.
    
        NOTE: This method just call the GetSupercellFCFromDyn, look at its documentation for further info.
    
        Returns
        -------
            fc_supercell : ndarray 3nat_sc x 3nat_sc
                The force constant matrix in the supercell.
                If it is a supercell structure, it is use that structure to determine the supercell array
            super_structure : Structure()
                If given, it is used to generate the supercell. Note that in this
                case the supercell_array argument is ignored
        
        """
        
        nq = len(self.q_tot)
        nat = self.structure.N_atoms
        nat_sc = nat * nq
        
        if super_structure is None:
            super_structure = self.structure.generate_supercell(supercell_array)

        # Check the consistency of the argument with the number of q point
        if nat_sc != super_structure.N_atoms:
            raise ValueError("Error, the super_structure number of atoms %d does not match %d computed from the q points." % (super_structure.N_atoms, nat_sc))
            
        dynmat = np.zeros( (nq, 3*nat, 3*nat), dtype = np.complex128, order = "F")
        
        # Fill the dynamical matrix
        for i, q in enumerate(self.q_tot):
            dynmat[i, :,:] = self.dynmats[i]
            
        fc = GetSupercellFCFromDyn(dynmat, np.array(self.q_tot), self.structure, super_structure, img_thr = img_thr)
        return fc
        
#        
#        # Define the number of q points, atoms and unit cell atoms
#        nq = len(self.q_tot)
#        nat = self.structure.N_atoms
#        nat_sc = nq*nat
#        
#        # Check if the supercell array matches the number of q points
#        if np.prod(supercell_array) != nq:
#            raise ValueError("Error, the number of supercell %d must match the number of q points %d." % (np.prod(supercell_array), nq))
#        
#        dynmat = np.zeros( (nq, 3*nat, 3*nat), dtype = np.complex128, order = "F")
#        fc = np.zeros((3*nat_sc, 3*nat_sc), dtype = np.complex128)
#        
#        print "NQ:", nq
#        
#        R_vectors_cart = np.zeros((nq,3), dtype = np.float64, order = "F")
#        q_vect = np.zeros((nq,3), dtype = np.float64, order = "F")
#        
#        
#
#        # Fill the dynamical matrix
#        for i, q in enumerate(self.q_tot):
#            dynmat[i, :,:] = self.dynmats[i]
#            
#            a_x = i % supercell_array[0]
#            a_y = (i / supercell_array[0]) % supercell_array[1]
#            a_z = i / (supercell_array[0] * supercell_array[1])
#            R_vectors_cart[i,:] = a_x * self.structure.unit_cell[0,:] + a_y * self.structure.unit_cell[1,:] + a_z * self.structure.unit_cell[2,:]
#            
#            q_vect[i,:] = 2*np.pi * q / self.alat
#            
#        
#        # For now, to test, just the unit cell
#        for i in range(nq):
#            start_index = 3 * nat * i
#            for j in range(nq):
#                end_index = 3 * nat * j                
#                q_dot_R = np.einsum("ab, b", q_vect, R_vectors_cart[j,:] - R_vectors_cart[i,:])
#                #print "%d, %d => q dot R = " % (i, j), np.exp(1j * q_dot_R)
#                fc[end_index: end_index + 3*nat,  start_index: start_index + 3*nat ] += np.einsum("abc, a", dynmat, np.exp(1j* q_dot_R)) / nq
#            
#            
#            #np.sum(dynmat * np.exp(), axis = 0) / nq
#        print "Imaginary:", np.sqrt(np.sum(np.imag(fc)**2))
#        
#        return fc

    def GetSupercell(self):
        """
        GET SUPERCELL
        =============

        Return the supercell along which this matrix has been generated.

        Results
        -------
            supercell : list of 3 int
                The supercell in each direction.
        """
        return symmetries.GetSupercellFromQlist(self.q_tot, self.structure.unit_cell)
    
    def Interpolate(self, coarse_grid, fine_grid, support_dyn_coarse = None, 
                    support_dyn_fine = None, symmetrize = False):
        """
        INTERPOLATE THE DYNAMICAL MATRIX IN A FINER Q MESH
        ==================================================
        
        This method interpolates the dynamical matrix in a finer mesh.
        It is possible to use a different dynamical matrix as a support,
        then only the difference of the current dynamical matrix 
        with the support is interpolated. In this way you can easier achieve convergence.

        NOTE: This method ignores effective charges. 
        If you want to account for effective charges you should use the ForceTensor.Tensor2 class
        to interpolate.

        
        Parameters
        ----------
            coarse_grid : ndarray(size=3, dtype = int)
                The current q point mesh size
            fine_grid : ndarray(size=3, dtype = int)
                The final q point mesh size
            support_dyn_coarse : Phonons(), optional
                A dynamical matrix used as a support in the same q grid as this one.
                Note that the q points must coincide with the one of this matrix.
            support_dyn_fine : Phonons(), optional
                The support dynamical matrix in the finer cell. 
                If given, the fine_grid is read 
                by the q points of this matrix, and must be compatible
                with the fine_grid.
            symmetrize : bool, optional
                If true activate the symmetrization for the new matrix
        
        Results
        -------
            interpolated_dyn : Phonons()
                The dynamical matrix interpolated.
        """

        if self.effective_charges is not None:
            WARN_TXT="""
WARNING: Effective charges are not accounted by this method
         You should generate a ForceTensor.Tensor2 object
         To account for the interpolation of long-range forces.
            """

            print(WARN_TXT)
            warnings.warn(WARN_TXT, DeprecationWarning)

        
        # Check if the support dynamical matrix is given:
        is_dync = support_dyn_coarse is not None
        is_dynf = support_dyn_fine is not None
        if is_dync != is_dynf:
            raise ValueError("Error, you must provide both support matrix")
                
        nqtot = np.prod(fine_grid)
        
        # Get the q list
        q_list = symmetries.GetQGrid(self.structure.unit_cell, fine_grid)
        #print "The q list:"
        #print q_list
        
        if is_dync and is_dynf:
            # Check if the is_dynf has the correct number of q points
            if nqtot != len(support_dyn_fine.q_tot):
                raise ValueError("Error, the number of q points of the support must coincide with the fine grid")
            
            # Check if the support dyn course q points coincides
            bg = Methods.get_reciprocal_vectors(self.structure.unit_cell)
            for iq, q in enumerate(self.q_tot):
                if Methods.get_min_dist_into_cell(bg, q, support_dyn_coarse.q_tot[iq]) > __EPSILON__:
                    print ("ERROR, NOT MATCHING Q:")
                    print ("self q1 = ", q)
                    print ("support coarse q2 = ", support_dyn_coarse.q_tot[iq])
                    raise ValueError("Error, the coarse support grid as a q point that does not match the self one")
        
            
            # Overwrite the q list
            q_list = support_dyn_fine.q_tot[:]
        
        
        # Prepare the super variables
        if not is_dynf:
            new_dynmat = Phonons(self.structure.copy(), nqtot)
            new_dynmat.q_stars = [[]]
            new_dynmat.initialized = True
            new_dynmat.nqirr = 1
            new_dynmat.alat = self.alat
        else:
            new_dynmat = support_dyn_fine.Copy()
        
        
        super_structure = self.structure.generate_supercell(fine_grid)
        superstruct_coarse = self.structure.generate_supercell(coarse_grid)
        
        nat = self.structure.N_atoms
        fcq = np.zeros( (len(self.q_tot), 3 * nat, 3*nat), dtype = np.complex128)
        for iq, q in enumerate(self.q_tot):
            fcq[iq, :, :] = self.dynmats[iq].copy()
            if is_dync:
                fcq[iq, :, :] -= support_dyn_coarse.dynmats[iq]
                
        # Get the real space force constant matrix
        #r_fcq = GetSupercellFCFromDyn(fcq, np.array(self.q_tot), self.structure, super_structure)
        r_fcq = GetSupercellFCFromDyn(fcq, np.array(self.q_tot), self.structure, superstruct_coarse)

        #r_fcq = self.GetRealSpaceFC(coarse_grid)
            
            
        q_star_i = 0
        passed_qstar = 0
        for iq, q in enumerate(q_list):
            new_dynmat.q_tot[iq][:] = q
            
            # Use the same star as the support matrix
            if is_dynf:
                if iq - passed_qstar == len(support_dyn_fine.q_stars[q_star_i]):
                    q_star_i += 1
                    passed_qstar = iq 
                    
            print ("WORKING ON:", q)
            new_dynmat.q_stars[q_star_i].append(q)
            new_dynmat.dynmats[iq] += InterpolateDynFC(r_fcq, coarse_grid, self.structure, self.structure.generate_supercell(coarse_grid), q)
        
        
        new_dynmat.AdjustQStar()
        
        if symmetrize:
            new_dynmat.Symmetrize()
        
        return new_dynmat
            
    
    def AdjustQStar(self, use_spglib = False):
        """
        ADJUST THE Q STAR
        =================
        
        This function uses the quantum espresso symmetry finder to
        divide the q points into the proper q stars, reordering the current dynamical matrix.


        Parameters
        ----------
            use_spglib : bool
                If true, the SPGLIB is used to perform the symmetrization.
                Otherwise the quantum espresso default symmetry route is used.
                NOTE: Still does not work. Needed to adjust the get q_star to use the spglib symmetries.
        """
        
        # Initialize the symmetries
        qe_sym = symmetries.QE_Symmetry(self.structure)

        if use_spglib:

            qe_sym.SetupFromSPGLIB()
        else:
            qe_sym.SetupQPoint()
        
        # Get the q_stars
        q_stars, q_order = qe_sym.SetupQStar(self.q_tot)
        
        # Reorder the dynamical matrix
        new_dynmats = []
        q_tot = []
        for i in range(len(q_order)):
            iq = q_order[i]
            q = self.q_tot[iq]
            new_dynmats.append(self.dynmats[iq])
        
        self.dynmats = new_dynmats
        self.q_stars = q_stars
        self.q_tot = [y for x in q_stars for y in x] # Unwrap the q points
        self.nqirr = len(q_stars)

        # # Now, the q_stars respect the correct fourier convention
        # q_tot = []
        # for q_star in q_stars:
        #     for q in q_star:
        #         q_tot.append(q)
        # self.q_tot = q_tot
            
    def SymmetrizeSupercell(self, supercell_size = None):
        """
        Testing function, it applies symmetries in the supercell.
        """

        if supercell_size == None:
            supercell_size = self.GetSupercell()

        
        if not __SPGLIB__:
            raise ImportError("Error, the SymmetrizeSupercell method of the Phonon class requires spglib")
        
        superdyn = self.GenerateSupercellDyn(supercell_size)

        # Apply the sum rule
        symmetries.CustomASR(superdyn.dynmats[0])

        qe_sym = symmetries.QE_Symmetry(superdyn.structure)
        qe_sym.SetupFromSPGLIB()
        #qe_sym.SetupQPoint()
        qe_sym.ApplySymmetriesToV2(superdyn.dynmats[0])
        
        #spgsym = spglib.get_symmetry(superdyn.structure.get_ase_atoms())
        #syms = symmetries.GetSymmetriesFromSPGLIB(spgsym, False)
        #superdyn.ForceSymmetries(syms)
        
        # Get the dynamical matrix back
        fcq = GetDynQFromFCSupercell(superdyn.dynmats[0], np.array(self.q_tot), self.structure, superdyn.structure)
        
        for iq, q in enumerate(self.q_tot):
            self.dynmats[iq] = fcq[iq, :, :]

        # Symmetrize also the effective charges and the Raman Tensor if any
        if not self.effective_charges is None:
            qe_sym.ApplySymmetryToEffCharge(self.effective_charges)
        if not self.raman_tensor is None:
            qe_sym.ApplySymmetryToRamanTensor(self.raman_tensor)
        
    def Symmetrize(self, verbose = False, asr = "custom", use_spglib = False):
        """
        SYMMETRIZE THE DYNAMICAL MATRIX
        ===============================
        
        This subroutine uses the QE symmetrization procedure to obtain
        a full symmetrized dynamical matrix.
        
        Parameters
        ----------
            verbose : bool
                If true a lot of info regarding the symmetrization are printed.
            asr : string
                The kind of the acustic sum rule. Allowed are 'crystal', 'simple' or 'custom'.
                for crystal and simple refer to the quantum-espresso guide.
            use_spglib : bool
                If True, the simmetrization is performed with SPGLIB in the supercell
        """

        if use_spglib:
            self.SymmetrizeSupercell()
        
        qe_sym = symmetries.QE_Symmetry(self.structure)
        
        fcq = np.array(self.dynmats, dtype = np.complex128)
        qe_sym.SymmetrizeFCQ(fcq, self.q_stars, asr = asr, verbose = verbose)
        
        for iq,q in enumerate(self.q_tot):
            self.dynmats[iq] = fcq[iq, :, :]

        # Symmetrize also the effective charges and the Raman Tensor if any
        if not self.effective_charges is None:
            qe_sym.ApplySymmetryToEffCharge(self.effective_charges)
        if not self.raman_tensor is None:
            qe_sym.ApplySymmetryToRamanTensor(self.raman_tensor)

    

    def ApplySumRule(self, kind = "custom"):
        """
        ACUSTIC SUM RULE
        ================
        
        The acustic sum rule is a way to impose translational symmetries on the dynamical matrix.
        It affects also the effective charges if any (the total effective charge must be zero).
        For the dynamical matrix it allows to have the self interaction terms:
        .. math::
        
            \\Phi_{n_a, n_a}^{x,y} = - \\sum_{n_b \\neq n_a} \\Phi_{n_a,n_b}^{x,y} 
        
        Parameters
        ----------
            kind : string
                - "custom" : The polarization vectors asigned to the translation are removed from the 
                    gamma dynamical matrix.
                - "normal" : The equation written in this doc_string is applied.
                A NotImplementedError is raised if kind differs from these types.
        """

        # Apply the sum rule on the dynamical matrix
        if kind == "custom":
            # Apply the sum rule
            symmetries.CustomASR(self.dynmats[0])
        elif kind == "normal":
            nb = np.arange(self.structure.N_atoms) 
            for i in range(9):
                x = i / 3
                y = i % 3
                for na in range(self.structure.N_atoms):
                    sum_value = np.sum(self.dynmats[0][3 * na + x, 3 * nb[(nb != na)] + y])
                    self.dynmats[0][3 * na + x, 3 * na + y] =  - sum_value
        else:
            raise NotImplementedError("Error, the specified kind for the sum rule is unknown {}".format(kind))
                    

        # Apply the sum rule on the effective charge
        if self.effective_charges != None:
            total_charge = np.sum(self.effective_charges, axis = 0)

            # Subtract to each atom an average of the total charges
            self.effective_charges = np.einsum("aij, ij -> aij", self.effective_charges,  - total_charge / self.structure.N_atoms)
    

    def GetIRActive(self, use_spglib = False):
        """
        GET IF A MODE IS IR ACTIVE
        ==========================

        This subroutine uses group theory to get if a mode is IR active.

        Parameters
        ----------
            use_spglib : bool
                If True, spglib is used for group theory.
                Good if you are in a supercell.

        Results
        -------
            is_ir_active : ndarray (size = 3*nat)
                Returns a bool array with True for each mode at gamma
                that is IR active.
        """

        there_are_eff_charges = True
        if self.effective_charges is None:
            there_are_eff_charges = False 
            self.effective_charges = np.zeros((self.structure.N_atoms, 3,3), dtype = np.double)
            self.effective_charges = np.random.uniform( size = self.effective_charges.shape)

            # Get the symmetries
            qe_sym = symmetries.QE_Symmetry(self.structure)
            if use_spglib:
                qe_sym.SetupFromSPGLIB()
            else:
                qe_sym.SetupQPoint()
            
            # Symmetrize the effective charges
            qe_sym.ApplySymmetryToEffCharge(self.effective_charges)

        # Simulate the IR signal
        Ir = self.GetIRIntensities()
        is_ir_active = Ir > 1e-8    

        # Delete the random effective charges if added
        if not there_are_eff_charges:
            self.effective_charges = None

        return is_ir_active



    def ApplySymmetry(self, symmat, irt = None):
        """
        APPLY SYMMETRY
        ==============
        
        This function apply a symmetry to the force constant matrix
        The matrix must be a 3 rows x 4 columns array containing the rotation and the subsequent translation of the vectors.
        
        The symmetry check is performed by comparing the two force constant matrix within the given threshold.
        
        .. math::
            
            \\Phi_{s(a)s(b)}^{ij} = \\sum_{h,k = 1}^3 S_{ik} S_{jh} \\Phi_{ab}^{kh}
            
            \\Phi = S \\Phi S^\\dagger
        
        where :math:`s(a)` is the atom in which the :math:`a` atom is mapped by the symmetry.
        
        Note: this works only in supercells at gamma point
        
        Parameters
        ----------
            symmat : ndarray 3x4
                The symmetry matrix to be checked. the last column contains the translations. Trans
            irt : ndarray (size = N_atoms)
                The atoms the symmetry is mapping to.
                
        Results
        -------
            ndarray 3Nat x 3Nat
                The new force constant matrix after the application of the symmetries
        """
        #A_TO_BOHR = 1.889725989
        
        
        # Check if the matrix has been initialized
        if len(self.dynmats) == 0:
            raise ValueError("Error, the phonon force constant has not been initialized. Please consider loading the phonon info.")
            
        if self.nqirr != 1:
            raise ValueError("Error, this method only works for gamma point calculations")
            
            
        # Get the way atoms are echanged
        if irt is None:
            aux_struct = self.structure.copy()
            aux_struct.apply_symmetry(symmat, delete_original = True)
            aux_struct.fix_coords_in_unit_cell()

            eq_atoms = self.structure.get_equivalent_atoms(aux_struct)
        else:
            eq_atoms = irt
        #print eq_atoms
        
        # Get the number of atoms
        n_atoms = self.structure.N_atoms
        
        # Get only the rotational part of the symmetry
        new_s_mat = symmat[:3, :3]
        
        out_fc = np.zeros(np.shape(self.dynmats[0]), dtype = np.complex128)
        in_fc = self.dynmats[0]
        
        # Apply the symmetry to the force constant matrix
        for na in range(n_atoms):
            for nb in range(0, n_atoms):
                # Get the atoms projection of the symmetries
                s_na = eq_atoms[na]
                s_nb = eq_atoms[nb]
                
                # Extract the matrix referring to na and nb atoms
                current_m = in_fc[3 * na : 3*na + 3, 3*nb : 3*nb + 3]
                
                # Conver the matrix in crystalline
                new_m = Methods.convert_matrix_cart_cryst(current_m, self.structure.unit_cell * A_TO_BOHR)
                
                # Apply the symmetry
                #new_m_sym = new_s_mat.dot(new_m.dot( new_s_mat.transpose()))
                new_m_sym = new_s_mat.transpose().dot(new_m.dot( new_s_mat))

                #new_m_sym =new_m.copy()
                
                # Convert back to cartesian coordinates
                new_m = Methods.convert_matrix_cart_cryst(new_m_sym, self.structure.unit_cell * A_TO_BOHR, cryst_to_cart=True)
                
                #print "%d -> %d , %d -> %d)" % (na, s_na, nb, s_nb)#, "d = %.5f" % np.real(np.sqrt(np.sum( (new_m - current_m)**2)))
                
                # Write the matrix into the output
                out_fc[3 * s_na : 3*s_na + 3, 3*s_nb : 3* s_nb + 3] = new_m.copy()
               
                
                #out_fc[3 * s_nb : 3*s_nb + 3, 3*s_na : 3 * s_na + 3] = np.conj(new_m.copy().transpose())
                #print "Test of the transpose. d = ", np.real(np.sqrt(np.sum( (in_fc[3 * nb : 3*nb + 3, 3*na : 3*na + 3].transpose() - out_fc[3 * nb : 3*nb + 3, 3*na : 3 * na + 3])**2)))
        
        # Return the symmetrized result
        #print "Total distance:", np.sqrt(np.sum( (out_fc - np.real(in_fc))**2))
        return out_fc
        
    
        
    def ForceSymmetries(self, symmetries, irt = None, apply_sum_rule = True):
        """
        FORCE THE PHONON TO RESPECT THE SYMMETRIES
        ==========================================
        
        This method forces the phonon dynamical matrix to respect
        the given symmetries.
        
        It uses the method ApplySymmetry to manipulate the force constant matrix.
        
        Note: This method only affect the force constant matrix, the structure is supposed to respect the symmetries.
        
        Note: This works only with gamma matrices (i.e. supercells)
        
        Parameters
        ----------
            symmetries : list of ndarray 3x4
                List of the symmetries matrices. The last column is the fractional translation.
            irt : ndarray(size = (N_sym, N_atoms_sc), dtype = np.intc)
                For each symmetry s, the atom i is mapped into the atom irt[s, i]
                If None, irt is recomputed with the symmetries module.
            apply_sum_rule: bool
                If true the default sum rule is applied.
        """
        
        # Apply the symmetries
        new_fc = np.zeros( np.shape(self.dynmats[0]), dtype = np.complex128 )

        
        self.structure.fix_coords_in_unit_cell()
        for i, sym in enumerate(symmetries):
            # Check if the structure satisfy the symmetry
            if not self.structure.check_symmetry(sym):
                print (sym)
                new_sym = sym.copy()
                new_sym[:, :3] = np.transpose( sym[:, :3])
                print ("Satisfy transpose?", self.structure.check_symmetry(new_sym))
                raise ValueError("Error, the given structure do not satisfy the %d-th symmetry." % (i+1))
            
            # Get the force constant
            current_irt = None
            if not irt is None:
                current_irt = irt[i, :]
            current_fc = self.ApplySymmetry(sym, irt = current_irt)
            
            print (i)
            
            # Try to add the sum rule here
            #newP = self.Copy()
            #newP.dynmats[0] = current_fc
#            #newP.ApplySumRule()
#            
#            distance = np.sum( (self.dynmats[0] - current_fc)**2)
#            distance = np.real(np.sqrt(distance))
#            
            #print "%d) d = " % (i+1), distance
    
            new_fc += current_fc
        
        # Average all the symmetrized structures
        new_fc /= len(symmetries)
        
        
        print ("DIST_SYM_FORC:", np.sqrt(np.sum( (new_fc - self.dynmats[0])**2)))
        self.dynmats[0] = new_fc.copy()
        
                    
        # Print the phonons all toghether
        #print "\n".join( ["\t".join("%.4e" % (xval - freqs[0,j]) for xval in freqs[:, j]) for j in range(3 * self.structure.N_atoms)])
        
        # Apply the acustic sum rule
        if apply_sum_rule:
            self.ApplySumRule()

    def DiagonalizeSupercell(self, verbose = False):
        r"""
        DYAGONALIZE THE DYNAMICAL MATRIX IN THE SUPERCELL
        =================================================

        This method dyagonalizes the dynamical matrix using the supercell approach.

        In this way we simply generate the polarization vector in the supercell
        using those in the unit cell.

        This is performed using the following equation:

        .. math ::

            e_\mu^0(R_0) = \frac{\sqrt{|\tilde e_{q\nu}^a|^2}}{N_q}

            e_\mu^a(R_a) = \frac{\cos(\vec q\cdot \Delta R_{a0}) \Re\left[\tilde e_{q\nu}^a\tilde {e_{q\nu}^b}^\dagger] - \sin(\vec q\cdot \Delta R_{a0}) \Im\left[\tilde e_{q\nu}^a\tilde {e_{q\nu}^b}^\dagger]}{e_\mu^0(R_0)N_q}
        
        Here the :math:`\tilde e_{q\nu}` are the complex polarization vectors in the q point so that :math:`\omega_{q\nu} = \omega_{\mu}`. 
        
        Results
        -------
            w_mu : ndarray( size = (n_modes), dtype = np.double)
                Frequencies in the supercell
            e_mu : ndarray( size = (3*Nat_sc, n_modes), dtype = np.double, order = "F")
                Polarization vectors in the supercell
        """

        supercell_size = len(self.q_tot)
        nat = self.structure.N_atoms

        nmodes = 3*nat*supercell_size
        nat_sc = nat*supercell_size 

        w_array = np.zeros( nmodes, dtype = np.double)
        e_pols_sc = np.zeros( (nmodes, nmodes), dtype = np.double, order = "F")

        # Get the structure in the supercell
        super_structure = self.structure.generate_supercell(self.GetSupercell())

        # Get the supercell correspondence vector
        itau = super_structure.get_itau(self.structure) - 1 # Fort2Py

        # Get the itau in the contracted indices (3*nat_sc -> 3*nat)
        itau_modes = (np.tile(np.array(itau) * 3, (3,1)).T + np.arange(3)).ravel()

        # Get the position in the supercell
        R_vec = np.zeros((nmodes, 3), dtype = np.double)
        for i in range(nat_sc):
            R_vec[3*i : 3*i+3, :] = np.tile(super_structure.coords[i, :] - self.structure.coords[itau[i], :], (3,1))
        
        i_mu = 0
        bg = self.structure.get_reciprocal_vectors() / (2*np.pi)
        for iq, q in enumerate(self.q_tot):
            # Check if the current q point has been seen (we do not distinguish between q and -q)
            skip_this_q = False
            for jq, q_prev in enumerate(self.q_tot):
                if jq >= iq:
                    break
                
                # Check if q and q_prev are related by a G-q operation
                dist = Methods.get_min_dist_into_cell(bg, -q, q_prev)
                if dist < __EPSILON__:
                    skip_this_q = True
                    break
            
            if skip_this_q:
                continue

            
            # Check if this q = -q + G
            is_minus_q = False 
            if Methods.get_min_dist_into_cell(bg, q, -q) < 1e-6:
                is_minus_q = True

                # The dynamical matrix must be real
                re_part = np.real(self.dynmats[iq])

                assert np.max(np.abs(np.imag(self.dynmats[iq]))) < __EPSILON__, "Error, at point {} (q = -q + G) the dynamical matrix is complex".format(iq)

                # Enforce reality to avoid complex polarization vectors
                self.dynmats[iq] = re_part

            # Diagonalize the matrix in the given q point
            wq, eq = self.DyagDinQ(iq)

            # Iterate over the frequencies of the given q point
            nm_q = i_mu
            for i_qnu, w_qnu in enumerate(wq):

                tilde_e_qnu =  eq[:, i_qnu]

                # If this is a minus_q, enforce reality of the vector
                # To correctly fix the gauge
                if is_minus_q:
                    # Get the phase factor from the first non zero value
                    phase_gauge = 0
                    for e_a in tilde_e_qnu:
                        if np.abs(e_a) > __EPSILON__:
                            phase_gauge = np.angle(e_a)
                            break
                    
                    # Work only if it is not already real
                    if np.abs(phase_gauge) > __EPSILON__ and np.abs(phase_gauge - np.pi) > __EPSILON__: 
                        # Apply the phase factor to the polarization vector
                        tilde_e_qnu *= np.exp(-1j * phase_gauge)

                        # Check if the polarization vector is real
                        re_tilde_e_qnu = np.real(tilde_e_qnu)

                        print("Phase:", phase_gauge)
                        print("Vector:", tilde_e_qnu)
                        assert np.max(np.abs(re_tilde_e_qnu - tilde_e_qnu)) < __EPSILON__, "Error while enforcing reality of {}".format(tilde_e_qnu)

                        tilde_e_qnu = re_tilde_e_qnu

                phase = R_vec.dot(q) * 2 * np.pi
                c_e_sc = tilde_e_qnu[itau_modes] * np.exp(1j*phase) / np.sqrt(supercell_size)
                c_e_sc_mq = np.conj(c_e_sc)

                # Get the real and imaginary part
                evec_1 = np.real(.5 * (c_e_sc + c_e_sc_mq))
                evec_2 = np.real((c_e_sc - c_e_sc_mq) / ( 2*1j))

                # Check if they are not zero
                norm1 = evec_1.dot(evec_1)
                norm2 = evec_2.dot(evec_2)
                scalar_dot = 0
                EPSILON = 1e-5

                if norm2 > EPSILON and norm1 > EPSILON:
                    scalar_dot = evec_1.dot(evec_2) / np.sqrt(norm1 * norm2)

                if verbose:
                    print("IQ: {}, MODE: {} has norm1 = {} |  norm2 = {} | scalar_dot = {}".format(iq, i_qnu, np.sqrt(norm1), np.sqrt(norm2), scalar_dot))

                # Check if add to the polarization both 1 and 2
                add_1 = False
                add_2 = False 

                if norm1 > EPSILON:
                    add_1 = True
                            
                if norm2 > EPSILON:
                    add_2 = True

                if is_minus_q: 
                    if add_1 and add_2:
                        if np.abs(np.abs(scalar_dot) - 1) > EPSILON:
                            raise ValueError("Error, with q = -q + G, the two vectors should be linearly dependent")

                        # In this case remove the one with lower norm (higher numerical accuracy)
                        if norm1 > norm2:
                            add_2 = False
                        else:
                            add_1 = False

            

                # If this is a q != -q point, this q point must contribute also for -q
                # Thus twice the elements should be present.
                if not is_minus_q:
                    if not (add_1 and add_2):
                        raise ValueError("Error, the q_point = {} {} {} should contribute also for -q, something went wrong".format(*list(q)))    
                        

                if add_1 and add_2:
                    # Since both real and imaginary should match in this case 
                    # Add only one of them
                    if is_minus_q:
                        add_2 = False 

                if verbose:
                    print("    add_1 = {}; add_2 = {}".format(add_1, add_2))


                # Add the vectors
                if add_1:
                    w_array[i_mu] = w_qnu
                    e_pols_sc[:, i_mu] = evec_1 / np.sqrt(norm1)
                    i_mu += 1
                if add_2:
                    w_array[i_mu] = w_qnu
                    e_pols_sc[:, i_mu] = evec_2 / np.sqrt(norm2)
                    i_mu += 1
                    

                # # Add the second vector
                # if norm1 > EPSILON:
                #     #q_cryst = Methods.covariant_coordinates(bg, q)
                #     #print ("IMU: {}, IQ: {}, IQNU: {}, TOTQ: {}, Q = {}, N1 = {:.3e}, N2 = {:.3e}, DOT = {:.3e}".format(i_mu, iq, i_qnu, len(self.q_tot), q_cryst, norm1, norm2, evec_1.dot(evec_2)))
                #     w_array[i_mu] = w_qnu
                #     e_pols_sc[:, i_mu] = evec_1 / np.sqrt(norm1)
                #     i_mu += 1
                    
                #     # If there is another q point
                #     if not is_minus_q: #scalar_dot < EPSILON:         
                #         if norm2 < EPSILON:
                #             raise ValueError("Error, the q_point = {} {} {} should contribute also for -q, something went wrong".format(*list(q)))    
                        

                #         w_array[i_mu] = w_qnu
                #         e_pols_sc[:, i_mu] = evec_2 / np.sqrt(norm2)
                #         i_mu += 1
                # else:
                #     w_array[i_mu] = w_qnu
                #     e_pols_sc[:, i_mu] = evec_2 / np.sqrt(norm2)
                #     i_mu += 1
            
            # Print how many vectors have been extracted
            if verbose:
                print("The {} / {} q point produced {} nodes".format(iq, len(self.q_tot), i_mu - nm_q))
                



        # Sort the frequencies
        sort_mask = np.argsort(w_array)
        w_array = w_array[sort_mask]
        e_pols_sc = e_pols_sc[:, sort_mask]


        # Get the check for the polarization vector normalization
        assert np.max(np.abs(np.einsum("ab, ab->b", e_pols_sc, e_pols_sc) - 1)) < __EPSILON__

        return w_array, e_pols_sc

        


    def ReadInfoFromESPRESSO(self, filename, read_dielectric_tensor = True, read_eff_charges = True, read_raman_tensor = True):
        """
        READ INFO FROM ESPRESSO
        =======================

        This method reads the effective charges, the dielectric tensor as well as 
        the Raman tensor from an espresso phonon output file.
        It is usefull if you want to run the electric field perturbation without computing
        all the phonon spectrum, deriving only with respect to the electric field.


        Parameters
        ----------
            filename : string
                Path to the standard output of the ph.x calculation.
            read_dielectric_constant: bool
                If False, the dielectric tensor will be ignored
            read_effective_charge : bool
                If False, the effective charges will be ignored
            read_raman_tensor : bool
                If False, the Raman tensor will be ignored.
        """

        if not os.path.exists(filename):
            raise IOError("Error, the given file {} does not exist".format(filename))

        # Read all the file
        f = open(filename, "r")
        lines = [l.strip() for l in f.readlines()]
        f.close()

        # The triggers to know what I am reading
        reading_dielectric = False
        reading_eff_charges = False 
        reading_raman = False

        reading_index = 0
        reading_atom = 0
        reading_pol = 0


        if read_dielectric_tensor and len([x for x in lines if "Dielectric constant in " in x]):
            self.dielectric_tensor = np.zeros((3,3), dtype = np.double)
        if read_eff_charges and len([x for x in lines if "Effective charges" in x]):
            self.effective_charges = np.zeros( (self.structure.N_atoms, 3, 3), dtype = np.double)
        if read_raman_tensor and len([x for x in lines if "Raman tensor" in x]):
            self.raman_tensor = np.zeros((3,3, 3* self.structure.N_atoms), dtype = np.double)

        # Start the analysis
        for line in lines:            
            data = line.split()
            if len(data) == 0:
                continue

            # Check the number of atoms coincides
            if "atoms/cell" in line:
                nat = int(data[4])
                if nat != self.structure.N_atoms:
                    raise ValueError("Error, this Phonon has {} atoms, while the {} calculations contains {} atoms".format(self.structure.N_atoms, filename, nat))

            # Check if we are reading the dielectric
            if "Dielectric constant in " in line:
                reading_dielectric = True 
                reading_eff_charges = False 
                reading_raman = False
                reading_index = 0
                reading_atom = 0
                reading_pol = 0
            elif "Effective charges" in line:
                reading_dielectric = False 
                reading_eff_charges = True 
                reading_raman = False
                reading_index = 0
                reading_atom = 0
                reading_pol = 0
            elif "Raman tensor (A^2)" in line:
                reading_dielectric = False 
                reading_eff_charges = False 
                reading_raman = True
                reading_index = 0
                reading_atom = 0
                reading_pol = 0


            # Check if we must read the dielectric file
            if reading_dielectric and read_dielectric_tensor:
                if len(data) == 5 and data[0] == "(":
                    self.dielectric_tensor[reading_index, :] = [float(x) for x in data[1:4]]
                    reading_index += 1

            if reading_eff_charges and read_eff_charges:
                if data[0] == "atom":
                    reading_atom = int(data[1]) - 1
                    reading_index = 0
                    # Check the consistency of the atom type
                    atm_type = data[2]
                    if self.structure.atoms[reading_atom] != atm_type:
                        error = """
Error while reading {}:
    atom index {} shoud be {}, while it is {} (index {})
""".format(filename, reading_atom, self.structure.atoms[reading_atom], atm_type, reading_atom+1)
                        raise ValueError(error)
                if len(data) == 6 and data[0][0] == 'E':
                    self.effective_charges[reading_atom, reading_index, :] = [float(x) for x in data[2:5]]
                    reading_index += 1
                
                # Check if we ended
                if reading_atom == self.structure.N_atoms - 1 and reading_index == 3:
                    reading_eff_charges = False


            # Reading the raman
            if reading_raman and read_raman_tensor:
                if data[0] == "atom":
                    reading_atom = int(data[2]) - 1
                    reading_index = 0
                    reading_pol = int(data[4]) - 1

                    if reading_atom >= self.structure.N_atoms:
                        error_msg = """
    Error, trying to read atom {} from inputfile {}.
    I expect a maximum of {} atoms from this structure.
""".format(reading_atom + 1, filename, self.structure.N_atoms)
                        raise ValueError(error_msg)
                
                if len(data) == 3:
                    is_good_line = False
                    try:
                        float(data[0])
                        is_good_line = True
                    except:
                        pass

                    if is_good_line:
                        numbers = [float(x) for x in data]
                        self.raman_tensor[reading_index, :, 3 * reading_atom + reading_pol] = numbers
                        reading_index += 1
            


        


def ImposeSCTranslations(fc_supercell, unit_cell_structure, supercell_structure, itau = None):
    """
    IMPOSE TRANSLATION IN THE SUPERCELL
    ===================================
    
    This subroutine imposes the unit cell translations of the supercell force constant matrix.
    Note that it is very different from the acustic sum rule.
    
    .. math::
        
        C_{k\\alpha,k'\\beta}(a,b) = C_{k\\alpha,k'\\beta}(0, b-a)
        
    
    This is obtained by averaging the result
    
    .. math::
        
        C_{k\\\alpha, k'\\beta}(a,b) = \\sum_{c \\in }
    
    Parameters
    ----------
        fc_supercell : ndarray (3nat_sc x 3nat_sc)
            The input-output force constant matrix in real space.
        unit_cell_structure: Structure()
            The structure in the unit cell
        supercell_structure : Structure()
            The structure of the supercell
        itau : optional, ndarray (int)
            The equivalence between unit_cell and supercell atoms. If None it is 
            extracted by the given structures. Note it must be in fortran language
    """
    
    
    if itau is None:
        # Get the fortran one
        itau = supercell_structure.get_itau(unit_cell_structure)
        
    nat_sc = supercell_structure.N_atoms
    fc_tmp = np.zeros( (3,3, nat_sc, nat_sc), dtype = np.float64, order = "F")
    tau_sc_cryst = np.zeros( (3, nat_sc), dtype = np.float64, order = "F")
    
    for i in range(nat_sc):   
        tau_sc_cryst[:,i] = Methods.covariant_coordinates(supercell_structure.unit_cell, supercell_structure.coords[i, :])
    
    # Fill the force constant matrix
    for i in range(nat_sc):
        for j in range(nat_sc):
            fc_tmp[:,:, i, j] = fc_supercell[3*i : 3*i + 3, 3*j: 3*j+3]
    
    # Call the fortran suboruitne
    symph.impose_trans_sc(fc_tmp, tau_sc_cryst, itau, nat_sc)
    
    #Revert it in the original force constant matrix
    for i in range(nat_sc):
        for j in range(nat_sc):
            fc_supercell[3*i : 3*i + 3, 3*j: 3*j+3] = fc_tmp[:,:, i, j]
    
    
    
        

def GetSupercellFCFromDyn(dynmat, q_tot, unit_cell_structure, supercell_structure, itau = None, img_thr = 1e-6):
    """
    GET THE REAL SPACE FORCE CONSTANT 
    =================================
    
    This subroutine uses the fourier transformation to get the real space force constant,
    starting from the fourer space matrix.
    
    .. math::
        
        C_{k\\alpha,k'\\beta}(0, b) = \\frac{1}{N_q} \\sum_q \\tilde C_{k\\alpha k'\\beta}(q) e^{i\\vec q \\cdot \\vec R_b}
        
    Then the translationa property is applied.
    
    .. math::
        
        C_{k\\alpha,k'\\beta}(a, b) = C_{k\\alpha,k'\\beta}(0, b-a)
        
    Here :math:`k` is the atom index in the unit cell, :math:`a` is the supercell index, :math:`\\alpha` is the
    cartesian indices.
    
    
    Parameters
    ----------
        dynmat : ndarray (nq, 3nat, 3nat, dtype = np.complex128)
            The dynamical matrix at each q point. Note nq must be complete, not only the irreducible.
        q_tot : ndarray ( nq, 3)
            The q vectors in Angstrom^-1
        unit_cell_structure : Structure()
            The reference structure of the unit cell.
        supercell_structure : Structure()
            The reference structure of the supercell. It is used to keep the same indices of the atomic positions.
            Note, it is required that consecutive atoms are placed sequently
        itau : Ndarray(nat_sc) , optional
            the correspondance between the supercell atoms and the unit cell one.
            If None is recomputed
    Returns
    -------
        fc_supercell : ndarray 3nat_sc x 3nat_sc
            The force constant matrix in the supercell.
    
    """
    
    # Define the number of q points, atoms and unit cell atoms
    nq = len(q_tot)
    nat = np.shape(dynmat)[1] //3
    nat_sc = nq*nat

        
    if itau is None:
        itau = supercell_structure.get_itau(unit_cell_structure)-1
    
    #dynmat = np.zeros( (nq, 3*nat, 3*nat), dtype = np.complex128, order = "F")
    fc = np.zeros((3*nat_sc, 3*nat_sc), dtype = np.complex128)


    fc = symph.fast_ft_real_space_from_dynq(unit_cell_structure.coords, supercell_structure.coords, itau+1, np.array(q_tot), dynmat, unit_cell_structure.N_atoms, supercell_structure.N_atoms, q_tot.shape[0])

    
    
    """ 
    for i in range(nat_sc):
        i_uc = itau[i]
        t1 = time.time()
        for j in range(nat_sc):
            j_uc = itau[j]
            R = supercell_structure.coords[i, :] - unit_cell_structure.coords[i_uc,:]
            R -= supercell_structure.coords[j, :] - unit_cell_structure.coords[j_uc,:]
            
            # q_dot_R is 1d array that for each q contains the scalar product with R
            q_dot_R = q_tot.dot(R)
            
            t2 = time.time()
            fc[3*i : 3*i + 3, 3*j : 3*j + 3] += np.einsum("abc, a", dynmat[:, 3*i_uc : 3*i_uc + 3, 3*j_uc: 3*j_uc + 3], np.exp(1j * 2*np.pi * q_dot_R)) / nq
            t3 = time.time()

            print("Time to do a single cycle: ", t3 - t2)
            print("Total number of cycles = {} / {}".format(nat_sc * i + j + 1, nat_sc**2)) 
        print("Time for a whole cycle: ", t3 - t1) """
#    
#    # For now, to test, just the unit cell
#    for i in range(nq):
#        start_index = 3*nat*i
#        for j in range(nq):
#            end_index = 3*nat*j
#            
#            q_dot_R = np.sum(q_tot[i,:] * R_vectors_cart[j,:])
#        
#            fc[end_index: end_index + 3*nat,  start_index: start_index + 3*nat ] += dynmat[i,:,:] *  np.exp(1j* 2 * np.pi* q_dot_R) / nq
#        
        
        #np.sum(dynmat * np.exp(), axis = 0) / nq
    #print "Imaginary:", np.sqrt(np.sum(np.imag(fc)**2))

    # Check the imaginary part
    imag = np.sqrt(np.sum(np.imag(fc)**2))
    ASSERT_ERROR = """
    Error, the imaginary part of the real space force constant 
    is not zero. IMAG={}
    """
    assert imag < img_thr, ASSERT_ERROR.format(imag)
    
    return fc



def GetDynQFromFCSupercell(fc_supercell, q_tot, unit_cell_structure, supercell_structure, itau = None):
    r"""
    GET THE DYNAMICAL MATRICES
    ==========================
    
    This subroutine uses the fourier transformation to get the dynamical matrices,
    starting from the real space force constant.
    
    .. math::
        
        \tilde C_{k\alpha k'\beta}(q) = \sum_{b}C_{k\alpha,k'\beta}(0, b)e^{i\vec q \cdot \vec R_b}
        
        
    Here :math:`k` is the atom index in the unit cell, :math:`a` is the supercell index, :math:`\alpha` is the
    cartesian indices.
    
    
    Parameters
    ----------
        fc_supercell : ndarray 3nat_sc x 3nat_sc
            The dynamical matrix at each q point. Note nq must be complete, not only the irreducible.
        q_tot : ndarray ( nq, 3)
            The q vectors in Angstrom^-1
        unit_cell_structure : Structure()
            The structure of the unit cell
        supercell_structure : Structure()
            The structure of the supercell
    Returns
    -------
        dynmat : ndarray (nq, 3nat, 3nat, dtype = np.complex128) 
            The force constant matrix in the supercell.
    
    """
    
    # Define the number of q points, atoms and unit cell atoms
    nq = np.shape(q_tot)[0]
    nat_sc = np.shape(fc_supercell)[0]//3
    nat = nat_sc // nq
    
    if itau is None:
        itau = supercell_structure.get_itau(unit_cell_structure)-1
    
    
    #dynmat = np.zeros( (nq, 3*nat, 3*nat), dtype = np.complex128, order = "F")
    dynmat = np.zeros((nq, 3*nat, 3*nat), dtype = np.complex128)
    
    #print "NQ:", nq
    
    
    for i in range(nat_sc):
        i_uc = itau[i]
        for j in range(nat_sc):
            j_uc = itau[j]
            R = supercell_structure.coords[i, :] - unit_cell_structure.coords[i_uc,:]
            R -= supercell_structure.coords[j, :] - unit_cell_structure.coords[j_uc,:]
            
            # q_dot_R is 1d array that for each q contains the scalar product with R
            q_dot_R = q_tot.dot(R)
            
            dynmat[:,3*i_uc: 3*i_uc +3,3*j_uc: 3*j_uc + 3] += np.einsum("a, bc",  np.exp(-1j * 2*np.pi * q_dot_R), fc_supercell[3*i : 3*i + 3, 3*j : 3*j + 3]) / nq
            
#    
#    # Fill the dynamical matrix
#    for i in range(nq):
#        
#        a_x = i % supercell_array[0]
#        a_y = (i / supercell_array[0]) % supercell_array[1]
#        a_z = i / (supercell_array[0] * supercell_array[1])
#        R_vectors_cart[i,:] = a_x * unit_cell[0,:] + a_y * unit_cell[1,:] + a_z * unit_cell[2,:]
#        
#        
#    
#    # For now, to test, just the unit cell
#    for i in range(nq):
#        start_index = 3 * nat * i   
#        q_dot_R = np.einsum("ab, b", q_tot, R_vectors_cart[i,:])
#        #print "%d, %d => q dot R = " % (i, j), np.exp(1j * q_dot_R)
#        dynmat[:,:,:] += np.einsum("bc, a->abc", fc_supercell[:3 *nat, start_index : start_index + 3*nat], np.exp(1j* 2 * np.pi* q_dot_R))
        
    
        
    
    return dynmat



def InterpolateDynFC(starting_fc, coarse_grid, unit_cell_structure, super_cell_structure, q_point):
    """
    INTERPOLATE FORCE CONSTANT MATRIX
    =================================
    
    Interpolate the real space force constant matrix in a bigger supercell. 
    This can be used to obtain a dynamical matrix in many other q points.
    This function uses the quantum espresso matdyn.x subroutines.
    
    Parameters
    ----------
        starting_fc : ndarray(size=(3*natsc , 3*natsc), dtype = float64)
            Array of the force constant matrix in real space.
        coarse_grid : ndarray(size=3, dtype=int)
            The dimension of the supercell that defines the starting_fc.
        unit_cell_structure : Structure()
            The structure in the unit cell
        super_cell_structure : Structure()
            The structure in the super cell
        q_point : ndarray(size=3, dtype=float64)
            The q point in which you want to interpolate the dynamical matrix.
    
    Results
    -------
        dyn_mat : ndarray(size=(3*nat, 3*nat), dtype = complex128)
            The interpolated dynamical matrix in the provided q point.
    """
    # Get some info about the size
    supercell_size = np.prod(coarse_grid)
    natsc = np.shape(starting_fc)[0]  // 3
    nat = natsc // supercell_size
    
    #print "nat:", nat
    #print "natsc:", natsc
    
    
    # Get the force constant in an appropriate supercell
    QE_frc = np.zeros((coarse_grid[0], coarse_grid[1], coarse_grid[2], 3, 3, nat, nat), dtype = np.float64, order = "F")
    QE_fc = np.zeros((3,3,natsc, natsc), dtype = np.float64, order = "F")
    QE_itau = super_cell_structure.get_itau(unit_cell_structure)
    QE_tau = np.zeros((3, nat), dtype = np.float64, order = "F")
    QE_tau_sc = np.zeros((3, natsc), dtype = np.float64, order = "F")
    QE_at = np.zeros((3,3), dtype = np.float64, order = "F")
    QE_at_sc = np.zeros((3,3), dtype = np.float64, order = "F")
    
    for i in range(natsc):
        for j in range(natsc):
            QE_fc[:,:, i, j] = starting_fc[3*i : 3*(i+1), 3*j : 3*(j+1)]

    
    QE_at[:,:] = unit_cell_structure.unit_cell.transpose()
    QE_at_sc[:,:] = super_cell_structure.unit_cell.transpose()
    QE_tau[:,:] = unit_cell_structure.coords.transpose()
    QE_tau_sc[:,:] = super_cell_structure.coords.transpose()
    
    #print "ENTERING IN GET_FRC"
    QE_frc[:,:,:,:,:,:,:] = symph.get_frc(QE_fc, QE_tau, QE_tau_sc, QE_at, QE_itau, 
          coarse_grid[0], coarse_grid[1], coarse_grid[2], nat, natsc)
    #print "EXITING IN GET_FRC"
    
    # Initialize the interpolation
    nrwsx = 200
    QE_rws = np.zeros((4, nrwsx), dtype = np.float64, order = "F")
    #print "ENTERING IN WSINIT"
    nrws = symph.wsinit(QE_rws, QE_at_sc, nrwsx)
    #print "EXTING FROM WSINIT"
    
    # Perform the interpolation
    QE_q = np.array(q_point, dtype = np.float64)
    #print "ENTERING:"
    #print "TAU SHAPE:", np.shape(QE_tau)
    #print "FRC SHAPE:", np.shape(QE_frc)
    new_dyn = symph.frc_blk(QE_q, QE_tau, QE_frc, QE_at, QE_rws, nrws, nat,
                            coarse_grid[0], coarse_grid[1], coarse_grid[2])
    
    # Conver the dynamical matrix in the Cellconstructor format
    output_dyn = np.zeros( (3*nat, 3*nat), dtype = np.complex128)
    for i in range(nat):
        for j in range(nat):
            output_dyn[3*i : 3*(i+1), 3*j: 3*(j+1)]= new_dyn[:,:, i, j]
            
    return output_dyn
    
    

def get_dyn_from_ase_phonons(ase_ph):
    """
    GET THE DYNAMICAL MATRIX FROM ASE
    =================================

    This function converts an ASE phonons object into the cellconstructor Phonons.

    Parameters
    ----------
        ase_ph : ase.phonons.Phonons()
            The ASE Phonons. It must be already computed
    
    Results
    -------
        dyn : CC.Phonons.Phonons()
            The dynamical matrix
    """


    FC = ase_ph.get_force_constant()
    
    supercell_size = ase_ph.N_c

    # Get the structure
    structure = Structure.Structure()
    structure.generate_from_ase_atoms(ase_ph.atoms)

    # Check if the structure has the unit cell
    if np.linalg.det(structure.unit_cell) == 0:
        ERROR_MSG = """
    Error, the ASE strucure passed to method 'get_dyn_from_ase_phonons' 
           does not have a valid unit cell. 
           If you are computing a isolated molecule, 
           you have to define an unit_cell that contains the molecule 
           (it will not affect the calculation)
    """
        raise ValueError(ERROR_MSG)

    # Get the supercell structure and itau
    nat_sc = structure.N_atoms * np.prod(supercell_size)
    supercell_structure = structure.generate_supercell(supercell_size)
    # Get the equivalent atom in the unit cell vs atoms in the supercell
    itau = supercell_structure.get_itau(structure) - 1 # Fort -> Py (indexing)

    # Get the lattice vectors
    R_cN = ase_ph.lattice_vectors()
    R_cN = np.array(R_cN).T

    # Get the lattice in cartesian units
    R_cN = R_cN.dot(structure.unit_cell)

    N_sup = np.prod(supercell_size)

    q_grid = symmetries.GetQGrid(structure.unit_cell, supercell_size)

    # Put gamma as the first vector
    gamma_index = np.argmin(np.sum(np.array(q_grid)**2, axis = 1))
    q_grid[gamma_index] = q_grid[0].copy()
    q_grid[0] = np.zeros(3, dtype = np.double)


    # Prepare the dynamical matrix
    dyn = Phonons(structure, len(q_grid))
    dyn.q_tot = q_grid 

    # Each q point in a different star
    dyn.q_stars = [ [q] for q in q_grid]    

    # Generate the dynamical matrix in the supercell
    fc_sup = np.zeros( (3*nat_sc, 3*nat_sc), dtype = np.double)

    # TODO: This can be slow
    #       It could be speeded up by getting directly the eigenmodes at the wanted q points.
    for ia in range(nat_sc):
        for ib in range(ia, nat_sc):
            # lattice vector
            R_1 = supercell_structure.coords[ia, :] - structure.coords[itau [ia], :]
            R_2 = supercell_structure.coords[ib, :] - structure.coords[itau [ib], :]
            delta_R = R_2 - R_1

            i_block = Methods.identify_vector(supercell_structure.unit_cell, R_cN, delta_R)

            if i_block == None:
                ERR_MSG="""
ERROR, the ASE Phonons seems not to contain a supercell vector needed for the 
       force constant matrix:
Lattice vector needed : {:16.8f} {:16.8f} {:16.8f}
List of ASE vectors: {}""".format(delta_R[0], delta_R[1], delta_R[2], R_cN)
                raise ValueError(ERR_MSG)
            
            for xa in range(3):
                for xb in range(3):
                    # We found the block
                    fc_sup[3*ia + xa, 3*ib + xb] = FC[i_block, 3*itau[ia] + xa, 3*itau[ib] + xb]
                    fc_sup[3*ib + xb, 3*ia + xa] = fc_sup[3*ia + xa, 3*ib + xb]

    # Now get the dynamical matrix in the correct q point
    dynq = GetDynQFromFCSupercell(fc_sup, np.array(dyn.q_tot), structure, supercell_structure)

    for iq in range(len(dyn.q_tot)):
        # Convert from eV/A^2 into Ry/Bohr^2
        dyn.dynmats[iq] = dynq[iq, :, :] * BOHR_TO_ANGSTROM**2 / RY_TO_EV

    # Now adjust the q stars to match the symmetries
    dyn.AdjustQStar()


    return dyn
