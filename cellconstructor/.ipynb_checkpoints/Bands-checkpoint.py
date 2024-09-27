from __future__ import print_function
from __future__ import division

"""
This module contains the info about the electronic band structure of a particular structure.
With this module it is possible to compute some optical properties.
"""

import os
import Structure
import Methods

import numpy as np
import scipy, scipy.interpolate

try:
    import ase, ase.units
    kB = ase.units.kB
except:
    kB = 8.617330337217213e-05

class Bands:

    def __init__(self, structure):
        """
        The band class must be associated to a Cellconstructor structure
        """

        self.structure = structure
        self.kpts_cryst = None
        self.N_k = 0
        self.N_bands = 0
        self.band_energies = None
        self.band_occupations = None

        # Here the final values
        self.good_Nk = 0
        self.good_kpts = None
        self.good_band_energies = None
        self.good_occupations = None

    def setup_from_ase_calc(self, ase_calc):
        """
        Setup the band structure starting from an ASE calculator object.
        """

        raise NotImplementedError("Error, this function must be implemented.")
        pass

    def standard_interpolation(self, new_kgrid, kind = "linear"):
        """
        BAND INTERPOLATION
         ==================

        This function performs an interpolation of the bands using a regular grid (scipy).

        Parameters
        ----------
           new_kgrid : ndarray(NK_new, 3)

        """

        self.good_kpts = np.copy(new_kgrid)
        self.good_Nk, dumb = np.shape(new_kgrid)
        self.good_band_energies = np.zeros((self.good_Nk, self.N_bands), dtype = np.double)
        self.good_occupations = np.zeros((self.good_Nk, self.N_bands), dtype = np.double)

        for i in range(self.N_bands):
            interp_func = scipy.interpolate.RegularGridInterpolator(self.kpts_cryst, self.band_energies[:, i], method = kind)
            self.good_band_energies[:, i] = interp_func(self.good_kpts)

    def compute_occupation_numbers(self, T):
        """
        OCCUPATION NUMBERS
        ==================

        This is the function that computes the occupation number for the interpolated data.
        You need to interpolate the data
        
        Parameters
        ----------
            T : float
               Temperature (in K)
        """

        raise NotImplementedError("Error, function not yet implemented")


    def get_band_ipersurface(self, band_index):
        """
        Get an 3d representing the band made as [Nx, Ny, Nz]
        """

        tot_kx, tot_ky, tot_kz = self.good_kpts.T

        kx, index = np.unique(tot_kx, return_index = True)
        ky, index= np.unique(tot_ky, return_index = True)
        kz, index = np.unique(tot_kz, return_index = True)

        

        

        

    def get_group_velocity(self):
        """
        Compute the fermi velocity for each band


        Results
        -------
            v_k = ndarray( size = (N_k, N_bands, 3))
              The vector for each band that is the fermi velocity
        """

        raise NotImplementedError("Error, function not yet implemented")
    

    def get_conductivity(self, T):
        """
        COMPUTE THE STATIC CONDUCTIVITY
        ===============================

        This function exploits the non interactive particle model from the Sommerfeld theory to
        compute the conductivity.

        It will compute the conductivity per relaxation time as:

        .. math ::
    
            \sigma(T) = \frac{1}{\Omega N_k} \sum_{nk} \frac{\partial f}{\partial \varepsilon}(T, \varepsilon_{nk}) |v_{nk}|^2

        where :math:`\Omega` is the unit cell volume, :math:`f` is the fermi occupation function and :math:`v_{nk}` is the
        fermi velocity of the nk band.

        NOTE: You must have interpolated the data and computed the occupation numbers.

        Parameters
        ----------
           T : float
              Temperature, in K

        Results
        -------
        conductivity : float
        """

        volume = np.linalg.det(self.structure.unit_cell) # Volume in A^3

        v_fermi = self.get_fermi_velocity()
        
        cond = 0

        raise NotImplementedError("Error, not yet implemented.")
        # for i in range(self.N_bands):
        #     cond += v_fermi[i,:].dot(v_fermi[i, :]) * get_fermi_derivative(self.fermi_energy, 
            
        
        
        

    def setup_from_espresso_output(self, filename):
        """
        Setup the band structure from a quantum espresso standard output (high verbosity requested)

        Notice that the k points must be on a regular crystal grid, and all printed in output .
        
        Parameters
        ----------
            filename : string
              path to the QE pw.x output. A non self-consistent calculation is suggested.
        """


        # Check if the file exists
        if not os.path.exists(filename):
            raise IOError("Error, the selected file {} does not exist.".format(filename))

        # Start to read the file
        f = open(filename, "r")
        lines = [line.strip() for line in f.readlines()]
        f.close()


        reading_k_points = False
        reading_bands = False
        reading_occupations = False
        reading_energies = False
        occupation_read = False

        band_index = -1
        occupations = []
        energies = []
        current_array = []

        kpts = []

        
        for i, line in enumerate(lines):
            data = lines.split()
            
            # Check if we must start reading the k points
            if len(kpts) == 0 and reading_k_points == False:
                if lines[0] == "cryst.":
                    reading_k_points = True

            if reading_k_points:
                # We finish to read all the k points
                if len(line) == 0:
                    reading_k_points = False

                kpt = np.zeros(3, dtype = np.double)
                kpt[0] = float(data[4])
                kpt[1] = float(data[5])
                kpt[2] = float(data[6])
                
                kpts.append(kpt)

            if not reading_k_points and "End" in line and "calculation" in line:
                # We must start reading the bands
                reading_bands = True
                continue
            
            # Check if we must update the band index
            if reading_k_points and "bands" in line:
                if band_index >= 0:
                    if not occupation_read:
                        raise IOError("Error, I have not found the occupation number. Be sure that QE has been executed in high verbosity.\nFile: {}".format(filename))
                    occupations.append(current_array)
                band_index += 1
                current_array = []

            # Chekc if we must read the occupation numbers
            if reading_k_points and "occupation" in line:
                energies.append(current_array)
                current_array = []
                occupation_read = True

            if reading_k_points:
                # Check if this line is filled by numbers
                all_numbers = False
                try:
                    numbers = [float(x) for x in data]
                    all_numbers = True
                except:
                    pass

                if all_numbers:
                    for x in data:
                        current_array.append(float(x))


            # Check if we are reading the fermi energy
            if "Fermi" in line:
                self.fermi_energy = float(data[4])
                reading_k_points = False
                occupations.append(current_array)
                break
                

        # Check if the reading proceded correctly
        self.kpts_cryst = np.array(kpts)
        self.band_energies = np.array(energies)
        self.band_occupations = np.arrany(occupations)

        N_k, dumb = np.shape(self.kpts_cryst)
        self.N_k = N_k

        dumb1, N_bands = np.shape(self.band_energies)
        dumb2, dumb3 = np.shape(self.band_occupations)

        assert dumb1 == N_k, "The number of k points in input did not match the one in output."
        assert N_bands == dumb3, "The number of bands did non match the occupation number"


        self.N_bands = N_bands


def get_fermi_function(energy, mu, T):
    return 1 / (np.exp( (energy - mu) / (kB*T)) + 1)
        
def get_fermi_derivative(energy, mu, T):
    """
    Get the derivative of the fermi function.
    """

    return - get_fermi_function(energy, mu, T)**2 * np.exp(  (energy - mu) / (kB*T)) / (kB*T)

    
