from __future__ import print_function
from __future__ import division

"""
This file setup the anharmonic force fields as ASE calculators.
These can be used for examples of calculations.
"""

import Methods
import numpy as np

__ASE__ = True
try:
    import ase
    import ase.calculators.calculator as clt  
except:
    __ASE__ = False 
    raise ImportError("Error, ASE is required to load this module.")


RyToEv = 13.605691932782346

class RockSalt(clt.Calculator):
    """
    This is a calculator to be used for rocksalt structures, with main directions along x,y and z.

    To use it, you need to initialize it. 
    You must provide the harmonic dynamical matrices of the structure, and the anharmonic terms p3 and p4.
    
    Be aware, the order of the atoms matters!
    You can use it as a standard ASE calculator. But you must first setup the parameters with
    set_force_field_parameters.
    """
    def __init__(self, restart = None, ignore_bad_restart_file = False, label = None, atoms = None, **kwargs):
        # Set everything up with the parent class
        clt.Calculator.__init__(self, restart, ignore_bad_restart_file, label, atoms, **kwargs)

        # Override the implemented properties
        self.implemented_properties = ["energy", "forces"]

        self.harm_dyn = None
        self.reference_structure = None
        self.nn_list = None

        self.p3 = 0
        self.p4 = 0
        self.p4x = 0

    def set_force_field_parameters(self, harmonic_dyn, p3=0, p4=0, p4x=0):
        """
        SETUP FORCE FIELD
        =================

        We use the dynamical matrix tu setup the harmonic part of the force field, p3 and p4 are
        defined in the Bianco paper on Structural Phase transition.
        """

        self.harm_dyn = harmonic_dyn.Copy()
        self.reference_structure = self.harm_dyn.structure.generate_supercell(harmonic_dyn.GetSupercell())

        # For each atom in the reference structure  find the near neighbour
        nat_sc = self.reference_structure.N_atoms
        self.reference_structure.fix_coords_in_unit_cell()
        
        self.nn_list = -np.ones((nat_sc, 6), dtype = np.intc)
        for i in range(nat_sc):
            self.nn_list[i, 0] = Methods.get_directed_nn(self.reference_structure, i, np.array([1,0,0]))
            self.nn_list[i, 1] = Methods.get_directed_nn(self.reference_structure, i, np.array([-1,0,0]))
            self.nn_list[i, 2] = Methods.get_directed_nn(self.reference_structure, i, np.array([0,1,0]))
            self.nn_list[i, 3] = Methods.get_directed_nn(self.reference_structure, i, np.array([0,-1,0]))
            self.nn_list[i, 4] = Methods.get_directed_nn(self.reference_structure, i, np.array([0,0,1]))
            self.nn_list[i, 5] = Methods.get_directed_nn(self.reference_structure, i, np.array([0,0,-1]))

        # Delete the atoms whose near neighbour are not returned
        for i in range(nat_sc):
            for j in range(6):
                close_atom = self.nn_list[i, j]
                if not i in self.nn_list[close_atom, :]:
                    self.nn_list[i, j] = -1
                    
        
        self.p3 = p3
        self.p4 = p4

        if p4x != 0:
            raise NotImplementedError("Error, p4x not jet implemented")
        
        self.p4x = p4x
        
    def calculate(self, atoms = None, properties = ["energy"], system_changes = ['positions', 'numbers', 'cell', 'pbc', 'initial_charges', 'initial_magmoms']):
        # Set everything up using the parent class
        clt.Calculator.calculate(self, atoms, properties, system_changes)


        # For each atom in the system use the potential
        nat = len(self.atoms)

        # Apply the toy model force field
        energy = 0
        forces = np.zeros((len(atoms), 3))

        # Get the harmonic energy
        u_disp = atoms.get_positions() - self.reference_structure.coords
        energy, forces = self.harm_dyn.get_energy_forces(None, displacement = u_disp.ravel(), supercell = self.harm_dyn.GetSupercell())

        # Convert from Ry to eV both energy and forces
        energy *= RyToEv
        forces *= RyToEv

        for i in range(nat):
            for alpha in range(3):
                s_next = self.nn_list[i, alpha*2]
                s_prev = self.nn_list[i, alpha*2 + 1]
                
                A_sap = 0
                A_sam = 0
                if s_next != -1:
                    A_sap = (u_disp[s_next, alpha] - u_disp[i, alpha]) / np.sqrt(2)
                if s_prev != -1:
                    A_sam = (u_disp[s_prev, alpha] - u_disp[i, alpha]) / np.sqrt(2)

                # Add the third order term
                energy += self.p3 * (A_sap**3 - A_sam**3)
                forces[i, alpha] += 6 * self.p3* (A_sap**2 - A_sam**2) / np.sqrt(2)

                # Add the fourth order term
                energy += self.p4 * (A_sap**4 + A_sam**4)
                forces[i, alpha] += 8 * self.p4 * (A_sap**3 + A_sam**3) / np.sqrt(2)

        
        self.results = {"energy" : energy, "forces": forces}

        