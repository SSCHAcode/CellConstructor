from __future__ import print_function
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Structure

import ase
from ase.optimize import BFGS
from ase.phonons import Phonons

from ase.units import Bohr, Ry


INFO = """
In this tutoria we load a dynamical matrix from the ase calculator.
You need CP2K installed and callable, as the calculator we use will make use of CP2K force-field.
"""

from CP2K_toy_model_calculator import CP2K_water_calculator as CP2K


def get_phonons_from_ase():

    # Create a simple ASE calculator
    print(INFO)

    # Load the water structure
    water_struct = CC.Structure.Structure()
    water_struct.read_scf("water.scf")

    # Get the ASE object
    ase_water = water_struct.get_ase_atoms()

    # Attach the calculator and relax the structure
    ase_water.set_calculator(CP2K())

    optimizer = BFGS(ase_water)
    optimizer.run(fmax = 0.01)

    # Get the Harmonic dynamical matrix
    water_ph = Phonons(ase_water, CP2K(),
                       supercell = (1,1,1),
                       delta = 0.05)

    water_ph.run()
    water_ph.read(acoustic = True)
    water_ph.clean()

    # Regenerate the relaxed structure
    water_struct.generate_from_ase_atoms(ase_water)

    # Generate the CellConstructor Dynamical Matrix
    dyn_water = CC.Phonons.Phonons(water_struct)

    # Copy the ASE force constant into the dynamical matrix
    dyn_water.dynmats[0][:,:] = water_ph.get_force_constant()[0, :, :]

    # Convert into Ry / Bohr^2
    dyn_water.dynmats[0][:,:] *= Bohr**2 / Ry
    
    # Save the dynamical matrix in the quantum espresso format
    dyn_water.save_qe("dyn_water")
    dyn_water.Symmetrize() # Apply the symmetries

    # Diagonalize the dynamical matrix
    w, pols = dyn_water.DiagonalizeSupercell()
    for i in range(len(w)):
        print("{:2d}) {:16.8f} cm-1".format(i+1,
                                            w[i] * CC.Units.RY_TO_CM))
        
    
if __name__ == "__main__":
    
    get_phonons_from_ase()
    

    
