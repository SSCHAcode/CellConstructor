from __future__ import print_function, division

import cellconstructor as CC
import cellconstructor.Phonons

# Load ASE
import ase
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.phonons import Phonons

import numpy as np

def test_ase_phonons():
    # COMPUTE THE PHONONS WITH ASE

    # Setup crystal and EMT calculator
    atoms = bulk('Al', 'fcc', a=4.05)

    # Phonon calculator
    N = 6
    ph = Phonons(atoms, EMT(), supercell=(N, N, N), delta=0.05)
    ph.run()

    # Read forces and assemble the dynamical matrix
    ph.read(acoustic=True)
    ph.clean()

    # Now load on CellConstructor
    dyn = CC.Phonons.get_dyn_from_ase_phonons(ph)

    w, pols = dyn.DiagonalizeSupercell()


    assert np.min(w) * CC.Units.RY_TO_CM > -1, "Something wrong happened in the phonon calculation"


if __name__ == "__main__":
    test_ase_phonons()

