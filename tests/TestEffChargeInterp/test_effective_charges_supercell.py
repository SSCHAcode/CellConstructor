import cellconstructor as CC
import cellconstructor.Phonons 
import cellconstructor.ForceTensor


import sys, os
import time
import numpy as np 
import pytest

def test_effective_charges_supercell():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dyn", 32)

    # Get enforce the ASR on the effective charges
    dyn.Symmetrize()
    ef_charge = dyn.effective_charges.copy()

    super_dyn = dyn.GenerateSupercellDyn()

    # Check the effective charges ASR 
    for i in range(3):
        asr_check = np.sum(ef_charge[:, i, :])

        assert np.abs(asr_check) < 1e-6, "ASR not enforced for the uc effective charges on component %d" % i

        # Check also the new effective charges
        asr_check = np.sum(super_dyn.effective_charges[:, i, :])
        assert np.abs(asr_check) < 1e-6, "ASR not enforced for the super cell effective charges on component %d" % i


    # Check the effective charges
    assert np.allclose(ef_charge, super_dyn.effective_charges[:dyn.structure.N_atoms, :, :]), "Effective charges not correctly copied"

    print("Correctly copied the effective charges")


if __name__ == "__main__":
    test_effective_charges_supercell()
