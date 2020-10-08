import sys, os
import cellconstructor as CC
import cellconstructor.Phonons
import pytest

import numpy as np

try:
    __SPGLIB__ = True
    import spglib
except:
    __SPGLIB__ = False
    
def test_spglib_symmetrization():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Skip the test if spglib is not installed
    if not __SPGLIB__:
        pytest.skip("This test requires SPGLIB installed")

    
    # Load the dyn
    dyn = CC.Phonons.Phonons("SnTe_sscha", 3)

    # Symmetrize with quantum espresso
    dyn.Symmetrize()

    w, pols = dyn.DiagonalizeSupercell()

    # Check identity on the polarization vectors
    identity = np.einsum("ai, bi", pols, pols)
    I = np.eye(identity.shape[0])
    assert np.max(np.abs(identity - I)) < 1e-10, "Test identity on polarization with QE symmetrization failed"

    # Generate the supercell and symmetrize with spglib
    new_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    new_dyn.Symmetrize(use_spglib = True)
    w2, pols = new_dyn.DiagonalizeSupercell()
    
    identity = np.einsum("ai, bi", pols, pols)
    I = np.eye(identity.shape[0])
    assert np.max(np.abs(identity - I)) < 1e-10, "Test identity on polarization with SPGLIB symmetrization (supercell) failed"
    

    # Symmetrize with spglib (nothing should happen)
    dyn.Symmetrize(use_spglib = True)

    w3, pols = dyn.DiagonalizeSupercell()
    
    # Check identity on the polarization vectors
    identity = np.einsum("ai, bi", pols, pols)
    I = np.eye(identity.shape[0])
    assert np.max(np.abs(identity - I)) < 1e-10, "Test identity on polarization with SPGLIB symmetrization (unit_cell) failed"


if __name__ == "__main__":
    test_spglib_symmetrization()
