from __future__ import print_function
from __future__ import division
import pytest

import sys, os
import cellconstructor as CC
import cellconstructor.Phonons
from cellconstructor.Units import *

# Perform the test with the following parameters
TEST_DYN = [("../TestSymmetriesSupercell/SnSe.dyn.2x2x2", 3, (4,4,1)),
            ("../TestSymmetriesSupercell/skydyn_", 4, (6,6,1))]

@pytest.mark.parametrize("dyn_name, nqirr, target_cell", TEST_DYN)
def test_interp_and_diag(dyn_name, nqirr, target_cell):
    # Move in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    
    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons(dyn_name, nqirr)

    # Perform the interpolation
    interp_dyn = dyn.Interpolate(dyn.GetSupercell(),
                                 target_cell)

    
    # Try to save and reload the matrix
    interp_dyn.save_qe("__trial__")
    new_interp_dyn = CC.Phonons.Phonons("__trial__", interp_dyn.nqirr)

    # Compute supercell eigenvalues and eigenvectors
    w_new, p_new = new_interp_dyn.DiagonalizeSupercell()

    # Compute the old eigenvalues and eigenvectors (before the saving)
    w_old, p_old = interp_dyn.DiagonalizeSupercell()

    # They should be equal.
    good_mask = np.abs(w_new - w_old) < 1e-7
    print("Freq before saving-reloading | Freq after saving-reloading")
    print("\n".join(["{:16.8f} | {:16.8f}  Good? {}".format(w_old[i] * RY_TO_CM, w_new[i] * RY_TO_CM, good_mask[i]) for i in range(len(w_old))]))
    assert np.max(np.abs(w_new - w_old)) < 1e-7

    # Ok, it seems that the transformation went correctly
    # Lets try to generate the supercell matrix
    supercell_dyn = interp_dyn.GenerateSupercellDyn(target_cell)

    # Ok, Fourier transform went correctly
    # Lets diagonalize it once again
    w_again, p_again = supercell_dyn.DiagonalizeSupercell()
    good_mask = np.abs(w_again - w_old) < 1e-6
    print("Freq before saving-reloading | Freq after saving-reloading")
    print("\n".join(["{:16.8f} | {:16.8f}  Good? {}".format(w_again[i] * RY_TO_CM, w_new[i] * RY_TO_CM, good_mask[i]) for i in range(len(w_old))]))
    assert np.max(np.abs(w_new - w_old)) < 1e-6

    

    
if __name__ == "__main__":
    test_interp_and_diag(*TEST_DYN[0])
