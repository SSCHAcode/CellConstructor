import numpy as np
import pytest

import cellconstructor as CC
import cellconstructor.ForceTensor
import cellconstructor.Phonons


def _make_test_structure():
    s = CC.Structure.Structure(2)
    s.atoms = ["Si", "Si"]
    s.coords[0, :] = [0.0, 0.0, 0.0]
    s.coords[1, :] = [1.3575, 1.3575, 1.3575]
    s.unit_cell = np.eye(3) * 5.43
    s.has_unit_cell = True
    s.masses = {"Si": 28.0855 / CC.Units.MASS_RY_TO_UMA}
    return s


def test_setup_from_uninitialized_phonons():
    """
    Passing an empty Phonons() object (no structure loaded) to
    SetupFromPhonons must raise a clear error, not crash cryptically.
    """
    uc = _make_test_structure()
    sc = uc.generate_supercell((2, 2, 2))
    tensor = CC.ForceTensor.Tensor2(uc, sc, (2, 2, 2))

    empty_phon = CC.Phonons.Phonons()

    with pytest.raises(ValueError,
                       match="not initialized|no structure"):
        tensor.SetupFromPhonons(empty_phon)


def test_setup_wrong_supercell():
    """
    Passing a phonon with a different supercell to SetupFromPhonons
    must raise an error about supercell mismatch.
    """
    uc = _make_test_structure()
    sc_222 = uc.generate_supercell((2, 2, 2))
    tensor = CC.ForceTensor.Tensor2(uc, sc_222, (2, 2, 2))

    # Create a phonon on a 1x1x1 cell (smaller supercell)
    phon = CC.Phonons.Phonons(uc, nqirr=1)
    assert list(phon.GetSupercell()) == [1, 1, 1]

    with pytest.raises((ValueError, AssertionError, IOError),
                       match="supercell"):
        tensor.SetupFromPhonons(phon)


if __name__ == "__main__":
    test_setup_from_uninitialized_phonons()
    test_setup_wrong_supercell()
    print("ok")
