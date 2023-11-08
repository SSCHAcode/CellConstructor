import pytest
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.visualize
import sys, os
import numpy as np

def test_asr_1d(verbose = False):
    # Change the directory to the one of the current script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Load the phonons
    dyn = CC.Phonons.Phonons("dynmat1", full_name=True)

    # Randomize to avoid rotations that mess up the test
    dyn.structure.coords += np.random.normal(0, 0.01, dyn.structure.coords.shape)


    dyn.structure.one_dim_axis = 2

    # Apply the ASR
    dyn.Symmetrize()

    w, p = dyn.DiagonalizeSupercell()
    asr_modes = dyn.structure.get_asr_modes(p)
    if verbose:
        print(asr_modes)
        dyn.save_qe("1ddyn_asr")

    assert np.all(asr_modes[:4])
    assert np.all(~asr_modes[4:])


if __name__ == "__main__":
    test_asr_1d(verbose = True)



