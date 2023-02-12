#!/usr/bin/env python3

import sys, os
import pytest
import numpy as np

import cellconstructor as CC
import cellconstructor.Structure



def test_equivalent_atoms(debug=False):
    # Set a seed for reproducibility
    np.random.seed(0)

    struct = CC.Structure.Structure(2)
    struct.coords[1, :] = 0.5
    struct.has_unit_cell = True
    struct.unit_cell = np.eye(3)

    new_struct = struct.generate_supercell((3,3,3))
    reference = new_struct.copy()
    new_struct.coords[:,:] += np.random.normal(0, 0.1, new_struct.coords.shape)

    # Shuffle the rows of new_struct.coords
    irt = np.random.permutation(new_struct.coords.shape[0])
    new_struct.coords = new_struct.coords[irt, :]

    irt_code = new_struct.get_equivalent_atoms(reference)
    irt_debug = new_struct.get_equivalent_atoms(reference,
                                               debug=True)

    if debug:
        print("IRT:", irt)
        print("IRT_CODE:", irt_code)
        print("IRT_DEBUG:", irt_debug)

    assert np.all(np.array(irt_code) == np.array(irt_debug)), "Pristine structure wrong atom assignment"
    new_struct.coords[:,:] += np.random.normal(0, 0.1, new_struct.coords.shape)

    irt_code = reference.get_equivalent_atoms(new_struct)
    irt_debug = reference.get_equivalent_atoms(new_struct,
                                               debug=True)
    assert np.all(np.array(irt_code) == np.array(irt_debug)), "Noised structure wrong atom assignment"




if __name__ == "__main__":
    test_equivalent_atoms(debug=True)
