#!/usr/bin/env python3

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Structure

import ase, ase.build
import ase.calculators.emt

import numpy as np
import pytest


@pytest.mark.parametrize("supercell", [(2,2,2), (3,3,3)])
def test_phonons_finite_displacements(supercell, debug=False):
    """Test the phonons using finite displacements"""
    timer = CC.Timer.Timer(active=True)

    # Build a MgO structure using ASE
    atoms = ase.build.bulk('Cu', 'fcc', a=3.6)

    # Convert to cellconstructor
    struct = CC.Structure.Structure()
    struct.generate_from_ase_atoms(atoms)

    my_struct = struct.generate_supercell((2,2,2))
    my_struct.unit_cell[0, :] = struct.unit_cell[1, :] + struct.unit_cell[2, :]
    my_struct.unit_cell[1, :] = struct.unit_cell[0, :] + struct.unit_cell[2, :]
    my_struct.unit_cell[2, :] = struct.unit_cell[0, :] + struct.unit_cell[1, :]
    my_struct.fix_coords_in_unit_cell(delete_copies=True)

    struct = my_struct

    # Generate the EMT calculator for ASE
    calc = ase.calculators.emt.EMT()

    # Get the dynamical matrix using finite displacements
    dyn = CC.Phonons.compute_phonons_finite_displacements(struct, calc,
                                                          supercell=supercell, use_symmetries=False)

    # Compute the dynamical matrix using the symmetrized
    dyn2 = CC.Phonons.compute_phonons_finite_displacements_sym(struct, calc,
                                                               supercell=supercell,
                                                               debug=True,
                                                               timer=timer)

    w_good, pol_good = dyn.DiagonalizeSupercell()
    w_bad, pol_bad = dyn2.DiagonalizeSupercell()

    dyn.Symmetrize(use_spglib = True)
    dyn2.Symmetrize(use_spglib = True)

    if debug:
        print("\n".join(["{:3}) {:10.5f} {:10.5f} cm-1".format(i,
                                                               w_good[i] * CC.Units.RY_TO_CM,
                                                               w_bad[i] * CC.Units.RY_TO_CM)
                         for i in range(len(w_good))]))

    if timer is not None:
        timer.print_report()

    # Check that the two are equal
    for iq, dyn in enumerate(dyn.dynmats):
        assert np.allclose(dyn, dyn2.dynmats[iq], atol=1e-3)


if __name__ == "__main__":
    test_phonons_finite_displacements((6,6,6), debug=True)
