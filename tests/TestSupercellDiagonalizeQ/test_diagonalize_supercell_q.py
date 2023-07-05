import sys, os
import numpy as np
import cellconstructor as CC, cellconstructor.Phonons
import ase, ase.calculators, ase.calculators.emt

def test_diagonalize_supercell_q(verbose = False):
    # Fix the seed to assure reproducibility 
    np.random.seed(0)

    # Set the current working directory
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    supercell = (4, 4, 4)
    # Load gold but build a crazy dynamical matrix just to test a low symmetry group
    # R3m (without inversion)
    struct = CC.Structure.Structure(2)
    a_param = 4
    struct.unit_cell = np.eye(3) * a_param
    struct.atoms[0] = "Au"
    struct.atoms[1] = "Ag"
    struct.coords[1, :] = np.ones(3) * a_param / 2 + 0.2
    struct.build_masses()
    
    calculator = ase.calculators.emt.EMT()

    # Get a dynamical matrix
    dynmat = CC.Phonons.compute_phonons_finite_displacements(
        struct,
        calculator, 
        supercell = supercell)
 
    dynmat.AdjustQStar()
    dynmat.Symmetrize()
    dynmat.ForcePositiveDefinite()
    
    # Now test the polarization and frequency diagonalization
    w, pols, wq, pq = dynmat.DiagonalizeSupercell(return_qmodes = True)

    nq = len(dynmat.q_tot)
    for i in range(nq):
        w_test, p_test = dynmat.DyagDinQ(i)

        assert np.allclose(wq[:, i], w_test)
        assert np.allclose(pq[:, :, i], p_test)


if __name__ == "__main__":
    test_diagonalize_supercell_q(verbose = True)