from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries
import sys, os
import pytest
import numpy as np
try:
    import spglib
    __SPGLIB__ = True
except:
    __SPGLIB__ = False
    
def test_double_symmetrization(verbose = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    if not __SPGLIB__:
        pytest.skip("spglib needed for this test")

    # Load the SnTe matrix
    dyn = CC.Phonons.Phonons("SnTe_sscha", 3)
    dyn.Symmetrize(use_spglib = True)
    
    w, pols = dyn.DiagonalizeSupercell()

    # get the structure in the supercell to obtain the symmetrization matrix
    ss = dyn.structure.generate_supercell(dyn.GetSupercell())

    # Get the simmetries
    spglib_syms = spglib.get_symmetry(ss.get_ase_atoms())
    syms = CC.symmetries.GetSymmetriesFromSPGLIB(spglib_syms)

    # Get the symmetries in the polarization basis
    syms_pols = CC.symmetries.GetSymmetriesOnModes(syms, ss, pols)

    n_syms = len(syms)

    # Generate the dynamical matrix in the supercell
    dyn_ss = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    fc = dyn_ss.dynmats[0]

    m = np.tile(ss.get_masses_array(), (3,1)).T.ravel()
    mm =  np.outer(m,m)

    dc = fc / np.sqrt(mm)

    # Pass in mode space
    dc_mode = np.conj(pols.T).dot(dc.dot(pols)).astype(np.double)

    dc_mode_other = np.diag(w**2)
    thr = np.max(np.abs(dc_mode - dc_mode_other))
    thr2 = np.max(np.abs(np.diag(dc_mode) - w**2))
    #assert thr < 1e-12, "DC violated by {} | tr2 = {}".format(thr, thr2)

    dc_mode = dc_mode_other
    print(dc_mode)
    

    print(np.max(np.abs(syms_pols[0, :, :] - np.eye(len(m)))))

    random_v = np.zeros(m.shape, dtype = np.double)
    random_v[:] = np.random.normal(size = m.shape)

    epol = np.einsum("ia,i -> ia", pols, 1 / np.sqrt(m))
    epol_t = np.einsum("ia,i -> ia", pols, np.sqrt(m))
    random_v_pols = epol.T.dot(random_v)
    test_transform = epol_t.dot(random_v_pols)

    # Check identity
    identity = np.einsum("ba, bc -> ac", pols, pols)
    thr = np.max(np.abs(identity - np.eye(identity.shape[0])))
    np.savetxt("identity_orthogonal.txt", identity)
    print ("Orthogonality of polarization violated by {}".format(thr))
    assert thr < 1e-9, "Orthogonality of polarization violated by {}".format(thr)

    thr = np.max(np.abs( test_transform - random_v))
    print("Identity with threshold: {}".format(thr))
    assert thr < 1e-9, "Error, identity failed with threshold {}".format(thr)

    # Try to make one direction and back

    
    for i in range(n_syms):
        new_dc = syms_pols[i, : ,:].T.dot(dc_mode.dot(syms_pols[i, :, :]))

        max_diff = np.max(np.abs(dc_mode - new_dc))
        assert max_diff < 1e-5, "Error while applying sym {} of {}".format(i, max_diff)

        sym_mat = CC.symmetries.GetSymmetryMatrix(syms[i], ss)
        v_1 = sym_mat.dot(random_v)


        v_2_pol = syms_pols[i, :, :].dot(random_v_pols)
        v_2 = epol_t.dot(v_2_pol)

        if verbose:
            np.savetxt("sym_{}.txt".format(i), sym_mat)
            np.savetxt("sym_{}_pol.txt".format(i), syms_pols[i,:,:])

        thr = np.max(np.abs(v_1 - v_2))
        assert thr < 1e-5, "Sym {} violated the trheshold by {}".format(i, thr)


        
    

    
if __name__ == "__main__":
    test_double_symmetrization(verbose = True)
