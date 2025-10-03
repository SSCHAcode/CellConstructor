from __future__ import print_function
from __future__ import division
import symph
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
    nat = ss.N_atoms

    # Get the simmetries
    spglib_syms = spglib.get_symmetry(ss.get_spglib_cell())
    syms = CC.symmetries.GetSymmetriesFromSPGLIB(spglib_syms)
    
    m = np.tile(ss.get_masses_array(), (3,1)).T.ravel()
    mm =  np.outer(m,m)
    
    epol = np.einsum("ia,i -> ia", pols, 1/ np.sqrt(m))
    epol_t = np.einsum("ia,i -> ia", pols,  np.sqrt(m))

    # Get the symmetries in the polarization basis
    syms_pols = CC.symmetries.GetSymmetriesOnModes(syms, ss, pols)

    n_syms = len(syms)
    n_modes = len(w)

    # Generate the dynamical matrix in the supercell
    dyn_ss = dyn.GenerateSupercellDyn(dyn.GetSupercell())
    fc = dyn_ss.dynmats[0]


    dc = fc / np.sqrt(mm)

    # Pass in mode space
    dc_mode = np.conj(pols.T).dot(dc.dot(pols)).astype(np.double)

    dc_mode_other = np.diag(w**2)
    thr = np.max(np.abs(dc_mode - dc_mode_other))
    thr2 = np.max(np.abs(np.diag(dc_mode) - w**2))
    #assert thr < 1e-12, "DC violated by {} | tr2 = {}".format(thr, thr2)

    dc_mode = dc_mode_other

    random_v = np.zeros(m.shape, dtype = np.double)
    random_v[:] = np.random.normal(size = m.shape)

    random_v_pols = epol.T.dot(random_v)
    test_transform = epol_t.dot(random_v_pols)

    # Check identity
    identity = np.einsum("ba, bc -> ac", pols, pols)
    thr = np.max(np.abs(identity - np.eye(identity.shape[0])))
    np.savetxt("identity_orthogonal.txt", identity)
    if verbose: print ("Orthogonality of polarization violated by {}".format(thr))
    assert thr < 1e-9, "Orthogonality of polarization violated by {}".format(thr)

    thr = np.max(np.abs( test_transform - random_v))
    if verbose: print("Identity with threshold: {}".format(thr))
    assert thr < 1e-9, "Error, identity failed with threshold {}".format(thr)

    # Try to make one direction and back

    
    for i in range(n_syms):
        new_dc = syms_pols[i, : ,:].T.dot(dc_mode.dot(syms_pols[i, :, :]))

        max_diff = np.max(np.abs(dc_mode - new_dc))
        assert max_diff < 1e-5, "Error while applying sym {} of {}".format(i, max_diff)

        sym_mat = CC.symmetries.GetSymmetryMatrix(syms[i], ss)
        sym_mat_cryst = CC.symmetries.GetSymmetryMatrix(syms[i], ss, crystal = True)
        sym_pol_my = epol.T.dot(sym_mat.dot(epol_t))

        # Check if the phi matrix is really symmetric
        # to the given symmetry
        new_fc = sym_mat.dot(fc.dot(sym_mat.T))
        new_fc2 = sym_mat.T.dot(fc.dot(sym_mat))

        thr = np.max(np.abs(new_fc - fc))
        thr2 = np.max(np.abs(new_fc2 - fc))

        # Try to use the spglib equations
        qe_sym = CC.symmetries.QE_Symmetry(ss)
        qe_sym.SetupFromSPGLIB()

        new_fc_s = fc.copy()
        new_fc_spglib = np.zeros( (3,3, qe_sym.QE_nat, qe_sym.QE_nat), dtype = np.double, order ="F")
        for ii in range(qe_sym.QE_nat):
            for j in range(qe_sym.QE_nat):
                new_fc_spglib[:, :, ii, j] = fc[3*ii : 3*(ii+1), 3*j : 3*(j+1)]

        QE_s = np.zeros((3,3,48), dtype = np.intc)
        QE_s[:,:,0] = syms[i][:,:3].T
        QE_irt = np.zeros((48, ss.N_atoms), dtype = np.intc)
        irt = CC.symmetries.GetIRT(ss, syms[i]) + 1
        print(irt)
        QE_irt[0, :] = irt
        
        symph.sym_v2(new_fc_spglib,
                     qe_sym.QE_at,
                     qe_sym.QE_bg,
                     QE_s,
                     QE_irt,
                     1,
                     qe_sym.QE_nat)
        
        for ii in range(qe_sym.QE_nat):
            for j in range(qe_sym.QE_nat):
                new_fc_s[3*ii : 3*(ii+1), 3*j : 3*(j+1)] = new_fc_spglib[:, :, ii, j]

        thr3 = np.max(np.abs(fc - new_fc_s))


        # Apply the symmetry with my matrix also in crystal coordinates
        fc_cryst = np.zeros(fc.shape, dtype = np.double)
        new_fc_cs = np.zeros(fc.shape, dtype = np.double)
        for ii in range(ss.N_atoms):
            for jj in range(ss.N_atoms):
                fmat = fc[3*ii: 3*ii + 3, 3*jj: 3*jj + 3]
                fc_cryst[3*ii : 3*ii + 3, 3*jj: 3*jj + 3] = CC.Methods.convert_matrix_cart_cryst(fmat,
                                                                                                 ss.unit_cell)
        new_fc_cryst = sym_mat_cryst.T.dot(fc_cryst.dot(sym_mat_cryst))
        
        for ii in range(ss.N_atoms):
            for jj in range(ss.N_atoms):
                fmat = new_fc_cryst[3*ii: 3*ii + 3, 3*jj: 3*jj + 3]
                new_fc_cs[3*ii : 3*ii + 3, 3*jj: 3*jj + 3] = CC.Methods.convert_matrix_cart_cryst(fmat,
                                                                                                  ss.unit_cell,
                                                                                                  True)

        thr4 = np.max(np.abs(new_fc_cs - fc))
        # Try to apply the symmetry in crystal space
        
        if verbose:
            print("The symmetry {} is satisfied up to: {} | {} | {} | {}".format(i, thr, thr2, thr3, thr4))
            np.savetxt("sym_{}.txt".format(i), syms[i])
            np.savetxt("sym_{}_real.txt".format(i), sym_mat)
            np.savetxt("sym_{}_cryst.txt".format(i), sym_mat_cryst)
            np.savetxt("sym_{}_pol.txt".format(i), syms_pols[i,:,:])
            np.savetxt("sym_{}_pol_conv.txt".format(i), sym_pol_my)
            np.savetxt("epol.txt", epol)
            np.savetxt("epol_t.txt", epol_t)

        assert np.min([thr, thr2, thr3, thr4]) < 1e-8, "Error, the symmetry {} is not imposed [by {} | {} | {} | {}]".format(i, thr, thr2, thr3, thr4)
        
        v_1 = sym_mat.dot(random_v)

        # First of all, lets check if the symmetry matrix is equivalent to applysymmetryvector:
        v_1_tmp = CC.symmetries.ApplySymmetryToVector(syms[i], random_v.reshape((nat, 3)),
                                                      ss.unit_cell,
                                                      CC.symmetries.GetIRT(ss, syms[i])).ravel()

        thr = np.max(np.abs(v_1_tmp - v_1))
        if verbose:
            print("Good Symmetry Matrix: {} ) {}".format(i, thr))
        assert thr < 1e-5, "Error, the ApplySymmetryVector works differently than GetSymmetryMatrix by {} (id = {})".format(thr, i)


        v_2_pol = syms_pols[i, :, :].dot(random_v_pols)
        v_2 = epol_t.dot(v_2_pol)


        thr = np.max(np.abs(v_1 - v_2))
        assert thr < 1e-5, "Sym {} violated the trheshold by {}".format(i, thr)


        # Now test the consistency rule
        ws_mat = np.einsum("ab, a, b -> ab",
                           syms_pols[i, 3:, 3:],
                           w[3:], 1 / w[3:])
        ws_mat2 = np.einsum("ab, a, b -> ab",
                            syms_pols[i, 3:, 3:],
                            1 / w[3:], w[3:])

        I = ws_mat.T.dot(ws_mat)
        I2 = ws_mat2.dot(ws_mat2.T)

        # Check the two identities
        thr_1 = np.max(np.abs(I - np.eye(n_modes - 3)))
        thr_2 = np.max(np.abs(I - np.eye(n_modes - 3)))

        if verbose:
            print("Identity rule for sym {} = 1: {} | 2: {}".format(i, thr_1, thr_2))

        assert thr_1 < 1e-7 or thr_2 < 1e-7, "Error, the symmetry does not satisfy the polarization rule."


        

        
    

    
if __name__ == "__main__":
    test_double_symmetrization(verbose = True)
