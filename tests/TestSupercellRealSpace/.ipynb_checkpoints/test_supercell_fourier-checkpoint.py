# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons


import sys, os

def test_supercell_fourier():

    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    SUPER_DYN = "../TestPhononSupercell/dynmat"
    NQIRR = 8
    SUPERCELL = (3, 3, 2)


    dyn = CC.Phonons.Phonons(SUPER_DYN, NQIRR)


    fc = dyn.GetRealSpaceFC(SUPERCELL)
    fc_new = fc.copy()


    print("Real space:")
    print(fc[:6, :6])

    print("First one:")
    print(dyn.dynmats[0])


    print ("Distances")
    super_structure =  dyn.structure.generate_supercell(SUPERCELL)
    m =super_structure.get_masses_array()
    nq = np.prod(SUPERCELL)
    nat_sc = dyn.structure.N_atoms *nq

    _m_ = np.zeros(3*nat_sc)
    for i in range(nat_sc):
        _m_[3 * i : 3*i + 3] = m[i]

    m_mat = np.outer(1 / np.sqrt(_m_), 1 / np.sqrt(_m_))

    fc *= m_mat

    w_tot = np.sqrt(np.abs(np.real(np.linalg.eigvals(fc))))
    w_tot.sort()

    w_old = np.zeros(len(w_tot))

    for i in range(nq):
        w,p = dyn.DyagDinQ(i)
        w_old[ i * len(w) : (i+1) * len(w)] = w

    w_old.sort()    
    print ("Freq:")
    print ("\n".join ( [" %.5f vs %.5f" % (w_tot[i] * CC.Phonons.RY_TO_CM, w_old[i] * CC.Phonons.RY_TO_CM) for i in range (len(w_tot))]))


    # Try to revert the code

    dynmats_new = CC.Phonons.GetDynQFromFCSupercell(fc_new, np.array(dyn.q_tot), dyn.structure, super_structure)
    d2 = CC.Phonons.GetDynQFromFCSupercell_parallel(fc_new, np.array(dyn.q_tot), dyn.structure, super_structure)


    dyn_sc_new = CC.Phonons.GetSupercellFCFromDyn(dynmats_new, np.array(dyn.q_tot), dyn.structure, super_structure)
    dyn_sc_new2 = CC.Phonons.GetSupercellFCFromDyn(d2, np.array(dyn.q_tot), dyn.structure, super_structure)

    dist1 = np.max(np.abs(dyn_sc_new - fc_new))
    dist2 = np.max(np.abs(dyn_sc_new2 - fc_new))
    print ("Distance reverted:", dist1)
    print ("Distance reverted:", dist2)

    assert dist1 < 1e-10, 'Error in the fourier transform'
    assert dist2 < 1e-10, 'Error in the parallel fourier transform'

    #print "\n".join ( ["RATIO: %.5f " % (w_tot[i] / w_old[i] ) for i in range (len(w_tot))])


if __name__ == "__main__":
    test_supercell_fourier()
