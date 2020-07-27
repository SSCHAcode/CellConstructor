import cellconstructor as CC
import cellconstructor.Phonons 
import cellconstructor.ForceTensor


import sys, os
import time
import numpy as np 
import pytest

def test_nonanal(verbose = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dyn", 32)

    t2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
    t2.SetupFromPhonons(dyn)

    time1 = time.time()
    t2.Center()
    time2 = time.time()
    print("Time to the full centerng:", time2 - time1, "s")

    q_small_dir = np.random.normal(size = 3)
    q_small_dir /= np.sqrt(q_small_dir.dot(q_small_dir))

    
    dyn_interp_standard = t2.Interpolate(q_small_dir * 1e-7, asr = False)
    dyn_interp_nonanal = t2.Interpolate(np.zeros(3), asr = False,
                                        q_direct = q_small_dir)

    dist = np.max(np.abs(dyn_interp_standard - dyn_interp_nonanal))

    

    if verbose:
        m = dyn.structure.get_masses_array()
        m = np.tile(m, (3,1)).T.ravel()

        d_1 = dyn_interp_standard / np.sqrt(np.outer(m,m))
        d_2 = dyn_interp_nonanal / np.sqrt(np.outer(m,m))

        w2_1 = np.linalg.eigvalsh(d_1)
        w2_2 = np.linalg.eigvalsh(d_2)

        w_1 = np.sqrt(np.abs(w2_1)) * np.sign(w2_1) * CC.Units.RY_TO_CM
        w_2 = np.sqrt(np.abs(w2_2)) * np.sign(w2_2) * CC.Units.RY_TO_CM

        dyn.q_tot = [np.zeros(3)]
        dyn.q_stars = [[np.zeros(3)]]
        dyn.nqirr = 1
        dyn.dynmats[0] = dyn_interp_standard
        dyn.save_qe("d_interp_standard")
        dyn.dynmats[0] = dyn_interp_nonanal
        dyn.save_qe("d_interp_nonanal")
        

        print("\n".join(["w_{:3d} = {:16.8f} | {:16.8f} cm-1".format(i, w_1[i], w_2[i])
                         for i in range(len(w_1))])) 

        print("Distance:", dist)
    assert dist < 1e-5, "Error, the nonanal function of interpolating effective charges is not working correctly.\n Distance from expected: {}".format(dist)

if __name__ == "__main__":
    test_nonanal(True)

