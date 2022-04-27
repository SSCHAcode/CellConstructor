import pytest
import sys, os
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.ForceTensor

import numpy as np

# Perform the test with the following parameters
TEST_DYN = [("../TestSymmetriesSupercell/SnSe.dyn.2x2x2", 3),
            ("../TestSymmetriesSupercell/skydyn_", 4),
            ("dyn", 4)]

@pytest.mark.parametrize("dyn_name, nqirr", TEST_DYN)
def test_interpolate_on_itself(dyn_name, nqirr, verbose = False):
    # Move in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons(dyn_name, nqirr)
    dyn.Symmetrize()

    t2 = CC.ForceTensor.Tensor2(dyn.structure,
                                dyn.structure.generate_supercell(dyn.GetSupercell()),
                                dyn.GetSupercell())
    t2.SetupFromPhonons(dyn)
    t2.Center(Far = 3)
    #t2.Apply_ASR()

    m = dyn.structure.get_masses_array()
    m = np.tile(m, (3,1)).T.ravel()
    
    for iq, q in enumerate(dyn.q_tot):
        fc = t2.Interpolate(-q)
        dynq = fc / np.sqrt(np.outer(m, m))

        w_tensor = np.linalg.eigvalsh(dynq)
        w_tensor = np.sqrt(np.abs(w_tensor)) * np.sign(w_tensor)
        w, p = dyn.DyagDinQ(iq)

        w_tensor *= CC.Units.RY_TO_CM
        w *= CC.Units.RY_TO_CM

        if verbose:
            print("q = {}".format(q))
            print("\n".join(["{:4d})  {:8.3f} cm-1 | {:8.3f} cm-1".format(k, w[k], w_tensor[k]) for k in range(dyn.structure.N_atoms *3)]))
            print()
            print()

        assert np.max(np.abs(w - w_tensor)) < 1e-2, "Error on point q = {}".format(q)

        


if __name__ == "__main__":
    
    test_interpolate_on_itself(*TEST_DYN[-2], verbose = True)

        
    

    

    
