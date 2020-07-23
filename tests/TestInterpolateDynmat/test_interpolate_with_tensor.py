from __future__ import print_function
from __future__ import division
import pytest

import sys, os
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ForceTensor
from cellconstructor.Units import *

__EPSILON__ = 1e-7

# Perform the test with the following parameters
TEST_DYN = [("../TestSymmetriesSupercell/SnSe.dyn.2x2x2", 3, (4,4,1)),
            ("../TestSymmetriesSupercell/skydyn_", 4, (6,6,1))]

@pytest.mark.parametrize("dyn_name, nqirr, target_cell", TEST_DYN)
def test_interpolate_with_tensor(dyn_name, nqirr, target_cell):
    # Move in the directory of the script
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    
    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons(dyn_name, nqirr)

    # Perform the interpolation
    interp_dyn = dyn.Interpolate(dyn.GetSupercell(),
                                 target_cell)
    interp_dyn.Symmetrize()

    
    # Try to save and reload the matrix
    interp_dyn.save_qe("__trial__")
    new_interp_dyn = CC.Phonons.Phonons("__trial__", interp_dyn.nqirr)

    # Now perform the interpolation by using the tensor utility
    t2 = CC.ForceTensor.Tensor2(dyn.structure,
                                dyn.structure.generate_supercell(dyn.GetSupercell()),
                                dyn.GetSupercell())

    t2.SetupFromPhonons(dyn)

    # Check if I reload the frequencies before centering everything matches
    print("Check before the centering")
    for iq, q in enumerate(dyn.q_tot):
        fc_mat = np.conj(t2.Interpolate(q, asr = False))
        dist = dyn.dynmats[iq] - fc_mat
        dist = np.max(np.abs(dist))
        _m_ = np.tile(dyn.structure.get_masses_array(), (3,1)).T.ravel()
        dynmat = fc_mat / np.sqrt(np.outer(_m_, _m_))

        w2 = np.linalg.eigvalsh(dynmat)
        w = np.sqrt(np.abs(w2)) * np.sign(w2)
        w_old,_ = dyn.DyagDinQ(iq)

        print("Q:", q)
        print("\n".join(["{:3d}) {:10.4f} | {:10.4f} cm-1".format(j, w[j] * RY_TO_CM,
                                                                  w_old[j] * RY_TO_CM)
                         for j in range(len(w))]))

        print("DIST:", dist)
        assert dist < __EPSILON__, "Error, the q = {} is wrong.".format(q)

    t2.Center() # Apply the recentering

    print("Check after the centering")
    # Perform the same check after the centering.
    # It must work, as the grid is commensurate, so nothing should change
    for iq, q in enumerate(dyn.q_tot):
        fc_mat = np.conj(t2.Interpolate(q, asr = False))
        dist = dyn.dynmats[iq] - fc_mat
        dist = np.max(np.abs(dist))
        _m_ = np.tile(dyn.structure.get_masses_array(), (3,1)).T.ravel()
        dynmat = fc_mat / np.sqrt(np.outer(_m_, _m_))

        w2 = np.linalg.eigvalsh(dynmat)
        w = np.sqrt(np.abs(w2)) * np.sign(w2)
        w_old,_ = dyn.DyagDinQ(iq)

        print("Q:", q)
        print("\n".join(["{:3d}) {:10.4f} | {:10.4f} cm-1".format(j, w[j] * RY_TO_CM,
                                                                  w_old[j] * RY_TO_CM)
                         for j in range(len(w))]))

        assert dist < __EPSILON__, "Error, the q = {} is wrong.".format(q)
    

    # Ok, it seems that everything is working properly on the original grid
    # Lets try to see if we match the espresso interpolation result
    new_dyn = t2.GeneratePhonons(target_cell)
    new_dyn.save_qe("__trial2__")
    new_dyn.Symmetrize()

    print("Testing the interpolation...")
    for iq, q in enumerate(interp_dyn.q_tot):
        dist = interp_dyn.dynmats[iq] - new_dyn.dynmats[iq]
        dist = np.max(np.abs(dist))
        w,_ = new_dyn.DyagDinQ(iq)
        w_old,_ = interp_dyn.DyagDinQ(iq)

        print("{:3d}) Q:".format(iq), q, "Q NEW:", new_dyn.q_tot[iq])
        print("\n".join(["{:3d}) {:10.4f} | {:10.4f} cm-1".format(j, w[j] * RY_TO_CM,
                                                                  w_old[j] * RY_TO_CM)
                         for j in range(len(w))]))

        assert dist < __EPSILON__, "Error, the q = {} is wrong by {}.".format(q, dist)
    

    # Check if the frequencies matches
    w_qe,_ = interp_dyn.DiagonalizeSupercell()
    w_fc,_ = new_dyn.DiagonalizeSupercell()

    print("ALl modes:")
    print("\n".join(["{:3d}) {:10.4f} | {:10.4f} cm-1".format(j, w_fc[j] * RY_TO_CM,
                                                              w_qe[j] * RY_TO_CM)
                     for j in range(len(w_qe))]))
    

    for i in range(len(w_qe)):
        minindex = np.argmin( np.abs(w_qe[i] - w_fc) )
        delta = np.abs(w_qe[i] - w_fc[minindex])

        assert delta < __EPSILON__, "Error, frequency {} of the original interpolation not found in the new dynamical matrix.".format(w_qe[i] * RY_TO_CM)
        
        

    
if __name__ == "__main__":
    test_interpolate_with_tensor(*TEST_DYN[0])
