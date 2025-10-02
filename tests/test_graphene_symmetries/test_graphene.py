import cellconstructor as CC
import cellconstructor.Phonons

import sscha, sscha.Ensemble, sscha.SchaMinimizer

import numpy as np
import sys, os


# A mock test that should not crash when we try to impose symmetries on a valid dynamical matrix
def test_root_step_identity():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    dyn = CC.Phonons.Phonons("dyn_mono_10x10x1_full")
    dyn.Symmetrize()

     
if __name__ == "__main__":
    test_root_step_identity()

    
