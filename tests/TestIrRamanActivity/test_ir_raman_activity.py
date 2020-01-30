from __future__ import print_function
from __future__ import division

import pytest
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries

import sys, os


INFO = """
Test the IR activity of common ice. 
We use a simple program to test what are the IR active modes of ice XI.
"""

#@pytest.mark.skip(reason="Not implemented")
def test_ir_activity():
    # Change to the local path
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dynmat")

    # Symmetrize and apply the acoustic sum rule
    dyn.Symmetrize()

    # Get the frequencies and the polarization vectors
    ws, pols = dyn.DyagDinQ(0)


    # Analyze the polarization vectors to look for IR active modes:
    ir_active_modes = dyn.GetIRActive()
    raman_active_modes = dyn.GetRamanActive()

    # Print the mode frequency
    print()
    for i, w in enumerate(ws * CC.Phonons.RY_TO_CM):
        print("{:4d}) {:16.8f} cm-1 | IR active? {} | Raman active? {}".format(i, w ,
                                                            ir_active_modes[i],
                                                            raman_active_modes[i]))
    print("Done.")



if __name__ == "__main__":
    print(INFO)
    test_ir_activity()
