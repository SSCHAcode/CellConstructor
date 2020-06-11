import cellconstructor as CC
import cellconstructor.Phonons
import numpy as np
import sys, os
import spglib

NQIRR = 30

def test_adjust_qstar():
    # Go to the current directory
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dyn_P.350", 112)

    # Adjust the Q star
    dyn.AdjustQStar(use_spglib = False)

    q_star_espresso = len(dyn.q_stars)
    
    assert q_star_espresso == NQIRR 

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("dyn_P.350", 112)

    # Adjust the Q star
    dyn.AdjustQStar(use_spglib = True)

    q_star_spglib = len(dyn.q_stars)
    
    assert q_star_spglib == NQIRR 
    
    


if __name__ == "__main__":
    test_adjust_qstar()
