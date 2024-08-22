import cellconstructor as CC, cellconstructor.Phonons
import numpy as np
import pytest
import os
import spglib
import ase, ase.visualize

def test_phonons_bad(verbose=False):
    # Change the directory into the one of the script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load the phonons
    dyn = CC.Phonons.Phonons("1T_TiSe2_", 4)

    # Symmetrize using spglib
    ase.visualize.view(dyn.structure.get_ase_atoms())
    syms = CC.symmetries.GetSymmetriesFromSPGLIB(spglib.get_symmetry(dyn.structure.get_ase_atoms(), 0.05))
    dyn.structure.impose_symmetries(syms)

    dyn.FixQPoints()

    # Print all the q-points
    
    qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
    qe_sym.SetupQPoint(verbose=True)
    qe_sym.SetupQStar(dyn.q_tot, verbose=verbose)



    # Check the q-star
    dyn.AdjustQStar()


if __name__ == "__main__":
    test_phonons_bad(verbose=True)
