import sys, os
import cellconstructor as CC
import cellconstructor.Phonons


def test_diagonalize_supercell():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    # Load the dyn
    dyn = CC.Phonons.Phonons("SnTe_sscha", 3)
    dyn.Symmetrize(use_spglib = True)

    w, p = dyn.DiagonalizeSupercell(verbose = True)


if __name__ == "__main__":
    test_diagonalize_supercell()
    
