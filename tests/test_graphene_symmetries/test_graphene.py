import cellconstructor as CC
import cellconstructor.Phonons

import numpy as np
import sys, os


# A mock test that should not crash when we try to impose symmetries on a valid dynamical matrix
def test_root_step_identity(verbose=False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    dyn = CC.Phonons.Phonons("dyn_mono_10x10x1_full")

    # The real problem is the structure symmetrization
    qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
    qe_sym.SetupQPoint(verbose=verbose)

    qe_symmat = qe_sym.GetSymmetries()
    #fix_translations(qe_symmat)

    if verbose:
        print("Original coords:")
        print(CC.Methods.covariant_coordinates(dyn.structure.unit_cell, dyn.structure.coords))

    for i, symmat in enumerate(qe_symmat):
        new_struct = dyn.structure.copy()

        if verbose:
            print()
            print(f"Symmetry {i}: ")
            print(symmat)

            # Try to fix the translations


            new_struct.apply_symmetry(symmat, delete_original=True)
            print("coords:")
            print(CC.Methods.covariant_coordinates(new_struct.unit_cell, new_struct.coords))
            print("Equivalent atoms:") 
            print(dyn.structure.get_equivalent_atoms(new_struct))

            for k in range(new_struct.N_atoms):
                for h in range(k, new_struct.N_atoms):
                    dist = CC.Methods.get_min_dist_into_cell(new_struct.unit_cell, dyn.structure.coords[k, :], new_struct.coords[h, :])
                    print(f"Distance {k} - {h} = {dist}")
        


    dyn.Symmetrize()

# def fix_translations(symmetries):
#     for i, sym in enumerate(symmetries):
#         mask = sym[:, 3] < 0
#         sym[:, 3][mask] += 1


     
if __name__ == "__main__":
    test_root_step_identity(verbose=True)

    
