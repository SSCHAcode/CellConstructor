import cellconstructor as CC
import cellconstructor.Phonons
import numpy as np

def test_random_generate():

    dyn = CC.Phonons.Phonons("ffield_dynq", 3)
    dyn.ForcePositiveDefinite()
    dyn.Symmetrize()
    
    structs = dyn.ExtractRandomStructures(size = 2, T = 150)
    for s in structs:
        s.fix_coords_in_unit_cell()

        for i in range(s.N_atoms):
            for j in range(s.N_atoms):
                if i != j:
                    d = np.sqrt( np.sum( (s.coords[i,:] - s.coords[j, :])**2))
                    assert d > 0.2


if __name__ == "__main__":
    test_random_generate()
    
