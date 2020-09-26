import cellconstructor as CC
import cellconstructor.Phonons

import sys, os
import numpy as np

EPS = 1e-8

def test_get_harmonic_energy_force():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the dynamical matrix
    dyn = CC.Phonons.Phonons("PbTe.dyn", 8)
    super_struct = dyn.structure.generate_supercell(dyn.GetSupercell())
    
    # Generate a set of structures
    structs = dyn.ExtractRandomStructures(20, 300)
    xats = np.array([x.coords for x in structs])
    u_disps = xats - np.tile(super_struct.coords, (len(structs), 1,1))
    u_disps = u_disps.reshape((len(structs), 3 * super_struct.N_atoms))

    super_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())

    en1, forc1 = dyn.get_energy_forces(None, displacement = u_disps)
    en2, forc2 = super_dyn.get_energy_forces(None, displacement = u_disps)

    en_dist = np.max(np.abs(en1 - en2))
    assert en_dist < EPS, "Error, energy difference between two methods: {}".format(en_dist)
    f_dist = np.max(np.abs(forc1 - forc2))
    assert f_dist < EPS, "Error, the force difference between two methods: {}".format(f_dist)

    
if __name__ == "__main__":
    test_get_harmonic_energy_force()
