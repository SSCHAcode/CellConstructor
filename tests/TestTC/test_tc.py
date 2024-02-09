from __future__ import print_function
from __future__ import division

import numpy as np
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ForceTensor
import cellconstructor.ThermalConductivity
import sys, os
import time

def test_tc():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    dyn_prefix = 'final_dyn'
    nqirr = 4

    SSCHA_TO_MS = cellconstructor.ThermalConductivity.SSCHA_TO_MS
    RY_TO_THZ = cellconstructor.ThermalConductivity.SSCHA_TO_THZ
    dyn = CC.Phonons.Phonons(dyn_prefix, nqirr)

    supercell = dyn.GetSupercell()

    fc3 = CC.ForceTensor.Tensor3(dyn.structure,
    dyn.structure.generate_supercell(supercell), supercell)

    d3 = np.load("d3_realspace_sym.npy")*2.0
    fc3.SetupFromTensor(d3)
    fc3 = CC.ThermalConductivity.centering_fc3(fc3)

    mesh = [10,10,10]
    smear = 0.1/RY_TO_THZ

    tc = CC.ThermalConductivity.ThermalConductivity(dyn, fc3,
    kpoint_grid = mesh, scattering_grid = mesh, smearing_scale = None,
    smearing_type = 'constant', cp_mode = 'quantum', off_diag = True)

    temperatures = np.linspace(200,1200,6,dtype=float)
    start_time = time.time()
    tc.setup_harmonic_properties(smear)
    tc.write_harmonic_properties_to_file()

    tc.calculate_kappa(mode = 'SRTA', temperatures = temperatures,
    write_lifetimes = True, gauss_smearing = True, offdiag_mode = 'wigner',
    kappa_filename = 'Thermal_conductivity_SRTA', lf_method = 'fortran-P')

    tc1 = CC.ThermalConductivity.load_thermal_conductivity('standard.pkl')
    keys =  list(tc.kappa.keys())
    keys1 =  list(tc.kappa.keys())
    for key in keys:
        for key1 in keys1:
            if(key == key1):
                np.testing.assert_allclose(tc.kappa[key], tc1.kappa[key1], atol = 1.0e-6)
if __name__ == "__main__":
    test_tc()
