import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.ForceTensor, cellconstructor.Spectral
import numpy as np
import ase, ase.dft, ase.dft.kpoints
import sys, os


def test_load_fc3():
    # TODO: Run with mpirun to properly test it
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    TEMPERATURE = 450
    ENERGY_START = 0
    ENERGY_END = 150
    D_ENERGY = 0.05

    SMEARING_START = .5
    SMEARING_END = 20
    NSMEARING = 2

    KPTS = [2]
    Q_POINT = np.array([.5, .5, 0]) # M


    # Load the dynamcal matrix
    dyn = CC.Phonons.Phonons("converged", 4)

    # Load the d3
    supercell = dyn.GetSupercell()
    tensor3 = CC.ForceTensor.Tensor3(dyn.structure,
                                     dyn.structure.generate_supercell(supercell),
                                     supercell)

    tensor3.SetupFromFile(fname="FC3",file_format='D3Q')
    return

    # Convert the k point in A^-1
    bg = dyn.structure.get_reciprocal_vectors()  / (2 * np.pi)
    q_path = [CC.Methods.cryst_to_cart(Q_POINT, bg)]


    for i, nk in enumerate(KPTS):
        print("Studying nk = {}".format(nk))
        # Interpolate on a 20x20x20
        k_grid = [nk, nk, nk]

        # Compute the dynaical correction
        CC.Spectral.get_diag_dynamic_correction_along_path(dyn=dyn, 
                                                           tensor3=tensor3,  
                                                           k_grid=k_grid, 
                                                           q_path=q_path,
                                                           T = TEMPERATURE, 
                                                           e1=ENERGY_END, de=D_ENERGY, e0=ENERGY_START,
                                                           sm1=SMEARING_END, nsm=NSMEARING, sm0=SMEARING_START,
                                                           filename_sp = 'conv_spectral_M_nk{}'.format(nk))





if __name__ == "__main__":
    test_load_fc3()    
