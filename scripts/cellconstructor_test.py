#!python

from __future__ import print_function
from __future__ import division

import unittest
import numpy as np
import sys, os

try:
    # Python2
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen


import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.Manipulate
import cellconstructor.ForceTensor

__SPGLIB__ = True
try:
    import spglib
except:
    __SPGLIB__ = False

__ASE__ = True
try:
    import ase
    import ase.visualize
except:
    __ASE__ = False


# Class of a distorted rocksalt
class BCCToyModel:
    """
    This is a NaCl toy model
    """
    def __init__(self):
        self.lattice_param = 5.6402 # A
        self.N_atoms = 2
        #self.w_freq = 200 #cm-1
        
        self.structure = None
        self.dynmat = None

        # The effective charges
        self.z0 = np.zeros((self.N_atoms, 3, 3))
        self.dz_dr = 1

        # Build all
        self.build()
        
    def build(self):
        """
        Build the dynamical matrix
        and the structure
        """
        

        self.structure = CC.Structure.Structure(self.N_atoms)
        self.structure.coords[1,:] = .51 * self.lattice_param

        self.structure.has_unit_cell = True
        self.structure.unit_cell = np.eye(3) * self.lattice_param
        self.structure.atoms[0] = "Na"
        self.structure.atoms[1] = "Cl"

        self.structure.masses = {"Na": 20953.89349715178,
                                 "Cl": 302313.43272048925}

        # Build the dynamical matrix
        self.dynmat = CC.Phonons.Phonons(self.structure)

        self.dynmat.dynmats[0] = np.random.uniform(size=(3 * self.N_atoms, 3*self.N_atoms))
        self.dynmat.dynmats[0] += self.dynmat.dynmats[0].T
        
        self.dynmat.Symmetrize()
        self.dynmat.ForcePositiveDefinite()

        # Setup the effective charges
        self.dynmat.effective_charges = self.z0
        

    def get_z(self, structure):
        """
        Get the Z effective charge from the structure
        """

        # Get the displacement of the structure
        u_disp = structure.coords - self.structure.coords
        return self.get_z_u(u_disp)

    def get_z_u(self, u_disp):
        """
        Returns the effective charge given the u_disp.
        
        We remove the q square
        """

        projector = np.ones(3 * self.N_atoms)
        projector[3:] *= -1
        projector /= np.sqrt(projector.dot(projector))
        v_disp = projector.dot(u_disp.ravel()) * CC.Units.A_TO_BOHR

        pol = np.array([1,1,1])

        z_eff_all = np.outer(projector, pol) * 2 * v_disp * self.dz_dr
        z_eff_all = z_eff_all.reshape((self.N_atoms, 3,3))
        z_eff = np.einsum("abc->acb", z_eff_all)
        

        

        # z_eff = np.zeros(3, self.N_atoms, 3))
        
        # # Project the displacement along the only active mode
        
        # u_prime = u_disp.reshape((self.N_atoms, 3)) * CC.Units.A_TO_BOHR
        # trans = np.sum(u_prime, axis = 0) / self.N_atoms
        # v_disp = u_prime - np.tile(trans, (self.N_atoms,1))

        # z_eff = np.zeros((3, self.N_atoms, 3))
        # z_eff  = np.tile(self.dz_dr * v_disp, (3, 1)).reshape((3, self.N_atoms, 3))
        
        # # Exchange axes
        # z_eff = np.einsum("ibc ->bic", z_eff)
        return z_eff





# This function retrives a testing dynamical matrix from the web
def DownloadDynSnSe():
    """
    DOWNLOAD THE DYN FROM WEB
    =========================

    We use urllib to download from the web a testing dynamical matrix.

    Results
    -------
        dyn : CC.Phonons.Phonons()
            The dynamical matrix for test
    """
    NQ = 3
    for i in range(1,NQ +1):
        # Download from the web the dynamical matrices
        dynfile = urlopen("https://raw.githubusercontent.com/mesonepigreco/CellConstructor/master/tests/TestSymmetriesSupercell/SnSe.dyn.2x2x2%d" % i)
        with open("dyn.SnSe.%d" % i,'wb') as output:
            output.write(dynfile.read())

    # Load the dynamical matrices
    dyn = CC.Phonons.Phonons("dyn.SnSe.", NQ)

    # Lets remove the downloaded file
    for i in range(1,NQ +1):
        os.remove("dyn.SnSe.%d" % i)

    return dyn


# This function retrives a testing dynamical matrix from the web
def DownloadDynSky():
    """
    DOWNLOAD THE DYN FROM WEB
    =========================

    We use urllib to download from the web a testing dynamical matrix.

    Results
    -------
        dyn : CC.Phonons.Phonons()
            The dynamical matrix for test
    """
    NQ = 4
    for i in range(1,NQ +1):
        # Download from the web the dynamical matrices
        dynfile = urlopen("https://raw.githubusercontent.com/mesonepigreco/CellConstructor/master/tests/TestSymmetriesSupercell/skydyn_%d" %i)
        #dynfile = urllib2.urlopen("https://raw.githubusercontent.com/mesonepigreco/CellConstructor/master/tests/TestSymmetriesSupercell/newsscha_odd%d" % i)
        with open("dyn.Sky.%d" % i,'wb') as output:
            output.write(dynfile.read())

    # Load the dynamical matrices
    dyn = CC.Phonons.Phonons("dyn.Sky.", NQ)

    # Lets remove the downloaded file
    for i in range(1,NQ +1):
        os.remove("dyn.Sky.%d" % i)

    return dyn
    
    

class TestStructureMethods(unittest.TestCase):

    def setUp(self):
        # Prepare the structure for the tests
        struct_ice = CC.Structure.Structure(12)

        struct_ice.atoms = ["O", "H", "H"] * 4
        struct_ice.coords[0, :] = [2.1897386602476634,  1.2676932473237612,  0.4274022656440920]
        struct_ice.coords[1, :] = [2.1897386602476634,  1.2543789748068450 , 1.4310249717986974]
        struct_ice.coords[2, :] = [2.1897386602476634,  0.3101819276910475,  0.1262494311426678]
        struct_ice.coords[3, :] = [4.3794773204953268,  2.5395546109658751,  6.6906885580208124]
        struct_ice.coords[4, :] = [5.1833303728919438,  2.0453283392179049,  7.0329865156648150]
        struct_ice.coords[5, :] = [3.5756242725987106,  2.0453283392179049,  7.0329865156648150]
        struct_ice.coords[6, :] = [4.3794773204953268,  2.5362441767130308,  4.0053346514190915]
        struct_ice.coords[7, :] = [4.3794773204953268,  2.5495584492299472,  5.0089573620736969]
        struct_ice.coords[8, :] = [4.3794773204953268,  3.4937555008457450,  3.7041818169176670]
        struct_ice.coords[9, :] = [2.1897386602476634,  1.2643828130709174,  3.1127561677458120]
        struct_ice.coords[10, :] = [1.3858856123510472,  1.7586090848188873,  3.4550541253898150]
        struct_ice.coords[11, :] = [2.9935917126442799,  1.7586090848188873,  3.4550541253898150]

        struct_ice.has_unit_cell = True
        struct_ice.unit_cell = np.zeros((3, 3), dtype = np.float64)
        struct_ice.unit_cell[0,0] = 4.3794773204953268
        struct_ice.unit_cell[1,0] = 2.1897386602476634
        struct_ice.unit_cell[1,1] = 3.8039374195367919
        struct_ice.unit_cell[2,2] = 7.1558647760499996

        #ase.visualize.view(struct_ice.get_ase_atoms())

        self.struct_ice = struct_ice

        # Get a rocksalt structure
        self.rocksalt = BCCToyModel()

        # Get a simple cubic structure
        self.struct_simple_cubic = CC.Structure.Structure(1)
        self.struct_simple_cubic.atoms = ["H"]
        self.struct_simple_cubic.has_unit_cell = True
        self.struct_simple_cubic.unit_cell = np.eye(3)
        self.struct_simple_cubic.masses = {"H": 938}


        # Download the complex dynamical matrices from internet
        self.dynSnSe = DownloadDynSnSe()
        self.dynSky = DownloadDynSky()

        # Test a simple NaCl setup
        a = 5.41
        self.NaCl = CC.Structure.Structure(2)
        self.NaCl.atoms = ["Na","Cl"]
        self.NaCl.coords[1,:] = (a/2, a/2, a/2)
        self.NaCl.unit_cell[0,:] = (a/2, a/2, 0)
        self.NaCl.unit_cell[1,:] = (a/2, 0, a/2)
        self.NaCl.unit_cell[2,:] = (0, a/2, a/2)
        self.NaCl.has_unit_cell = True
        self.NaCl.set_masses({"Na": 22.989769, "Cl": 35.453})
        self.NaCldyn = CC.Phonons.Phonons(self.NaCl)
        self.NaCldyn.dynmats[0][:,:] = np.random.uniform(size = (6, 6))
        self.NaCldyn.dynmats[0] += self.NaCldyn.dynmats[0].T
        self.NaCldyn.Symmetrize()




        
    def test_generate_supercell(self):

        #ase.visualize.view(self.struct_ice.get_ase_atoms())
        
        #print("Generating supercell...",)
        super_structure = self.struct_ice.generate_supercell((2,2,2))
        #print(" Done.")

        # Test if restricting the correct cell works
        ortho_cell = self.struct_ice.unit_cell.copy()
        ortho_cell[1,:] = 2*ortho_cell[1,:] - ortho_cell[0,:]
        #print("Restricting supercell to convertional one...",)
        super_structure.unit_cell = ortho_cell
        super_structure.fix_coords_in_unit_cell()
        #print(" Done.")

        #print("N atoms in the conventional cell = {}".format(super_structure.N_atoms))
        #print("N atoms in the primitive cell = {}".format(self.struct_ice.N_atoms))
        self.assertEqual(super_structure.N_atoms, self.struct_ice.N_atoms*2)
        
    def test_spglib_symmetries(self):
        self.assertTrue(__SPGLIB__)
        self.assertTrue(__ASE__)

        if __SPGLIB__ and __ASE__:
            spacegroup = spglib.get_spacegroup(self.struct_ice.get_ase_atoms())
            self.assertEqual(spacegroup, "Cmc2_1 (36)")


    @unittest.skip("not yet implemented test")
    def test_IR_modes(self):
        """
        This test loads a dynamical matrix with effective charges and computes the 
        IR activity for each mode. We benchmark it against dynmat.x of quantum espresso.
        """
        raise NotImplementedError("Error, this test must be implemented")

    @unittest.skip("not yet working function")
    def test_IR_Raman_on_NaCl(self):
        """
        We generate a simple NaCl structure and predict using symmetries if the mode is IR or Raman active
        """
        self.setUp()
        self.NaCldyn.Symmetrize()

        for i in range(100):
            get_ir_activity = self.NaCldyn.GetIRActive()
            assert (get_ir_activity == [False, False, False, True, True, True]).all(), get_ir_activity

        for i in range(100):
            get_raman_activity = self.NaCldyn.GetRamanActive()
            assert not get_raman_activity.any()

        # Check if the activity vector is actually good
        self.NaCldyn.dynmats[0][:,:] = np.random.uniform(size = (3*self.NaCl.N_atoms, 3*self.NaCl.N_atoms))
        self.NaCldyn.effective_charges = np.random.uniform(size = (self.NaCl.N_atoms, 3, 3))
        self.NaCldyn.raman_tensor = np.random.uniform(size = (3, 3, 3 * self.NaCl.N_atoms))

        self.NaCldyn.Symmetrize()

        # Check that the raman tensor is really zero
        assert np.max(np.abs(self.NaCldyn.raman_tensor)) < 1e-10

        # Get the IR responce
        ir_responce = self.NaCldyn.GetIRIntensities()

        # Check that if the intensities along the degenerate modes are all equal
        assert np.max(np.abs(ir_responce[3:] - ir_responce[3])) < 1e-10


    def test_harmonic_energy_forces_fast(self):
        """
        Test the computation of harmonic forces with the new fast method (that is done in the polarization space)
        against the one done in the full real space.
        """

        # Get the Dynamical matrix
        dyn = self.dynSnSe.Copy()

        N_RANDOM = 100
        nat = dyn.structure.N_atoms
        nat_sc = int(nat * np.prod(dyn.GetSupercell()))

        random_structures = dyn.ExtractRandomStructures(N_RANDOM, 100)
        __EPSIL__ = 1e-8

        # Get all the structures at once
        u_disps = np.zeros( (N_RANDOM, 3 * nat_sc), dtype = np.double)
        forces = np.zeros((N_RANDOM, nat_sc, 3), dtype = np.double)
        energies = np.zeros(N_RANDOM, dtype = np.double)
        

        for i, s in enumerate(random_structures):
            u_disps[i, :] = (s.coords - dyn.structure.generate_supercell(dyn.GetSupercell()).coords).ravel()
            energy_new, force_new = dyn.get_energy_forces(s)
            energy_old, force_old = dyn.get_energy_forces(s, use_unit_cell = False)

            assert np.abs(energy_new - energy_old) < __EPSIL__, "Error, the energy are not correctly computed: {:.16e} Ry vs {:.16r} Ry".format(energy_new, energy_old)
            diff = np.max(np.abs(force_new - force_old))
            forces[i, :, :] = force_new
            energies[i] = energy_new
            assert diff < __EPSIL__, "Error, the forces are not correctly computed: max difference is {} Ry / A".format(diff)

        new_energy, all_forces = dyn.get_energy_forces(None, displacement = u_disps, use_unit_cell = True)
        diff = np.max(np.abs(all_forces - forces))

        assert diff < __EPSIL__, "Error, the forces computed all toghether differs of {} Ry / A".format(diff)

        diff = np.max(np.abs(energies - new_energy))
        assert diff < __EPSIL__, "Error, the energies computed all togheter differs of {} Ry / A".format(diff)



    def test_change_phonon_cell(self):
        """
        This tries to change the unit cell of a dynamical matrix and see if the fourier transmformation goes correctly
        """

        dyn = self.dynSnSe.Copy()

        new_cell = dyn.structure.unit_cell.copy()
        new_cell[0,:] *= 1.1
        new_cell[1,:] *= 0.8
        new_cell[2,:] *= 1.03

        dyn.AdjustToNewCell(new_cell)

        # Generate the dynamical matrix in the supercell
        superdyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())

        # Now return back in q space
        dynq = CC.Phonons.GetDynQFromFCSupercell(superdyn.dynmats[0], np.array(dyn.q_tot), dyn.structure, superdyn.structure)

        __tollerance__ = 1e-6
        # Check if the dynq agrees with the correct one
        for iq in range(len(dyn.q_tot)):
            eps = np.max(np.abs(dyn.dynmats[iq] - dynq[iq, :, :]))
            self.assertTrue(eps < __tollerance__)

    def test_read_write_phonons(self):
        """
        We test if the writing and reading back a dynamical matrix 
        works correctly.

        We also add a raman tensor, effective charges and a random dielectric tensor
        """

        # Perform the check for all the dynamical matrices that may have some problems
        for i, dyn in enumerate([d.Copy() for d in [self.dynSky, self.dynSnSe]]):
            dyn = self.dynSky.Copy()

            # Create also the tensors
            dyn.dielectric_tensor = np.random.uniform(size = (3,3))
            dyn.effective_charges = np.random.uniform(size = (dyn.structure.N_atoms, 3, 3))
            dyn.raman_tensor = np.random.uniform(size = (3,3,3 * dyn.structure.N_atoms))


            nqirr = len(dyn.q_stars)
            root_name = "tmp_dyn_{}_".format(i)
            dyn.save_qe(root_name)

            # Check the number of saved point
            files = [f for f in os.listdir(".") if root_name in f]

            print("I read the following files: ", " ".join(files))

            for j in range(nqirr):
                dyn_name = "{}{}".format(root_name, j+1)
                
                self.assertTrue(dyn_name in files)

            # Try to reload the matrix
            new_dyn = CC.Phonons.Phonons(root_name, nqirr = nqirr)

            # Compare the two matrices
            __tol__ = 1e-7
            for iq, q in enumerate(dyn.q_tot):
                eps = np.max(np.abs(dyn.dynmats[iq] - new_dyn.dynmats[iq]))
                self.assertTrue(eps < __tol__)

                # Test the q point
                eps = np.abs(np.abs(q - new_dyn.q_tot[iq]))

            # Check the dielectric tensor, effective charges and raman tensor
            eps =  np.max(np.abs(dyn.dielectric_tensor - new_dyn.dielectric_tensor))
            self.assertTrue(eps < __tol__)
            eps =  np.max(np.abs(dyn.raman_tensor - new_dyn.raman_tensor))
            self.assertTrue(eps < __tol__)
            eps =  np.max(np.abs(dyn.effective_charges - new_dyn.effective_charges))
            self.assertTrue(eps < __tol__)
                


        





    def test_qe_get_symmetries(self):
        syms = CC.symmetries.QE_Symmetry(self.struct_ice)
        syms.SetupQPoint()

        self.assertEqual(syms.QE_nsymq, 4)

        # Test on the simple cubic
        qe_sym = CC.symmetries.QE_Symmetry(self.struct_simple_cubic)
        qe_sym.SetupQPoint()

        self.assertEqual(qe_sym.QE_nsymq, 48)

    def test_tensor_interpolation(self):
        """
        Test the interpolation using the Tensor
        module against the interpolation of the quantum espresso
        module.
        """

        dyn = self.dynSnSe.Copy()

        new_cell = (4,4,1)

        qe_inter_dyn = dyn.Interpolate(dyn.GetSupercell(), new_cell)

        # Interpolate using the Tensor library
        t2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
        t2.SetupFromPhonons(dyn)
        t2.Center()
        tensor_inter_dyn = t2.GeneratePhonons(new_cell)

        # Apply the symmetries to both
        tensor_inter_dyn.Symmetrize()
        qe_inter_dyn.Symmetrize()

        for iq, q in enumerate(tensor_inter_dyn.q_tot):
            dist = qe_inter_dyn.dynmats[iq] - tensor_inter_dyn.dynmats[iq]
            dist = np.max(np.abs(dist))

            #print("Testing at q = ", q, " | ", qe_inter_dyn.q_tot[iq])
            self.assertTrue(dist < 1e-6)
        


    def test_qgrid_interpolation(self):
        """
        We test if the interpolation and the generation of 
        a q grid is consistent.
        """

        dyn = self.dynSnSe.Copy()

        new_dyn = dyn.Interpolate(dyn.GetSupercell(), (4,4,1))
        superdyn = new_dyn.GenerateSupercellDyn(new_dyn.GetSupercell())

        # Now we return back
        dynq = CC.Phonons.GetDynQFromFCSupercell(superdyn.dynmats[0], np.array(new_dyn.q_tot), new_dyn.structure, superdyn.structure)

        for iq, q in enumerate(new_dyn.q_tot):
            zero = dynq[iq, :, :] - new_dyn.dynmats[iq]
            zero = np.max(np.abs(zero))
            self.assertTrue(zero < 1e-8)


    def test_stress_symmetries(self):
        syms = CC.symmetries.QE_Symmetry(self.struct_ice)
        syms.ChangeThreshold(1e-1)
        syms.SetupQPoint()

        random_matrix = np.zeros((3,3), dtype = np.float64)
        random_matrix[:,:] = np.random.uniform(size= (3,3))

        syms.ApplySymmetryToMatrix(random_matrix)

        __epsil__ = 1e-8

        # Test hermitianity
        self.assertTrue(np.sum(np.abs(random_matrix - random_matrix.T)) < __epsil__)
        
        voight_stress = np.zeros(6, dtype = np.float64)
        voight_stress[:3] = np.diag(random_matrix)
        voight_stress[3] = random_matrix[1,2]
        voight_stress[4] = random_matrix[0,2]
        voight_stress[5] = random_matrix[0,1]

        # Test orthorombic
        self.assertTrue(np.sum(np.abs(voight_stress[3:])) < __epsil__)

    def test_fourier(self):
        """
        Generate a random dynamical matrix, apply translations,
        and then generate do forward and backward the transformation
        """
        SUPERCELL = (4,4,1)

        # Use the unit cell of SnSe
        dyn = self.dynSnSe

        # Generate a super structure
        superstructure= dyn.structure.generate_supercell(SUPERCELL)
        natsc = superstructure.N_atoms

        # Here the random dynamical matrix
        fc_random = np.random.uniform(size = (3*natsc, 3*natsc))
        fc_random += fc_random.T

        # Apply the translations
        CC.symmetries.ApplyTranslationsToSupercell(fc_random, superstructure, SUPERCELL)

        # Get the expected q grid
        q_grid = CC.symmetries.GetQGrid(dyn.structure.unit_cell, SUPERCELL)

        # Go into the q space
        dynq = CC.Phonons.GetDynQFromFCSupercell(fc_random, np.array(q_grid),
                                                dyn.structure,
                                                superstructure)


        # Return back to the real space fc
        fc_new = CC.Phonons.GetSupercellFCFromDyn(dynq, np.array(q_grid),
                                                dyn.structure,
                                                superstructure)

        # Check the difference between before and after
        self.assertTrue(np.max(np.abs( fc_new - fc_random)) < 1e-10)


    @unittest.skip("Tensor Library still to be finished")
    def test_tensor_translational_invariance(self):
        """
        Test if the tensor class generates 
        a translational invariant tensor when interpolating
        """

        tensor = CC.ForceTensor.Tensor2(self.dynSnSe.structure)
        tensor.SetupFromPhonons(self.dynSnSe)

        # Interpolate to supercell (4,4,1)
        SUPERCELL = (4,4,1)
        force_fc = tensor.GenerateSupercellTensor(SUPERCELL)

        superstructure = self.dynSnSe.structure.generate_supercell(SUPERCELL)
        symmetrize_fc = force_fc.copy()

        # Impose the translations on the new supercell structure
        CC.symmetries.ApplyTranslationsToSupercell(symmetrize_fc, superstructure, SUPERCELL)
        
        self.assertTrue(np.max(np.abs(force_fc - symmetrize_fc)) < 1e-10)
        
    def test_q_star_with_minus_q(self):
        """
        This subroutine tests the interpolation in a supercell 
        where there is a q point so that
        q != -q + G

        In which there is no inversion symmetry (so -q is not in the star)
        Then we try to both recover the correct q star, symmetrize and
        check the symmetrization in the supercell
        """


        dyn = CC.Phonons.Phonons(self.struct_ice)
        
        # Get a random dynamical matrix
        fc_random = np.complex128(np.random.uniform(size = (3 * self.struct_ice.N_atoms, 3 * self.struct_ice.N_atoms)))
        fc_random += fc_random.T 
        
        
        dyn.dynmats = [fc_random]
        dyn.q_tot = [np.array([0,0,0])]
        dyn.q_stars = [[np.array([0,0,0])]]

        # Perform the symmetrization
        #dyn.Symmetrize()

        # Now perform the interpolation
        SUPERCELL = (1,1,3)
        new_dyn = dyn.Interpolate((1,1,1), SUPERCELL)
        new_dyn.Symmetrize()
        super_dyn = new_dyn.GenerateSupercellDyn(new_dyn.GetSupercell())
        fc1 = super_dyn.dynmats[0].copy()
        new_dyn.SymmetrizeSupercell()
        super_dyn = new_dyn.GenerateSupercellDyn(new_dyn.GetSupercell())
        fc2 = super_dyn.dynmats[0].copy()

        self.assertTrue( np.sqrt(np.sum( (fc1 - fc2)**2)) < 1e-6)

    @unittest.skip("Second order effective charges to be implemented")
    def test_second_order_effective_charges(self):
        """
        Here we test the effective charges second order tensor
        using the rocksalt toy model
        """

        n_config = 10000
        structures = self.rocksalt.dynmat.ExtractRandomStructures(n_config)
        eff_charges = [self.rocksalt.get_z(s) for s in structures]

        dM_drdr = CC.Manipulate.GetSecondOrderDipoleMoment(self.rocksalt.dynmat, structures, eff_charges, T = 0)

        # Do the numerical differences
        dx = 0.01
        dM_num = np.zeros((3*self.rocksalt.N_atoms, 3*self.rocksalt.N_atoms,3))

        for x1 in range(3*self.rocksalt.N_atoms):
            nat1 = int(x1 / 3)
            cart1 = x1 % 3

            struct = self.rocksalt.structure.copy()
            struct.coords[nat1, cart1] += dx

            z = self.rocksalt.get_z(struct)
            for i in range(3):
                dM_num[x1, :, i] = z[:, i, :].ravel()
        dM_num /= dx * CC.Units.A_TO_BOHR

        distance = np.sqrt(np.sum((dM_drdr - dM_num)**2))
        #print("DISTANCE:", distance)

        self.assertTrue(distance < 0.05)



    def test_phonons_supercell(self):
        # Build a supercell dynamical matrix
        supercell_size = (2,2,2)
        nat_sc = np.prod(supercell_size)
        fc_random = np.zeros((3*nat_sc, 3*nat_sc), dtype = np.complex128)
        fc_random[:,:] = np.random.uniform(size = (3*nat_sc, 3*nat_sc))
        CC.symmetries.CustomASR(fc_random)
        back = fc_random.copy()
        CC.symmetries.CustomASR(back)

        __epsil__ = 1e-8
        delta = np.sum( (fc_random - back)**2)
        # Acustic sum rule
        self.assertTrue(delta < __epsil__)

        # Get the irreducible q points
        qe_sym = CC.symmetries.QE_Symmetry(self.struct_simple_cubic)
        qe_sym.SetupQPoint()

        q_irr = qe_sym.GetQIrr(supercell_size)
        self.assertEqual(len(q_irr), 4)

        n_qirr = len(q_irr)
        q_tot = CC.symmetries.GetQGrid(self.struct_simple_cubic.unit_cell, supercell_size)
        
        dynq = CC.Phonons.GetDynQFromFCSupercell(fc_random, np.array(q_tot), self.struct_simple_cubic, self.struct_simple_cubic.generate_supercell(supercell_size))


        dyn = CC.Phonons.Phonons()
        for i in range(nat_sc):
            dyn.dynmats.append(dynq[i, :, :])
            dyn.q_tot.append(q_tot[i])
        

        dyn.structure = self.struct_simple_cubic
        dyn.AdjustQStar()

        dyn.Symmetrize()
        dyn.ForcePositiveDefinite()
        
        new_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
        w, p = new_dyn.DyagDinQ(0)

        dyn.Symmetrize()
        new_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())
        w_new, p = new_dyn.DyagDinQ(0)

        delta = np.sum( (w[3:] - w_new[3:])**2)
        self.assertTrue(delta < __epsil__)

    def test_polarization_vectors_supercell(self):
        # Download the dynamical matrix from internet
        dyn = DownloadDynSnSe()
        
        dyn.Symmetrize()

        # Get the dynamical matrix in the supercell
        super_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())

        # Get the polarization vectors in the supercell without generating the big matrix
        w, pols = dyn.DiagonalizeSupercell()

        # Get the masses
        m = np.tile(super_dyn.structure.get_masses_array(), (3, 1)).T.ravel()

        # Get the polarization vectors times the masses
        new_v = np.einsum("ab, a-> ab", pols, np.sqrt(m))

        # Generate the supercell dyn from the polarization vectors
        sup_dyn = np.einsum("i, ai, bi->ab", w**2, new_v, new_v)

        # Compare the two dynamical matrices
        __tollerance__ = 1e-6

        # Print the scalar product between the polarization vectors
        n_modes = len(w)
        s_mat = pols.T.dot(pols)
        delta_mat = s_mat - np.eye(n_modes)
        #np.savetxt("s_mat.dat", s_mat)
        self.assertTrue( np.sum(delta_mat**2) < __tollerance__)


        delta = np.sum(np.abs(super_dyn.dynmats[0] - sup_dyn)) 

        #print("DELTA:", delta)

        self.assertTrue( delta < __tollerance__)

    def test_dyn_realspace_and_back(self):
        """
        Test in which we take a particularly odd dynamical matrix
        and we generate the dynamical matrix in realspace, and then we 
        return back in q space. The two must match.
        """

        dyn = self.dynSky.Copy()
        superdyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())

        dynq = CC.Phonons.GetDynQFromFCSupercell(superdyn.dynmats[0], np.array(dyn.q_tot), \
            dyn.structure, superdyn.structure)
        
        for iq, q in enumerate(dyn.q_tot):
            delta = dynq[iq,:,:] - dyn.dynmats[iq]
            delta = np.sqrt(np.sum(np.abs(delta)**2))
            self.assertAlmostEqual(delta, 0)

    def test_upsilon_matrix(self):
        """
        In this test we compute the upsilon matrix (inverse phonon covariance) 
        in the supercell both by generating the supercell 
        and by computing upsilon using the DiagonalizeSupercell method.
        These two ways shold give the same result.
        """

        # Get the dynamical matrix
        dyn = self.dynSnSe.Copy()
        T = 300 # K

        # Generate the supercell dynamical matrix
        super_dyn = dyn.GenerateSupercellDyn(dyn.GetSupercell())

        # Compute Upsilon in both ways
        ups_from_supercell = super_dyn.GetUpsilonMatrix(T)
        ups_from_dyn = dyn.GetUpsilonMatrix(T)

        # Perform the comparison
        delta = np.sqrt(np.sum( (ups_from_supercell - ups_from_dyn)**2))
        self.assertAlmostEqual(delta, 0, delta = 1e-5)
        

    def test_symmetries_realspace_supercell(self):
        # Download from internet
        #NQ = 3
        dyn = self.dynSnSe.Copy()

        # Lets add a small random noise at gamma
        for iq, q in enumerate(dyn.q_tot):
            dyn.dynmats[iq][:,:] += np.random.uniform( size = np.shape(dyn.dynmats[0]))


        # Generate the dynamical matrix from the supercell
        dyn_supercell = dyn.GenerateSupercellDyn(dyn.GetSupercell())

        # Get the symmetries in the supercell using SPGLIB
        spg_sym = CC.symmetries.QE_Symmetry(dyn_supercell.structure)
        spg_sym.SetupFromSPGLIB()

        # Perform the symmetrization
        spg_sym.ApplySymmetriesToV2(dyn_supercell.dynmats[0])

        # Apply the custom sum rule
        dyn_supercell.ApplySumRule()

        # Convert back to the original dynamical matrix
        fcq = CC.Phonons.GetDynQFromFCSupercell(dyn_supercell.dynmats[0], np.array(dyn.q_tot), \
            dyn.structure, dyn_supercell.structure)

        # Create the symmetrized dynamical matrix
        new_dyn = dyn.Copy()
        for iq, q in enumerate(dyn.q_tot):
            new_dyn.dynmats[iq] = fcq[iq, :, :]
        
        # Symmetrize using the standard tools
        dyn.Symmetrize()

        threshold = 1e-3

        # Now compare the spectrum between the two matrices
        for iq,q in enumerate(dyn.q_tot):
            w_spglib, dumb = new_dyn.DyagDinQ(iq)
            w_standard, dumb = dyn.DyagDinQ(iq)

            w_spglib *= CC.Units.RY_TO_CM
            w_standard *= CC.Units.RY_TO_CM

            self.assertTrue(np.max(np.abs(w_spglib - w_standard)) < threshold)

        # Lets try to see if the spglib symmetrization correctly symemtrizes also a vector
        random_vector = np.zeros(np.shape(dyn.structure.coords), dtype = np.double)
        random_vector[:,:] = np.random.uniform( size = np.shape(random_vector))

        rv2 = random_vector.copy()

        # Get the symmetries in the unit cell using spglib
        qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
        qe_sym.SetupFromSPGLIB()
        qe_sym.SymmetrizeVector(random_vector)

        # Get the symmetries using quantum espresso
        qe_sym.SetupQPoint()
        qe_sym.SymmetrizeVector(rv2)

        # Check if rv2 is equal to random_vector
        self.assertTrue( np.sum( (random_vector - rv2)**2) < 1e-8)

    def test_multiple_spglib_symmetries(self):
        """
        This test tries to apply many times the symmetrization.
        If after two application the dynamical matrix changes, 
        there is a problem in the symmetrization.
        """
        dyn = DownloadDynSky()

        dyn.SymmetrizeSupercell()

        new_dyn = dyn.Copy()

        new_dyn.SymmetrizeSupercell()

        for iq, q in enumerate(dyn.q_tot):
            delta = dyn.dynmats[iq] - new_dyn.dynmats[iq]
            delta = np.sqrt(np.sum(np.abs(delta)**2))
            #print("Testing iq = {}, q = {}".format(i, iq))
            self.assertAlmostEqual(delta, 0)
        


    def test_spglib_phonon_symmetries(self):
        # Get the Sky dyn
        # This is a phonon matrix that gave a lot of problem
        # In the symmetrization in the supercell
        dyn = self.dynSky.Copy()

        # Symmetrize using quantum espresso
        dyn_qe = dyn.Copy()
        dyn_qe.Symmetrize()

        # Symmetrize using spblig
        dyn_spglib = dyn.Copy()
        dyn_spglib.SymmetrizeSupercell()
        #__thr__ = 1e-8

        #dyn_qe.save_qe("trial_qe")
        #dyn_spglib.save_qe("trial_spglib")

        # Compare
        for i, iq in enumerate(dyn.q_tot):
            delta = dyn_qe.dynmats[i] - dyn_spglib.dynmats[i]
            delta = np.sqrt(np.sum(np.abs(delta)**2))
            #print("Testing iq = {}, q = {}".format(i, iq))
            self.assertAlmostEqual(delta, 0)
        
        


if __name__ == "__main__":
    # Make everything reproducible
    np.random.seed(1)

    # Run all the tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStructureMethods)
    unittest.TextTestRunner(verbosity=2).run(suite)
