#!python

from __future__ import print_function
from __future__ import division

import unittest
import urllib2
import numpy as np
import sys, os

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons

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


        # Get a simple cubic structure
        self.struct_simple_cubic = CC.Structure.Structure(1)
        self.struct_simple_cubic.atoms = ["H"]
        self.struct_simple_cubic.has_unit_cell = True
        self.struct_simple_cubic.unit_cell = np.eye(3)
        self.struct_simple_cubic.masses = {"H": 938}


        
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

    def test_qe_get_symmetries(self):
        syms = CC.symmetries.QE_Symmetry(self.struct_ice)
        syms.SetupQPoint()

        self.assertEqual(syms.QE_nsymq, 4)

        # Test on the simple cubic
        qe_sym = CC.symmetries.QE_Symmetry(self.struct_simple_cubic)
        qe_sym.SetupQPoint()

        self.assertEqual(qe_sym.QE_nsymq, 48)

        

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


    def test_symmetries_realspace_supercell(self):
        # Download from internet
        NQ = 3
        for i in range(1,NQ +1):
            # Download from the web the dynamical matrices
            dynfile = urllib2.urlopen("https://raw.githubusercontent.com/mesonepigreco/CellConstructor/master/tests/TestSymmetriesSupercell/SnSe.dyn.2x2x2%d" % i)
            with open("dyn.SnSe.%d" % i,'wb') as output:
                output.write(dynfile.read())

        # Load the dynamical matrices
        dyn = CC.Phonons.Phonons("dyn.SnSe.", NQ)

        # Lets remove the downloaded file
        for i in range(1,NQ +1):
            os.remove("dyn.SnSe.%d" % i)

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








        

        



# Run all the tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestStructureMethods)
unittest.TextTestRunner(verbosity=2).run(suite)
