# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import spglib
import cellconstructor as CC
import cellconstructor.Structure


import sys, os

import pytest

def test_supercell_replica():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)


    # Load the structure
    struct = CC.Structure.Structure()
    struct.read_scf("unit_cell_structure.scf")

    # Generate a supercell
    super_struct = struct.generate_supercell((2,2,1))
    print ("Space group before:")
    print (spglib.get_spacegroup(super_struct.get_spglib_cell()),)
    print (len(spglib.get_symmetry(super_struct.get_spglib_cell())["translations"]))

    # Get the symmetries in the supercell using spglib
    spglib_syms = spglib.get_symmetry(super_struct.get_spglib_cell())
    syms = CC.symmetries.GetSymmetriesFromSPGLIB(spglib_syms, False)
    nsyms = len(syms)

    # Generate a random distorted super structure
    d_structure = super_struct.copy()
    d_structure.coords += np.random.normal(scale = 0.1, size=np.shape(d_structure.coords))


    # Get the new pool of structures
    new_d_structures = []
    for i in range(nsyms):
        # Get irt
        irt = CC.symmetries.GetIRT(super_struct, syms[i])
        #print "Symmetry ", i
        #print len(set(irt))

        u_disp = d_structure.coords - super_struct.coords
        new_u_disp = CC.symmetries.ApplySymmetryToVector(syms[i], u_disp, super_struct.unit_cell, irt[:])
        tmp = super_struct.copy()
        tmp.coords += new_u_disp
        tmp.save_scf("replica_%d.scf" % i)
        new_d_structures.append(tmp)


    # Average all the displacements to see if the symmetries are recovered correctly
    new_structure = super_struct.copy()
    new_structure.coords = np.sum([x.coords for x in new_d_structures], axis = 0) / nsyms

    # Get again the symmetries
    print ("Symmetries after the sum:")
    print (spglib.get_spacegroup(new_structure.get_spglib_cell()), )
    print (len(spglib.get_symmetry(new_structure.get_spglib_cell())["translations"]))

    # Lets check if the structure is the same as before 
    # Should be 0 only if the symmeties are enaugh to have 0 force.
    print ("Difference from the first one:")
    print (np.sqrt(np.sum((new_structure.coords - super_struct.coords)**2)))

if __name__ == "__main__":
    test_supercell_replica()
    
