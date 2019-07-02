# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.Manipulate
from ase.visualize import view
import cellconstructor.symmetries

import numpy as np
import matplotlib.pyplot as plt
import spglib
import time

"""
This script check the symmetries of a dynamical matrix.
In the end the symmetry is constrained.
"""
RyToCm = 109691.40235

# Read the dynamical matrix
PH = CC.Phonons.Phonons("hydrogen_dyn", nqirr = 1)

print ("Loaded hydrogen_dyn1")
print ("Symmetry group:", spglib.get_spacegroup(PH.structure.get_ase_atoms(), 0.01))


# Get info about the symmetries of the structure
symmetries = spglib.get_symmetry(PH.structure.get_ase_atoms(), 0.01)
print ("Number of symmetries:", len(symmetries["rotations"]))
 
# Convert the spglib symmetries into the cellconstructor format
sym_mats = CC.symmetries.GetSymmetriesFromSPGLIB(symmetries)

# Impose the symmetries on the structure
PH.structure.fix_coords_in_unit_cell()
PH.structure.impose_symmetries(sym_mats)

view(PH.structure.get_ase_atoms())
 

# Get frequencies of the original matrix
w, pols = PH.DyagDinQ(0)
PH_new = PH.Copy()

# Force the symmetrization
#PH_new.SymmetrizeSupercell((1,1,1))
qe_sym = CC.symmetries.QE_Symmetry(PH.structure)
qe_sym.SetupQPoint(verbose = True)
#qe_sym.SymmetrizeDynQ(PH_new.dynmats[0], np.array([0,0,0]))
qe_sym.ApplySymmetriesToV2(PH_new.dynmats[0])
#CC.symmetries.CustomASR(PH_new.dynmats[0])

new_w, new_pols = PH_new.DyagDinQ(0)

# Symmetrize using the quantum espresso
PH.Symmetrize()
w_qe, p_qe = PH.DyagDinQ(0)

print ("Python Symmetries | QE Symmetries | Old Matrix")
print ("\n".join(["%12.2f\t%12.2f\t%12.2f  cm-1" % (new_w[k]*RyToCm, w[k] * RyToCm, w_qe[k] * RyToCm) for k in range(0, len(w))]))
        
        
