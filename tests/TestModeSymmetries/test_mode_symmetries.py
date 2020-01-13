from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Phonons

import sys, os

total_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(total_path)

# Load the dynamical matrix
dyn = CC.Phonons.Phonons("../TestPhononSupercell/dynmat")

# Apply the symmetries
dyn.Symmetrize()

# Load the symmetries from the structure
qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
qe_sym.SetupQPoint(verbose = True)
symmetries = qe_sym.GetSymmetries()

# Get frequencies and polarization vectors
w, pols = dyn.DyagDinQ(0)

# Get the symmetry matrix in the polarization space
sim_modes = CC.symmetries.GetSymmetriesOnModes(symmetries, dyn.structure, pols)
#sim_modes2 = CC.symmetries.GetSymmetriesOnModesFast(symmetries, dyn.structure, pols)
for i in range(len(symmetries)):
    print("Symmetry:")
    print(symmetries[i])
    print("Interaction matrix:")
    print("\n".join(["{:16.4f} cm-1  | diag_value = {:10.4f}".format(w[j] * CC.Phonons.RY_TO_CM, sim_modes[i,j,j]) for j in range(len(w))]))
