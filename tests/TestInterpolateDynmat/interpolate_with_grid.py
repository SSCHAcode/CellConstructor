# -*- coding: utf-8 -*-

"""
Here we interpolate a dynamical matrix in a finer grid.
"""

import cellconstructor as CC
import cellconstructor.Phonons

# Load the dynamical matrix
dyn = CC.Phonons.Phonons("dyn_prova")

harm_coarse = CC.Phonons.Phonons("harm_support")
harm_fine = CC.Phonons.Phonons("harm_support", 2)

# Interpolate on a 3x3x2 supercell
new_dyn = dyn.Interpolate((1,1,1), (2,2,1), harm_coarse, harm_fine, symmetrize=False)

symqe = CC.symmetries.QE_Symmetry(new_dyn.structure)
symqe.SetupQPoint(verbose=True)


print new_dyn.q_stars
q_stars, q_index = symqe.SetupQStar(new_dyn.q_tot)
# Save the new dynamical matrix
new_dyn.save_qe("new_dyn")