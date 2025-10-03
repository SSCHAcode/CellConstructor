from __future__ import print_function
from __future__ import division

__DOC__ = r"""
In this example we try to separate the modes that are at different q vectors.

To do so, we first import the dynamical matrix in the supercell 3x3x2 of ice XI,
then we generate the vectors in the supercell in a given q point (commensurate),
then we generate the supercell in real space of the dynamical matrix, and select
the modes that belongs to the chose mode. 
"""
import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons
import sys, os

total_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(total_path)

# Print the doc in the stdout
print(__DOC__)

# Load the ice XI dynamical matrix
dynq = CC.Phonons.Phonons("dynmat", 8)
dynq.Symmetrize()

# Generate the dynamical matrix in real space
dyn_realspace = dynq.GenerateSupercellDyn( dynq.GetSupercell() )

# Get frequencies and polarization vectors in the supercell
w_sc, pols_sc = dyn_realspace.DyagDinQ(0)
nat = dynq.structure.N_atoms
nat_sc = dyn_realspace.structure.N_atoms

# Lets get all the possible q points for the given supercell.
q_grid = CC.symmetries.GetQGrid(dynq.structure.unit_cell, dynq.GetSupercell())

# Separate the different q points
pols_sc = CC.symmetries.AdjustSupercellPolarizationVectors(w_sc, pols_sc, q_grid, dyn_realspace.structure, nat)

# Test if we can rebuild the dynamical matrix properly
m = np.tile(dyn_realspace.structure.get_masses_array(), (3,1)).T.ravel()
fc_matrix = np.einsum("a, ba, ca, b, c -> bc", w_sc**2, pols_sc, pols_sc, np.sqrt(m), np.sqrt(m))
dyn_realspace.dynmats[0] = fc_matrix
dyn_realspace.save_qe("RealspaceAdjusted")


# Lets pick one q vector
q_vector = q_grid[4]
print("The selected q vector is: ", q_vector)

# Lets generate the basis of the modes along this vector
projector = np.zeros( (3*nat_sc, 3*nat_sc), dtype = np.float64)
for i in range(nat):
    for j in range(3):
        e_mu = np.zeros( 3*nat_sc, dtype = np.float64)

        for k in range(np.prod(dynq.GetSupercell())):
            atm_index = nat*k + i
            delta_r = dyn_realspace.structure.coords[atm_index, :] - dyn_realspace.structure.coords[i, :]
            e_mu[3*atm_index + j] = np.cos(q_vector.dot(delta_r)*2*np.pi)

        # Normalize
        e_mu /= np.sqrt(np.conj(e_mu).dot(e_mu))
        projector += np.outer(e_mu, np.conj(e_mu)) # |e_mu><e_mu|


# Print the component of each vector inside the q space.
print("\n".join(["{:4d}){:16.4f} cm-1 has projection {:10.4f}".format(i, w_sc[i] * CC.Phonons.RY_TO_CM,
                                                                      np.real(np.conj(pols_sc[:, i]).dot(projector.dot(pols_sc[:,i]))))
                 for i in range(3*nat_sc)]))


# Identify the symmetry in the polarization basis
try:
    import spglib
except:
    print("Please, install spglib if you want to run the test on the symmetries.")
    exit(0)

spglib_sym = spglib.get_symmetry(dyn_realspace.structure.get_spglib_cell())
symmetries = CC.symmetries.GetSymmetriesFromSPGLIB(spglib_sym, False)


# Get the symmetries in the new polarization basis
pol_symmetries = CC.symmetries.GetSymmetriesOnModes(symmetries, dyn_realspace.structure, pols_sc)

# Exclude translations
pol_symmetries = pol_symmetries[:, 3:, 3:]
w_sc = w_sc[3:]

# Build a mask that can extract degenerate modes
deg_mask = np.abs(w_sc[1:] - w_sc[:-1]) < 1e-8
deg_mask1 = np.concatenate(([False], deg_mask))
deg_mask2 = np.concatenate((deg_mask, [False]))
deg_realmask = deg_mask1 | deg_mask2

Nsym, dumb1, dumb2 = np.shape(pol_symmetries)
for i in range(Nsym):
    print("Symmetry {}:".format(i+1))
    print(pol_symmetries[i, deg_realmask, deg_realmask])
