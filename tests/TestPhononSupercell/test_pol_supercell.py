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

# Lets pick one q vector
q_vector = q_grid[2]
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
        e_mu /= np.sqrt(e_mu.dot(e_mu))
        projector += np.outer(e_mu, e_mu) # |e_mu><e_mu|


# Print the component of each vector inside the q space.
print("\n".join(["{:.4f} cm-1 has projection {:.4f}".format(w_sc[i] * CC.Phonons.RY_TO_CM,
                                                          pols_sc[:, i].dot(projector.dot(pols_sc[:,i])))
                 for i in range(3*nat_sc)]))
                                                        
