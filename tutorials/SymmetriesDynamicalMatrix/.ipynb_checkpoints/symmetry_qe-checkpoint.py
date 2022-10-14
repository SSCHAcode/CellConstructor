from __future__ import print_function

# Import numpy
import numpy as np

# Import cellconstructor
import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.symmetries

# Define a rocksalt structure
bcc = CC.Structure.Structure(2)
bcc.coords[1,:] = 5.6402 * np.array([.5, .5, .5]) # Shift the second atom in the center
bcc.atoms = ["Na", "Cl"]
bcc.unit_cell = np.eye(3) * 5.6402 # A cubic cell of 5.64 A edge
bcc.has_unit_cell = True # Setup periodic boundary conditions

# Setup the mass on the two atoms (Ry units)
bcc.masses = {"Na": 20953.89349715178,
              "Cl": 302313.43272048925}



# Lets generate the random dynamical matrix
dynamical_matrix = CC.Phonons.Phonons(bcc)
dynamical_matrix.dynmats[0] = np.random.uniform(size = (3 * bcc.N_atoms,
                                                        3* bcc.N_atoms))

# Force the random matrix to be hermitian (so we can diagonalize it)
dynamical_matrix.dynmats[0] += dynamical_matrix.dynmats[0].T
                               
# Lets compute the phonon frequencies without symmetries
w, pols = dynamical_matrix.DiagonalizeSupercell()

# Print on the screen the random frequencies
print("Non symmetric frequencies:")
print("\n".join(["{:d}) {:.4f} cm-1".format(i, w * CC.Units.RY_TO_CM)for i,w in enumerate(w)]))

# Symmetrize the dynamical matrix
dynamical_matrix.Symmetrize() # Use QE to symmetrize

# Recompute the frequencies and print them in output
w, pols = dynamical_matrix.DiagonalizeSupercell()
print()
print("frequencies after the symmetrization:")
print("\n".join(["{:d}) {:.4f} cm-1".format(i, w * CC.Units.RY_TO_CM) for i,w in enumerate(w)]))





