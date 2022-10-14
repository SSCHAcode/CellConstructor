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

# Initialize the symmetry class
syms = CC.symmetries.QE_Symmetry(bcc)
syms.SetupFromSPGLIB() # Setup the espresso symmetries on spglib

# Generate the real space dynamical matrix
superdyn = dynamical_matrix.GenerateSupercellDyn(dynamical_matrix.GetSupercell())
# Apply the symmetries to the real space matrix
CC.symmetries.CustomASR(superdyn.dynmats[0])
syms.ApplySymmetriesToV2(superdyn.dynmats[0])
# Get back the dyanmical matrix in q space
dynq = CC.Phonons.GetDynQFromFCSupercell(superdyn.dynmats[0],
                                         np.array(dynamical_matrix.q_tot),
                                         dynamical_matrix.structure,
                                         superdyn.structure)

# Copy each q point of the symmetrized dynamical matrix into
# the original one
for i in range(len(dynamical_matrix.q_tot)):
    dynamical_matrix.dynmats[i] = dynq[i,:,:]


# Recompute the frequencies and print them in output
w, pols = dynamical_matrix.DiagonalizeSupercell()
print()
print("frequencies after the symmetrization:")
print("\n".join(["{:d}) {:.4f} cm-1".format(i, w * CC.Units.RY_TO_CM) for i,w in enumerate(w)]))





