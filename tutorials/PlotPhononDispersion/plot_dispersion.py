import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ForceTensor

import ase, ase.dft.kpoints
import sys, os

import numpy as np
import matplotlib.pyplot as plt

# The dynamical matrix
PATH_TO_DYN="../QuasiHarmonicApproximation/V804/dynmat"
NQIRR = 8

# Load the dynamical matrix
dyn = CC.Phonons.Phonons(PATH_TO_DYN, NQIRR)

# Optionally you can display the BZ
# With the standard paths
# to choose the path
# Just uncomment the following three lines
"""  SHOW THE BRILLUIN ZONE
ase_atoms = dyn.structure.get_ase_atoms()
lattice = ase_atoms.cell.get_bravais_lattice()
lattice.plot_bz(show = True)
"""

# Select the path
PATH = "GYTAZG"
N_POINTS = 1000



# -------- HERE THE CORE SCRIPT ------------
band_path = ase.dft.kpoints.bandpath(PATH,
                                     dyn.structure.unit_cell,
                                     N_POINTS)

# Get the q points of the path
q_path = band_path.cartesian_kpts()

# Get the values of x axis for plotting the band path
x_axis, xticks, xlabels = band_path.get_linear_kpoint_axis()


# Now we need to Fourier interpolate the dynamical matrix
# In the selected path
# For this porupouse we create a Tensor2 object
t2 = CC.ForceTensor.Tensor2(dyn.structure,
                            dyn.structure.generate_supercell(dyn.GetSupercell()),
                            dyn.GetSupercell())
t2.SetupFromPhonons(dyn)

# We now center the tensor and apply the sum rule after centering
t2.Center(Far = 3)
t2.Apply_ASR()


# Now we need to perform the interpolation, dyagonalizing the dynamical matrix for each q point of the path
n_modes = 3 * dyn.structure.N_atoms
ws = np.zeros((N_POINTS, n_modes), dtype = np.double)
m = dyn.structure.get_masses_array()
m = np.tile(m, (3,1)).T.ravel()

for i in range(N_POINTS):
    # For each point in the path

    # Interpoalte the dynamical matrix
    fc = t2.Interpolate(-q_path[i, :])

    # Mass rescale the force constant matrix
    dynq = fc / np.outer(np.sqrt(m), np.sqrt(m))

    # Diagonalize the dynamical matrix
    w2 = np.linalg.eigvalsh(dynq)
    ws[i, :] = np.sqrt(np.abs(w2)) * np.sign(w2) * CC.Units.RY_TO_CM
    


# ============= PLOT THE FIGURE =================
plt.figure()

# Plot all the modes
for i in range(n_modes):
    plt.plot(x_axis, ws[:,i])

ax = plt.gca()
# Plot vertical lines for each high symmetry points
for x in xticks:
    ax.axvline(x, 0, 1, color = "k", lw = 0.4)

# Set the x labels to the high symmetry points
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)

ax.set_ylabel("Energy [cm-1]")
ax.set_xlabel("q path")

plt.tight_layout()
plt.show()
