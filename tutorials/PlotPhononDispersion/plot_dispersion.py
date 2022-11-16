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


# Perform the interpolation
frequencies = CC.ForceTensor.get_phonons_in_qpath(dyn, q_path)

# ============= PLOT THE FIGURE =================
fig = plt.figure(dpi = 200)
ax = plt.gca()

# Plot all the modes
for i in range(frequencies.shape[-1]):
    ax.plot(x_axis, frequencies[:,i])

# Plot vertical lines for each high symmetry points
for x in xticks:
    ax.axvline(x, 0, 1, color = "k", lw = 0.4)

# Set the x labels to the high symmetry points
ax.set_xticks(xticks)
ax.set_xticklabels(xlabels)

ax.set_ylabel("Energy [cm-1]")
ax.set_xlabel("q path")

fig.tight_layout()
fig.savefig("dispersion.png")
fig.savefig("dispersion.eps")
plt.show()
