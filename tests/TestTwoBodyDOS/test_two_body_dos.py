from __future__ import print_function

import cellconstructor as CC 
import cellconstructor.Phonons
import cellconstructor.Methods

from matplotlib.pyplot import *
from numpy import *
import scipy,scipy.signal

import sys, os

total_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(total_path)

"""
Compute the two body phonon DOS.
This is usefull to have an hint on the possible effect of a non zero scattering
on the phonon lifetimes.
"""

# Load the dynamical matrix only at gamma for this test
dyn = CC.Phonons.Phonons("dynmat", 1)
dyn.Symmetrize()

# Create the array of frequencies (in Ry)
w_array = linspace(0, 4000, 10000) / CC.Phonons.RY_TO_CM

# Get the two body DOS at gamma
Gamma = 3 / CC.Phonons.RY_TO_CM
DOS = dyn.get_two_phonon_dos(w_array, Gamma, 50, exclude_acoustic = True)


# Delete nan
nan_clean_mask = ~isnan(DOS)
DOS = DOS[nan_clean_mask]
w_array = w_array[nan_clean_mask]

# Plot the results
figure(dpi = 150)
title("Two body phonon-dos")
xlabel("Frequency [cm-1]")
ylabel("DOS")
plot(w_array * CC.Phonons.RY_TO_CM, DOS, label ="- Imag part")

# To check for consistency, we add a series of vertical lines to match the vibrational modes
# First we extract all the frequencies
w, pols = dyn.DyagDinQ(0)
# Then we remove the translations (w = 0)
trans = CC.Methods.get_translations(pols, dyn.structure.get_masses_array())
w = w[~trans] * CC.Phonons.RY_TO_CM

# Now we plot a vertical line for each mode (dashed black lines)
vlines(w, 0, max(DOS)*1.1, linestyles = "--", color = "k")


# KRAMERS-KRONIG of the dos (the - in the dos is to obtain back the original sign of the dos)
dos_cmplz = 1j*scipy.signal.hilbert(DOS - min(DOS))
plot(w_array * CC.Phonons.RY_TO_CM, real(dos_cmplz), label = "Real part")
legend()
tight_layout()
#show()

