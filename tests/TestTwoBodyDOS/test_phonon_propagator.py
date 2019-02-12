import cellconstructor as CC 
import cellconstructor.Phonons
import cellconstructor.Methods

from matplotlib.pyplot import *
from numpy import *
import numpy as np
import scipy,scipy.signal

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
T = 50
DOS = dyn.get_two_phonon_dos(w_array, Gamma, T, exclude_acustic = True)


# Delete nan
nan_clean_mask = ~isnan(DOS)
DOS = DOS[nan_clean_mask]
w_array = w_array[nan_clean_mask]

# Get the phonon propagator
w_prop = linspace(0, max(w_array), 1000)
chi_prop = np.zeros(shape(w_prop), dtype = np.complex128)
for i, w in enumerate(w_prop):
    chi_prop[i] = np.sum(dyn.get_phonon_propagator(w, T, np.zeros(0,0,0), np.zeros(0,0,0), Gamma))
    

    
# Plot the results
figure(dpi = 150)
title("Two body phonon-dos")
xlabel("Frequency [cm-1]")
ylabel("DOS")
plot(w_array * CC.Phonons.RY_TO_CM, DOS, label ="- Imag part")
plot(w_prop * CC.Phonons.RY_TO_CM, imag(chi_prop), label = "Imag form prop")

# To check for consistency, we add a series of vertical lines to match the vibrational modes
# First we extract all the frequencies
w, pols = dyn.DyagDinQ(0)
# Then we remove the translations (w = 0)
trans = CC.Methods.get_translations(pols, dyn.structure.get_masses_array())
w = w[~trans] * CC.Phonons.RY_TO_CM
print w
# Now we plot a vertical line for each mode (dashed black lines)
vlines(w, 0, max(DOS)*1.1, linestyles = "--", color = "k")


# KRAMERS-KRONIG of the dos (the - in the dos is to obtain back the original sign of the dos)
dos_cmplz = 1j*scipy.signal.hilbert(DOS - min(DOS))
plot(w_array * CC.Phonons.RY_TO_CM, real(dos_cmplz), label = "Real part")
legend()
tight_layout()
show()

