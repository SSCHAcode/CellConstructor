from __future__ import print_function
from __future__ import division


import numpy as np

__INFO__ = """
Here all the conversion between common units are stored.

The default units are Angstrom, eV, and the atomic unit mass (1/12 of C12 nucleus)

However, phonons are saved in Ry/bohr^2 (to mantain the quantum-espresso compatibility)
therefore some unsefull conversion with Ry and Ha atomic units are provided.
"""

A_TO_BOHR = np.float64(1.889725989)
BOHR_TO_ANGSTROM = 1 / A_TO_BOHR 
RY_TO_CM = np.float64(109737.36034769034314)#109691.40235
ELECTRON_MASS_UMA = np.float64(1822.8885468045)
MASS_RY_TO_UMA = 2 / ELECTRON_MASS_UMA
HBAR = np.float64(0.06465415105180661)
K_B = np.float64(8.617330337217213e-05)
RY_TO_EV = np.float64(13.605693012183622)
HA_TO_EV = 2 * RY_TO_EV
RY_PER_BOHR3_TO_GPA = 14710.513242194795
RY_PER_BOHR3_TO_EV_PER_A3 = 91.81576765360094
RY_TO_KELVIN = RY_TO_EV / K_B
GPA_TO_EV_PER_A3 = RY_PER_BOHR3_TO_EV_PER_A3 / RY_PER_BOHR3_TO_GPA
