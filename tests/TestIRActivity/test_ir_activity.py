from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries

INFO = """
Test the IR activity of common ice. 
We use a simple program to test what are the IR active modes of ice XI.
"""

print(INFO)

# Load the dynamical matrix
dyn = CC.Phonons.Phonons("dynmat")

# Symmetrize and apply the acoustic sum rule
dyn.Symmetrize()

# Get the frequencies and the polarization vectors
ws, pols = dyn.DyagDinQ(0)

# Extract the symmetries (using the build-in quantum-espresso module)
qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
qe_sym.SetupQPoint()
symmetries = qe_sym.GetSymmetries()

# Analyze the polarization vectors to look for IR active modes:
ir_active_modes = CC.symmetries.GetIRActiveModes(symmetries,
                                                 dyn.structure,
                                                 pols)

# Print the mode frequency
print()
for i, w in enumerate(ws * CC.Phonons.RY_TO_CM):
    print("{:4d}) {:16.8f} cm-1 | IR active? {}".format(i, w ,
                                                        ir_active_modes[i]))
print("Done.")
                                                        
    
