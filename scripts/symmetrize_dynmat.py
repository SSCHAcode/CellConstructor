#!python

from __future__ import print_function

__doc__ = """
This script perform the automatic symmetrization of the passed dynamical matrix
(It finds out by itself the number of q points, pass the root name)

USAGE:
>>> symmetrize_dynmat  <dynmat_root_name> [<threshold>]

where <dynmat_root_name> is the root of the dynmat. For example, lets take a 
dynamical matrix with 4 irreducible q points saved as:

dyn1 dyn2 dyn3 dyn4 

You will need to call the script as:
>>> symmetrize_dynmat dyn

The new dynamical matrix will be saved as:
s_dyn1 s_dyn2 s_dyn3 s_dyn4


The <threshold> optional argument is the threshold for symmetry identification in
the structure. This is used by the QE subroutine.
"""

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries

import sys, os
import numpy as np

if len(sys.argv) not in [2,3]:
    print (__doc__)
    raise ValueError("Error, this script needs 1 or 2 arguments.")


# Finds out how many q points are present
nqirr = 0
running = True
while running:
    if os.path.exists("%s%d" % (sys.argv[1], nqirr + 1)):
        nqirr += 1
    else:
        running = False

if nqirr == 0:
    raise ValueError("Error, no dynamical matrix found")

# Load the dynamical matrix
dyn = CC.Phonons.Phonons(sys.argv[1], nqirr)

fcq = np.array(dyn.dynmats)
qe_sym = CC.symmetries.QE_Symmetry(dyn.structure)
if len(sys.argv) == 3:
    thr = float(sys.argv[2])
    qe_sym.ChangeThreshold(thr)
    
qe_sym.SymmetrizeFCQ(fcq, dyn.q_stars, asr = "custom")
for iq, q in enumerate(dyn.q_tot):
    dyn.dynmats[iq] = fcq[iq, :, :]
    
dyn.save_qe("s_%s" % sys.argv[1])
