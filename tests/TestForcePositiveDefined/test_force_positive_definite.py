# -*- coding: utf-8 -*-

"""
In this example a non positive defined dynamical matrix is
processed and then transformed in a positive defined matrix.
This is very usefull to start a SSCHA calculation starting from
instable or interpolated matrices.
"""

import cellconstructor as CC
import cellconstructor.Phonons

# Ry to Cm-1 conversion 
RyToCm = 109691.40235

dyns = CC.Phonons.Phonons("sscha_2x1x2_", nqirr = 3)

# Print the current frequencies
countq = 0
for iq in range(dyns.nqirr):
    # Print the qpoint and frequencies
    print "q = ", " ".join([str(x) for x in list(dyns.q_stars[iq][0])])
    
    print "Frequencies [cm-1]:"
    w, pols = dyns.DyagDinQ(countq)
    
    print w * RyToCm    
    countq += len(dyns.q_stars[iq])

# Constrain the matrix to be positive defined
dyns.ForcePositiveDefinite()


# Print the new frequencies
countq = 0
for iq in range(dyns.nqirr):
    # Print the qpoint and frequencies
    print "q = ", " ".join([str(x) for x in list(dyns.q_stars[iq][0])])
    
    print "Frequencies [cm-1]:"
    w, pols = dyns.DyagDinQ(countq)
    
    print w * RyToCm    
    countq += len(dyns.q_stars[iq])


# Save the new matrices
dyns.save_qe("new_sscha_2x1x2_")