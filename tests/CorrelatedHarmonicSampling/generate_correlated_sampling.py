# -*- coding: utf-8 -*-

"""
This program takes in input the two dynamical matrices at two different volumes
and brings the population1 of atomic displacements generated according to dyn1
into a population2 of displacements generated according to population2
"""

import numpy as np
import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Manipulate
from ase.visualize import view

#How many configuration are stored in the directory population1?
N_POP = 10

# Read the displacements 
disps = []
for i in range(N_POP):
    disps.append(np.loadtxt("population1/disp_%d.dat" % (i+1)))
    
# Now read the two dynamical matrices
dyn1 = CC.Phonons.Phonons("dyn1", full_name = True)
dyn2 = CC.Phonons.Phonons("dyn2", full_name = True)

# Get the two upsilon matrices
ups1 = dyn1.GetUpsilonMatrix(0)
ups2 = dyn2.GetUpsilonMatrix(0)

# Show the ase of the two structure in the dynamical matrix
view(dyn1.structure.get_ase_atoms())
view(dyn2.structure.get_ase_atoms())

# Look at the scalar products
me = np.arange(36)
me[24] = 25
me[25] = 24
me[34] = 35
me[35] = 34
me[30] = 32
me[32] = 30
me[14] = 12
me[15] = 14
me[12] = 13
me[13] = 15
scalar_prod = CC.Manipulate.GetScalarProductPolVects(dyn1, dyn2, me)

print scalar_prod
print "Modes lower than EPS:"
print np.arange(36)[(scalar_prod < 6e-1) & (scalar_prod > -6e-1)]


# Now Perform the change of the structures
new_disps = CC.Manipulate.TransformStructure(dyn1, dyn2, 0, disps, me)

# Trivial tentative

# Print the transformed structure on the dyrectory population2
print "Check the probability of the two ensembles:"
for i in range(N_POP):
    np.savetxt("population2/disp_%d.dat" % (i+1), new_disps[i], fmt = "%20.8f")
    
    # Print the probability before and after the change
    print "%.8f -> %.8f" % (dyn1.GetProbability(disps[i], 0, ups1), 
                            dyn2.GetProbability(new_disps[i], 0, ups2))

print "Done."