# -*- coding: utf-8 -*-
from __future__ import print_function
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
print ("Plotting the structures")
view(dyn1.structure.get_ase_atoms())
view(dyn2.structure.get_ase_atoms())
print ("")

# Check the scalar product between two polarization vectors.
scalar_prod = CC.Manipulate.GetScalarProductPolVects(dyn1, dyn2)
print ("Scalar product between modes before ordering:")
print (scalar_prod)
print ("Modes that do not match:")
print (np.arange(36)[(scalar_prod < 6e-1) & (scalar_prod > -6e-1)])

# Order correctly the modes to match the polarization vectors and direction.
me, ms = CC.Manipulate.ChooseParamForTransformStructure(dyn1, dyn2)

# Check the scalar product between two polarization vectors.
scalar_prod = CC.Manipulate.GetScalarProductPolVects(dyn1, dyn2, me, ms)
print ("")
print ("Scalar product between modes after ordering:")
print (scalar_prod)
print ("Modes that do not match:")
print (np.arange(36)[(scalar_prod < 6e-1) & (scalar_prod > -6e-1)])

print ("See how the ordering is necessary to avoid mis-alignment between the modes of the two dyns")
print ("They are necessary to have all the modes orthogonal each other.")

# Now Perform the change of the structures
new_disps = CC.Manipulate.TransformStructure(dyn1, dyn2, 0, disps, me, ms)

# Trivial tentative

# Print the transformed structure on the dyrectory population2
print ("Saving the new structure into population2")
for i in range(N_POP):
    np.savetxt("population2/disp_%d.dat" % (i+1), new_disps[i], fmt = "%20.8f")
    
    # Save also the scf file
    struct = dyn2.structure.copy()
    struct.coords += new_disps[i]
    
    # Avoid saving the unit cell
    # NOTE THE ATOMS ARE IN ANSGROM
    # TO USE A COSTUM alat, specify also the alat unit (in angstrom)
    struct.save_scf("population2/disp_%d.scf" % (i+1), alat = 1, avoid_header = True)

print ("Done.")
