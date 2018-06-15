# -*- coding: utf-8 -*-

import cellconstructor as CC
import numpy as np
from ase.visualize import view
import matplotlib.pyplot as plt
import cellconstructor.Structure

"""
This example will load a quantum espresso structure of phase C2c-24 of
high pressure hydrogen and will compute the distances between the H-H bounded
molecules.
"""
# The structure has a uniform scaling with alat equal to
alat = 2.838000235


# Load the hydrogen structure (rescaling both the cell)
HydIII = CC.Structure.Structure()
HydIII.read_scf("Hydrogen.scf", alat)

# Show the structure
view(HydIII.get_ase_atoms())

# Extract the hydrogen molecules
distance = 0.687
tollerance = 0.2

# Get all the hydrogen molecules
Mols, indices = HydIII.GetBiatomicMolecules(["H", "H"], distance, tollerance, True)

print "Found %s molecules:" % len(Mols)

# Now compute the H-H distances between the molecules
dist = []
for i, mol in enumerate(Mols):
    d = np.sqrt(np.sum((mol.coords[0,:] - mol.coords[1,:])**2))
    dist.append(d)

print "Here all the H-H distances:"
print dist

# Use matplotlib to plot a pretty hystogram
plt.title("H-H distance histogram")
plt.xlabel(r"dist [$\AA$]")
plt.ylabel(r"freq")
plt.hist(dist, 20)
plt.tight_layout()
plt.show()