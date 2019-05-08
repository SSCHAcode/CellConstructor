from __future__ import print_function

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons

__ASE__ = True
try:
    import ase, ase.visualize
except:
    __ASE__ = False

"""
Get the closest neighbour of the 6-th atom (oxigen) along the z direction
"""

# Load the structure
structure = CC.Structure.Structure()
structure.read_scf("ice.scf")

# View the structure in ASE
if __ASE__:
    ase.visualize.view(structure.get_ase_atoms())

ATOM_ID = 6
search_direction = np.array([0,0,1])
print( "The closest atom to %d is:" % ATOM_ID)
print (CC.Methods.get_directed_nn(structure, ATOM_ID, search_direction))
