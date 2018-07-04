# -*- coding: utf-8 -*-

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.Methods

try:
    import spglib
except:
    raise ValueError("Error, to run this example you need to install spglib")
    
    
"""
This code load a dynamical matrix with the structure that barely 
satisfy a C2/c monoclinic group (with a 0.04 threshold) and
constrain the symmetries to allow programs like quantum espresso
to detect symmetries correctly.


NOTE: To recognize symmetries this example uses spglib.
"""

# initialize the dynamical matrix
dyn = CC.Phonons.Phonons("old_dyn", full_name=True)

# Print the symmetry group at high threshold
GROUP = spglib.get_spacegroup(dyn.structure.get_ase_atoms(), 0.04)
print "Space group with high threshold:", spglib.get_spacegroup(dyn.structure.get_ase_atoms())
print "Space group with low threshold:", GROUP

# Get the symmetries from the new spacegroup
symmetries = spglib.get_symmetry(dyn.structure.get_ase_atoms(), 0.04)

# Transform the spglib symmetries into the CellConstructor data type
sym_mats = CC.Methods.GetSymmetriesFromSPGLIB(symmetries, True)
print [t[:,3] for t in sym_mats]
# Force the symmetrization
dyn.structure.impose_symmetries(sym_mats)

# Check once again the symetry
print "New space group with high threshold:", spglib.get_spacegroup(dyn.structure.get_ase_atoms())
