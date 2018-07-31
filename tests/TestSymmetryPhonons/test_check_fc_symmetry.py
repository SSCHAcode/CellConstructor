# -*- coding: utf-8 -*-

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons
import cellconstructor.Manipulate
from ase.visualize import view

import numpy as np
import spglib

"""
This script check the symmetries of a dynamical matrix.
"""
RyToCm = 109691.40235

# Read the dynamical matrix
PH = CC.Phonons.Phonons("hydrogen_dyn", nqirr = 1)

# Get info about the symmetries of the structure
symmetries = spglib.get_symmetry(PH.structure.get_ase_atoms(), 0.01)

# Convert the spglib symmetries into the cellconstructor format
sym_mats = CC.Methods.GetSymmetriesFromSPGLIB(symmetries)

# Impose the symmetries on the structure
PH.structure.fix_coords_in_unit_cell()
PH.structure.impose_symmetries(sym_mats)

view(PH.structure.get_ase_atoms())



# Get frequencies of the original matrix
w, pols = PH.DyagDinQ(0)

PH_new = PH.Copy()

ofreqs, caiser2 = np.linalg.eig(PH.dynmats[0])

# For each symmetry, get the new force constant matrix
for i, sym_mat in enumerate(sym_mats):
    # Apply the symmetry
    new_fc = PH.ApplySymmetry(sym_mat)
    
    
    # Build a new Phonon matrix
    PH_new.dynmats[0] = new_fc.copy()
    #PH_new.ApplySumRule()
    
    # Get the frequencies of the new matrix
    new_w, new_pols = PH_new.DyagDinQ(0)
    freqs, caiser = np.linalg.eig(new_fc)
    np.sort(freqs)
    
    PH_new.save_qe("symm_%d" % (i+1), full_name=True)
    
    # Print the distance between the two matrices
    distance = np.sum( (PH.dynmats[0] - PH_new.dynmats[0])**2)
    distance = np.real(np.sqrt(distance))
    
    # Print the data on the symmetries
    print "Iteration %d." % (i+1)
    print "Symmetry:"
    print sym_mat
    
    print ""
    print "Frequencies of the new symmetry | old frequencies"
    print "\n".join(["%12.2f\t%12.2f   cm-1" % (new_w[k]*RyToCm, w[k] * RyToCm) for k in range(0, len(w))])
    
    
    #print "Check corrispondance between eigenvalues and eigenvectors:"
    #me, ms = CC.Manipulate.ChooseParamForTransformStructure(PH, PH_new, n_max = 100)
    #print " ".join(["%d" % x for x in me])
    #print " ".join(["%.2e" % x for x in ms])
    
    #print "Other eigens:"
    #print "\n".join(["%12.2f\t%12.2f   XX" % (np.sort(freqs)[k]*RyToCm, np.sort(ofreqs)[k] * RyToCm) for k in range(0, len(w))])

    print ""
    print " Distance between the two matrices:", distance #/ PH.structure.N_atoms ** 2
    print ""
    print ""
    print " ------------------------------------------------- "
    print ""

print "Done"

# Force the symmetrization
PH.ForceSymmetries(sym_mats)

new_w, new_pols = PH.DyagDinQ(0)
print "Frequencies of the new symmetry | old frequencies"
print "\n".join(["%12.2f\t%12.2f   cm-1" % (new_w[k]*RyToCm, w[k] * RyToCm) for k in range(0, len(w))])