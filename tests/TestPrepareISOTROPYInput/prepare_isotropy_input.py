# -*- coding: utf-8 -*-

"""
Here we test how to generate an input file for the findsym program from the ISOTROPY
suite. This can be used as an alternative tool to find the symmetry group of a 
given structure
"""

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.symmetries


# Load the trial structure
structure = CC.Structure.Structure()
structure.read_scf("trial_structure.scf")

# Prepare the ISOTROPY input file
CC.symmetries.PrepareISOTROPYFindSymInput(structure, "isotropy_findsym.in")

print "File prepared in 'isotropy_findsym.in'"