# -*- coding: utf-8 -*-

import cellconstructor as CC
import cellconstructor.Structure
import cellconstructor.Phonons

"""
This script check the symmetries of a dynamical matrix.
"""


# Read the dynamical matrix
PH = CC.Phonons.Phonons("hydrogen_dyn", nqirr = 1)

# Get info about the symmetries of the structure
