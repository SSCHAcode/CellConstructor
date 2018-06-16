#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:10:21 2017

@author: darth-vader
"""
import numpy as np
import os


CURRENT_PATH = os.path.realpath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_PATH)

def get_symmetries_from_ita(ita, red=False):
    """
    This function returns a matrix containing the symmetries from the given ITA code of the Group.
    The corresponding ITA/group label can be found on the Bilbao Crystallographic Server.
    
    Parameters
    ----------
        - ita : int
             The ITA code that identifies the group symmetry.
        - red : bool (default = False)
            If red is True then load the symmetries only in the smallest unit cell (orthorombic)
    Results
    -------
        - symmetries : list
            A list of 3 rows x 4 columns matrices (ndarray), containing the symmetry operations 
            of the chosen group.
    """
    
    if ita <= 0:
        raise ValueError("Error, ITA group %d is not valid." % ita)
      
    filename="%s/SymData/%d.dat" % (CURRENT_DIR, ita)
    if red:
        filename="%s/SymData/%d_red.dat" % (CURRENT_DIR, ita)

    
    if not os.path.exists(filename):
        print "Error, ITA group not yet implemented."
        print "You can download the symmetries for this group from the Bilbao Crystallographic Server"
        print "And just add the %d.dat file into the SymData folder of the current program." % ita
        print "It should take less than five minutes."
        
        raise ValueError("Error, ITA group  %d not yet implemented. Check stdout on how to solve this problem." % ita)
    
    fp = open(filename, "r")
    
    # Get the number of symemtries
    n_sym = int(fp.readline().strip())
    fp.close()
    
    symdata = np.loadtxt(filename, skiprows = 1)
    symmetries = []

    for i in range(n_sym):
        symmetries.append(symdata[3*i:3*(i+1), :])
    
    return symmetries
