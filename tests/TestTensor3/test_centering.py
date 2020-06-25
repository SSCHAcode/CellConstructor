from __future__ import print_function
from __future__ import division

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ForceTensor

import pytest
import sys, os

def test_centering():
    # Go in the current directory
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Load the structure and the dynamical matrix
    dyn = CC.Phonons.Phonons("dyn")
    
    # Load the tensor (binary format)
    d3 = np.load("d3_realspace_sym.npy")


    # Now we can define a tensor
    tensor = CC.ForceTensor.Tensor3(dyn.structure, dyn.structure, dyn.GetSupercell())

    # We initialize the tensor with the 3n,3n,3n numpy matrix
    print("Initializing the tensor3...")
    tensor.SetupFromTensor(d3)

    tensor.WriteOnFile("tensor_original.dat")

    print("Centering...")
    tensor.Center()

    # Now we save the result
    tensor.WriteOnFile("tensor_centered.dat")
    
    print("Done.")

    print("R2:")
    print(tensor.r_vector3.T[:10])



if __name__ == "__main__":
    test_centering()
    
    
