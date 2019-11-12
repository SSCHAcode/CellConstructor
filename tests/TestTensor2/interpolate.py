from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ForceTensor

import numpy as np

"""
In this example we use the tensor utility to do the interpolation 
of a dynamical matrix.
The interpolation is preceeded with a test that the Tensor2 matrix
transforms back the tensor into itself if the same unit cell
is provided
"""

dyn = CC.Phonons.Phonons("../TestPhononSupercell/dynmat")

tensor = CC.ForceTensor.Tensor2(dyn.structure)
tensor.SetupFromPhonons(dyn)

for i in range(dyn.structure.N_atoms):
    for j in range(0,dyn.structure.N_atoms):
        print("Atoms {} {} | r_vector = {}".format(i +1, j+1, tensor.r_vectors[i,j,:]))
        for k in range(3):
            print("{:12.6f} {:12.6f} {:12.6f}".format(*list(tensor.tensor[i, j, k, :])))
            
new_tensor = tensor.GenerateSupercellTensor((1,1,1))

new_dyn = tensor.GeneratePhonons((2,2,1))
new_dyn.save_qe("prova")

other_dyn = dyn.Interpolate((1,1,1), (2,2,1))
other_dyn.save_qe("compare")

diff = new_tensor - dyn.dynmats[0]
print("Diff tensor:")
print(diff)

print("Difference:")
print(np.sqrt(np.sum(diff.dot(diff.T))))
# # Print the transformed file
# f = file("after_tensor.dat", "w")

# for i in range(dyn.structure.N_atoms):
#     for j in range(0,dyn.structure.N_atoms):
#         f.write("Atoms {} {}\n".format(i +1, j+1))
#         for k in range(3):
#             f.write("{:12.6f} {:12.6f} {:12.6f}\n".format(*list(new_tensor[3*i + k, 3*j: 3*j+3])))
#         f.write("\n")
# f.close()




