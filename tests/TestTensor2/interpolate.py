from __future__ import print_function
from __future__ import division

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.ForceTensor

import ase
from ase.visualize import view

import numpy as np

"""
In this example we use the tensor utility to do the interpolation 
of a dynamical matrix.
The interpolation is preceeded with a test that the Tensor2 matrix
transforms back the tensor into itself if the same unit cell
is provided
"""

SUPERCELL = (2,2,1)

dyn = CC.Phonons.Phonons("../TestPhononSupercell/dynmat")

other_dyn = dyn.Interpolate((1,1,1), SUPERCELL)
other_dyn.save_qe("compare")
other_supercell = other_dyn.GenerateSupercellDyn(SUPERCELL)

tensor = CC.ForceTensor.Tensor2(dyn.structure)
tensor.SetupFromPhonons(dyn)

for i in range(dyn.structure.N_atoms):
    for j in range(0,dyn.structure.N_atoms):
        print("Atoms {} {} | r_vector = {}".format(i +1, j+1, tensor.r_vectors[i,j,:]))
        for k in range(3):
            print("{:12.6f} {:12.6f} {:12.6f}".format(*list(tensor.tensor[i, j, k, :])))

            
new_tensor = tensor.GenerateSupercellTensor(SUPERCELL)

new_dyn = tensor.GeneratePhonons(SUPERCELL)
new_dyn.save_qe("prova")

diff = new_tensor - other_supercell.dynmats[0]
print("Diff tensor:")
print(diff)


total_difference = np.max(np.abs(diff)) 


# Check if the supercell satisfy the translational symmetry
compare = other_supercell.dynmats[0]
#compare = new_tensor
new_tensor_old = compare.copy()


ws,p = other_supercell.DiagonalizeSupercell()

print("Unit cell:")
print(other_supercell.structure.unit_cell)

CC.symmetries.ApplyTranslationsToSupercell(compare, other_supercell.structure, SUPERCELL)

print("Frequencies after supercell symmetrization:")
wp,p = other_supercell.DiagonalizeSupercell()
print("\n".join(["{:3d}) {:10.3f} cm-1 vs {:10.3f} cm-1".format(i, ws[i] * CC.Units.RY_TO_CM, wp[i] * CC.Units.RY_TO_CM) for i, w in enumerate(ws)]))

print("Difference after supercell symmetrization:")
print(np.max(np.abs(compare - new_tensor_old)))

print("Difference betwee Tensor and QE interpolation:")
print(total_difference)

print("Blocks that have a difference:")
diff_mask = np.abs(diff) > 1e-8
diff_indices = np.arange(np.prod(diff_mask.shape))[diff_mask.ravel()]
maxx, _ = np.shape(diff)
print("\n".join(["{:3d} {:3d} | atoms {:2d} {:2d} ".format(int(x / maxx), x % maxx, int(int(x/maxx) / 3), int(x % maxx / 3)) for x in diff_indices]))


# Lets pick (4, 11)
print()
print("Tensor (4,11):")
print(new_tensor[ 4*3 : 5*3, 11*3 : 12*3])
print("superdyn:")
print(other_supercell.dynmats[0][ 4*3 : 5*3, 11*3 : 12*3])


# # Print the transformed file
# f = file("after_tensor.dat", "w")

# for i in range(dyn.structure.N_atoms):
#     for j in range(0,dyn.structure.N_atoms):
#         f.write("Atoms {} {}\n".format(i +1, j+1))
#         for k in range(3):
#             f.write("{:12.6f} {:12.6f} {:12.6f}\n".format(*list(new_tensor[3*i + k, 3*j: 3*j+3])))
#         f.write("\n")
# f.close()




