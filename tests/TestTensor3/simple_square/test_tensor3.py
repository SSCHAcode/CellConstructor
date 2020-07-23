from __future__ import print_function

import cellconstructor as CC
import cellconstructor.ForceTensor
import cellconstructor.Structure

import numpy as np

import itertools

simple_structure = CC.Structure.Structure(1)
#simple_structure.coords[1, :] = (0.5, 0.5, 0.5)
simple_structure.unit_cell = np.eye(3)
simple_structure.unit_cell[2,2]=10
simple_structure.has_unit_cell = True


def test_save_simple_tensor():
    
    supercell_size = (2,2,1)
    #supercell_size = (2,1,1)
    supercell_structure = simple_structure.generate_supercell(supercell_size)
    nat_sc = supercell_structure.N_atoms
    
    
    
    
    for na in range(simple_structure.N_atoms):    
            for l,m,n in itertools.product(range(2),range(2),range(1)):
                # Get the atom in the supercell corresponding to the one in the unit cell
                na_vect = simple_structure.coords[na, :] + simple_structure.unit_cell.T.dot([l,m,n])
                na_sc = np.argmin( [np.sum( (supercell_structure.coords[k, :] - na_vect)**2) for k in range(nat_sc)])
                print (na,l,m,n,na_sc)
    
    
    
    t3_matrix = np.zeros( (3*nat_sc, 3*nat_sc, 3*nat_sc))
    
    #t3_matrix[5,1, 1] = t3_matrix[1,5, 1] =t3_matrix[1,1, 5] = 2.3 
    
    #t3_matrix[10,1, 1] = t3_matrix[1,10, 1] = t3_matrix[1,1, 10] =3.5
    
    ##################
#  tot ind  atomo sc - cart
    
         #0 0 - x
         #1 0 - y
         #2 0 - z
         #--------
         #3 1 - x
         #4 1 - y
         #5 1 - z
         #6 2 - x
         #7 2 - y
         #8 2 - z
       #  9 3 - x
       # 10 3 - y
       # 11 3 - z
       
       
    #primo indice da 0 a 2
    
    #t3_matrix[1,1,1] = 1.0
    #t3_matrix[0,0,0] = 1.0 
    t3_matrix[0,6,9] = 1.0     
    
    tensor1 = CC.ForceTensor.Tensor3(simple_structure, supercell_structure, supercell_size)
    #tensor2 = CC.ForceTensor.Tensor3(simple_structure, supercell_structure, supercell_size)
    #tensor3 = CC.ForceTensor.Tensor3(simple_structure, supercell_structure, supercell_size)
    
    tensor1.SetupFromTensor(t3_matrix)
    #tensor2.Setup(t3_matrix)    
    #tensor3.Setup(t3_matrix)        
    

    
    
    #tensor.WriteOnFile("prova_3RD_now")
    #tensor.ReadFromFile("prova_3RD_now")
    #tensor.WriteOnFile("prova_3RD_reload")

    tensor1.Center(Far=1)
    #tensor2.CenteringF(1)
    #tensor3.__old_centering2__(1)
    
    tensor1.WriteOnFile("prova_3RD_centrato")
    #tensor2.WriteOnFile("prova_3RD_centratoF")
    #tensor3.WriteOnFile("prova_3RD_centrato_old2")

if __name__ == "__main__":
    test_save_simple_tensor()
    
