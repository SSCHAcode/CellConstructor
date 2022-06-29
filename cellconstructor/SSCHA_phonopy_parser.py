import numpy as np
import phonopy

import cellconstructor as CC

import cellconstructor.Structure as Structure
import cellconstructor.ForceTensor
import cellconstructor.Phonons as Phonons
import cellconstructor.Methods as Methods
import cellconstructor.symmetries as symmetries
import cellconstructor.Spectral
from cellconstructor.Units import *
from phono3py.phonon3.fc3 import set_permutation_symmetry_compact_fc3, set_permutation_symmetry_fc3, set_translational_invariance_compact_fc3, set_translational_invariance_fc3, cutoff_fc3_by_zero, distribute_fc3
from phono3py.phonon3.dataset import get_displacements_and_forces_fc3
from phonopy.harmonic.force_constants import compact_fc_to_full_fc

EV_TO_J = 1.602176634e-19
AU = 1.66053906660e-27

# Check if two vectors are same if we account for periodic boundary conditions
def same_vectors(vec1, vec2, cell):
    rvec1 = np.dot(vec1, cell)
    rvec2 = np.dot(vec2, cell)
    if(np.linalg.norm(rvec1 - rvec2) < 1.0-6):
        return True
    else:
        same = False
        for ix in range(-1,2):
            if(not same):
                for iy in range(-1,2):
                    if(not same):
                        for iz in range(-1,2):
                            rvec3 = rvec2 + np.array([ix,iy,iz])
                            if(np.linalg.norm(rvec1 - rvec3) < 1.0-6):
                                same = True
                                break
    return same

# Rewrite phonopy force constants to a format identical to CC.ForceTensor one
# Better use sscha_phonons_from_phonopy !
def phonopy_fc2_to_tensor2(fc2, phonon):
    s2u = phonon.supercell.s2u_map.copy()
    u2s = phonon.supercell.u2s_map.copy()
    nat_uc = len(u2s)
    nat = len(s2u)
    if(nat%nat_uc == 0):
        nuc = np.int(np.round(float(nat)/float(nat_uc)))
    else:
        print('Number of unit cell could not be determined from the number of atoms!', nat, nat_uc)
    if(len(fc2) == nat_uc):
        indices = np.arange(nat_uc)
    elif(len(fc2) == nat):
        indices = u2s.copy()
    else:
        raise RuntimeError('Unexpected number of pairs in second order force constants! ', len(fc2))
    ind_s = np.zeros_like(s2u)
    for iat in range(nat):
        for jat in range(nat_uc):
            if(s2u[iat] == u2s[jat]):
                ind_s[iat] = jat
    vecs = phonon.supercell.scaled_positions - phonon.supercell.scaled_positions[0]
    for iat in range(nat):
        for ix in range(3):
            if(vecs[iat][ix] < -1.0e-3):
                vecs[iat][ix] += 1.0
            elif(vecs[iat][ix] > 0.999):
                vecs[iat][ix] -= 1.0

    vecs = np.round(vecs, 5)
    rvec = vecs[0:nuc]
#    rvec = np.dot(rvec, phonon.supercell.cell)
#    vecs = np.dot(vecs, phonon.supercell.cell)
    invcell = np.linalg.inv(phonon.primitive.cell)
    ind_u = np.zeros_like(s2u)
    for iat in range(nat):
        found = False
        lvec = vecs[iat] - vecs[s2u[iat]]
        for ix in range(3):
            if(lvec[ix] < -1.0e-3):
                lvec[ix] += 1.0
            elif(lvec[ix] > 0.999):
                lvec[ix] -= 1.0
        for iuc in range(nuc):
            if(np.linalg.norm(lvec - rvec[iuc]) < 1.0e-4):
                ind_u[iat] = iuc
                found = True
                break
        if(not found):
            print('Could not find the unit cell  of atom: ', iat)
            print(lvec)
    print(ind_u)
    sscha_fc2 = np.zeros((nuc, 3*nat_uc, 3*nat_uc))
    print(ind_s)
    print(indices)
    for iat in range(nat_uc):
        for jat in range(nat):
            iuc = ind_u[jat]
            sscha_fc2[iuc, 3*iat:3*(iat+1), 3*ind_s[jat]:3*(ind_s[jat] + 1)] = fc2[indices[iat]][jat]
    rvec = np.dot(rvec, phonon.supercell.cell)

    return sscha_fc2, rvec

# Generate CC.Structure from PhonopyAtoms
def get_sscha_structure_from_phonopy(phatoms):

    sscha_structure = Structure.Structure(nat = len(phatoms.positions))
    sscha_structure.coords = phatoms.positions
    sscha_structure.N_atoms = len(phatoms.positions)
    sscha_structure.atoms = phatoms.symbols
    sscha_structure.unit_cell = phatoms.cell
    sscha_structure.has_unit_cell = True
    for iat in range(sscha_structure.N_atoms):
        sscha_structure.masses[sscha_structure.atoms[iat]] = phatoms.masses[iat]/MASS_RY_TO_UMA

    return sscha_structure

# Get Cellconstructor.Phonons from Phonopy object
def sscha_phonons_from_phonopy(phonon):

    sscha_structure = get_sscha_structure_from_phonopy(phonon.primitive) # get structure
    sscha_supercell = get_sscha_structure_from_phonopy(phonon.supercell)

    nat_uc = sscha_structure.N_atoms
    nat_sc = sscha_supercell.N_atoms

    q_grid = symmetries.GetQGrid(sscha_structure.unit_cell, np.diag(phonon.supercell_matrix))
    gamma_index = np.argmin(np.sum(np.array(q_grid)**2, axis = 1))
    q_grid[gamma_index] = q_grid[0].copy()
    q_grid[0] = np.zeros(3, dtype = np.double)
    dyn = Phonons.Phonons(sscha_structure, len(q_grid))
    dyn.q_tot = q_grid 

    if(len(phonon.force_constants) == nat_sc):
        fc2 = phonon.force_constants.copy()/RY_TO_EV*BOHR_TO_ANGSTROM**2
    elif(len(phonon.force_constants) == nat_uc):
        fc2 = compact_fc_to_full_fc(phonon, phonon.force_constants)
        fc2 *= BOHR_TO_ANGSTROM**2/RY_TO_EV
    else:
        raise RuntimeError('Number of force constants does not match expected phonopy formats! ')

    sscha_fc2 = np.zeros( (3*nat_sc, 3*nat_sc), dtype = np.double)
    for iat in range(nat_sc):
        for jat in range(nat_sc):
            sscha_fc2[3*iat:3*(iat+1), 3*jat:3*(jat+1)] = fc2[iat, jat]

    dynq = Phonons.GetDynQFromFCSupercell(sscha_fc2, np.array(dyn.q_tot), sscha_structure, sscha_supercell)
    dyn.dynmats = dynq
    dyn.AdjustQStar()

    return dyn


# Get ForceTensor.Tensor3 from Phono3py object 
def phonopy_fc3_to_tensor3(tc, apply_symmetries = True):
    
    unitcell = get_sscha_structure_from_phonopy(tc.primitive)
    supercell = get_sscha_structure_from_phonopy(tc.supercell)
    uc_nat = unitcell.N_atoms
    sc_nat = supercell.N_atoms
    if(uc_nat in tc.fc3.shape[0:3]):
        print('Compact forceconstants.')
        tc.generate_displacements()
        supercells = tc.supercells_with_displacements
        tc.fc3 = force_constants_3rd_order
        fcart_dummy = []
        for isup in range(len(supercells)):
            fcart_dummy.append(np.zeros_like(supercells[isup].scaled_positions))
        tc.forces = fcart_dummy
        disps, _ = get_displacements_and_forces_fc3(tc.dataset)
        first_disp_atoms = np.unique([x["number"] for x in tc.dataset["first_atoms"]])
        s2p_map = tc.primitive.s2p_map
        p2s_map = tc.primitive.p2s_map
        p2p_map = tc.primitive.p2p_map
        s2compact = np.array([p2p_map[i] for i in s2p_map], dtype="int_")
        for i in first_disp_atoms:
            assert i in p2s_map
        target_atoms = [i for i in p2s_map if i not in first_disp_atoms]
        rotations = tc.symmetry.symmetry_operations["rotations"]
        permutations = tc.symmetry.atomic_permutations
        distribute_fc3(tc.fc3, first_disp_atoms, target_atoms, phonon.supercell.cell.T, rotations, permutations, s2compact, verbose=False)

    supercell_matrix = (np.diag(tc.supercell_matrix).astype(int)).tolist()
    #print(supercell_matrix)
    #print(unitcell.unit_cell)
    #print(unitcell.coords)
    supercell_structure = unitcell.generate_supercell(supercell_matrix)
    #print(supercell_structure.unit_cell)
    #print(supercell_structure.coords)
    atom_mapping = np.zeros(len(supercell.coords), dtype=int)
    already_there = [False for x in range(len(supercell.coords))]
    for iat in range(len(supercell.coords)):
        found_atom = False
        for jat in range(len(supercell.coords)):
            if(np.linalg.norm(supercell.coords[iat] - supercell_structure.coords[jat]) < 1.0e-5 and not already_there[jat]):
                atom_mapping[iat] = jat
                found_atom = True
                already_there[jat] = True
                break
            elif(np.linalg.norm(supercell.coords[iat] - supercell_structure.coords[jat]) < 1.0e-5 and already_there[jat]):
                raise RuntimeError('Already matched this atom!')
        if(not found_atom):
            print('Could not find ' + str(iat + 1) + ' atom in the structure!')
    if(not np.all(already_there)):
        raise RuntimeError('Did not match all atoms...')
#    print(atom_mapping)
    tensor3 = CC.ForceTensor.Tensor3(unitcell, supercell_structure, supercell_matrix)
    #print(tensor3.supercell_structure.unit_cell)
    #print(tensor3.supercell_structure.coords)
    aux_tensor = np.zeros((3*sc_nat, 3*sc_nat, 3*sc_nat))
    for iat in range(sc_nat):
        iat1 = atom_mapping[iat]
        for jat in range(sc_nat):
            jat1 = atom_mapping[jat]
            for kat in range(sc_nat):
                kat1 = atom_mapping[kat]
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            aux_tensor[iat1*3+i, jat1*3+j, kat1*3+k] = tc.fc3[iat,jat,kat,i,j,k]
    d3 = np.asfortranarray(aux_tensor)
    if(apply_symmetries):
        qe_sym = CC.symmetries.QE_Symmetry(supercell_structure)
        qe_sym.SetupFromSPGLIB()
        qe_sym.ApplySymmetryToTensor3(d3)
    d3 *= BOHR_TO_ANGSTROM**3/RY_TO_EV
    tensor3.SetupFromTensor(d3)
    np.save("d3_realspace_sym.npy", d3)

    print('Translated phonopy 3rd order force constants.')
    return tensor3

def tensor2_to_phonopy_fc2(SSCHA_tensor, phonon):

    SSCHA_force_constants = np.array(SSCHA_tensor.tensor.copy())
    red_vecs_SSCHA = np.round(SSCHA_tensor.x_r_vector2.copy().astype(float), 4)
    red_vecs_phonopy = np.round(np.dot(phonon.supercell.positions, np.linalg.inv(phonon.primitive.cell)), 4)
    dims = np.diag(phonon.supercell_matrix)
    nuc = np.prod(dims)
    if(nuc != np.shape(red_vecs_SSCHA)[1]):
        print('Phonopy and SSCHA have different number of unit cells!')
        print('Make sure you do not center() SSCHA tensor before attempting this conversion!')
        raise RuntimeError('Non-conforming shapes of matrices!')
    nat = len(phonon.primitive.positions)
    s2u = phonon.supercell.s2u_map.copy()
    primitive_cell_position = red_vecs_phonopy[s2u]
    for iat in range(len(red_vecs_phonopy)):
        red_vecs_phonopy[iat] = red_vecs_phonopy[iat] - primitive_cell_position[iat]
        for i in range(3):
            if(red_vecs_phonopy[iat][i] >= dims[i]):
                red_vecs_phonopy[iat][i] = red_vecs_phonopy[iat][i] - dims[i]
            elif(red_vecs_phonopy[iat][i] < 0.0):
                red_vecs_phonopy[iat][i] = red_vecs_phonopy[iat][i] + dims[i]
    unit_cell_order = np.zeros(np.shape(red_vecs_SSCHA)[1], dtype = int)
    for iuc in range(np.shape(red_vecs_SSCHA)[1]):
        for iat in range(nuc):
            if(np.linalg.norm(red_vecs_SSCHA[:,iuc] - red_vecs_phonopy[iat]) < 1.0e-4):
                unit_cell_order[iuc] = iat
    phonopy_tensor = np.zeros((nat,nuc*nat,3,3))
    for iuc in range(nuc):
        for iat in range(nat):
            for jat in range(nat):
                phonopy_tensor[iat][jat*nuc + unit_cell_order[iuc]] = SSCHA_force_constants[iuc,3*iat:3*(iat+1),3*jat:3*(jat+1)]
    return phonopy_tensor*RY_TO_EV/BOHR_TO_ANGSTROM**2

def tensor3_to_phonopy_fc3(SSCHA_tensor, phonon):

    SSCHA_force_constants = np.array(SSCHA_tensor.tensor.copy())
    red_vecs_SSCHA_1 = np.round(SSCHA_tensor.x_r_vector2.copy().astype(float), 4)
    red_vecs_SSCHA_2 = np.round(SSCHA_tensor.x_r_vector3.copy().astype(float), 4)
    red_vecs_phonopy = np.round(np.dot(phonon.supercell.positions, np.linalg.inv(phonon.primitive.cell)), 4)
    dims = np.diag(phonon.supercell_matrix)
    nuc = np.prod(dims)
    if(nuc**2 != np.shape(red_vecs_SSCHA_1)[1]):
        print('Phonopy and SSCHA have different number of unit cells!')
        print('Make sure you do not center() SSCHA tensor before attempting this conversion!')
        raise RuntimeError('Non-conforming shapes of matrices!')
    nat = len(phonon.primitive.positions)
    s2u = phonon.supercell.s2u_map.copy()
    primitive_cell_position = red_vecs_phonopy[s2u]
    for iat in range(len(red_vecs_phonopy)):
        red_vecs_phonopy[iat] = red_vecs_phonopy[iat] - primitive_cell_position[iat]
        for i in range(3):
            if(red_vecs_phonopy[iat][i] >= dims[i]):
                red_vecs_phonopy[iat][i] = red_vecs_phonopy[iat][i] - dims[i]
            elif(red_vecs_phonopy[iat][i] < 0.0):
                red_vecs_phonopy[iat][i] = red_vecs_phonopy[iat][i] + dims[i]
    unit_cell_order_1 = np.zeros(np.shape(red_vecs_SSCHA_1)[1], dtype = int)
    unit_cell_order_2 = np.zeros(np.shape(red_vecs_SSCHA_2)[1], dtype = int)
    for iuc in range(np.shape(red_vecs_SSCHA_1)[1]):
        for iat in range(nuc):
            if(np.linalg.norm(red_vecs_SSCHA_1[:,iuc] - red_vecs_phonopy[iat]) < 1.0e-4):
                unit_cell_order_1[iuc] = iat
    for iuc in range(np.shape(red_vecs_SSCHA_2)[1]):
        for iat in range(nuc):
            if(np.linalg.norm(red_vecs_SSCHA_2[:,iuc] - red_vecs_phonopy[iat]) < 1.0e-4):
                unit_cell_order_2[iuc] = iat
    phonopy_tensor = np.zeros((nat,nuc*nat,nuc*nat,3,3,3))
    for iuc in range(nuc**2):
        for iat in range(nat):
            for jat in range(nat):
                for kat in range(nat):
                    phonopy_tensor[iat][jat*nuc + unit_cell_order_1[iuc]][kat*nuc + unit_cell_order_2[iuc]] = SSCHA_force_constants[iuc,3*iat:3*(iat+1),3*jat:3*(jat+1), 3*kat:3*(kat+1)]
    return phonopy_tensor*RY_TO_EV/BOHR_TO_ANGSTROM**3
