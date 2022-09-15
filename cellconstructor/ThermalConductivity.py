#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division 

import numpy as np
import os, sys
import scipy, scipy.optimize
from scipy import integrate
import h5py
import psutil

# Import the Fortran Code
import symph
import thirdorder
import itertools
import thermal_conductivity

import cellconstructor as CC
import cellconstructor.Phonons as Phonons 
import cellconstructor.Methods as Methods 
import cellconstructor.symmetries as symmetries
import cellconstructor.Spectral
from cellconstructor.Units import A_TO_BOHR, BOHR_TO_ANGSTROM, RY_TO_CM, ELECTRON_MASS_UMA, MASS_RY_TO_UMA, HBAR, K_B, RY_TO_EV
import spglib
from scipy import stats
import warnings

import time


try:
    from mpi4py import MPI
    __MPI__ = True
except:
    __MPI__ = False
    
try:
    import spglib
    __SPGLIB__ = True
except:
    __SPGLIB__ = False

__EPSILON__ = 1e-5 
__EPSILON_W__ = 3e-9
HBAR_JS = 1.054571817e-34
HPLANCK = HBAR_JS*2.0*np.pi
RY_TO_J = 2.1798741e-18
HBAR_RY = HBAR_JS/RY_TO_J
AU = 1.66053906660e-27
EV_TO_J = 1.602176634e-19
RY_TO_THZ = RY_TO_CM/0.0299792458
SSCHA_TO_THZ = np.sqrt(RY_TO_J/AU/MASS_RY_TO_UMA)/BOHR_TO_ANGSTROM/100.0/2.0/np.pi 
SSCHA_TO_MS = np.sqrt(RY_TO_J/AU/MASS_RY_TO_UMA)/BOHR_TO_ANGSTROM
KB = 1.380649e-23
STR_FMT = '<15'

# Abundance of isotopes in nature taken from https://www.chem.ualberta.ca/~massspec/atomic_mass_abund.pdf
natural_isotopes = {\
        'H' :[[0.9999, 1.0078], [0.0001, 2.0141]],\
        'He':[[1.0000, 4.0026]],\
        'Li':[[0.0759, 6.0151], [0.9241, 7.0160]],\
        'Be':[[1.0000, 9.0122]],\
        'B' :[[0.1990, 10.0129], [0.8010, 11.0093]],\
        'C' :[[0.9893, 12.0000], [0.0107, 13.0033]],\
        'N' :[[0.9963, 14.0030], [0.0037, 15.0001]],\
        'O' :[[0.9976, 15.9949], [0.0003, 16.9991], [0.0021, 17.9991]],\
        'F' :[[1.0000, 18.9984]],\
        'Ne':[[0.9048, 19.9924], [0.0027, 20.9938], [0.0925, 21.9913]],\
        'Na':[[1.0000, 22.9898]],\
        'Mg':[[0.7899, 23.9850], [0.1000, 24.9858], [0.1101, 25.9825]],\
        'Al':[[1.0000, 26.9815]],\
        'Si':[[0.9223, 27.9769], [0.0468, 28.9764], [0.0309, 29.9737]],\
        'P' :[[1.0000, 30.9737]],\
        'S' :[[0.9493, 31.9720], [0.0076, 32.9714], [0.0429, 33.9679], [0.0002, 35.9671]],\
        'Cl':[[0.7578, 34.9688], [0.2422, 36.9659]],\
        'Ar':[[0.0034, 35.9675], [0.0006, 37.9627], [0.9960, 39.9624]],\
        'K' :[[0.9326, 38.9637], [0.0001, 39.9640], [0.0672, 40.9618]],\
        'Ca':[[0.9694, 39.9626], [0.0065, 41.9586], [0.0209, 43.9555], [0.0001, 42.9588], [0.0001, 47.9525]],\
        'Sc':[[1.0000, 44.9559]],\
        'Ti':[[0.0825, 45.9526], [0.0744, 46.9518], [0.7372, 47.9479], [0.0541, 48.9479], [0.0518, 49.9448]],\
        'Va':[[0.0025, 49.9472], [0.9975, 50.9440]],\
        'Cr':[[0.0435, 49.9460], [0.8379, 51.9405], [0.0950, 52.9406], [0.0236, 53.9389]],\
        'Mn':[[1.0000, 54.9380]],\
        'Fe':[[0.0585, 53.9396], [0.9175, 55.9349], [0.0212, 56.9354], [0.0028, 57.9332]],\
        'Co':[[1.0000, 58.9332]],\
        'Ni':[[0.6808, 57.9353], [0.2622, 59.9308], [0.0114, 60.9311], [0.0363, 61.9283], [0.0093, 63.9279]],\
        'Cu':[[0.6917, 62.9296], [0.3083, 64.9278]],\
        'Zn':[[0.4863, 63.9291], [0.2790, 65.9260], [0.0410, 66.9271], [0.1875, 67.9248], [0.0062, 69.9253]],\
        'Ga':[[0.6011, 68.9256], [0.3989, 70.9247]],\
        'Ge':[[0.2084, 69.9242], [0.2754, 71.9220], [0.0773, 72.9234], [0.3628, 73.9211], [0.0761, 75.9214]],\
        'As':[[1.0000, 74.9216]],\
        'Se':[[0.0089, 73.9224], [0.0937, 75.9192], [0.0763, 76.9199], [0.2377, 77.9173], [0.4961, 79.9165], [0.0873, 81.9167]],\
        'Br':[[0.5069, 78.9183], [0.4931, 80.9163]],\
        'Kr':[[0.0035, 77.9204], [0.0228, 79.9164], [0.1158, 81.9135], [0.1149, 82.9141], [0.5700, 83.9115], [0.1730, 85.9106]],\
        'Rb':[[0.7217, 84.9118], [0.2783, 86.9092]],\
        'Sr':[[0.0056, 83.9134], [0.0986, 85.9093], [0.0700, 86.9089], [0.8258, 87.9056]],\
        'Y' :[[1.0000, 88.9058]],\
        'Zr':[[0.5145, 89.9047], [0.1122, 90.9056], [0.1715, 91.9050], [0.1738, 93.9063], [0.0280, 95.9083]],\
        'Nb':[[1.0000, 92.9064]],\
        'Mo':[[0.1484, 91.9068], [0.0925, 93.9050], [0.1592, 94.9058], [0.1668, 95.9046], [0.0955, 96.9060], [0.2413, 97.9054], [0.0963, 99.9075]],\
        'Tc':[[1.0000, 97.9072]],\
        'Ru':[[0.0554, 95.9076], [0.0187, 97.9053], [0.1276, 98.9059], [0.1260, 99.9042], [0.1706, 100.9056], [0.3155, 101.9043], [0.1862, 103.9054]],\
        'Rh':[[1.0000, 102.9055]],\
        'Pd':[[0.0102, 101.9056], [0.1114, 103.9040], [0.2233, 104.9051], [0.2733, 105.9035], [0.2646, 107.9040], [0.1172, 109.9052]],\
        'Ag':[[0.5184, 106.9051], [0.4816, 108.9048]],\
        'Cd':[[0.0125, 105.9066], [0.0089, 107.9042], [0.1249, 109.9030], [0.1280, 110.9042], [0.2413, 111.9028], [0.1222, 112.9044], [0.2873, 113.9033], [0.0749, 115.9048]],\
        'In':[[0.0429, 112.9041], [0.9571, 114.9039]],\
        'Sn':[[0.0097, 111.9048], [0.0066, 113.9028], [0.0034, 114.9033], [0.1454, 115.9017], [0.0768, 116.9029], [0.2422, 117.9016], [0.0859, 118.9033], [0.3258, 119.9022], [0.0463, 121.9034], [0.0579, 123.9053]],\
        'Sb':[[0.5721, 120.9038], [0.4279, 122.9042]],\
        'Te':[[0.0009, 119.9040], [0.0255, 121.9030], [0.0089, 122.9042], [0.0474, 123.9028], [0.0707, 124.9044], [0.1884, 125.9033], [0.3174, 127.9045], [0.3408, 129.9062]],\
        'I' :[[1.0000, 126.9045]],}
# This should be continued!

def get_spglib_cell(dyn):

    """

    Get spglib cell from Cellconstructor.Phonons()

    """

    pos = np.dot(dyn.structure.coords, np.linalg.inv(dyn.structure.unit_cell))
    numbers = [0 for x in range(len(pos))]
    symbols_unique = np.unique(dyn.structure.atoms)
    for iat in range(len(dyn.structure.atoms)):
        for jat in range(len(symbols_unique)):
            if(dyn.structure.atoms[iat] == symbols_unique[jat]):
                numbers[iat] == jat
    return (dyn.structure.unit_cell, pos, numbers)

    ########################################################################################################################################

def check_if_rotation(rot, thr):

    """

    Check if rot matrix is rotation

    """

    if(np.abs(np.linalg.det(rot)) - 1.0 > thr):
        print('Magnitude of the determinant of the rotation matrix larger than 1.0! Should not happen!')
        print(np.abs(np.linalg.det(rot)))
    try:
        if(np.any(np.linalg.inv(rot) - rot.T > 1.0e-4)):
            print('Transpose and inverse of rotation matrix are not the same! Should not happen!')
            print(np.linalg.inv(rot))
            print(rot.T)
    except:
        print('Something wrong when trying to calculate inverse of rotation matrix. Possibly singular!')
        print(rot)
        raise RuntimeError('Exit!')

    #######################################################################################################################################

def gaussian(x, x0, sigma):

    """

    Definition of gaussian function

    x0 - location
    sigma - square root of variance

    """

    return np.exp(-0.5*(x-x0)**2/sigma**2)/np.sqrt(2.0*np.pi)/sigma

    ####################################################################################################################################

def heat_capacity(freqs, temperature, hbar1, kb1, cp_mode = 'quantum'):

    """

    Calculate mode heat capacity

    freqs - frequency of the phonon mode
    temperature - temperature at which to calculate heat capacity
    hbar1 - reduced Planck's constant in appropriate units
    kb1   - Boltzmann constant in appropriate units
    mode - how to treat phonon populations, quantum - bose einstein, classical - kbT/hbar\omega

    """
    if(cp_mode == 'quantum'):
        if(freqs > 0.0):
            x1 = np.exp(hbar1*freqs/kb1/temperature)
            return (hbar1*freqs)**2*x1/kb1/temperature**2/(x1-1.0)**2
        else:
            return 0.0
    elif(cp_mode == 'classical'):
        return kb1
   
    ####################################################################################################################################
    
def bose_einstein(freqs, temperature, hbar1, kb1, cp_mode = 'quantum'):

    if(cp_mode == 'quantum'):
        if(freqs > 0.0):
            x1 = np.exp(hbar1*freqs/kb1/temperature)
            return 1.0/(x1-1.0)
        else:
            return 0.0
    elif(cp_mode == 'classical'):
        return kb1/temperature

    #####################################################################################################################################

def same_vector(vec1, vec2, cell):

    """

    Check if two vectors are the same. PBC rules apply

    vec1, vec2 - two vectors to compare
    cell       - lattice vectors for which PBC must hold

    """


    invcell = np.linalg.inv(cell)
    rvec1 = np.dot(vec1, invcell)
    rvec1 -= np.rint(rvec1)
    rvec2 = np.dot(vec2, invcell)
    rvec2 -= np.rint(rvec2)
    res = False

    if(np.linalg.norm(rvec1 - rvec2) < 1.0e-4):
        res = True

    if(not res):
        for i in range(-1, 2):
            if(not res):
                for j in range(-1, 2):
                    if(not res):
                        for k in range(-1, 2):
                            rvec21 = rvec2 + np.array([float(i), float(j), float(k)])
                            if(np.linalg.norm(rvec1 - rvec21) < 1.0e-4):
                                res = True
                                break
    return res

######################################################################################################################################

def check_degeneracy(x, tol):

    """

    Simple check if there are degenerate phonon modes for a given k - point.

    x   - frequencies at the given k - point
    tol - tolerance to be satisfied 

    """

    x1 = x.copy()
    x1 = x1.tolist()
    degs = []
    for i in range(len(x1)):
        if(not any(i in sl for sl in degs)):
            curr_degs = []
            curr_degs.append(i)
            for j in range(i+1, len(x1)):
                if(np.abs(x1[i]-x1[j]) < tol):
                    curr_degs.append(j)
            degs.append(curr_degs)
    return degs
            
def get_kpoints_in_path(path, nkpts, kprim):
    segments = path['path']
    coords = path['point_coords']
    tot_length = 0.0
    for i in range(len(segments)):
        qpt1 = coords[segments[i][1]]
        qpt2 = coords[segments[i][0]]
        qpt1 = np.dot(qpt1, kprim)
        qpt2 = np.dot(qpt2, kprim)
        tot_length += np.linalg.norm(qpt1 - qpt2)
    dl = tot_length/float(nkpts)
    distance = []
    kpoints = []
    for i in range(len(segments)):
        qpt1 = coords[segments[i][1]]
        qpt2 = coords[segments[i][0]]
        qpt1 = np.dot(qpt1, kprim)
        qpt2 = np.dot(qpt2, kprim)
        length = np.linalg.norm(qpt1 - qpt2)
        kpoints.append(qpt2)
        if(i == 0):
            distance.append(0.0)
        else:
            distance.append(distance[-1])
        start_dist = distance[-1]
        nkpts1 = np.int(np.floor(length/dl))
        for j in range(nkpts1):
            newqpt = qpt2 + (qpt1 - qpt2)/float(nkpts1)*float(j + 1)
            kpoints.append(newqpt)
            distance.append(start_dist + np.linalg.norm(qpt1 - qpt2)/float(nkpts1)*float(j + 1))
    distance = np.array(distance)
    print('k-point path taken: ')
    for i in range(len(segments)):
        print(segments[i], end = ' -> ')
    print('')
    distance /= distance[-1]
    return kpoints, distance, segments


def stupid_centering_fc3_v3(tensor3, check_for_symmetries = True, Far = 1):
    #print(psutil.virtual_memory().percent)
    rprim = tensor3.unitcell_structure.unit_cell.copy()
    irprim = np.linalg.inv(rprim)
    rsup = tensor3.supercell_structure.unit_cell.copy()
    irsup = np.linalg.inv(rsup)
    #print(rprim)
    positions = tensor3.unitcell_structure.coords.copy()
    xpos = np.dot(positions, np.linalg.inv(rprim))
    natom = len(xpos)
    #print(xpos)
    symbols = tensor3.unitcell_structure.atoms
    unique_symbols = np.unique(symbols)
    unique_numbers = np.arange(len(unique_symbols), dtype=int) + 1
    numbers = np.zeros(len(symbols))
    for iat in range(len(symbols)):
        for jat in range(len(unique_symbols)):
            if(symbols[iat] == unique_symbols[jat]):
                numbers[iat] = unique_numbers[jat]
    #print(numbers)
    cell = (rprim, xpos, numbers)
    if(tensor3.n_R == tensor3.n_sup**2):
        print('ForceTensor most likely not previously centered! ')
        if(check_for_symmetries):
            permutation = thermal_conductivity.third_order_cond_centering.check_permutation_symmetry(tensor3.tensor, tensor3.r_vector2.T, tensor3.r_vector3.T, tensor3.n_R, natom)
            print(psutil.virtual_memory().percent)
            if(not permutation):
                print('Permutation symmetry not satisfied. Forcing symmetry! ')
                fc3 = thermal_conductivity.third_order_cond_centering.apply_permutation_symmetry(tensor3.tensor, tensor3.r_vector2.T, tensor3.r_vector3.T, tensor3.n_R, natom)
                tensor3.tensor = fc3
            permutation = thermal_conductivity.third_order_cond_centering.check_permutation_symmetry(tensor3.tensor, tensor3.r_vector2.T, tensor3.r_vector3.T, tensor3.n_R, natom)
        else:
            permutation = True
        if(permutation):
            print('Permutation symmetry satisfied. Centering ...')
            hfc3, hr_vector2, hr_vector3, tot_trip = thermal_conductivity.third_order_cond_centering.find_triplets(tensor3.tensor, tensor3.r_vector2.T, tensor3.r_vector3.T, rsup, irsup, positions, Far, tensor3.n_R, natom)
            hfc3 = np.asfortranarray(hfc3)
            hr_vector3 = np.asfortranarray(hr_vector3)
            hr_vector2 = np.asfortranarray(hr_vector2)
            #print(np.shape(hfc3), np.shape(hr_vector3), np.shape(hr_vector2), tensor3.n_R, natom)
            #print(tot_trip)
            maxtrip = thermal_conductivity.third_order_cond_centering.number_of_triplets(hfc3, hr_vector2, hr_vector3, tot_trip, natom, tensor3.n_R)
           # print(maxtrip, tensor3.n_R, natom)
           # print(psutil.virtual_memory().percent)
           # print(np.shape(hfc3), np.shape(hr_vector2), np.shape(hr_vector3))
           # print('This should occupy: ' + format((np.prod(np.shape(hfc3)) + np.prod(np.shape(hr_vector2)) + np.prod(np.shape(hr_vector3)))*8.0/1024.0**3, '.3f') + ' GB of memory!')
            fc3, r_vector2, r_vector3, ntrip = thermal_conductivity.third_order_cond_centering.distribute_fc3(hfc3, hr_vector2, hr_vector3, tot_trip, maxtrip, natom, tensor3.n_R)
            #ntrip = 0
            #for i in range(len(r_vector2)):
            #    if(i != 0 and np.linalg.norm(r_vector2[i]) < 1.0e-6 and np.linalg.norm(r_vector3[i]) < 1.0e-6):
            #        ntrip = i
            #        break
            print('Final number of triplets: ', ntrip)
            tensor3.n_R = ntrip
            tensor3.r_vector2 = r_vector2[0:ntrip,:].T
            tensor3.r_vector3 = r_vector3[0:ntrip,:].T
            tensor3.x_r_vector2 = np.zeros_like(tensor3.r_vector2)
            tensor3.x_r_vector3 = np.zeros_like(tensor3.r_vector3)
            tensor3.tensor = fc3[0:ntrip]
            tensor3.x_r_vector2 = np.rint(np.dot(r_vector2[0:ntrip,:], irprim), dtype=float).T
            tensor3.x_r_vector3 = np.rint(np.dot(r_vector3[0:ntrip,:], irprim), dtype=float).T
            write_fc3(tensor3)
            if(check_for_symmetries):
                permutation = thermal_conductivity.third_order_cond_centering.check_permutation_symmetry(tensor3.tensor, tensor3.r_vector2.T, tensor3.r_vector3.T, tensor3.n_R, natom)
                if(not permutation):
                    print('After centering tensor does not satisfy permutation symmetry!')
            return tensor3
        else:
            raise RuntimeError('Permutation symmetry not satisfied again. Aborting ...')

def apply_asr(tensor3, tol = 1.0e-10):

    print('Applying ASR!')
    natom = tensor3.unitcell_structure.N_atoms

    fc3 = thermal_conductivity.third_order_cond_centering.apply_asr(tensor3.tensor, tensor3.r_vector2.T, tensor3.r_vector3.T, tensor3.n_R, natom)  
    tensor3.tensor = fc3
    permutation = thermal_conductivity.third_order_cond_centering.check_permutation_symmetry(tensor3.tensor, tensor3.r_vector2.T, tensor3.r_vector3.T, tensor3.n_R, natom)
    if(not permutation):
        print('After ASR tensor does not satisfy permutation symmetry!')

    return tensor3

def write_fc3(tensor3):
    natom = len(tensor3.unitcell_structure.coords)
    with open('fc3.dat', 'w+') as outfile:
        for i in range(tensor3.n_R):
            outfile.write('Triplet: '+ str(i+1) + '\n')
            for j in range(3):
                outfile.write(3*' ' + format(tensor3.r_vector2[j,i], '.8f'))
            outfile.write('\n')
            for j in range(3):
                outfile.write(3*' ' + format(tensor3.r_vector3[j,i], '.8f'))
            outfile.write('\n')
            for iat in range(natom):
                for j in range(3):
                    outfile.write('Coordinate '+ str(1+j + 3*iat) + '\n')
                    for jat in range(natom):
                        for j1 in range(3):
                            for kat in range(natom):
                                for k1 in range(3):
                                    outfile.write(3*' ' + format(tensor3.tensor[i,3*iat+j,3*jat+j1,3*kat+k1], '.8f'))
                            outfile.write('\n')

def stupid_centering_fc3_v2(tensor3, Far = 1):

    rprim = tensor3.unitcell_structure.unit_cell.copy()
    irprim = np.linalg.inv(rprim)
    rsup = tensor3.supercell_structure.unit_cell.copy()
    irsup = np.linalg.inv(rsup)
    #print(rprim)
    positions = tensor3.unitcell_structure.coords.copy()
    xpos = np.dot(positions, np.linalg.inv(rprim))
    natom = len(xpos)
    #print(xpos)
    symbols = tensor3.unitcell_structure.atoms
    unique_symbols = np.unique(symbols)
    unique_numbers = np.arange(len(unique_symbols), dtype=int) + 1
    numbers = np.zeros(len(symbols))
    for iat in range(len(symbols)):
        for jat in range(len(unique_symbols)):
            if(symbols[iat] == unique_symbols[jat]):
                numbers[iat] = unique_numbers[jat]
    #print(numbers)
    cell = (rprim, xpos, numbers)

    if(tensor3.n_R == tensor3.n_sup**2):
        print('Not previously centered. Stupid centering!')
        got = [False for x in range(tensor3.n_R)]
        pairs = []
        # Check if it satisfies the permutation symmetry
        for i in range(tensor3.n_R):
            rvec2 = tensor3.r_vector2[:,i].copy()
            rvec3 = tensor3.r_vector3[:,i].copy()
            if(np.linalg.norm(rvec2) < 1.0e-5 and np.linalg.norm(rvec3) < 1.0e-5):
                pairs.append([i,i])
                got[i] = True
            else:
                if(not got[i]):
                    for j in range(i, tensor3.n_R):
                        if(not got[j]):
                            rvec21 = tensor3.r_vector2[:,j].copy()
                            rvec31 = tensor3.r_vector3[:,j].copy()
                            if(np.linalg.norm(rvec21 - rvec3) < 1.0e-5 and np.linalg.norm(rvec31 - rvec2) < 1.0e-5):
                                pairs.append([i,j])
                                got[i] = True
                                got[j] = True
        if(not np.all(got)):
            for i in range(tensor3.n_R):
                if(not got[i]):
                    print(tensor3.x_r_vector2[:,i])
                    print(tensor3.x_r_vector3[:,i])
                    print('')
        else:
            print('Found all pairs')
            for ipair in range(len(pairs)):
                if(pairs[ipair][0] != pairs[ipair][1]):
                    ip = pairs[ipair][0]
                    jp = pairs[ipair][1]
                    for i in range(3*natom):
                        for iat in range(natom):
                            for jat in range(natom):
                                if(np.any(np.abs(tensor3.tensor[ip,i,3*iat:3*(iat + 1), 3*jat:3*(jat+1)] - tensor3.tensor[jp,i,3*jat:3*(jat + 1), 3*iat:3*(iat+1)].T) > 1.0e-10*np.amax(np.abs(tensor3.tensor)))):
                                    print('Permutation symmetry failed original!')
                                    print(tensor3.tensor[ip,i])
                                    print(tensor3.tensor[jp,i])
                                    print(tensor3.tensor[ip,i] - tensor3.tensor[jp,i].T)
                                    print(np.abs(tensor3.tensor[ip,i] - tensor3.tensor[jp,i].T) > 1.0e-8*np.amax(np.abs(tensor3.tensor[ip,i])))
                                    raise RuntimeError('Get me out!', np.amax(np.abs(tensor3.tensor[ip,i])))
                                    break
        new_r_vector2 = [[] for iat in range(natom**3)]
        new_r_vector3 = [[] for iat in range(natom**3)]
        new_tensor = [[] for iat in range(natom**3)]
        multiplicity = [[] for iat in range(natom**3)]
        print('Finished check. Centering...')
        for ir in range(tensor3.n_R):
            rvec2 = tensor3.r_vector2[:,ir].copy()
            rvec3 = tensor3.r_vector3[:,ir].copy()
            xvec2 = np.dot(rvec2, irsup)
            xvec3 = np.dot(rvec3, irsup)
            size2 = np.linalg.norm(rvec2)
            size20 = np.linalg.norm(rvec2)
            size3 = np.linalg.norm(rvec3)
            size30 = np.linalg.norm(rvec3)
            size = np.linalg.norm(rvec3 - rvec2) + size2 + size3
            rvec2_new = [[] for iat in range(natom**3)]
            rvec3_new = [[] for iat in range(natom**3)]
            # Find the shortest pair vectors in the mirror supercells -Far <= x <= Far
            for iat in range(natom):
                for jat in range(natom):
                    for kat in range(natom):
                        index = kat + jat*natom + natom**2*iat
                        rvec2 = tensor3.r_vector2[:,ir].copy() + positions[jat] - positions[iat]
                        rvec3 = tensor3.r_vector3[:,ir].copy() + positions[kat] - positions[iat]
                        xvec2 = np.dot(rvec2, irsup)
                        xvec3 = np.dot(rvec3, irsup)
                        size2 = np.linalg.norm(rvec2)
                        size20 = np.linalg.norm(rvec2)
                        size3 = np.linalg.norm(rvec3)
                        size30 = np.linalg.norm(rvec3)
                        for i in range(-Far,Far + 1):
                            for j in range(-Far, Far + 1):
                                for k in range(-Far, Far +1):
                                    xvec21 = xvec2 + np.array([i,j,k])
                                    rvec21 = np.dot(xvec21, rsup)
                                    size21 = np.linalg.norm(rvec21)
                                    if(size21 <= size2 + 1.0e-6):
                                        if(abs(size21 - size2) < 1.0e-6):
                                            uc_vec = tensor3.r_vector2[:,ir].copy()
                                            uc_vec = np.dot(uc_vec, irsup)
                                            uc_vec += np.array([i,j,k])
                                            uc_vec = np.dot(uc_vec, rsup)
                                            rvec2_new[index].append(uc_vec)
                                        else:
                                            uc_vec = tensor3.r_vector2[:,ir].copy()
                                            uc_vec = np.dot(uc_vec, irsup)
                                            uc_vec += np.array([i,j,k])
                                            uc_vec = np.dot(uc_vec, rsup)
                                            rvec2_new[index] = []
                                            rvec2_new[index].append(uc_vec)
                                        size2 = size21
                                        rvec2 = rvec21.copy()
                                    xvec31 = xvec3 + np.array([i,j,k])
                                    rvec31 = np.dot(xvec31, rsup)
                                    size31 = np.linalg.norm(rvec31)
                                    if(size31 <= size3 + 1.0e-6):
                                        if(abs(size31 - size3) < 1.0e-6):
                                            uc_vec = tensor3.r_vector3[:,ir].copy()
                                            uc_vec = np.dot(uc_vec, irsup)
                                            uc_vec += np.array([i,j,k])
                                            uc_vec = np.dot(uc_vec, rsup)
                                            rvec3_new[index].append(uc_vec)
                                        else:
                                            uc_vec = tensor3.r_vector3[:,ir].copy()
                                            uc_vec = np.dot(uc_vec, irsup)
                                            uc_vec += np.array([i,j,k])
                                            uc_vec = np.dot(uc_vec, rsup)
                                            rvec3_new[index] = []
                                            rvec3_new[index].append(uc_vec)
                                        size3 = size31
                                        rvec3 = rvec31.copy()
                # For each pair of the shortest pairs construct another entry to tensor3 and scale it with multiplicity
                        for iuc in range(len(rvec2_new[index])):
                            for juc in range(len(rvec3_new[index])):
                                already_there = False
                                for kuc in range(len(new_r_vector2[index])):
                                    if(np.linalg.norm(rvec2_new[index][iuc] - new_r_vector2[index][kuc]) < 1.0e-6 and \
                                            np.linalg.norm(rvec3_new[index][juc] - new_r_vector3[index][kuc]) < 1.0e-6):
                                        already_there = True
                                        print('Would double count this one!')
                                        break
                                if not already_there:
                                    new_r_vector2[index].append(rvec2_new[index][iuc])
                                    new_r_vector3[index].append(rvec3_new[index][juc])
                                    new_tensor[index].append(tensor3.tensor[ir]/float(len(rvec2_new[index])*len(rvec3_new[index])))
                                    multiplicity[index].append(len(rvec2_new[index])*len(rvec3_new[index]))
        n_R = [0 for x in range(natom**3)]
        for iat in range(natom):
            for jat in range(natom):
                for kat in range(natom):
                    n_R[kat + jat*natom +iat*natom**2] = len(new_r_vector2[kat + jat*natom + natom**2*iat])
        #Check if all n_R are equal
        if(n_R.count(n_R[0]) == len(n_R)):
            index0 = 0
        else:
            index0 = np.argsort(n_R)[-1]
            print(n_R)
            for ir in range(natom**3):
                print(len(multiplicity[ir]))
            #for ir in range(tensor3.n_R):
            #    print(multiplicity[:][ir])
            #print('Largest number of triplets for: ', index0)
            #for itrip in range(len(new_r_vector2)):
            #    if(itrip != index0):
            #        print('Checking triplet: ', itrip)
            #        for i in range(len(new_r_vector2[index0])):
            #            found = False
            #            for j in range(len(new_r_vector2[itrip])):
            #                if(np.linalg.norm(new_r_vector2[itrip][j] - new_r_vector2[index0][i]) < 1.0e-6):
            #                    found = True
            #                    break
            #            if(not found):
            #                print('For triplet: ', itrip)
            #                print('Could not find: ', np.dot(new_r_vector2[index0][i], irprim))
            #raise RuntimeError('Number of lattice vectors for all triplets is not the same! ')
        print(n_R)
        r_vector2 = new_r_vector2[index0].copy()
        r_vector3 = new_r_vector3[index0].copy()
        extra = 0
        for itrip in range(len(n_R)):
            if(itrip != index0):
                for iuc in range(len(new_r_vector2[itrip])):
                    already_there = False
                    for juc in range(len(r_vector2)):
                        if(np.linalg.norm(r_vector2[juc] - new_r_vector2[itrip][iuc]) < 1.0e-6 and \
                                np.linalg.norm(r_vector3[juc] - new_r_vector3[itrip][iuc]) < 1.0e-6):
                            already_there = True
                            break
                    if(not already_there):
                        extra += 1
                        r_vector2.append(new_r_vector2[itrip][iuc])
                        r_vector3.append(new_r_vector3[itrip][iuc])
        print('Added ' + str(extra) + ' new triplets!')
        fc3 = []
        for iuc in range(len(r_vector2)):
            fc3.append(np.zeros_like(tensor3.tensor[0]))
            for iat in range(natom):
                for jat in range(natom):
                    for kat in range(natom):
                        index = kat + jat*natom + iat*natom**2
                        found = False
                        for juc in range(len(new_r_vector2[index])):
                            if(np.linalg.norm(new_r_vector2[index][juc] - r_vector2[iuc]) < 1.0e-6 and \
                                    np.linalg.norm(new_r_vector3[index][juc] - r_vector3[iuc]) < 1.0e-6):
                                fc3[iuc][3*iat:3+3*iat,3*jat:3+3*jat,3*kat:3+3*kat] = new_tensor[index][juc][3*iat:3+3*iat,3*jat:3+3*jat,3*kat:3+3*kat]#*multiplicity[index][juc]/multiplicity[index][iuc]
                                found = True
                                break
                        #if(not found):
                            #print('Still can not find! Very weird')
                        #    match1 = False
                        #    match2 = False
                        #    for ir in range(tensor3.n_R):
                        #        rvec2 = tensor3.r_vector2[:,ir].copy() 
                        #        xvec2 = np.dot(rvec2, irsup)
                        #        rvec3 = tensor3.r_vector3[:,ir].copy() 
                        #        xvec3 = np.dot(rvec3, irsup)
                        #        for i1 in range(-1,2):
                        #            if(not match1):
                        #                for j1 in range(-1,2):
                        #                    if(not match1):
                        #                        for k1 in range(-1,2):
                        #                            xvec21 = xvec2 + np.array([i1,j1,k1])
                        #                            rvec21 = np.dot(xvec21, rsup)
                        #                            if(np.linalg.norm(rvec21 - r_vector2[iuc]) < 1.0e-6):
                        #                                match1 = True
                        #                                break
                        #        for i1 in range(-1,2):
                        #            if(not match2):
                        #                for j1 in range(-1,2):
                        #                    if(not match2):
                        #                        for k1 in range(-1,2):
                        #                            xvec31 = xvec3 + np.array([i1,j1,k1])
                        #                            rvec31 = np.dot(xvec31, rsup)
                        #                            if(np.linalg.norm(rvec31 - r_vector3[iuc]) < 1.0e-6):
                        #                                match2 = True
                        #                                break
                        #        if(match1 and match2):
                        #            print('Matched!', ir, index, float(iuc)/float(len(r_vector2)))
                        #            break
                        #    if(match1 and match2):
                        #        fc3[iuc][3*iat:3+3*iat,3*jat:3+3*jat,3*kat:3+3*kat] = tensor3.tensor[ir][3*iat:3+3*iat,3*jat:3+3*jat,3*kat:3+3*kat]/float(multiplicity[index][ir])                                    
                         #   else:
                         #       print(match1, match2)
                         #       raise RuntimeError('Could not find triplet!')
        print('Final number of triplets: ', len(r_vector2))
        tensor3.n_R = len(r_vector2)
        tensor3.r_vector2 = np.array(r_vector2).T
        tensor3.r_vector3 = np.array(r_vector3).T
        tensor3.x_r_vector2 = np.zeros_like(tensor3.r_vector2)
        tensor3.x_r_vector3 = np.zeros_like(tensor3.r_vector3)
        tensor3.tensor = np.array(fc3)
        write_fc3(tensor3)
        got = [False for x in range(tensor3.n_R)]
        pairs = []
        for i in range(tensor3.n_R):
            rvec2 = tensor3.r_vector2[:,i].copy()
            rvec3 = tensor3.r_vector3[:,i].copy()
            tensor3.x_r_vector2[:,i] = np.rint(np.dot(tensor3.r_vector2[:,i], irprim), dtype=float)
            tensor3.x_r_vector3[:,i] = np.rint(np.dot(tensor3.r_vector3[:,i], irprim), dtype=float)
            # Find pairs to check if this centering broke permutation symmetry
            if(np.linalg.norm(rvec2 - rvec3) < 1.0e-5):
                pairs.append([i,i])
                got[i] = True
            else:
                if(not got[i]):
                    for j in range(i, tensor3.n_R):
                        if(not got[j]):
                            rvec21 = tensor3.r_vector2[:,j].copy()
                            rvec31 = tensor3.r_vector3[:,j].copy()
                            if(np.linalg.norm(rvec21 - rvec3) < 1.0e-5 and np.linalg.norm(rvec31 - rvec2) < 1.0e-5):
                                pairs.append([i,j])
                                got[i] = True
                                got[j] = True
        if(not np.all(got)):
            for i in range(tensor3.n_R):
                if(not got[i]):
                    print(tensor3.x_r_vector2[:,i])
                    print(tensor3.x_r_vector3[:,i])
                    print('')
        else:
            print('Found all pairs')
            for ipair in range(len(pairs)):
                if(pairs[ipair][0] != pairs[ipair][1]):
                    ip = pairs[ipair][0]
                    jp = pairs[ipair][1]
                    for i in range(3*natom):
                        for iat in range(natom):
                            for jat in range(natom):
                                if(np.any(np.abs(tensor3.tensor[ip,i,3*iat:3*(iat + 1), 3*jat:3*(jat+1)] - tensor3.tensor[jp,i,3*jat:3*(jat + 1), 3*iat:3*(iat+1)].T) > 1.0e-10*np.amax(np.abs(tensor3.tensor)))):
                                    print('Permutation symmetry failed!')
                                    print(tensor3.tensor[ip,i])
                                    print(tensor3.tensor[jp,i])
                                    print(tensor3.tensor[ip,i] - tensor3.tensor[jp,i].T)
                                    print(np.abs(tensor3.tensor[ip,i] - tensor3.tensor[jp,i].T) > 1.0e-8*np.amax(np.abs(tensor3.tensor[ip,i])))
                                    raise RuntimeError('Get me out!', np.amax(np.abs(tensor3.tensor[ip,i])))
                                    break

    else:
        print('Probably already centered! Nothing to do!')

    return tensor3

def apply_permutation_symmetry(tensor3, pairs):
        for ipair in range(len(pairs)):
            if(pairs[ipair][0] != pairs[ipair][1]):
                ip = pairs[ipair][0]
                jp = pairs[ipair][1]
                for i in range(3*tensor3.nat):
                    for iat in range(tensor3.nat):
                        for jat in range(tensor3.nat):
                            #if(np.any(np.abs(tensor3.tensor[ip,i,3*iat:3*(iat + 1), 3*jat:3*(jat+1)] - tensor3.tensor[jp,i,3*jat:3*(jat + 1), 3*iat:3*(iat+1)].T) > 1.0e-10*np.amax(np.abs(tensor3.tensor)))):
                            tensor3.tensor[ip,i,3*iat:3*(iat + 1), 3*jat:3*(jat+1)] = (tensor3.tensor[ip,i,3*iat:3*(iat + 1), 3*jat:3*(jat+1)] + tensor3.tensor[jp,i,3*jat:3*(jat + 1), 3*iat:3*(iat+1)].T)/2.0
                            tensor3.tensor[jp,i,3*jat:3*(jat + 1), 3*iat:3*(iat+1)] = tensor3.tensor[ip,i,3*iat:3*(iat + 1), 3*jat:3*(jat+1)].T

def rotate_eigenvectors(ddm, eigs):

    _, eigvecs = np.linalg.eigh(np.dot(eigs.conj(), np.dot(ddm, eigs.T)))
    rot_eigvecs = np.dot(eigvecs.T, eigs)
    
    return rot_eigvecs

class ThermalConductivity:

    def __init__(self, dyn, tensor3, kpoint_grid = 2, scattering_grid = None, smearing_scale = 1.0, smearing_type = 'adaptive', cp_mode = 'quantum', off_diag = False):

        """

        This class contains necesary routines to calculate lattice thermal conductivity using SSCHA auxiliary 2nd and 3rd order force constants.

        Parameters:

        Necesary:

            dyn            : SSCHA dynamical matrix object 
            tensor3        : SSCHA 3rd order force constants

            kpoint_grid    : Initializes the grid for Brillouin zone integration. It is used in the calculation of lattice thermal conductivity and
            the calculation of the phonon lifetimes. Default is 2.
            smearing_scale : Scale for the smearing constant if adaptive smearing is used. Default value is 2.0
            smearing_type  : Type of smearing used. Could be constant (same for all phonon modes) or adaptive (scaled by the phonon group velocity and the q point density).
            cp_mode        : Flag determining how phonon occupation factors are calculated (quantum/classical), default is quantum
            off_diag       : Boolean parameter for the calculation of the off-diagonal elements of group velocity. 

        """

        self.dyn = dyn
        self.fc3 = tensor3
        if(isinstance(kpoint_grid, int)):
            self.kpoint_grid = np.array([kpoint_grid for x in range(3)])
        else:
            self.kpoint_grid = np.array(kpoint_grid).astype(int)
        if(scattering_grid is not None):
            if(isinstance(scattering_grid, int)):
                self.scattering_grid = np.array([scattering_grid for x in range(3)])
            else:
                self.scattering_grid = np.array(scattering_grid).astype(int)
        else:
            self.scattering_grid = self.kpoint_grid.copy()
        self.smearing_scale = smearing_scale
        self.unitcell = self.dyn.structure.unit_cell
        print('Primitive cell: ')
        print(self.unitcell)
        self.supercell = self.dyn.structure.generate_supercell(dyn.GetSupercell()).unit_cell
        print('Supercell: ')
        print(self.supercell)
        self.smearing_type = smearing_type
        self.cp_mode = cp_mode
        self.off_diag = off_diag
        self.volume = self.dyn.structure.get_volume()
        
        self.reciprocal_lattice = np.linalg.inv(self.unitcell).T
        self.force_constants = []
        self.ruc = []
        self.set_force_constants(dyn) # uncentered force constants !
        self.nuc = len(self.ruc)

        self.symmetry = symmetries.QE_Symmetry(self.dyn.structure)

        self.set_kpoints_alternative()
        self.nband = 3*self.dyn.structure.N_atoms
        self.delta_omega = 0.0

        self.freqs = np.zeros((self.nkpt, self.nband))
        self.gruneisen = np.zeros((self.nkpt, self.nband))
        self.eigvecs = np.zeros((self.nkpt, self.nband, self.nband), dtype=complex)
        if(self.off_diag):
            self.gvels = np.zeros((self.nkpt, self.nband, self.nband, 3))
        else:
            self.gvels = np.zeros((self.nkpt, self.nband, 3))
        self.sigmas = np.zeros_like(self.freqs)
        # Lifetimes, frequency shifts, lineshapes, heat_capacities and thermal conductivities are stored in dictionaries
        # Dictionary key is the temperature at which property is calculated on
        self.lifetimes = {}
        self.freqs_shifts = {}
        self.lineshapes = {}
        self.cp = {}
        self.kappa = {}
        self.got_scattering_rates_isotopes = False

    ##################################################################################################################################

    def save(self, filename = 'sscha_thermal_conductivity.h5'):

        """
            Routine to save most of the information needed for further postprocessing.
            
            filename : Title of the file the information is to be stored to
        
        """

        hf = h5py.File(filename, 'w')

        ne = None

        if(self.smearing_scale is not None):
            hf.create_dataset('smearing_scale', data = np.array([self.smearing_scale]))
        hf.create_dataset('kpoint_grid', data = self.kpoint_grid)
        hf.create_dataset('scattering_grid', data = self.scattering_grid)
        hf.create_dataset('unit_cell', data = self.unitcell)
        hf.create_dataset('supercell', data = self.supercell)
        dt = h5py.special_dtype(vlen=str)
        hf.create_dataset('smearing_type', data = self.smearing_type, dtype=dt)
        hf.create_dataset('cp_mode', data = self.cp_mode)
        hf.create_dataset('off_diag', data = self.off_diag)
        hf.create_dataset('reciprocal_lattice', data = self.reciprocal_lattice)
        hf.create_dataset('k_points', data = self.k_points)
        hf.create_dataset('qpoints', data = self.qpoints)
        hf.create_dataset('delta_omega', data = self.delta_omega)
        hf.create_dataset('nkpt', data = self.nkpt)
        hf.create_dataset('nband', data = self.nband)
        hf.create_dataset('nirrkpt', data = self.nirrkpt)
        
        irrqpts = [hf.create_group('irreducible_kpoint' + str(i + 1)) for i in range(self.nirrkpt)]
        for ikpt in range(self.nirrkpt):
            irrqpts[ikpt].create_dataset('k_point', data = self.irr_k_points[ikpt])
            irrqpts[ikpt].create_dataset('star', data = self.qstar_list[ikpt])
            irrqpts[ikpt].create_dataset('frequency', data = self.freqs[self.qstar_list[ikpt][0]])
            eigvecs = []
            gvels = []
            sigmas = []
            for istar in range(len(self.qstar[ikpt])):
                jkpt = self.qstar_list[ikpt][istar]
                eigvecs.append(self.eigvecs[jkpt])
                gvels.append(self.gvels[jkpt])
                sigmas.append(self.sigmas[jkpt])
            eigvecs = np.array(eigvecs)
            gvels = np.array(gvels)
            sigmas = np.array(sigmas)
            irrqpts[ikpt].create_dataset('eigenvectors', data = eigvecs)
            irrqpts[ikpt].create_dataset('group_velocities', data = gvels)
            irrqpts[ikpt].create_dataset('sigmas', data = sigmas)
            keys = []
            for key in self.lifetimes.keys():
                keys.append(key)
            if(len(keys) > 0):
                for ik in range(len(keys)):
                    irrqpts[ikpt].create_dataset('lifetimes_' + keys[ik], data = self.lifetimes[keys[ik]][self.qstar_list[ikpt][0]])
            keys = []
            for key in self.freqs_shifts.keys():
                keys.append(key)
            if(len(keys) > 0):
                for ik in range(len(keys)):
                    irrqpts[ikpt].create_dataset('freqs_shifts_' + keys[ik], data = self.freqs_shifts[keys[ik]][self.qstar_list[ikpt][0]])
            keys = []
            for key in self.lineshapes.keys():
                keys.append(key)
            if(len(keys) > 0):
                for ik in range(len(keys)):
                    if(ne is None):
                        ne = np.shape(self.lineshapes[keys[ik]])[-1]
                        hf.create_dataset('ne', data = ne)
                    else:
                        if(ne != np.shape(self.lineshapes[keys[ik]])[-1]):
                            raise RuntimeError('Number of energy/frequency points not same for all temperatures!')
                    irrqpts[ikpt].create_dataset('lineshapes_' + keys[ik], data = self.lineshapes[keys[ik]][self.qstar_list[ikpt][0]])
            keys = []
            for key in self.cp.keys():
                keys.append(key)
            if(len(keys) > 0):
                for ik in range(len(keys)):
                    irrqpts[ikpt].create_dataset('cp_' + keys[ik], data = self.cp[keys[ik]][self.qstar_list[ikpt][0]])
        keys = []
        for key in self.kappa.keys():
            keys.append(key)
        if(len(keys) > 0):
            for ik in range(len(keys)):
                hf.create_dataset('kappa_' + keys[ik], data = self.kappa[keys[ik]])
        hf.close()

    def load(self, filename):

        """
            Routine to read the information that one might need for postprocessing!

        """

        hf = h5py.File(filename, 'r')

        try:
            self.smearing_scale = np.array(hf.get('smearing_scale'))
        except:
            pass
        self.kpoint_grid = np.array(hf.get('kpoint_grid'))
        self.scattering_grid = np.array(hf.get('scattering_grid'))
        self.unitcell = np.array(hf.get('unit_cell'))
        self.supercell = np.array(hf.get('supercell'))
        self.smearing_type = np.array2string(np.array(hf.get('smearing_type')))[2:-1]
        self.cp_mode = np.array2string(np.array(hf.get('cp_mode')))[2:-1]
        self.off_diag = np.array(hf.get('off_diag')).item()
        self.reciprocal_lattice = np.array(hf.get('reciprocal_lattice'))
        self.k_points = np.array(hf.get('k_points'))
        self.qpoints = np.array(hf.get('qpoints'))
        self.delta_omega = np.array(hf.get('delta_omega')).item()
        self.nkpt = np.array(hf.get('nkpt')).item()
        self.nband = np.array(hf.get('nband')).item()
        self.nirrkpt = np.array(hf.get('nirrkpt')).item()
        ne = np.array(hf.get('ne')).item()
        self.freqs = np.zeros((self.nkpt, self.nband))
        self.gruneisen = np.zeros((self.nkpt, self.nband))
        self.eigvecs = np.zeros((self.nkpt, self.nband, self.nband), dtype=complex)
        if(self.off_diag):
            self.gvels = np.zeros((self.nkpt, self.nband, self.nband, 3))
        else:
            self.gvels = np.zeros((self.nkpt, self.nband, 3))
        self.sigmas = np.zeros_like(self.freqs)

        self.irr_k_points = []
        self.qstar_list = []
        for i in range(self.nirrkpt):
            self.irr_k_points.append(np.array(hf.get('irreducible_kpoint' + str(i + 1)).get('k_point')))
            self.qstar_list.append(np.array(hf.get('irreducible_kpoint' + str(i + 1)).get('star')))
            eigvecs = np.array(hf.get('irreducible_kpoint' + str(i + 1)).get('eigenvectors'))
            gvels = np.array(hf.get('irreducible_kpoint' + str(i + 1)).get('group_velocities'))
            sigmas = np.array(hf.get('irreducible_kpoint' + str(i + 1)).get('sigmas'))

            for istar in range(len(self.qstar_list[-1])):
                jkpt = self.qstar_list[-1][istar]
                self.freqs[jkpt] = np.array(hf.get('irreducible_kpoint' + str(i + 1)).get('frequency'))
                self.eigvecs[jkpt] = eigvecs[istar]
                self.gvels[jkpt] = gvels[istar]
                self.sigmas[jkpt] = sigmas[istar]
                keys = []
                for key in hf.get('irreducible_kpoint' + str(i + 1)).keys():
                    keys.append(key)
                for key in keys:
                    if('cp' in key):
                        temp = key.split('_')[-1]
                        if(temp not in self.cp.keys()):
                            self.cp[temp] = np.zeros((self.nkpt, self.nband))
                        self.cp[temp][jkpt] = np.array(hf.get('irreducible_kpoint' + str(i + 1)).get(key))
                    elif('lifetimes' in key):
                        temp = key.split('_')[-1]
                        if(temp not in self.lifetimes.keys()):
                            self.lifetimes[temp] = np.zeros((self.nkpt, self.nband))
                        self.lifetimes[temp][jkpt] = np.array(hf.get('irreducible_kpoint' + str(i + 1)).get(key))
                    elif('freqs_shifts' in key):
                        temp = key.split('_')[-1]
                        if(temp not in self.lifetimes.keys()):
                            self.freqs_shifts[temp] = np.zeros((self.nkpt, self.nband))
                        self.freqs_shifts[temp][jkpt] = np.array(hf.get('irreducible_kpoint' + str(i + 1)).get(key))
                    elif('lineshapes' in key):
                        temp = key.split('_')[-1]
                        if(temp not in self.lineshapes.keys()):
                            print('Reading')
                            self.lineshapes[temp] = np.zeros((self.nkpt, self.nband, ne))
                        self.lineshapes[temp][jkpt] = np.array(hf.get('irreducible_kpoint' + str(i + 1)).get(key))
                    
        for key in hf.keys():
            if('kappa' in key):
                temp = key.split('_')[-1]
                self.kappa[temp] = np.array(hf.get(key))

############################################################################################################################################

    def set_kpoints(self):

        """
        Sets up the k point grid. Finds irreducible points and their map to the full Brillouin zone.

        """

        time0 = time.time()
        time_init = time.time()
        self.k_points = CC.symmetries.GetQGrid(self.dyn.structure.unit_cell, self.kpoint_grid)
        self.qpoints = np.dot(np.array(self.k_points), self.unitcell.T)
        self.scattering_k_points = CC.symmetries.GetQGrid(self.dyn.structure.unit_cell, self.scattering_grid)
        self.scattering_qpoints = np.dot(np.array(self.scattering_k_points), self.unitcell.T)
        print('Generated grid in ' + format(time.time() - time0, '.2e') + ' seconds.')
        time0 = time.time()
        self.irr_k_points = self.symmetry.SelectIrreducibleQ(self.k_points) 
        print('Generated irreducible grid in ' + format(time.time() - time0, '.2e') + ' seconds.')
        time0 = time.time()
        self.qstar = []
        for iqpt in range(len(self.irr_k_points)):
            self.qstar.append(self.symmetry.GetQStar(self.irr_k_points[iqpt]))
        print('Generated q stars in ' + format(time.time() - time0, '.2e') + ' seconds.')
        time0 = time.time()
        self.qstar_list = []
        for iqstar in range(len(self.qstar)):
            curr_qstar_list = []
            for iqpt in range(len(self.qstar[iqstar])):
                found = False
                for jqpt in range(len(self.k_points)):
                    if(same_vector(np.array(self.qstar[iqstar][iqpt]), np.array(self.k_points[jqpt]), self.reciprocal_lattice)):
                        found = True
                        curr_qstar_list.append(jqpt)
                        break
                if(not found):
                    print('Could not find this q point in self.k_points!')
            self.qstar_list.append(curr_qstar_list)
        print('Generated g star list in ' + format(time.time() - time0, '.2e') + ' seconds.')
        time0 = time.time()
        #print(self.qstar_list)

        self.nkpt = np.shape(self.k_points)[0]
        self.scattering_nkpt = np.shape(self.scattering_k_points)[0]
        self.nirrkpt = np.shape(self.irr_k_points)[0]
        self.weights = np.zeros(self.nirrkpt, dtype = int)
        for iqpt in range(self.nirrkpt):
            self.weights[iqpt] = len(self.qstar_list[iqpt])
        print('Found ' + str(self.nkpt) + ' q points in the grid!')
        print('Found ' + str(self.nirrkpt) + ' q points in irreducible grid!')
        print('Setting up q points took ' + format(time.time() - time_init, '.2e') + ' seconds!')

    ##################################################################################################################################
    
    def set_kpoints_alternative(self):

        """
        Faster version of setting up of k points. Still too slow. Should try fortran version.

        """

        time0 = time.time()
        time_init = time.time()

        aux_kgrid = CC.symmetries.GetQGrid(self.dyn.structure.unit_cell, self.kpoint_grid)
        print('Generated grid in ' + format(time.time() - time0, '.2e') + ' seconds.')
        time0 = time.time()
        self.irr_k_points = self.symmetry.SelectIrreducibleQ(aux_kgrid) #self.symmetry.SetupQStar(self.k_points)
        print('Generated irreducible grid in ' + format(time.time() - time0, '.2e') + ' seconds.')
        time0 = time.time()
        self.qstar = []
        for iqpt in range(len(self.irr_k_points)):
            self.qstar.append(self.symmetry.GetQStar(self.irr_k_points[iqpt]))
        print('Generated q star in ' + format(time.time() - time0, '.2e') + ' seconds.')
        time0 = time.time()
        k_points = []
        self.qstar_list = []
        qpt_index = 0
        for iqpt in range(len(self.qstar)):
            curr_qstar_list = []
            for jqpt in range(len(self.qstar[iqpt])):
                curr_qstar_list.append(qpt_index)
                qpt_index += 1
                k_points.append(self.qstar[iqpt][jqpt])
            self.qstar_list.append(curr_qstar_list)
        print('Generated q star list in ' + format(time.time() - time0, '.2e') + ' seconds.')
        time0 = time.time()
        self.k_points = np.array(k_points)
        self.qpoints = np.dot(np.array(self.k_points), self.unitcell.T)
        self.scattering_k_points = CC.symmetries.GetQGrid(self.dyn.structure.unit_cell, self.scattering_grid)
        self.scattering_qpoints = np.dot(np.array(self.scattering_k_points), self.unitcell.T)
        self.nkpt = np.shape(self.k_points)[0]
        self.scattering_nkpt = np.shape(self.scattering_k_points)[0]
        self.nirrkpt = np.shape(self.irr_k_points)[0]
        self.weights = np.zeros(self.nirrkpt, dtype = int)
        for iqpt in range(self.nirrkpt):
            self.weights[iqpt] = len(self.qstar_list[iqpt])
        self.set_up_scattering_grids = False
        print('Found ' + str(self.nkpt) + ' q points in the grid!')
        print('Found ' + str(self.nirrkpt) + ' q points in irreducible grid!')
        print('Setting up q points took ' + format(time.time() - time_init, '.2e') + ' seconds!')

    ###################################################################################################################################

    def set_scattering_grids(self):

        """

        Setup scattering grids for each k - point. Only permutation symmetry is found to be stable. Python version - very slow.

        """

        start_time = time.time()
        cell = get_spglib_cell(self.dyn)
        tot_r = spglib.get_symmetry_dataset(cell)['rotations']
        nsym = len(tot_r)

        scatt_grids = []
        weights = []
        for iqpt in range(self.nirrkpt):
            curr_grid = []
            curr_w = []
            rot_q = []
            q1 = self.qpoints[self.qstar_list[iqpt][0]]
            for isym in range(nsym):
                if(same_vector(q1, np.dot(tot_r[isym], q1), np.eye(3))):
                    rot_q.append(tot_r[isym])
            curr_nsym = len(rot_q)
            print('Small group of ' + str(iqpt) + '. q point is ' + str(curr_nsym) + ' size!')
            for jqpt in range(self.nkpt):
                q2 = self.qpoints[jqpt]
                q3 = -1.0*q1 - q2
                pair_found = False
                for i in range(len(curr_grid)):
                    if(not pair_found):
                        #if(same_vector(q3, curr_grid[i][0], self.reciprocal_lattice) and same_vector(q2, curr_grid[i][1], self.reciprocal_lattice)):
                        if(same_vector(q3, curr_grid[i][0], np.eye(3)) and same_vector(q2, curr_grid[i][1], np.eye(3))):
                                pair_found = True
                                curr_w[i] += 1
                                break
                        else:
                            for isym in range(curr_nsym):
                                #if(same_vector(q3, np.dot(rot_q[isym], curr_grid[i][0]), self.reciprocal_lattice) and\
                                #    same_vector(q2, np.dot(rot_q[isym], curr_grid[i][1]), self.reciprocal_lattice)):
                                if(same_vector(q3, np.dot(rot_q[isym], curr_grid[i][0]), np.eye(3)) and\
                                    same_vector(q2, np.dot(rot_q[isym], curr_grid[i][1]), np.eye(3))):
                                        pair_found = True
                                        curr_w[i] += 1
                                        break
                                elif(same_vector(q3, np.dot(rot_q[isym], curr_grid[i][1]), np.eye(3)) and\
                                    same_vector(q2, np.dot(rot_q[isym], curr_grid[i][0]), np.eye(3))):
                                        pair_found = True
                                        curr_w[i] += 1
                                        break
                if(not pair_found):
                    curr_grid.append([q2, q3])
                    curr_w.append(1)
            scatt_grids.append(curr_grid)
            weights.append(curr_w)
        self.scattering_grids = []
        self.scattering_weights = []
        for iqpt in range(self.nirrkpt):
            if(sum(weights[iqpt]) != self.nkpt):
                print('WARNING! Sum of weights for ' + str(iqpt) + '. q point does not match total number of q points!')
                print(sum(weights[iqpt]), self.nkpt)
            curr_grid = np.array(scatt_grids[iqpt])
            self.scattering_grids.append(np.dot(curr_grid[:,0,:], self.reciprocal_lattice))
            self.scattering_weights.append(weights[iqpt])
            print('Number of scattering events for ' + str(iqpt + 1) + '. q point in irr zone is ' + str(len(self.scattering_grids[iqpt])) + '!')
        self.set_up_scattering_grids = True
        print('Set up scattering grids in ' + format(time.time() - start_time, '.1f') + ' seconds.')

    ###################################################################################################################################

    def set_scattering_grids_fortran(self):

        """

        Setup scattering grids by calling fortran routine. Much faster.

        """

        start_time = time.time()
        cell = get_spglib_cell(self.dyn)
        tot_r = spglib.get_symmetry_dataset(cell)['rotations']
        tot_t = spglib.get_symmetry_dataset(cell)['translations']
        rotations = []
        for i in range(len(tot_r)):
        #    print(tot_t[i])
            if(np.all(tot_t[i] < 1.0e-6)):
                rotations.append(tot_r[i])
        rotations = np.asfortranarray(rotations)
        nsym = len(rotations)
       # print(nsym)
       
        irrgrid = []
        for iqpt in range(self.nirrkpt):
            irrgrid.append(self.qpoints[self.qstar_list[iqpt][0]])
        irrgrid = np.asfortranarray(irrgrid)
        (scattering_grid, scattering_weight) = thermal_conductivity.scattering_grids.get_scattering_q_grid(rotations, irrgrid, self.qpoints, \
                self.scattering_qpoints, self.nirrkpt, self.nkpt, self.scattering_nkpt, nsym)
        self.scattering_grids = []
        self.scattering_weights = []
        for iqpt in range(self.nirrkpt):
            curr_grid = []
            curr_w = []
            for jqpt in range(self.scattering_nkpt):
                if(scattering_weight[iqpt][jqpt] > 0):
                    curr_grid.append(np.dot(scattering_grid[iqpt][jqpt], self.reciprocal_lattice))
                    curr_w.append(scattering_weight[iqpt][jqpt])
                else:
                    break
            self.scattering_grids.append(curr_grid)
            self.scattering_weights.append(curr_w)
            if(sum(curr_w) != self.scattering_nkpt):
                print('WARNING! Sum of weights for ' + str(iqpt + 1) + '. q point does not match total number of q points!')
                print(sum(curr_w), self.scattering_nkpt)
            print('Number of scattering events for ' + str(iqpt + 1) + '. q point in irr zone is ' + str(len(self.scattering_grids[iqpt])) + '!')
        self.set_up_scattering_grids = True
        print('Set up scattering grids in ' + format(time.time() - start_time, '.1f') + ' seconds.')
                
    ###################################################################################################################################

    def set_scattering_grids_simple(self):

        """

        Set scattering grids to be all points in the Brillouin zone.

        """

        self.scattering_grids = []
        self.scattering_weights = []
        start_time = time.time()
        for iqpt in range(self.nirrkpt):
            self.scattering_grids.append(self.k_points)
            self.scattering_weights.append(np.ones(self.nkpt))
            if(sum(self.scattering_weights[-1]) != self.nkpt):
                print('WARNING! Sum of weights for ' + str(iqpt + 1) + '. q point does not match total number of q points!')
                print(sum(self.scattering_weights[-1]), self.nkpt)
            print('Number of scattering events for ' + str(iqpt + 1) + '. q point in irr zone is ' + str(len(self.scattering_grids[iqpt])) + '!')
        self.set_up_scattering_grids = True
        print('Set up scattering grids in ' + format(time.time() - start_time, '.1f') + ' seconds.')
        

    ####################################################################################################################################

    def set_force_constants(self, dyn):

        """
        Translates the SSCHA second order tensor in a local version which is used for calculation of harmonic properties.

        dyn : Cellconstructor.Phonons() object
        """

        dyn.Symmetrize()
        self.fc2 = CC.ForceTensor.Tensor2(dyn.structure, dyn.structure.generate_supercell(dyn.GetSupercell()), dyn.GetSupercell())
        self.fc2.SetupFromPhonons(self.dyn)
        self.fc2.Center()
        self.fc2.Apply_ASR()

        self.force_constants = self.fc2.tensor.copy()
        self.ruc = self.fc2.r_vector2.T
        invcell = np.linalg.inv(self.supercell)
        

   ###################################################################################################################################

    def setup_smearings(self, smearing_value = 0.00005):

        """
        Sets up smearing factor for each phonon mode.

        if type == adaptive : smearing for each phonon mode is unique and depends on the group velocity of the mode and the k point density.
                              For modes with zero velocity, take value 10 times smaller than the average smearing.
        if type == constant : smearing is same for all of the modes 

        smearing_value : value of the smearing in case type == "constant"

        """

        if(self.smearing_type == 'adaptive'):
            if(np.all(self.freqs == 0.0) or np.all(self.gvels == 0.0)):
                print('Harmonic properties are all zero! Try reruning the harmonic calculation! ')
                return 1
            delta_q = 0.0
            for i in range(len(self.reciprocal_lattice)):
                if(np.linalg.norm(self.reciprocal_lattice[i]/float(self.kpoint_grid[i])) > delta_q):
                    delta_q = np.linalg.norm(self.reciprocal_lattice[i]/float(self.kpoint_grid[i]))

            for ikpt in range(self.nkpt):
                for iband in range(self.nband):
                    delta_v = np.linalg.norm(self.gvels[ikpt,iband])
                    if(delta_v > 0.0):
                        self.sigmas[ikpt][iband] = delta_v*delta_q*self.smearing_scale
                    else:
                        self.sigmas[ikpt][iband] = 0.0
            min_smear = np.amax(self.sigmas)/100.0 #np.amin(self.sigmas[self.sigmas > 1.0e-6])
            #print(min_smear)
            self.sigmas[self.sigmas < min_smear] = min_smear
            #self.sigmas[self.freqs <= np.amin(self.freqs)*10] = min_smear/10.0
        if(self.smearing_type == 'constant'):
            self.sigmas[:,:] = smearing_value

    ##################################################################################################################################

    def what_temperatures(self):

        """
        Routine to print which temperatures have already been calculated.

        """

        print('Heat capacities are calculated for: ')
        print(self.cp.keys())
        print('Phonon lifetimes are calculated for: ')
        print(self.lifetimes.keys())
        print('Phonon lineshapes are calculated for: ')
        print(self.lineshapes.keys())

    ###################################################################################################################################

    def get_spectral_kappa(self, temperature, ne = 100, prefered = 'lineshape'):

        """
        Routine to calculate frequency resolved lattice thermal conductivity $\kappa (\omega)$
        ne : number of frequency points in case we calculate spectral kappa from phonon lifetimes (in case of lineshapes we use energies defined for the calculation of lineshapes)
        prefered : in case both lifetimes and lineshapes are calculated defines which method to use to calculate spectral kappa
        """

        if(not self.off_diag):
            temp_key = format(temperature, '.1f')
            if(temp_key in self.lifetimes.keys() and temp_key in self.lineshapes.keys()):
                if(prefered == 'lineshape'):
                    print('Both lineshapes and lifetimes are calculated. Calculating spectral kappa from lineshapes!')
                    spec_kappa = self.calc_spectral_kappa_gk_diag(self, temperature)
                else:
                    print('Both lineshapes and lifetimes are calculated. Calculating spectral kappa from lifetimes!')
                    spec_kappa = self.calc_spectral_kappa_srta_diag(self, temperature, ne)
            elif(temp_key in self.lifetimes.keys()):
                spec_kappa = self.calc_spectral_kappa_srta_diag(self, temperature, ne)
            elif(temp_key in self.lineshapes.keys()):
                spec_kappa = self.calc_spectral_kappa_gk_diag(self, temperature)
            else:
                print('You have not calculated phonon lifetimes or lineshapes for this temperature. Can not calculate spectral kappa!')
                spec_kappa = (np.zeros(ne), np.zeros(ne))
        else:
            print('Calculation with offdiagonal elements have been initialized. Not possible to calculate spectral kappa! ')
            spec_kappa = (np.zeros(ne), np.zeros(ne))

        return spec_kappa

    ###################################################################################################################################

    def calc_spectral_kappa_gk_diag(self, temperature):

        """
        Calculate spectral kappa from lineshapes.

        temperature : temperature at which to calculate spectral kappa. Lineshapes should already been calculated.
        """

        ls_key = format(temperature, '.1f')
        spec_kappa = np.zeros((3,3,self.lineshapes[ls_key].shape[-1]))
        energies = np.arange(spec_kappa.shape[-1], dtype=float)*self.delta_omega + self.delta_omega
        exponents = np.exp(energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        integrands_plus = self.lineshapes[ls_key]**2*energies**2*exponents/(exponents - 1.0)**2
        exponents = np.exp(-1.0*energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        integrands_minus = self.lineshapes[ls_key]**2*energies**2*exponents/(exponents - 1.0)**2
        integrands = (integrands_plus + integrands_minus)
        
        if(self.off_diag):
            spec_kappa = np.einsum('ijjk,ijjl,ijm->klm', self.gvels, self.gvels,integrands)*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        else:
            spec_kappa = np.einsum('ijk,ijl,ijm->klm', self.gvels, self.gvels,integrands)*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        spec_kappa = spec_kappa*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi*(SSCHA_TO_THZ*2.0*np.pi)*1.0e12/2.0
        tot_kappa = np.sum(spec_kappa, axis = len(spec_kappa) - 1)*self.delta_omega
        print('Total kappa is: ', np.diag(tot_kappa))

        return energies*SSCHA_TO_THZ, spec_kappa/SSCHA_TO_THZ

    ##################################################################################################################################

    def calc_spectral_kappa_srta_diag(self, temperature, ne):

        """
        Calculate spectral kappa from lifetimes.

        temperature : temperature at which spectral kappa should be calculated.
        ne          : number of frequency points for spectal kappa to be calculated on.

        """

        key = format(temperature, '.1f')
        delta_en = np.amax(self.freqs)/float(ne)
        energies = np.arange(ne, dtype=float)*delta_en + delta_en
        spec_kappa = np.zeros((3,3,ne))
        for ien in range(ne):
            for iqpt in range(self.nkpt):
                for iband in range(self.nband):
                    weight = gaussian(energies[ien], self.freqs[iqpt, iband], self.sigmas[iqpt, iband])
                    if(self.off_diag):
                        spec_kappa[:,:,ien] += self.cp[key][iqpt,iband]*self.lifetimes[key][iqpt,iband]*weight*np.outer(self.gvels[iqpt,iband,iband], self.gvels[iqpt,iband, iband])
                    else:
                        spec_kappa[:,:,ien] += self.cp[key][iqpt,iband]*self.lifetimes[key][iqpt,iband]*weight*np.outer(self.gvels[iqpt,iband], self.gvels[iqpt,iband])
        spec_kappa = spec_kappa*SSCHA_TO_MS**2/self.volume/float(self.nkpt)*1.0e30#*(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        tot_kappa = np.sum(spec_kappa, axis = len(spec_kappa) - 1)*delta_en
        print('Total kappa is: ', np.diag(tot_kappa))

        return energies*SSCHA_TO_THZ, spec_kappa/SSCHA_TO_THZ

   ####################################################################################################################################

    def calculate_kappa(self, temperatures = [300.0], write_lifetimes = True, mode = 'SRTA', gauss_smearing = False, lf_method = 'fortran-LA', isotope_scattering = False, isotopes = None, \
            write_lineshapes=False, ne = 2000, kappa_filename = 'Thermal_conductivity'):

        """
        Main function that calculates lattice thermal conductivity.

        temperatures     : list of temperatures to be calculated in K. Warning: Code does not recognize difference in temperatures smaller than 0.1 K.
        write_lifetimes  : Boolean parameter for writing lifetimes as they are being calculated.
        mode             : Method to calculate lattice thermal conductivity:
            SRTA         : Single relaxation time approximation (NOT selfconsistent solution) solution of Boltzmann transport equation
            GK           : Green-Kubo method (npj Computational Materials volume 7, Article number: 57 (2021))
        gauss_smearing   : If true will use the Gaussian function to satisfy energy conservation insted of Lorentzian
        lf_method        : In case of mode == SRTA, specifies the way to calculate lifetimes. See method in get_lifetimes function.
        write_lineshapes : Boolean parameter to write phonon lineshapes as they are being calculated.
        ne               : Number of frequency points to calculate phonon lineshapes on in case of GK. \
                           Number of frequency points to solve self-consistent equation on in case of SRTA. \
                           Less anharmonic materials and lower temperatures will need more points (in case of GK).
        kappa_filename   : Name of the file to write the results to.
        """

        start_time = time.time()
        ntemp = len(temperatures)
        if(ntemp == 0):
            print('The temperature is not specified!')
            return 1

        kappa_file = open(kappa_filename, 'w+')
        tot_kappa = []
        if(not self.off_diag):
            kappa_file.write('#  ' + format('Temperature (K)', STR_FMT))
            kappa_file.write('   ' + format('Kappa xx (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa yy (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa zz (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa xy (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa yz (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa zx (W/mK)', STR_FMT))
            kappa_file.write('\n')
        else:
            kappa_file.write('#  ' + format('Temperature (K)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_diag xx (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_diag yy (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_diag zz (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_diag xy (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_diag yz (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_diag zx (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_offdiag xx (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_offdiag yy (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_offdiag zz (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_offdiag xy (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_offdiag yz (W/mK)', STR_FMT))
            kappa_file.write('   ' + format('Kappa_offdiag zx (W/mK)', STR_FMT))
            kappa_file.write('\n')
        
        for itemp in range(ntemp):
            tc_key = format(temperatures[itemp], '.1f')
            if(mode == 'SRTA'):
                if(not self.off_diag):
                    kappa = self.calculate_kappa_srta_diag(temperatures[itemp], ne, write_lifetimes, gauss_smearing = gauss_smearing, isotope_scattering=isotope_scattering, isotopes= isotopes, lf_method = lf_method)
                    kappa = kappa/self.volume/float(self.nkpt)*1.0e30
                    kappa_file.write(3*' ' + format(temperatures[itemp], '.12e'))
                    for icart in range(3):
                        kappa_file.write(3*' ' + format(kappa[icart][icart], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[0][1], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[1][2], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[2][0], '.12e'))
                    kappa_file.write('\n')
                    self.kappa[tc_key] = kappa
                else:
                    kappa_diag, kappa_nondiag = self.calculate_kappa_srta_offdiag(temperatures[itemp], ne, write_lifetimes, gauss_smearing = gauss_smearing, isotope_scattering=isotope_scattering, isotopes=isotopes, lf_method = lf_method)
                    kappa_diag = kappa_diag/self.volume/float(self.nkpt)*1.0e30
                    kappa_nondiag = kappa_nondiag/self.volume/float(self.nkpt)*1.0e30
                    kappa_file.write(3*' ' + format(temperatures[itemp], '.12e'))
                    for icart in range(3):
                        kappa_file.write(3*' ' + format(kappa_diag[icart][icart], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_diag[0][1], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_diag[1][2], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_diag[2][0], '.12e'))
                    for icart in range(3):
                        kappa_file.write(3*' ' + format(kappa_nondiag[icart][icart], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_nondiag[0][1], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_nondiag[1][2], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_nondiag[2][0], '.12e'))
                    kappa_file.write('\n')
                    self.kappa[tc_key] = kappa_diag + kappa_nondiag 
            elif(mode == 'GK'):
                self.delta_omega = np.amax(self.freqs)*2.0/float(ne)
                energies = np.arange(ne, dtype=float)*self.delta_omega + self.delta_omega
                if(not self.off_diag):
                    kappa = self.calculate_kappa_gk_diag(temperatures[itemp], write_lineshapes, energies, gauss_smearing = gauss_smearing)
                    kappa_file.write(3*' ' + format(temperatures[itemp], '.12e'))
                    for icart in range(3):
                        kappa_file.write(3*' ' + format(kappa[icart][icart], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[0][1], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[1][2], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[2][0], '.12e'))
                    kappa_file.write('\n')
                    self.kappa[tc_key] = kappa
                else:
                    kappa_diag, kappa_nondiag = self.calculate_kappa_gk_offdiag(temperatures[itemp], write_lineshapes, energies, gauss_smearing = gauss_smearing)
                    kappa_file.write(3*' ' + format(temperatures[itemp], '.12e'))
                    for icart in range(3):
                        kappa_file.write(3*' ' + format(kappa_diag[icart][icart], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_diag[0][1], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_diag[1][2], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_diag[2][0], '.12e'))
                    for icart in range(3):
                        kappa_file.write(3*' ' + format(kappa_nondiag[icart][icart], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_nondiag[0][1], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_nondiag[1][2], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_nondiag[2][0], '.12e'))
                    kappa_file.write('\n')
                    self.kappa[tc_key] = kappa_diag + kappa_nondiag 
            else:
                print('Can not recognize this method of calculating kappa! ')
                print(mode)
                kappa_file.close()
                return 1

        kappa_file.close()
        print('Calculated lattice thermal conductivity in ', time.time() - start_time, ' seconds!')

    ##################################################################################################################################

    def calculate_kappa_gk_diag(self, temperature, write_lineshapes, energies, gauss_smearing = False):

        """

        Calculation of lattice thermal conductivity using Green-Kubo method if only diagonal group velocities are available.

        temperature      : temperature at which kappa should be calculated.
        write_lineshapes : Boolean noting should we write phonon lineshapes on a file
        energies         : frequency points at which phonon lineshapes should be calculated

        """

        ls_key = format(temperature, '.1f')
        if(ls_key in self.lineshapes.keys()):
            print('Lineshapes for this temperature have already been calculated. Continuing ...')
        else:
            self.get_lineshapes(temperature, write_lineshapes, energies, method = 'fortran', gauss_smearing = gauss_smearing)
        kappa = 0.0
        exponents = np.exp(energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        #integrands_plus = self.lineshapes[ls_key]**2*energies**2*exponents/(exponents - 1.0)**2
        integrands_plus = self.lineshapes[ls_key]**2*exponents/(exponents - 1.0)**2
        exponents = np.exp(-1.0*energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        #integrands_minus = self.lineshapes[ls_key]**2*energies**2*exponents/(exponents - 1.0)**2
        integrands_minus = self.lineshapes[ls_key]**2*exponents/(exponents - 1.0)**2
#        print(integrands_plus.shape)
        #integrals = (np.sum(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.sum(integrands_minus, axis = len(integrands_plus.shape) - 1))*self.delta_omega*(SSCHA_TO_THZ*2.0*np.pi)*1.0e12/2.0
        integrals = (np.trapz(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.trapz(integrands_minus, axis = len(integrands_minus.shape) - 1))*self.delta_omega*(SSCHA_TO_THZ*2.0*np.pi)*1.0e12/2.0
        #integrals = (integrate.simps(integrands_plus, axis = len(integrands_plus.shape) - 1) + integrate.simps(integrands_minus, axis = len(integrands_minus.shape) - 1))*self.delta_omega*(SSCHA_TO_THZ*2.0*np.pi)*1.0e12/2.0
        kappa = np.einsum('ijk,ijl,ij,ij,ij->kl', self.gvels, self.gvels, integrals, self.freqs, self.freqs)*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        kappa += kappa.T
        kappa = kappa/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi

        return kappa

    ##################################################################################################################################

    def calculate_kappa_gk_offdiag(self, temperature, write_lineshapes, energies, gauss_smearing = False):

        """
        Calculation of lattice thermal conductivity using Green-Kubo method if both diagonal and off-diagonal group velocities are available.

        temperature      : temperature at which kappa should be calculated.
        write_lineshapes : Boolean noting should we write phonon lineshapes on a file
        energies         : frequency points at which phonon lineshapes should be calculated

        """

        ls_key = format(temperature, '.1f')
        if(ls_key in self.lineshapes.keys()):
            print('Lineshapes for this temperature have already been calculated. Continuing ...')
        else:
            self.get_lineshapes(temperature, write_lineshapes, energies, method = 'fortran', gauss_smearing = gauss_smearing)
        kappa_diag = np.zeros((3,3))
        exponents_plus = np.exp(energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        #integrands_plus = self.lineshapes[ls_key]**2*energies**2*exponents_plus/(exponents_plus - 1.0)**2
        integrands_plus = self.lineshapes[ls_key]**2*exponents_plus/(exponents_plus - 1.0)**2
        exponents_minus = np.exp(-1.0*energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        #integrands_minus = self.lineshapes[ls_key]**2*energies**2*exponents_minus/(exponents_minus - 1.0)**2
        integrands_minus = self.lineshapes[ls_key]**2*exponents_minus/(exponents_minus - 1.0)**2
        integrals = (np.sum(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.sum(integrands_minus, axis = len(integrands_plus.shape) - 1))*self.delta_omega*(SSCHA_TO_THZ)*1.0e12/2.0*2.0*np.pi
        kappa_diag = np.einsum('ijjk,ijjl,ij,ij,ij->kl', self.gvels, self.gvels, integrals, self.freqs, self.freqs)*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        kappa_diag += kappa_diag.T
        kappa_diag = kappa_diag/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi

        kappa_nondiag = np.zeros_like(kappa_diag)
        for iqpt in range(self.nkpt):
            for iband in range(self.nband-1):
                if(self.freqs[iqpt, iband] != 0.0):
                    for jband in range(iband, self.nband):
                        if(iband != jband and self.freqs[iqpt, jband] != 0.0):
                            integrands_plus = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_plus/(exponents_plus - 1.0)**2
                            integrands_minus = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_minus/(exponents_minus - 1.0)**2
                            integrals = (np.sum(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.sum(integrands_minus, axis = len(integrands_plus.shape) - 1))*self.delta_omega*(SSCHA_TO_THZ*2.0*np.pi)*1.0e12/4.0
                            kappa_nondiag += integrals*(self.freqs[iqpt, iband]**2 + self.freqs[iqpt, jband]**2)**2/self.freqs[iqpt][jband]/self.freqs[iqpt][iband]*np.outer(self.gvels[iqpt, iband, jband], self.gvels[iqpt, jband, iband])\
                                    *SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        kappa_nondiag += kappa_nondiag.T
        kappa_nondiag = kappa_nondiag/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi

        return kappa_diag, kappa_nondiag

    #################################################################################################################################

    def get_lineshapes(self, temperature, write_lineshapes, energies, method = 'fortran', gauss_smearing = False):

        """
        Calculate phonon lineshapes in full Brillouin zone.

        temperature      : temperature to calculate lineshapes on.
        write_lineshapes : Boolean parameter to write phonon lineshapes as they are being calculated.
        energies         : the list of frequencies for which lineshapes are calculated.
        method           : practically only determines how many times fortran routines are called. "fortran" should be much faster.
        gauss_smearing   : are we using Gaussian smearing as approximation for energy conservation
        """

        start_time = time.time()
        if(self.delta_omega == 0.0 and energies is not None):
            self.delta_omega = energies[1] - energies[0]
        ls_key = format(temperature, '.1f')

        if(method == 'python'):

            lineshapes = np.zeros((self.nkpt, self.nband, len(energies)))
            for ikpt in range(self.nirrkpt):
                jkpt = self.qstar_list[ikpt][0]
                print('Calculating lineshapes: ' + format(float(ikpt)/float(self.nirrkpt)*100.0, '.2f') + ' %')
                curr_ls = self.get_diag_dynamic_bubble(self.kpoint_grid, self.k_points[jkpt], self.sigmas[jkpt], energies, temperature)
                print('Normalization of lineshapes (should be 0.5): ')
                print(np.sum(curr_ls, axis=1)*energies[0])
                if(CC.Methods.is_gamma(self.dyn.structure.unit_cell, self.k_points[jkpt])):
                    for iband in range(self.nband):
                        if(self.freqs[jkpt, iband] < np.amax(self.freqs[jkpt])*1.0e-6):
                            curr_ls[iband] = 0.0

                if(write_lineshapes):
                    filename = 'Lineshape_irrkpt_' + str(jkpt) + '_T_' + format(temperature, '.1f')
                    self.write_lineshape(filename, curr_ls, jkpt, energies)

                for iqpt in range(len(self.qstar_list[ikpt])):
                    jqpt = self.qstar_list[ikpt][iqpt]
                    lineshapes[jqpt,:] = curr_ls*2.0
            print('Shape of lineshapes', lineshapes.shape)
            self.lineshapes[ls_key] = lineshapes

        elif(method == 'fortran'):

            lineshapes = np.zeros((self.nkpt, self.nband, len(energies)))
            if(not self.set_up_scattering_grids):
                #if(gauss_smearing):
                #    self.set_scattering_grids_simple()
                #else:
                self.set_scattering_grids_fortran()

            irrqgrid = np.zeros((3, self.nirrkpt))
            scattering_events = np.zeros(self.nirrkpt, dtype=int)
            sigmas = np.zeros((self.nirrkpt, self.nband))
            for ikpt in range(self.nirrkpt):
                irrqgrid[:,ikpt] = self.k_points[self.qstar_list[ikpt][0]].copy()
                scattering_events[ikpt] = len(self.scattering_grids[ikpt])
                sigmas[ikpt] = self.sigmas[self.qstar_list[ikpt][0]]
            irrqgrid = np.asfortranarray(irrqgrid)

            scattering_grids = []
            weights = []
            for ikpt in range(self.nirrkpt):
                for jkpt in range(len(self.scattering_grids[ikpt])):
                    scattering_grids.append(self.scattering_grids[ikpt][jkpt])
                    weights.append(self.scattering_weights[ikpt][jkpt])
            num_scattering_events = len(scattering_grids)
            if(sum(scattering_events) != num_scattering_events):
                print('Difference in number of scattering events!')
            if(sum(weights) != self.scattering_nkpt*self.nirrkpt):
                print('Unexpected number of weights!')
            scattering_grids = np.asfortranarray(scattering_grids).T
            weights = np.asfortranarray(weights)

            classical = False
            if(self.cp_mode == 'classical'):
                classical = True

            curr_ls = thermal_conductivity.get_lf.calculate_lineshapes(irrqgrid, scattering_grids, weights, scattering_events,\
                    self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, self.fc3.r_vector3, \
                    self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(),\
                    sigmas.T, np.zeros_like(sigmas.T, dtype=float), temperature, gauss_smearing, classical, energies, len(energies), self.nirrkpt, \
                    self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)

            for ikpt in range(self.nirrkpt):
                jkpt = self.qstar_list[ikpt][0]
                #print('Normalization of lineshapes (should be 0.5): ')
                #print(np.sum(curr_ls[ikpt], axis=1)*energies[0])
                if(CC.Methods.is_gamma(self.dyn.structure.unit_cell, self.k_points[jkpt])):
                    for iband in range(self.nband):
                        if(self.freqs[jkpt, iband] < np.amax(self.freqs[jkpt])*1.0e-6):
                            curr_ls[ikpt, iband] = 0.0
                if(write_lineshapes):
                    filename = 'Lineshape_irrkpt_' + str(jkpt) + '_T_' + format(temperature, '.1f')
                    self.write_lineshape(filename, curr_ls[ikpt], jkpt, energies)

                for iqpt in range(len(self.qstar_list[ikpt])):
                    jqpt = self.qstar_list[ikpt][iqpt]
                    lineshapes[jqpt,:,:] = curr_ls[ikpt,:,:]*2.0
            print('Shape of lineshapes', lineshapes.shape)
            self.lineshapes[ls_key] = lineshapes

        print('Calculated SSCHA lineshapes in: ', time.time() - start_time)

    #################################################################################################################################

    def get_lifetimes_selfconsistently(self, temperature, ne, gauss_smearing = False):

        """
        Calculate phonon lifetimes in full Brillouin zone self-consistently.

        temperature      : temperature to calculate lineshapes on.
        write_lifetimes  : Boolean parameter to write phonon lifetimes as they are being calculated.
        energies         : the list of frequencies for which lineshapes are calculated.
        gauss_smearing   : are we using Gaussian smearing as approximation for energy conservation
        """

        start_time = time.time()
        if(self.delta_omega == 0.0 and not 'energies' in locals()):
            self.delta_omega = np.amax(self.freqs)*2.0/float(ne)
            energies = np.arange(ne, dtype=float)*self.delta_omega + self.delta_omega
        elif(self.delta_omega != 0.0 and not 'energies' in locals()):
            energies = np.arange(ne, dtype=float)*self.delta_omega + self.delta_omega
        elif(self.delta_omega == 0.0 and energies is not None):
            self.delta_omega = energies[1] - energies[0]
        lf_key = format(temperature, '.1f')

        lifetimes = np.zeros((self.nkpt, self.nband))
        shifts = np.zeros_like(lifetimes)
        if(not self.set_up_scattering_grids):
            #if(gauss_smearing):
            #    self.set_scattering_grids_simple()
            #else:
            self.set_scattering_grids_fortran()

        irrqgrid = np.zeros((3, self.nirrkpt))
        scattering_events = np.zeros(self.nirrkpt, dtype=int)
        sigmas = np.zeros((self.nirrkpt, self.nband))
        for ikpt in range(self.nirrkpt):
            irrqgrid[:,ikpt] = self.k_points[self.qstar_list[ikpt][0]].copy()
            scattering_events[ikpt] = len(self.scattering_grids[ikpt])
            sigmas[ikpt] = self.sigmas[self.qstar_list[ikpt][0]]
        irrqgrid = np.asfortranarray(irrqgrid)

        scattering_grids = []
        weights = []
        for ikpt in range(self.nirrkpt):
            for jkpt in range(len(self.scattering_grids[ikpt])):
                scattering_grids.append(self.scattering_grids[ikpt][jkpt])
                weights.append(self.scattering_weights[ikpt][jkpt])
        num_scattering_events = len(scattering_grids)
        if(sum(scattering_events) != num_scattering_events):
            print('Difference in number of scattering events!')
        if(sum(weights) != self.scattering_nkpt*self.nirrkpt):
            print('Unexpected number of weights!')
        scattering_grids = np.asfortranarray(scattering_grids).T
        weights = np.asfortranarray(weights)

        classical = False
        if(self.cp_mode == 'classical'):
            classical = True

        selfengs = thermal_conductivity.get_lf.calculate_lifetimes_selfconsistently(irrqgrid, scattering_grids, weights, scattering_events,\
                self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, self.fc3.r_vector3, \
                self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(),\
                sigmas.T, np.zeros_like(sigmas.T, dtype=float), temperature, gauss_smearing, classical, energies, len(energies), self.nirrkpt, \
                self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)

        for ikpt in range(self.nirrkpt):
            for iqpt in range(len(self.qstar_list[ikpt])):
                jqpt = self.qstar_list[ikpt][iqpt]
                lifetimes[jqpt,:] = -1.0*np.divide(np.ones_like(selfengs[ikpt].imag, dtype=float), selfengs[ikpt].imag, out=np.zeros_like(selfengs[ikpt].imag), where=selfengs[ikpt].imag!=0.0)/2.0
                shifts[jqpt,:] = selfengs[ikpt].real

        self.lifetimes[lf_key] = lifetimes/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)
        self.freqs_shifts[lf_key] = shifts

        print('Calculated SSCHA lifetimes in: ', time.time() - start_time)

    ##################################################################################################################################

    def get_lineshapes_along_the_line(self, temperature, ne = 1000, filename = 'spectral_function_along_path', gauss_smearing = True, kpoints = None, start_nkpts = 100):

        """
        Calculate phonon lineshapes in full Brillouin zone.

        temperature      : temperature to calculate lineshapes on.
        ne               : Number of frequency points for the lineshapes
        gauss_smearing   : are we using Gaussian smearing as approximation for energy conservation
        kpoints          : the list of kpoints in reduced coordinates to calculate lineshapes for.
                           If not provided generate them using seekpath
        nkpts            : Number of k points along the path. Will differ from the final number of points!
        """

        start_time = time.time()

        tics = []
        if(kpoints is None):
            import seekpath
            rat = np.dot(self.dyn.structure.coords, np.linalg.inv(self.dyn.structure.unit_cell))
            sym = np.unique(self.dyn.structure.atoms)
            nt = np.zeros(len(rat))
            for i in range(len(rat)):
                for j in range(len(sym)):
                    if(self.dyn.structure.atoms[i] == sym[j]):
                        nt[i] = j + 1
                        break
            path = seekpath.getpaths.get_path((self.dyn.structure.unit_cell, rat, nt))
            kpoints, distances, segments = get_kpoints_in_path(path, start_nkpts, self.reciprocal_lattice)
            for i in range(len(distances)):
                if(i == 0):
                    tics.append(distances[0])
                elif(np.abs(distances[i] - distances[i-1]) < 1.0e-12):
                    tics.append(distances[i])
            tics.append(distances[-1])
            if(len(tics) - 1 != len(segments)):
                print('Number of tics and segments does not match! Weird!')
                print(len(tics), len(segments) + 1)
            kpoints = np.array(kpoints)
        else:
            kpoints = kpoints#np.dot(kpoints, self.reciprocal_lattice)
        nkpts = len(kpoints)
        freqs = np.zeros((nkpts, self.nband))
        for ikpt in range(nkpts):
            freqs[ikpt], _ = self.get_frequency_at_q(kpoints[ikpt])

        maxfreq = np.amax(freqs)*2.1
        energies = np.arange(ne, dtype=float)/float(ne)*maxfreq

        lineshapes = np.zeros((nkpts, self.nband, ne))

        irrqgrid = kpoints.T
        scattering_events = np.zeros(nkpts, dtype=int)
        sigmas = np.zeros((nkpts, self.nband))
        sigmas[:,:] = self.sigmas[0,0]
        for ikpt in range(nkpts):
            scattering_events[ikpt] = len(self.scattering_qpoints)
        irrqgrid = np.asfortranarray(irrqgrid)

        scattering_grids = []
        weights = []
        for ikpt in range(nkpts):
            for jkpt in range(len(self.scattering_qpoints)):
                scattering_grids.append(self.scattering_k_points[jkpt])
                weights.append(1)
        num_scattering_events = len(scattering_grids)
        if(sum(scattering_events) != num_scattering_events):
            print('Difference in number of scattering events!')
            print(sum(scattering_events), num_scattering_events)
        if(sum(weights) != self.scattering_nkpt*nkpts):
            print('Unexpected number of weights!')
            print(sum(weights), self.scattering_nkpt*nkpts)
        scattering_grids = np.asfortranarray(scattering_grids).T
        weights = np.asfortranarray(weights)

        classical = False
        if(self.cp_mode == 'classical'):
            classical = True

        curr_ls = thermal_conductivity.get_lf.calculate_lineshapes(irrqgrid, scattering_grids, weights, scattering_events,\
                self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, self.fc3.r_vector3, \
                self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(),\
                sigmas.T, np.zeros_like(sigmas.T, dtype=float), temperature, gauss_smearing, classical, energies, len(energies), nkpts, \
                self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)

        for ikpt in range(nkpts):
            if(CC.Methods.is_gamma(self.dyn.structure.unit_cell, kpoints[ikpt])):
                for iband in range(self.nband):
                    if(freqs[ikpt, iband] < np.amax(freqs[ikpt])*1.0e-6):
                        curr_ls[ikpt, iband] = 0.0
            lineshapes[ikpt,:,:] = curr_ls[ikpt,:,:]*2.0

        with open('Qpoints_along_line', 'w+') as outfile:
            for ikpt in range(nkpts):
                qpt = np.dot(kpoints[ikpt], np.linalg.inv(self.reciprocal_lattice))
                for i in range(3):
                    outfile.write(3*' ' + format(qpt[i], '.12f'))
                outfile.write('\n')

        with open(filename, 'w+') as outfile:
            outfile.write('# Path and tics: \n')
            outfile.write('# ' + segments[0][0] + '  ' + format(tics[0], '.8f'))
            for i in range(len(segments) - 1):
                if(segments[i][1] == segments[i + 1][0]):
                    outfile.write('  ' + segments[i][1] + '  ' + format(tics[i+1], '.8f'))
                else:
                    outfile.write('  ' + segments[i][1] + ' | ' + segments[i+1][0] + '  ' + format(tics[i+1], '.8f'))
            outfile.write('  ' + segments[len(segments)-1][1] + '  ' + format(tics[len(segments)], '.8f') + '\n')
            outfile.write('# normalized distance       energy (THz)         lineshape (1/THz) \n')
            for ikpt in range(nkpts):
                for ie in range(ne):
                    outfile.write('  ' + format(distances[ikpt], '.12e'))
                    outfile.write('  ' + format(energies[ie]*SSCHA_TO_THZ, '.12e'))
                    for iband in range(self.nband):
                        outfile.write('  ' + format(lineshapes[ikpt,iband,ie]/SSCHA_TO_THZ, '.12e'))
                    outfile.write('\n')
                outfile.write('\n')

        with open(filename + '_phonons', 'w+') as outfile:
            outfile.write('# Path and tics: \n')
            outfile.write('# ' + segments[0][0] + '  ' + format(tics[0], '.8f'))
            for i in range(len(segments) - 1):
                if(segments[i][1] == segments[i + 1][0]):
                    outfile.write('  ' + segments[i][1] + '  ' + format(tics[i+1], '.8f'))
                else:
                    outfile.write('  ' + segments[i][1] + ' | ' + segments[i+1][0] + '  ' + format(tics[i+1], '.8f'))
            outfile.write('  ' + segments[-1][1] + '  ' + format(tics[-1], '.8f') + '\n')
            outfile.write('# normalized distance       frequency (THz)          \n')
            for ikpt in range(nkpts):
                outfile.write('  ' + format(distances[ikpt], '.12e'))
                for iband in range(self.nband):
                    outfile.write('  ' + format(freqs[ikpt,iband]*SSCHA_TO_THZ, '.12e'))
                outfile.write('\n')

        print('Calculated SSCHA lineshapes in: ', time.time() - start_time)

    ##################################################################################################################################
    def write_lineshape(self, filename, curr_ls, jkpt, energies):

        """

        Function to write phonon lineshapes onto a file.

        filename : title of the file at which lineshape is to be written.
        curr_ls  : lineshape to be written
        jkpt     : the index of the k point for which lineshapes are to be written
        energies : frequencies at which lineshapes have been calculated 

        """

        with open(filename, 'w+') as outfile:
            outfile.write('# SSCHA frequencies (THz) \n')
            outfile.write('#' + 15*' ')
            for iband in range(self.nband):
                outfile.write(3*' ' + format(self.freqs[jkpt, iband]*SSCHA_TO_THZ, '.12e'))
            outfile.write('\n')
            outfile.write('#  ' + format('Omega (THz)', STR_FMT))
            outfile.write('   ' + format('Spectral function (1/THz)', STR_FMT))
            outfile.write('\n')
            for ien in range(len(energies)):
                outfile.write(3*' ' + format(energies[ien]*SSCHA_TO_THZ, '.12e'))
                for iband in range(self.nband):
                    outfile.write(3*' ' + format(curr_ls[iband, ien]/SSCHA_TO_THZ, '.12e'))
                outfile.write('\n')

    ##################################################################################################################################

    def get_heat_capacity(self, temperature):

        """
        Calculate phonon mode heat capacity at temperature.

        """

        cp_key = format(temperature, '.1f')
        cp = np.zeros_like(self.freqs)
        for ikpt in range(self.nkpt):
            for iband in range(self.nband):
                cp[ikpt, iband] = heat_capacity(self.freqs[ikpt, iband]*SSCHA_TO_THZ*1.0e12, temperature, HPLANCK, KB, cp_mode = self.cp_mode)
        self.cp[cp_key] = cp


    ##################################################################################################################################

    def calculate_kappa_srta_diag(self, temperature, ne, write_lifetimes, gauss_smearing = False, isotope_scattering = True, isotopes = None, lf_method = 'fortran-LA'):

        """
        Calculate lattice thermal conductivity using single relaxation time approximation at temperature. Calculates only including diagonal term.

        """

        lf_key = format(temperature, '.1f')
        cp_key = format(temperature, '.1f')
        if(lf_key in self.lifetimes.keys()):
            print('Lifetimes for this temperature have already been calculated. Continuing ...')
        else:
            print('Calculating phonon lifetimes for ' + format(temperature, '.1f') + ' K temperature!')
            self.get_lifetimes(temperature, ne, gauss_smearing = gauss_smearing, isotope_scattering = isotope_scattering, isotopes = isotopes, method = lf_method)
        if(cp_key in self.cp.keys()):
            print('Phonon mode heat capacities for this temperature have already been calculated. Continuing ...')
        else:
            print('Calculating phonon mode heat capacities for ' + format(temperature, '.1f') + ' K temperature!')
            self.get_heat_capacity(temperature)

        if(write_lifetimes):
            self.write_transport_properties_to_file(temperature, isotope_scattering)
            
        kappa = np.einsum('ij,ijk,ijl,ij->kl',self.cp[cp_key],self.gvels,self.gvels,self.lifetimes[lf_key])
        kappa += kappa.T
        kappa = kappa/2.0*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2

        return kappa

    ##################################################################################################################################

    def calculate_kappa_srta_offdiag(self, temperature, ne, write_lifetimes, gauss_smearing = False, isotope_scattering = False, isotopes = None, lf_method = 'fortran-LA'):

        """
        Calculates both diagonal and off diagonal contribution to the lattice thermal conductivity (Nature Physics volume 15, pages 809–813 (2019)).
        Quite slow!

        """

        lf_key = format(temperature, '.1f')
        cp_key = format(temperature, '.1f')
        if(lf_key in self.lifetimes.keys()):
            print('Lifetimes for this temperature have already been calculated. Continuing ...')
        else:
            self.get_lifetimes(temperature, ne, gauss_smearing = gauss_smearing, isotope_scattering = isotope_scattering, isotopes = isotopes, method = lf_method)
        if(cp_key in self.cp.keys()):
            print('Phonon mode heat capacities for this temperature have already been calculated. Continuing ...')
        else:
            self.get_heat_capacity(temperature)
        scatt_rates = np.divide(np.ones_like(self.lifetimes[lf_key], dtype=float), self.lifetimes[lf_key], out=np.zeros_like(self.lifetimes[lf_key]), where=self.lifetimes[lf_key]!=0.0)/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)
#        scatt_rates = 1.0/(self.lifetimes[lf_key]*SSCHA_TO_THZ*2.0*np.pi*1.0e12)
        if(write_lifetimes):
            self.write_transport_properties_to_file(temperature, isotope_scattering)
        kappa_diag = np.einsum('ij,ijjk,ijjl,ij->kl',self.cp[cp_key],self.gvels,self.gvels,self.lifetimes[lf_key])

        kappa_nondiag = np.zeros_like(kappa_diag)
        for iqpt in range(self.nkpt):
            for iband in range(self.nband - 1):
                if(self.freqs[iqpt, iband] != 0.0):
                    for jband in range(iband + 1, self.nband):
                        if(self.freqs[iqpt, jband] != 0.0):
                            vel_fact = np.sqrt(2.0*self.freqs[iqpt, jband]*self.freqs[iqpt, iband])/(self.freqs[iqpt, jband] + self.freqs[iqpt, iband]) # as per Eq.34 in Caldarelli et al
                            kappa_nondiag += (self.freqs[iqpt, iband] + self.freqs[iqpt, jband])*(scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])*\
                                    (self.freqs[iqpt, jband]*self.cp[cp_key][iqpt, iband] + self.freqs[iqpt, iband]*self.cp[cp_key][iqpt, jband])*np.outer(self.gvels[iqpt, iband, jband], self.gvels[iqpt, jband, iband])*vel_fact**2/\
                                    self.freqs[iqpt,iband]/self.freqs[iqpt, jband]/2.0/(4.0*(self.freqs[iqpt,iband] - self.freqs[iqpt,jband])**2 + (scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])**2)
        kappa_nondiag = 2.0*kappa_nondiag/SSCHA_TO_THZ/1.0e12

        kappa_diag += kappa_diag.T
        kappa_nondiag += kappa_nondiag.T
        kappa_diag = kappa_diag/2.0*SSCHA_TO_MS**2#*(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        kappa_nondiag = kappa_nondiag/2.0*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2

        return kappa_diag, kappa_nondiag

    ################################################################################################################################################################################

    def calculate_kappa_srta_offdiag_isaeva(self, temperature, write_lifetimes, gauss_smearing = False, isotope_scattering = False, isotopes = None, lf_method = 'fortran-LA'):

        """
        Calculates both diagonal and off diagonal contribution to the lattice thermal conductivity (Nature Communications volume 10, Article number: 3853 (2019)).
        Quite slow!

        """

        lf_key = format(temperature, '.1f')
        cp_key = format(temperature, '.1f')
        if(lf_key in self.lifetimes.keys()):
            print('Lifetimes for this temperature have already been calculated. Continuing ...')
        else:
            self.get_lifetimes(temperature, gauss_smearing = gauss_smearing, isotope_scattering = isotope_scattering, isotopes = isotopes, method = lf_method)
        if(cp_key in self.cp.keys()):
            print('Phonon mode heat capacities for this temperature have already been calculated. Continuing ...')
        else:
            self.get_heat_capacity(temperature)
        scatt_rates = np.divide(np.ones_like(self.lifetimes[lf_key], dtype=float), self.lifetimes[lf_key], out=np.zeros_like(self.lifetimes[lf_key]), where=self.lifetimes[lf_key]!=0.0)/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)
#        scatt_rates = 1.0/(self.lifetimes[lf_key]*SSCHA_TO_THZ*2.0*np.pi*1.0e12)
        if(write_lifetimes):
            self.write_transport_properties_to_file(temperature, isotope_scattering)
        kappa_diag = np.einsum('ij,ijjk,ijjl,ij->kl',self.cp[cp_key],self.gvels,self.gvels,self.lifetimes[lf_key])

        pops = np.zeros_like(self.freqs)
        for iqpt in range(self.nkpt):
            for iband in range(self.nband):
                pops[iqpt, iband] = bose_einstein(self.freqs[iqpt, iband]*SSCHA_TO_THZ*1.0e12, temperature, HPLANCK, KB, cp_mode = self.cp_mode)

        kappa_nondiag = np.zeros_like(kappa_diag)
        for iqpt in range(self.nkpt):
            for iband in range(self.nband - 1):
                if(self.freqs[iqpt, iband] != 0.0):
                    for jband in range(iband + 1, self.nband):
                        if(self.freqs[iqpt, jband] != 0.0 and self.freqs[iqpt, jband] - self.freqs[iqpt, iband] != 0.0):
                            kappa_nondiag += self.freqs[iqpt, iband]*self.freqs[iqpt, jband]*(pops[iqpt, iband] - pops[iqpt, jband])/(self.freqs[iqpt, jband] - self.freqs[iqpt, iband])*\
                                np.outer(self.gvels[iqpt, iband, jband], self.gvels[iqpt, jband, iband])*(scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])/\
                                ((self.freqs[iqpt,iband] - self.freqs[iqpt,jband])**2 + (scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])**2)

        kappa_nondiag = 2.0*kappa_nondiag*HPLANCK/temperature

        kappa_diag += kappa_diag.T
        kappa_nondiag += kappa_nondiag.T
        kappa_diag = kappa_diag/2.0*SSCHA_TO_MS**2#*(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        kappa_nondiag = kappa_nondiag/2.0*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2

        return kappa_diag, kappa_nondiag

   ####################################################################################################################################

    def get_frequencies(self):
        
        """
        Get frequencies on a grid which is used for Brillouin zone integration (in THz).
   
        """
   
        return self.freqs*SSCHA_TO_THZ

    ####################################################################################################################################

    def get_scattering_rates_isotope(self, isotopes = None):


        start_time = time.time()
        self.scattering_rates_isotope = np.zeros((self.nkpt, self.nband))
        if(isotopes is None):
            isotopes = []
            for i in range(len(self.dyn.structure.atoms)):
                isotopes.append(natural_isotopes[self.dyn.structure.atoms[i]])
        for i in range(len(isotopes)):
            tot_w = 0.0
            for j in range(len(isotopes[i])):
                tot_w += isotopes[i][j][0]
            if(abs(tot_w - 1.0) > 1.0e-3):
                print('Sum of the isotopes percentages is not one!')

        av_mass = np.zeros(len(isotopes))
        for i in range(len(isotopes)):
            for j in range(len(isotopes[i])):
                av_mass[i] += isotopes[i][j][0]*isotopes[i][j][1]
            print('Average mass of the ' + self.dyn.structure.atoms[i] + ' is ' + format(av_mass[i], '.4f') + '.')

        g_factor = np.zeros(len(isotopes))
        for i in range(len(isotopes)):
            for j in range(len(isotopes[i])):
                g_factor[i] += isotopes[i][j][0]*(1.0 - isotopes[i][j][1]/av_mass[i])**2
            print('G factor of ' + self.dyn.structure.atoms[i] + ' is ' + format(g_factor[i], '.4f') + '.') 

        for ikpt in range(self.nirrkpt):
            jkpt = self.qstar_list[ikpt][0]
            curr_scatt_rate = self.get_scattering_rates_isotope_at_q(jkpt, g_factor, av_mass)
            for iqpt in range(len(self.qstar_list[ikpt])):
                jqpt = self.qstar_list[ikpt][iqpt]
                self.scattering_rates_isotope[jqpt,:] = curr_scatt_rate

        self.got_scattering_rates_isotopes = True
        print('Calculated scattering rates due to the isotope scattering in ' + format(time.time() - start_time, '.2e'), ' seconds!')
        
    ####################################################################################################################################

    def get_scattering_rates_isotope_at_q(self, iqpt, g_factor, av_mass):

        scatt_rate = np.zeros(self.nband)
        for iband in range(self.nband):
            for jqpt in range(self.nkpt):
                for jband in range(self.nband):
                    factor = 0.0
                    for iat in range(len(self.dyn.structure.atoms)):
                        factor += g_factor[iat]*np.dot(self.eigvecs[iqpt,iband,3*iat:3*(iat+1)].conj(), self.eigvecs[jqpt,jband,3*iat:3*(iat+1)])**2
                    scatt_rate[iband] += gaussian(self.freqs[iqpt,iband], self.freqs[jqpt,jband], self.sigmas[iqpt, iband])*factor
                    if(np.isnan(scatt_rate[iband])):
                        print(gaussian(self.freqs[iqpt,iband], self.freqs[jqpt,jband], self.sigmas[iqpt, iband]), factor)
                        print(g_factor[iat])
                        raise RuntimeError('NAN!')
            scatt_rate[iband] = np.pi*scatt_rate[iband]/2.0/float(self.nkpt)*self.freqs[iqpt, iband]**2
        return scatt_rate
  
    ####################################################################################################################################
   
    def get_lifetimes(self, temperature, ne, gauss_smearing = False, isotope_scattering = True, isotopes = None, method = 'fortran-LA'):

        """
        Get phonon lifetimes in the full Brillouin zone at temperature.

        ne                 : Number of frequencies used in self-consistent solution.
        gauss_smearing     : If true will use the Gaussian function to satisfy conservation laws, instead Lorentzian (only fortran)
        isotope_scattering : If true will calculate the scattering rates due to isotope concentration
        isotopes           : The relative concentration and masses of isotopes
        method             : Method by which phonon lifetimes are to be calculated.
            fortran/python : practically means only how many times fortran routine is being called. "fortran" much faster.
            LA/P           : Approximation used for the lifetimes. 
                             LA means one shot approximation defined in J. Phys.: Condens. Matter 33 363001 . Default value.
                             P means perturbative approximation. The one used by most other codes!
            SC             : Solve for lifetimes and frequency shifts self-consistently! 
        """

        if(not self.set_up_scattering_grids):
            #if(gauss_smearing):
            #    self.set_scattering_grids_simple()
            #else:
            self.set_scattering_grids_fortran()

        if(isotope_scattering and not self.got_scattering_rates_isotopes):
            self.get_scattering_rates_isotope(isotopes)

        start_time = time.time()
        lf_key = format(temperature, '.1f')
        print('Calculating lifetimes at: ' + lf_key + ' K.')

        if(method == 'python-LA'):

            lifetimes = np.zeros((self.nkpt, self.nband))
            shifts = np.zeros((self.nkpt, self.nband))
            for ikpt in range(self.nirrkpt):
                jkpt = self.qstar_list[ikpt][0]
                print('Calculating lifetimes: ' + format(float(ikpt)/float(self.nirrkpt)*100.0, '.2f') + ' %')
                curr_freq, curr_shift, curr_lw = self.get_lifetimes_at_q(self.kpoint_grid, self.k_points[jkpt], self.sigmas[jkpt], temperature)
                curr_lf = np.divide(np.ones_like(curr_lw, dtype=float), curr_lw, out=np.zeros_like(curr_lw), where=curr_lw!=0.0)/2.0
                if(CC.Methods.is_gamma(self.dyn.structure.unit_cell, self.k_points[jkpt])):
                    for iband in range(self.nband):
                        if(self.freqs[jkpt, iband] < np.amax(self.freqs[jkpt])*1.0e-6):
                            curr_lf[iband] = 0.0
                            curr_shift[iband] = 0.0
                for iqpt in range(len(self.qstar_list[ikpt])):
                    jqpt = self.qstar_list[ikpt][iqpt]
                    lifetimes[jqpt,:] = curr_lf
                    shifts[jqpt,:] = curr_shift
            print('Shape of lifetimes', lifetimes.shape)
            self.lifetimes[lf_key] = lifetimes/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)
            self.freqs_shifts[lf_key] = shifts

        elif(method == 'python-P'):

            lifetimes = np.zeros((self.nkpt, self.nband))
            shifts = np.zeros((self.nkpt, self.nband))
            for ikpt in range(self.nirrkpt):
                jkpt = self.qstar_list[ikpt][0]
                print('Calculating lifetimes: ' + format(float(ikpt)/float(self.nirrkpt)*100.0, '.2f') + ' %')
                selfnrg = np.diag(self.get_just_diag_dynamic_bubble(self.kpoint_grid, self.k_points[jkpt], self.sigmas[jkpt], self.freqs[jkpt], temperature))
                curr_lf = -1.0*np.divide(selfnrg.imag, self.freqs[jkpt], out=np.zeros_like(self.freqs[jkpt]), where=self.freqs[jkpt]!=0.0)/2.0
                curr_shifts = selfnrg.real
                if(CC.Methods.is_gamma(self.dyn.structure.unit_cell, self.k_points[jkpt])):
                    for iband in range(self.nband):
                        if(self.freqs[jkpt, iband] < np.amax(self.freqs[jkpt])*1.0e-6):
                            curr_lf[iband] = 0.0
                            curr_shifts[iband] = 0.0
                for iqpt in range(len(self.qstar_list[ikpt])):
                    jqpt = self.qstar_list[ikpt][iqpt]
                    lifetimes[jqpt,:] = np.divide(np.ones_like(curr_lf, dtype=float), curr_lf, out=np.zeros_like(curr_lf), where=curr_lf!=0.0)/2.0
                    shifts[jqpt,:] = curr_shifts
            print('Shape of lifetimes', lifetimes.shape)
            self.lifetimes[lf_key] = lifetimes/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)
            self.freqs_shifts[lf_key] = shifts

        elif(method == 'fortran-LA'):

            print('Calculating lifetimes in fortran, lorentzian approximation!')
            irrqgrid = np.zeros((3, self.nirrkpt))
            scattering_events = np.zeros(self.nirrkpt, dtype=int)
            sigmas = np.zeros((self.nirrkpt, self.nband))
            for ikpt in range(self.nirrkpt):
                irrqgrid[:,ikpt] = self.k_points[self.qstar_list[ikpt][0]].copy()
                scattering_events[ikpt] = len(self.scattering_grids[ikpt])
                sigmas[ikpt] = self.sigmas[self.qstar_list[ikpt][0]]
            irrqgrid = np.asfortranarray(irrqgrid)
            lifetimes = np.zeros((self.nkpt, self.nband))
            shifts = np.zeros((self.nkpt, self.nband))

            scattering_grids = []
            weights = []
            for ikpt in range(self.nirrkpt):
                for jkpt in range(len(self.scattering_grids[ikpt])):
                    scattering_grids.append(self.scattering_grids[ikpt][jkpt])
                    weights.append(self.scattering_weights[ikpt][jkpt])
            num_scattering_events = len(scattering_grids)
            if(sum(scattering_events) != num_scattering_events):
                print('Difference in number of scattering events!')
            if(sum(weights) != self.nkpt*self.nirrkpt):
                print('Unexpected number of weights!')
            scattering_grids = np.asfortranarray(scattering_grids).T
            weights = np.asfortranarray(weights)

            classical = False
            if(self.cp_mode == 'classical'):
                classical = True

            selfengs = thermal_conductivity.get_lf.calculate_lifetimes(irrqgrid, scattering_grids, weights, scattering_events, \
                    self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, \
                    self.fc3.r_vector3, self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(), sigmas.T, temperature, \
                    gauss_smearing, classical, self.nirrkpt, self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor),\
                    num_scattering_events)

            for ikpt in range(self.nirrkpt):
                for iqpt in range(len(self.qstar_list[ikpt])):
                    jqpt = self.qstar_list[ikpt][iqpt]
                    lifetimes[jqpt,:] = -1.0*np.divide(np.ones_like(selfengs[ikpt].imag, dtype=float), selfengs[ikpt].imag, out=np.zeros_like(selfengs[ikpt].imag), where=selfengs[ikpt].imag!=0.0)/2.0
                    shifts[jqpt, :] = selfengs[ikpt].real
            self.lifetimes[lf_key] = lifetimes/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)
            self.freqs_shifts[lf_key] = shifts

        elif(method == 'fortran-P'):

            print('Calculating lifetimes in fortran, perturbative approximation!')
            irrqgrid = np.zeros((3, self.nirrkpt))
            scattering_events = np.zeros(self.nirrkpt, dtype=int)
            sigmas = np.zeros((self.nirrkpt, self.nband))
            for ikpt in range(self.nirrkpt):
                irrqgrid[:,ikpt] = self.k_points[self.qstar_list[ikpt][0]].copy()
                scattering_events[ikpt] = len(self.scattering_grids[ikpt])
                sigmas[ikpt] = self.sigmas[self.qstar_list[ikpt][0]]
            irrqgrid = np.asfortranarray(irrqgrid)
            lifetimes = np.zeros((self.nkpt, self.nband))
            shifts = np.zeros((self.nkpt, self.nband))

            scattering_grids = []
            weights = []
            for ikpt in range(self.nirrkpt):
                for jkpt in range(len(self.scattering_grids[ikpt])):
                    scattering_grids.append(self.scattering_grids[ikpt][jkpt])
                    weights.append(self.scattering_weights[ikpt][jkpt])
            num_scattering_events = len(scattering_grids)
            if(sum(scattering_events) != num_scattering_events):
                print('Difference in number of scattering events!')
            if(sum(weights) != self.nkpt*self.nirrkpt):
                print('Unexpected number of weights!')
            scattering_grids = np.asfortranarray(scattering_grids).T
            weights = np.asfortranarray(weights)

            classical = False
            if(self.cp_mode == 'classical'):
                classical = True

            selfengs = thermal_conductivity.get_lf.calculate_lifetimes_perturbative(irrqgrid, scattering_grids, weights, scattering_events,\
                    self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, \
                    self.fc3.r_vector3, self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(), sigmas.T, temperature, \
                    gauss_smearing, classical, self.nirrkpt, self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)

            for ikpt in range(self.nirrkpt):
                for iqpt in range(len(self.qstar_list[ikpt])):
                    jqpt = self.qstar_list[ikpt][iqpt]
                    lifetimes[jqpt,:] = -1.0*np.divide(np.ones_like(selfengs[ikpt].imag, dtype=float), selfengs[ikpt].imag, out=np.zeros_like(selfengs[ikpt].imag), where=selfengs[ikpt].imag!=0.0)/2.0
                    shifts[jqpt,:] = selfengs[ikpt].real

            self.lifetimes[lf_key] = lifetimes/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)
            self.freqs_shifts[lf_key] = shifts

        elif(method == 'SC'):
            self.get_lifetimes_selfconsistently(temperature, ne, gauss_smearing = gauss_smearing)
            
        else:
            print('Unrecognized method! Exit!')
            raise RuntimeError('No such method for calculating phonon lifetimes!')

        print('Calculated SSCHA lifetimes in: ', time.time() - start_time)

   ####################################################################################################################################

    def setup_harmonic_properties(self, smearing_value = 0.00005):

        """

        Sets up harmonic properties (calculates frequencies, group velocities and smearing parameters.)

        smearing_value : Value of the smearing in case smearing_method == "constant"
        """

        for ikpt, kpt in enumerate(self.k_points):
            self.freqs[ikpt], self.eigvecs[ikpt] = self.get_frequency_at_q(kpt)
            self.gvels[ikpt] = self.get_group_velocity(kpt, self.freqs[ikpt], self.eigvecs[ikpt])
            #self.gvels[ikpt] = self.get_group_velocity_finite_difference(kpt, self.freqs[ikpt], self.eigvecs[ikpt])

        #self.symmetrize_group_velocities_over_star()
        #self.check_group_velocities()
        #self.check_frequencies()
        self.setup_smearings(smearing_value)
        print('Harmonic properties are set up!')

    #################################################################################################################################

    def check_frequencies(self):

        """

        Routine to check whether the frequencies in q star are all the same

        """

        for istar in range(self.nirrkpt):
            freqs0 = self.freqs[self.qstar_list[istar][0]]
            for jqpt in range(1, len(self.qstar_list[istar])):
                freqs1 = self.freqs[self.qstar_list[istar][jqpt]]
                if(np.any(np.abs(freqs0 - freqs1) > 1.0e-6*np.amax(freqs0))):
                    print('WARNING! Frequencies in star not the same. ', istar, jqpt)
                    print(freqs0)
                    print(freqs1)

    #################################################################################################################################

    def check_group_velocities(self):

        """

        Check whether the group velocities for wave vectors in q star are all the same. 

        """

        tot_r = self.symmetry.QE_s[:,:,:self.symmetry.QE_nsymq]
        nsym = np.shape(tot_r)[-1]
        print('Total number of symmetry operations: ', nsym)


        for istar in range(self.nirrkpt):
            q0 = np.dot(self.k_points[self.qstar_list[istar][0]], np.linalg.inv(self.reciprocal_lattice))
            vel0 = self.gvels[self.qstar_list[istar][0]].copy()
            for jqpt in range(1, len(self.qstar_list[istar])):
                q1 = np.dot(self.k_points[self.qstar_list[istar][jqpt]], np.linalg.inv(self.reciprocal_lattice))
                rotation = np.zeros((3,3))
                found_rot = False
                for ir in range(0, nsym):
                    q2 = np.dot(tot_r[:,:,ir], q0)
                    q3 = np.dot(np.dot(-1.0*np.eye(3), tot_r[:,:,ir]), q0)
                    if(same_vector(q2, q1, np.eye(3))):
                        print('Pure rotation')
                        rotation = np.dot(np.linalg.inv(self.unitcell.T), np.dot(tot_r[:,:,ir], self.unitcell.T))
                        found_rot = True
                        break
                    elif(same_vector(q3, q1, np.eye(3))):
                        print('Inverse and rotation')
                        rotation = np.dot(np.linalg.inv(self.unitcell.T), np.dot(np.dot(-1.0*np.eye(3),tot_r[:,:,ir]), self.unitcell.T))
                        found_rot = True
                        break
                if(found_rot):
                    check_if_rotation(rotation, self.symmetry.threshold)
                    if(self.off_diag):
                        vel1 = np.einsum('ij,klj->kli', rotation, self.gvels[self.qstar_list[istar][jqpt]])
                    else:
                        vel1 = np.einsum('ij,kj->ki', rotation, self.gvels[self.qstar_list[istar][jqpt]])
                    if(np.any(vel0 - vel1) > 1.0e-4):
                        print('Velocities in star not agreeing!', istar, jqpt)
                        print(vel0)
                        print(vel1)
                else:
                    print('Could not find rotation between vectors in star! ', istar, jqpt, self.qstar_list[istar][0], self.qstar_list[istar][jqpt])

    #################################################################################################################################

    def get_group_velocity(self, q, freqs, eigvecs):

        """
        Calculate group velocity. Using analytical formula.

        q       : wave vector in real space (without 2pi factor)
        freqs   : frequencies for this wave vector
        eigvecs : eigenvectors for this wave vector

        """

        uc_positions = self.dyn.structure.coords.copy()

        is_q_gamma = CC.Methods.is_gamma(self.fc2.unitcell_structure.unit_cell, q)
        if(self.off_diag):
            tmp_gvel = np.zeros((self.nband, self.nband, 3))
            gvel = np.zeros((self.nband, self.nband, 3))
        else:
            tmp_gvel = np.zeros((self.nband, 3))
            gvel = np.zeros((self.nband, 3))
        m = np.tile(self.dyn.structure.get_masses_array(), (3,1)).T.ravel()
        mm_mat = np.sqrt(np.outer(m, m))
        mm_inv_mat = 1.0 / mm_mat

        degs = check_degeneracy(freqs, np.amax(freqs)*1.0e-8)

        for icart in range(3):
            auxfc = np.zeros_like(self.force_constants[0], dtype = complex)
            for iuc in range(len(self.force_constants)):
                for iat in range(len(uc_positions)):
                    for jat in range(len(uc_positions)):
                        ruc = -self.ruc[iuc] + uc_positions[iat] - uc_positions[jat]
                        phase = np.dot(ruc, q)*2.0*np.pi
                        auxfc[3*iat:3*(iat+1),3*jat:3*(jat+1)] += complex(0.0,1.0)*ruc[icart]*self.force_constants[iuc,3*iat:3*(iat+1),3*jat:3*(jat+1)]*np.exp(1j*phase)
            ddynmat = auxfc * mm_inv_mat
            if(icart == 0):
                dirdynmat = ddynmat.copy()
            rot_eigvecs = np.zeros_like(eigvecs)
            for ideg, deg in enumerate(degs):
                rot_eigvecs[deg, :] = rotate_eigenvectors(dirdynmat, eigvecs[deg, :])
            if(is_q_gamma):
                if(self.off_diag):
                    for iband in range(self.nband):
                        if(freqs[iband] != 0.0):
                            for jband in range(self.nband):
                                if(freqs[jband] != 0.0):
                                    #tmp_gvel[iband,jband][icart] = np.dot(eigvecs[iband].conj(), np.dot(ddynmat, eigvecs[jband])).real/2.0/np.sqrt(freqs[jband]*freqs[iband])#*np.sqrt(EV_TO_J/AU)
                                    tmp_gvel[iband,jband][icart] = np.dot(rot_eigvecs[iband].conj(), np.dot(ddynmat, rot_eigvecs[jband])).real/2.0/np.sqrt(freqs[jband]*freqs[iband])#*np.sqrt(EV_TO_J/AU)
                else:
                    for iband in range(self.nband):
                        if(freqs[iband] != 0.0):
                            #tmp_gvel[iband][icart] = np.dot(eigvecs[iband].conj(), np.dot(ddynmat, eigvecs[iband])).real/2.0/freqs[iband]#*np.sqrt(EV_TO_J/AU)
                            tmp_gvel[iband][icart] = np.dot(rot_eigvecs[iband].conj(), np.dot(ddynmat, rot_eigvecs[iband])).real/2.0/freqs[iband]#*np.sqrt(EV_TO_J/AU)
            else:
                if(self.off_diag):
                    for iband in range(self.nband):
                        for jband in range(self.nband):
                            #tmp_gvel[iband,jband][icart] = np.dot(eigvecs[iband].conj(), np.dot(ddynmat, eigvecs[jband])).real/2.0/np.sqrt(freqs[jband]*freqs[iband])#*np.sqrt(EV_TO_J/AU)
                            tmp_gvel[iband,jband][icart] = np.dot(rot_eigvecs[iband].conj(), np.dot(ddynmat, rot_eigvecs[jband])).real/2.0/np.sqrt(freqs[jband]*freqs[iband])#*np.sqrt(EV_TO_J/AU)
                else:
                    for iband in range(self.nband):
                        #tmp_gvel[iband][icart] = np.dot(eigvecs[iband].conj(), np.dot(ddynmat, eigvecs[iband])).real/2.0/freqs[iband]#*np.sqrt(EV_TO_J/AU)
                        tmp_gvel[iband][icart] = np.dot(rot_eigvecs[iband].conj(), np.dot(ddynmat, rot_eigvecs[iband])).real/2.0/freqs[iband]#*np.sqrt(EV_TO_J/AU)

        if(np.any(np.isnan(tmp_gvel))):
            raise RuntimeError('NaN is group velocity matrix!')
        gvel = self.symmetrize_group_velocity(tmp_gvel, q) 
    
        return gvel


    ##################################################################################################################################

    def get_group_velocity_finite_difference(self, q, freqs, eigvecs):

        """
        Calculate group velocity. Using analytical formula.

        q       : wave vector in real space (without 2pi factor)
        freqs   : frequencies for this wave vector
        eigvecs : eigenvectors for this wave vector

        """

        is_q_gamma = CC.Methods.is_gamma(self.fc2.unitcell_structure.unit_cell, q)
        if(self.off_diag):
            tmp_gvel = np.zeros((self.nband, self.nband, 3))
            gvel = np.zeros((self.nband, self.nband, 3))
        else:
            tmp_gvel = np.zeros((self.nband, 3))
            gvel = np.zeros((self.nband, 3))

        degs = check_degeneracy(freqs, np.amax(freqs)*1.0e-8)

        for icart in range(3):
            dynmat0 = self.get_dynamical_matrix(q)
            dq = np.zeros_like(q)
            dq[icart] = np.sum(np.linalg.norm(self.reciprocal_lattice[:,icart]))/1000.0
            q1 = q + dq
            dynmat1 = self.get_dynamical_matrix(q1)
            q2 = q - dq
            dynmat2 = self.get_dynamical_matrix(q2)
            ddynmat = (dynmat1 - dynmat2)/np.linalg.norm(dq)/2.0/2.0/np.pi
            if(icart == 0):
                dirdynmat = ddynmat.copy()
            rot_eigvecs = np.zeros_like(eigvecs)
            for ideg, deg in enumerate(degs):
                rot_eigvecs[deg, :] = rotate_eigenvectors(dirdynmat, eigvecs[deg, :])
            if(is_q_gamma):
                if(self.off_diag):
                    for iband in range(self.nband):
                        if(freqs[iband] != 0.0):
                            for jband in range(self.nband):
                                if(freqs[jband] != 0.0):
                                    #gvel[iband,jband][icart] = np.dot(eigvecs[iband].conj(), np.dot(ddynmat, eigvecs[jband])).real/2.0/np.sqrt(freqs[jband]*freqs[iband])#*np.sqrt(EV_TO_J/AU)
                                    gvel[iband,jband][icart] = np.dot(rot_eigvecs[iband].conj(), np.dot(ddynmat, rot_eigvecs[jband])).real/2.0/np.sqrt(freqs[jband]*freqs[iband])#*np.sqrt(EV_TO_J/AU)
                else:
                    for iband in range(self.nband):
                        if(freqs[iband] != 0.0):
                           # tmp_gvel[iband][icart] = np.dot(eigvecs[iband].conj(), np.dot(ddynmat, eigvecs[iband])).real/2.0/freqs[iband]#*np.sqrt(EV_TO_J/AU)
                            tmp_gvel[iband][icart] = np.dot(rot_eigvecs[iband].conj(), np.dot(ddynmat, rot_eigvecs[iband])).real/2.0/freqs[iband]#*np.sqrt(EV_TO_J/AU)
            else:
                if(self.off_diag):
                    for iband in range(self.nband):
                        for jband in range(self.nband):
                            #tmp_gvel[iband,jband][icart] = np.dot(eigvecs[iband].conj(), np.dot(ddynmat, eigvecs[jband])).real/2.0/np.sqrt(freqs[jband]*freqs[iband])#*np.sqrt(EV_TO_J/AU)
                            tmp_gvel[iband,jband][icart] = np.dot(rot_eigvecs[iband].conj(), np.dot(ddynmat, rot_eigvecs[jband])).real/2.0/np.sqrt(freqs[jband]*freqs[iband])#*np.sqrt(EV_TO_J/AU)
                else:
                    for iband in range(self.nband):
                        #tmp_gvel[iband][icart] = np.dot(eigvecs[iband].conj(), np.dot(ddynmat, eigvecs[iband])).real/2.0/freqs[iband]#*np.sqrt(EV_TO_J/AU)
                        tmp_gvel[iband][icart] = np.dot(rot_eigvecs[iband].conj(), np.dot(ddynmat, rot_eigvecs[iband])).real/2.0/freqs[iband]#*np.sqrt(EV_TO_J/AU)

        gvel = self.symmetrize_group_velocity(tmp_gvel, q) 
    
        return gvel

    ##################################################################################################################################

    def symmetrize_group_velocity(self, vels, q):

        """

        Symmetrize group velocites according to the little group of wave vector.

        vels : group velocities at this wave vector
        q    : wave vector in question!

        """

        qred = np.dot(q, np.linalg.inv(self.reciprocal_lattice))
        qred -= np.rint(qred)
        cell = get_spglib_cell(self.dyn)
        tot_r = spglib.get_symmetry_dataset(cell)['rotations']
        nsym = len(tot_r)
        for i in range(nsym):
            tot_r[i] = tot_r[i].T

        rot_q = []
        for i in range(nsym):
            diff = qred- np.dot(tot_r[i,:,:], qred)
            if (np.all(np.abs(diff) < self.symmetry.threshold)):
                rot_q.append(tot_r[i,:,:])
        if(len(rot_q) > 0):
            rot_vels = np.zeros_like(vels)
            for i in range(len(rot_q)):
                rot_q[i] = np.dot(self.reciprocal_lattice.T, np.dot(rot_q[i], np.linalg.inv(self.reciprocal_lattice.T)))
                check_if_rotation(rot_q[i], self.symmetry.threshold)
                if(self.off_diag):
                    rot_vels += np.einsum('ij,klj->kli', rot_q[i], vels)
                else:
                    rot_vels += np.einsum('ij,kj->ki', rot_q[i], vels)
            rot_vels /= float(len(rot_q))
        else:
            rot_vels = vels.copy()

        return rot_vels

    #################################################################################################################################

    def symmetrize_group_velocities_over_star(self):

        """

        Symmetrize group velocities over q star.

        """

        cell = get_spglib_cell(self.dyn)
        tot_r = spglib.get_symmetry_dataset(cell)['rotations']
        nsym = len(tot_r)
        for i in range(nsym):
            tot_r[i] = tot_r[i].T


        for istar in range(self.nirrkpt):
            q0 = np.dot(self.k_points[self.qstar_list[istar][0]], np.linalg.inv(self.reciprocal_lattice))
            vel0 = self.gvels[self.qstar_list[istar][0]].copy()
            rotations = []
            rotations.append(np.eye(3))
            for jqpt in range(1, len(self.qstar_list[istar])):
                q1 = np.dot(self.k_points[self.qstar_list[istar][jqpt]], np.linalg.inv(self.reciprocal_lattice))
                rotation = np.zeros((3,3))
                found_rot = False
                for ir in range(0, nsym):
                    q2 = np.dot(tot_r[ir,:,:], q1)
                    if(same_vector(q2, q0, np.eye(3))):
                        rotation = np.dot(self.reciprocal_lattice.T, np.dot(tot_r[ir,:,:], np.linalg.inv(self.reciprocal_lattice.T)))
                        found_rot = True
                        break
                if(found_rot):
                    check_if_rotation(rotation, self.symmetry.threshold)
                    rotations.append(rotation)
                    if(self.off_diag):
                        vel1 = np.einsum('ij,klj->kli', rotation, self.gvels[self.qstar_list[istar][jqpt]])
                    else:
                        vel1 = np.einsum('ij,kj->ki', rotation, self.gvels[self.qstar_list[istar][jqpt]])
                    vel0 += vel1
                else:
                    print('Could not find rotation between vectors in star! ', istar, jqpt, self.qstar_list[istar][0], self.qstar_list[istar][jqpt])
            if(len(rotations) != len(self.qstar_list[istar])):
                print('Number of rotations does not match number of q points in the star: ', len(rotations) + 1, len(self.qstar_list[istar]))
            vel0 = vel0/float(len(rotations))
            for jqpt in range(len(self.qstar_list[istar])):
                if(self.off_diag):
                    self.gvels[self.qstar_list[istar][jqpt]] = np.einsum('ij,klj->kli', np.linalg.inv(rotations[jqpt]), vel0)
                else:
                    self.gvels[self.qstar_list[istar][jqpt]] = np.einsum('ij,kj->ki', np.linalg.inv(rotations[jqpt]), vel0)

    #################################################################################################################################

    def get_frequency_at_q(self, q):

        """

        Get phonon frequencies and eigenvectors at wave vector q.

        q: wave vector in real space (no 2pi factor)

        """

        dynmat = self.get_dynamical_matrix(q)
        w2_q, pols_q = np.linalg.eigh(dynmat)
        is_q_gamma = CC.Methods.is_gamma(self.fc2.unitcell_structure.unit_cell, q)
        if(is_q_gamma):
            for iband in range(self.nband):
                if(w2_q[iband] < np.amax(w2_q)*1.0e-6):
                    w2_q[iband] = 0.0
        if(np.any(w2_q < 0.0)):
            print('At q: ')
            print(q)
            print(w2_q)
            raise RuntimeError('SSCHA frequency imaginary. Stopping now.')
        else:
            w_q = np.sqrt(w2_q)

        return w_q, pols_q.T
    
    ###################################################################################################################################

    def get_dynamical_matrix(self, q):

        """

        Get dynamical matrix at wave vector.

        q : wave vector without 2pi factor

        """

        uc_positions = self.dyn.structure.coords.copy()
        m = np.tile(self.dyn.structure.get_masses_array(), (3,1)).T.ravel()
        mm_mat = np.sqrt(np.outer(m, m))
        mm_inv_mat = 1.0 / mm_mat
        dynmat = np.zeros_like(self.force_constants[0], dtype = complex)
        #phases = np.einsum('ij,j->i', self.ruc, q)*2.0*np.pi
        #exponents = np.exp(1j*phases)
        #dynmat = np.einsum('ijk,i->jk', self.force_constants, exponents) * mm_inv_mat
        #for iat in range(len(uc_positions)):
        #    for jat in range(len(uc_positions)):
        #        if(iat != jat):
        #            extra_phase = np.dot(uc_positions[iat] - uc_positions[jat], q)*2.0*np.pi
        #            dynmat[3*iat:3*(iat+1),3*jat:3*(jat+1)] *= np.exp(1j*extra_phase)
        for ir in range(len(self.ruc)):
            for iat in range(len(uc_positions)):
                for jat in range(len(uc_positions)):
                    r = -1.0*self.ruc[ir] + uc_positions[iat] - uc_positions[jat]
                    phase = np.dot(r, q)*2.0*np.pi
                    dynmat[3*iat:3*(iat+1),3*jat:3*(jat+1)] += self.force_constants[ir,3*iat:3*(iat+1),3*jat:3*(jat+1)]*np.exp(1j*phase)
        dynmat = dynmat*mm_inv_mat
        dynmat = (dynmat + dynmat.conj().T)/2.0
        
        return dynmat

    ####################################################################################################################################

    def get_dos(self, de = 0.1):

        """
        Calculate phonon DOS using gaussian smearing.

        de : the distance between two frequency points is sampling of DOS (in THz)

        """

        de = de /(np.sqrt(RY_TO_J/AU/MASS_RY_TO_UMA)/BOHR_TO_ANGSTROM/100.0/2.0/np.pi)
        if(np.all(self.freqs == 0)):
            print('All frequencies are zero. Make sure that the harmonic properties are initialized!')
        else:
            frequencies = self.freqs.flatten()
            smearing = self.sigmas.flatten()
            nsamples = int(np.amax(self.freqs)*1.1/de)
            samples = np.arange(nsamples).astype(float)*de
            deltas = np.zeros((nsamples, len(frequencies)))
            for i in range(nsamples):
                for j in range(len(frequencies)):
                    if(frequencies[j] > 0.0):
                        deltas[i,j] = gaussian(frequencies[j], samples[i], smearing[j])
            dos = np.sum(deltas, axis = 1)

        dos = dos/float(len(self.freqs))

        return samples*SSCHA_TO_THZ, dos/SSCHA_TO_THZ

    ########################################################################################################################################

    def get_dos_from_lineshapes(self, temperature, de = 0.1):

        """
        Calculates phonon DOS from lineshapes at temperature.

        temperature : temperature in K.
        de          : sampling spacing of energies after convolution (in THz)

        """

        de = de/SSCHA_TO_THZ
        key = format(temperature, '.1f')
        if(key in self.lineshapes.keys()):
            ne = self.lineshapes[key].shape[-1]
            energies = np.arange(ne, dtype=float)*self.delta_omega + self.delta_omega
            dos = np.sum(np.sum(self.lineshapes[key], axis = 0), axis = 0)
            dos = dos/float(self.nkpt)
            print('Total DOS is (should be number of bands): ', np.sum(dos)*self.delta_omega)
        else:
            print('Lineshapes not calculated for this temperature! Can not calculate DOS! ')
            dos = 0
            energies = 0

        ne = int(np.amax(energies)/de)
        energies_smoothed = np.arange(ne, dtype=float)*de + de
        dos_smoothed = np.zeros(ne, dtype=float)
        for ien in range(ne):
            for jen in range(len(dos)):
                dos_smoothed[ien] += gaussian(energies_smoothed[ien], energies[jen], de/2.0)*dos[jen]
        int_dos = np.sum(dos_smoothed)*de
        dos_smoothed = dos_smoothed/int_dos*np.sum(dos)*self.delta_omega
        
        return energies*SSCHA_TO_THZ, dos/SSCHA_TO_THZ, energies_smoothed*SSCHA_TO_THZ, dos_smoothed/SSCHA_TO_THZ

    ########################################################################################################################################

    def get_mean_square_displacement_from_lineshapes(self, temperature = 300.0):

        """
        Calculates mean square displacement factor from phonon lineshapes. (<u*_{qs}(0)u_{qs}(t=0)>)

        temperature : temperature in K.

        """

        key = format(temperature, '.1f')
        if(key in self.lineshapes.keys()):
            ne = self.lineshapes[key].shape[-1]
            energies = np.arange(ne, dtype=float)*self.delta_omega + self.delta_omega
            exponents_plus = np.exp(energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
            exponents_minus = np.exp(-1.0*energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
            msd = np.sum(self.lineshapes[key]*(1.0/(exponents_plus - 1.0) - 1.0/(exponents_minus - 1.0)), axis = len(self.lineshapes[key].shape) - 1)*self.delta_omega
            with open('Phonon_MSD_from_lineshapes', 'w+') as outfile:
                outfile.write('#  ' + format('Frequency (THz)', STR_FMT))
                outfile.write('   ' + format('MSD anharmonic', STR_FMT))
                outfile.write('   ' + format('MSD harmonic', STR_FMT))
                outfile.write('\n')
                for iqpt in range(self.nkpt):
                    for iband in range(self.nband):
                        if(self.freqs[iqpt, iband] != 0.0):
                            outfile.write(3*' ' + format(self.freqs[iqpt][iband]*SSCHA_TO_THZ, '.12e'))
                            outfile.write(3*' ' + format(msd[iqpt][iband], '.12e'))
                            outfile.write(3*' ' + format(2.0/(np.exp(self.freqs[iqpt][iband]*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature) - 1.0) + 1.0, '.12e'))
                            outfile.write('\n')
        else:
            print('Lineshapes not calculated for this temperature! Can not calculate mean square displacements! ')
            dos = 0
            energies = 0
        
    ####################################################################################################################################
   
    def write_harmonic_properties_to_file(self, filename = 'Phonon_harmonic_properties'):

        """
        Write harmonic properties (frequency, group velocity, smearing parameters) from second order SSCHA tensor to file.

        """

        with open(filename, 'w+') as outfile:
            outfile.write('#   ' + format('Frequencies (THz)', STR_FMT))
            outfile.write('    ' + format('Group velocity x (m/s)', STR_FMT))
            outfile.write('    ' + format('Group velocity y (m/s)', STR_FMT))
            outfile.write('    ' + format('Group velocity z (m/s)', STR_FMT))
            outfile.write('    ' + format('Smearing par (THz)', STR_FMT))
            outfile.write('\n')
            for ikpt in range(self.nkpt):
                for iband in range(self.nband):
                    outfile.write(3*' ' + format(self.freqs[ikpt][iband]*SSCHA_TO_THZ, '.12e'))
                    for icart in range(3):
                        if(self.off_diag):
                            outfile.write(3*' ' + format(self.gvels[ikpt][iband,iband][icart]*SSCHA_TO_MS, '.12e'))
                        else:
                            outfile.write(3*' ' + format(self.gvels[ikpt][iband][icart]*SSCHA_TO_MS, '.12e'))
                    outfile.write(3*' ' + format(self.sigmas[ikpt][iband]*SSCHA_TO_THZ, '.12e'))
                    outfile.write('\n')

    ######################################################################################################################################

    def write_transport_properties_to_file(self, temperature, isotope_scattering, filename = 'Phonon_transport_properties_'):
        """
        Write transport properties (frequencies, lifetimes, heat capacities) from SSCHA tensors to file.

        """

        lf_key = format(temperature, '.1f')
        if(lf_key in self.lifetimes.keys()):
            with open(filename + lf_key, 'w+') as outfile:
                outfile.write('#  ' + format('Frequency (THz)', STR_FMT))
                outfile.write('   ' + format('Lifetime (ps)', STR_FMT))
                outfile.write('   ' + format('Freq. shift (THz)', STR_FMT))
                if(isotope_scattering):
                    outfile.write('   ' + format('Isotope scatt. rate (THz)', STR_FMT))
                outfile.write('   ' + format('Mode heat capacity (J/K)', STR_FMT))
                outfile.write('\n')
                for iqpt in range(self.nkpt):
                    for iband in range(self.nband):
                        outfile.write(3*' ' + format(self.freqs[iqpt, iband]*SSCHA_TO_THZ, '.12e'))
                        outfile.write(3*' ' + format(self.lifetimes[lf_key][iqpt, iband]*1.0e12, '.12e'))
                        outfile.write(3*' ' + format(self.freqs_shifts[lf_key][iqpt, iband]*SSCHA_TO_THZ, '.12e'))
                        if(isotope_scattering):
                            outfile.write(3*' ' + format(self.scattering_rates_isotope[iqpt, iband]*SSCHA_TO_THZ, '.12e'))
                        outfile.write(3*' ' + format(self.cp[lf_key][iqpt, iband], '.12e'))
                        outfile.write('\n')
        else:
            print('Lifetimes have not been calculated for this temperature! ')
    #####################################################################################################################################

    def get_lifetimes_at_q(self, k_grid, q, smear, T):

        """

        Get lifetime at a specific wave vector q. Will also give phonon shift.

        k_grid : k_grid to sum scattering events over
        q      : wave vector in question
        smear  : smearing factors for this wave vectors (dimension = (nband))
        T      : temperature in K

        """

            
        structure = self.fc2.unitcell_structure
        
        # Get the integration points 
        k_points = CC.symmetries.GetQGrid(structure.unit_cell, k_grid)
        
            
        # Get the phi2 in q
        phi2_q = self.fc2.Interpolate(q, asr = False)

        # dynamical matrix in q
        m = np.tile(structure.get_masses_array(), (3,1)).T.ravel()    
        mm_mat = np.sqrt(np.outer(m, m))
        mm_inv_mat = 1 / mm_mat
        #
        d2_q = phi2_q * mm_inv_mat
        
        # Diagonalize the dynamical matrix in q
        w2_q, pols_q = np.linalg.eigh(d2_q)
        
        # Check if the q point is gamma
        is_q_gamma = CC.Methods.is_gamma(structure.unit_cell, q)
        
        if is_q_gamma:
            w2_q[0:3]=0.0
        if not (w2_q >= 0.0).all():
            print('q= ',q, '    (2pi/A)')
            print('w(q)= ',np.sign(w2_q)*np.sqrt(np.abs(w2_q))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_q=np.sqrt(w2_q)  

        def compute_k(k):
            # phi3 in q, k, -q - k
            phi3=self.fc3.Interpolate(k,-q-k, asr = False)
            #print(phi3)
            # phi2 in k
            phi2_k = self.fc2.Interpolate(k, asr = False) 

            # phi2 in -q-k
            phi2_mq_mk = self.fc2.Interpolate(-q -k, asr = False)

            # dynamical matrices (divide by the masses)
            d2_k = phi2_k * mm_inv_mat
            d2_mq_mk = phi2_mq_mk * mm_inv_mat
            
            # Diagonalize the dynamical matrices
            w2_k, pols_k = np.linalg.eigh(d2_k)
            w2_mq_mk, pols_mq_mk = np.linalg.eigh(d2_mq_mk)
            
            
            is_k_gamma = CC.Methods.is_gamma(structure.unit_cell, k)
            is_mq_mk_gamma = CC.Methods.is_gamma(structure.unit_cell, -q-k)
            
            if is_k_gamma:
                w2_k[0:3]=0.0
            if not (w2_k >= 0.0).all():
                print('k= ',k, '    (2pi/A)')
                print('w(k)= ',np.sign(w2_k)*np.sqrt(np.abs(w2_k))*CC.Units.RY_TO_CM,'  (cm-1)')
                print('Cannot continue with SSCHA negative frequencies')
                exit()
            w_k=np.sqrt(w2_k)

            if is_mq_mk_gamma:
                w2_mq_mk[0:3]=0.0
            if not (w2_mq_mk >= 0.0).all():
                print('-q-k= ',-q-k, '    (2pi/A)')
                print('w(-q-k)= ',np.sign(w2_mq_mk)*np.sqrt(np.abs(w2_mq_mk))*CC.Units.RY_TO_CM,'  (cm-1)')
                print('Cannot continue with SSCHA negative frequencies')
                exit()
            w_mq_mk=np.sqrt(w2_mq_mk)
           
            # Dividing the phi3 by the sqare root of masses
            d3 = np.einsum("abc, a, b, c -> abc", phi3, 1/np.sqrt(m), 1/np.sqrt(m), 1/np.sqrt(m))
            #print(d3)
            # d3 in mode components
            #d3_pols = np.einsum("abc, ai, bj, ck -> ijk", d3, pols_q, pols_k, pols_mq_mk)
            d3_pols = np.einsum("abc, ai -> ibc", d3, pols_q)
            d3_pols = np.einsum("abc, bi -> aic", d3_pols, pols_k)
            d3_pols = np.einsum("abc, ci -> abi", d3_pols, pols_mq_mk)
            #print(d3_pols)

            n_mod=3*structure.N_atoms 
            # Fortran duty ====
            
            selfnrg  = thermal_conductivity.third_order_cond.compute_perturb_selfnrg_single(smear,T,
                                                                np.array([w_q,w_k,w_mq_mk]).T,
                                                                np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                                d3_pols,n_mod)
            return selfnrg    
        
        CC.Settings.SetupParallel()

        selfnrg =CC.Settings.GoParallel(compute_k, k_points, reduce_op = "+")
        # divide by the N_k factor
        selfnrg /= len(k_points) # (n_mod,nsigma)
        
        #w_q_ext=w_q[...,None]
            
        shift=np.divide(selfnrg.real, 2*w_q, out=np.zeros_like(selfnrg.real), where=w_q!=0)
        hwhm=np.divide(-selfnrg.imag, 2*w_q, out=np.zeros_like(selfnrg.imag), where=w_q!=0)

        return w_q, shift,hwhm

    ######################################################################################################################################

    def get_diag_dynamic_bubble(self, k_grid, q, smear, energies, T):

        """
        Get lineshape at a specific wave vector q.

        k_grid : k_grid to sum scattering events over
        q      : wave vector in question
        smear  : smearing factors for this wave vectors (dimension = (nband))
        T      : temperature in K

        """

        structure = self.fc2.unitcell_structure

        # Get the integration points
        k_points = CC.symmetries.GetQGrid(structure.unit_cell, k_grid)


        # Get the phi2 in q
        phi2_q = self.fc2.Interpolate(q, asr = False)
        # dynamical matrix in q
        m = np.tile(structure.get_masses_array(), (3,1)).T.ravel()
        mm_mat = np.sqrt(np.outer(m, m))
        mm_inv_mat = 1 / mm_mat
        #
        d2_q = phi2_q * mm_inv_mat

        # Diagonalize the dynamical matrix in q
        w2_q, pols_q = np.linalg.eigh(d2_q)

        # Check if the q point is gamma
        is_q_gamma = CC.Methods.is_gamma(structure.unit_cell, q)

        if is_q_gamma:
            w2_q[0:3]=0.0
        if not (w2_q >= 0.0).all():
            print('q= ',q, '    (2pi/A)')
            print('w(q)= ',np.sign(w2_q)*np.sqrt(np.abs(w2_q))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_q=np.sqrt(w2_q)

        # Allocate the memory for the bubble
        ne=energies.shape[0]
        nat=structure.N_atoms
        tmp_bubble = np.zeros((ne,self.nband), dtype = np.complex128, order = "F")

        def compute_k(k):
            # phi3 in q, k, -q - k
            phi3=self.fc3.Interpolate(k,-q-k, asr = False)
            # phi2 in k
            phi2_k = self.fc2.Interpolate(k, asr = False)

            # phi2 in -q-k
            phi2_mq_mk = self.fc2.Interpolate(-q -k, asr = False)

            # dynamical matrices (divide by the masses)
            d2_k = phi2_k * mm_inv_mat
            d2_mq_mk = phi2_mq_mk * mm_inv_mat

            # Diagonalize the dynamical matrices
            w2_k, pols_k = np.linalg.eigh(d2_k)
            w2_mq_mk, pols_mq_mk = np.linalg.eigh(d2_mq_mk)


            is_k_gamma = CC.Methods.is_gamma(structure.unit_cell, k)
            is_mq_mk_gamma = CC.Methods.is_gamma(structure.unit_cell, -q-k)

            if is_k_gamma:
                w2_k[0:3]=0.0
            if not (w2_k >= 0.0).all():
                print('k= ',k, '    (2pi/A)')
                print('w(k)= ',np.sign(w2_k)*np.sqrt(np.abs(w2_k))*CC.Units.RY_TO_CM,'  (cm-1)')
                print('Cannot continue with SSCHA negative frequencies')
                exit()
            w_k=np.sqrt(w2_k)

            if is_mq_mk_gamma:
                w2_mq_mk[0:3]=0.0
            if not (w2_mq_mk >= 0.0).all():
                print('-q-k= ',-q-k, '    (2pi/A)')
                print('w(-q-k)= ',np.sign(w2_mq_mk)*np.sqrt(np.abs(w2_mq_mk))*CC.Units.RY_TO_CM,'  (cm-1)')
                print('Cannot continue with SSCHA negative frequencies')
                exit()
            w_mq_mk=np.sqrt(w2_mq_mk)

            # Dividing the phi3 by the sqare root of masses
            d3 = np.einsum("abc, a, b, c -> abc", phi3, 1/np.sqrt(m), 1/np.sqrt(m), 1/np.sqrt(m))

            # d3 in mode components
            #d3_pols = np.einsum("abc, ai, bj, ck -> ijk", d3, pols_mq, pols_k, pols_q_mk)
            d3_pols = np.einsum("abc, ai -> ibc", d3, pols_q)
            d3_pols = np.einsum("abc, bi -> aic", d3_pols, pols_k)
            d3_pols = np.einsum("abc, ci -> abi", d3_pols, pols_mq_mk)

            # Fortran duty ====

            #
            tmp_bubble  = thermal_conductivity.third_order_cond.compute_diag_dynamic_bubble_single(energies,smear,T,
                                                                np.array([w_q,w_k,w_mq_mk]).T,
                                                                np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                                d3_pols,ne,n_mod=self.nband)

            return tmp_bubble

        CC.Settings.SetupParallel()

        d_bubble_mod =CC.Settings.GoParallel(compute_k, k_points, reduce_op = "+")

        # divide by the N_k factor
        d_bubble_mod /= len(k_points) # (ne,nsmear,n_mod)
        #
        spectralf=thermal_conductivity.third_order_cond.compute_spectralf_diag_single(np.zeros_like(w_q),energies,w_q,
                                                                       d_bubble_mod,
                                                                       nat,ne)


        return spectralf.T

    ######################################################################################################################################

    def get_just_diag_dynamic_bubble(self, k_grid, q, smear, energies, T):

        """
        Get lineshape at a specific wave vector q.

        k_grid : k_grid to sum scattering events over
        q      : wave vector in question
        smear  : smearing factors for this wave vectors (dimension = (nband))
        T      : temperature in K

        """

        structure = self.fc2.unitcell_structure

        # Get the integration points
        k_points = CC.symmetries.GetQGrid(structure.unit_cell, k_grid)


        # Get the phi2 in q
        phi2_q = self.fc2.Interpolate(q, asr = False)
        # dynamical matrix in q
        m = np.tile(structure.get_masses_array(), (3,1)).T.ravel()
        mm_mat = np.sqrt(np.outer(m, m))
        mm_inv_mat = 1 / mm_mat
        #
        d2_q = phi2_q * mm_inv_mat

        # Diagonalize the dynamical matrix in q
        w2_q, pols_q = np.linalg.eigh(d2_q)

        # Check if the q point is gamma
        is_q_gamma = CC.Methods.is_gamma(structure.unit_cell, q)

        if is_q_gamma:
            w2_q[0:3]=0.0
        if not (w2_q >= 0.0).all():
            print('q= ',q, '    (2pi/A)')
            print('w(q)= ',np.sign(w2_q)*np.sqrt(np.abs(w2_q))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_q=np.sqrt(w2_q)

        # Allocate the memory for the bubble
        ne=energies.shape[0]
        nat=structure.N_atoms
        tmp_bubble = np.zeros((ne,self.nband), dtype = np.complex128, order = "F")

        def compute_k(k):
            # phi3 in q, k, -q - k
            phi3=self.fc3.Interpolate(k,-q-k, asr = False)
            # phi2 in k
            phi2_k = self.fc2.Interpolate(k, asr = False)

            # phi2 in -q-k
            phi2_mq_mk = self.fc2.Interpolate(-q -k, asr = False)

            # dynamical matrices (divide by the masses)
            d2_k = phi2_k * mm_inv_mat
            d2_mq_mk = phi2_mq_mk * mm_inv_mat

            # Diagonalize the dynamical matrices
            w2_k, pols_k = np.linalg.eigh(d2_k)
            w2_mq_mk, pols_mq_mk = np.linalg.eigh(d2_mq_mk)


            is_k_gamma = CC.Methods.is_gamma(structure.unit_cell, k)
            is_mq_mk_gamma = CC.Methods.is_gamma(structure.unit_cell, -q-k)

            if is_k_gamma:
                w2_k[0:3]=0.0
            if not (w2_k >= 0.0).all():
                print('k= ',k, '    (2pi/A)')
                print('w(k)= ',np.sign(w2_k)*np.sqrt(np.abs(w2_k))*CC.Units.RY_TO_CM,'  (cm-1)')
                print('Cannot continue with SSCHA negative frequencies')
                exit()
            w_k=np.sqrt(w2_k)

            if is_mq_mk_gamma:
                w2_mq_mk[0:3]=0.0
            if not (w2_mq_mk >= 0.0).all():
                print('-q-k= ',-q-k, '    (2pi/A)')
                print('w(-q-k)= ',np.sign(w2_mq_mk)*np.sqrt(np.abs(w2_mq_mk))*CC.Units.RY_TO_CM,'  (cm-1)')
                print('Cannot continue with SSCHA negative frequencies')
                exit()
            w_mq_mk=np.sqrt(w2_mq_mk)

            # Dividing the phi3 by the sqare root of masses
            d3 = np.einsum("abc, a, b, c -> abc", phi3, 1/np.sqrt(m), 1/np.sqrt(m), 1/np.sqrt(m))

            # d3 in mode components
            #d3_pols = np.einsum("abc, ai, bj, ck -> ijk", d3, pols_mq, pols_k, pols_q_mk)
            d3_pols = np.einsum("abc, ai -> ibc", d3, pols_q)
            d3_pols = np.einsum("abc, bi -> aic", d3_pols, pols_k)
            d3_pols = np.einsum("abc, ci -> abi", d3_pols, pols_mq_mk)

            # Fortran duty ====

            #
            tmp_bubble  = thermal_conductivity.third_order_cond.compute_diag_dynamic_bubble_single(energies,smear,T,
                                                                np.array([w_q,w_k,w_mq_mk]).T,
                                                                np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                                d3_pols,ne,n_mod=self.nband)

            return tmp_bubble

        CC.Settings.SetupParallel()

        d_bubble_mod =CC.Settings.GoParallel(compute_k, k_points, reduce_op = "+")

        # divide by the N_k factor
        d_bubble_mod /= len(k_points) # (ne,nsmear,n_mod)
        #
        return d_bubble_mod.T

    #####################################################################################################################################

    def get_lifetimes_at_q_smaller_grid(self, k_grid, weights, q, smear, T):

        """

        Get lifetime at a specific wave vector q calculated on a smaller grid. Will also give phonon shift.
        k_grid : k_grid to sum scattering events over
        q      : wave vector in question
        smear  : smearing factors for this wave vectors (dimension = (nband))
        T      : temperature in K

        """

            
        structure = self.fc2.unitcell_structure
        
        # Get the integration points 
        #k_points = CC.symmetries.GetQGrid(structure.unit_cell, k_grid)
        
            
        # Get the phi2 in q
        phi2_q = self.fc2.Interpolate(q, asr = False)

        # dynamical matrix in q
        m = np.tile(structure.get_masses_array(), (3,1)).T.ravel()    
        mm_mat = np.sqrt(np.outer(m, m))
        mm_inv_mat = 1 / mm_mat
        #
        d2_q = phi2_q * mm_inv_mat
        
        # Diagonalize the dynamical matrix in q
        w2_q, pols_q = np.linalg.eigh(d2_q)
        
        # Check if the q point is gamma
        is_q_gamma = CC.Methods.is_gamma(structure.unit_cell, q)
        
        if is_q_gamma:
            w2_q[0:3]=0.0
        if not (w2_q >= 0.0).all():
            print('q= ',q, '    (2pi/A)')
            print('w(q)= ',np.sign(w2_q)*np.sqrt(np.abs(w2_q))*CC.Units.RY_TO_CM,'  (cm-1)')
            print('Cannot continue with SSCHA negative frequencies')
            exit()
        w_q=np.sqrt(w2_q)  

        def compute_k(inputs):
            k = inputs[0]
            weight = inputs[1]
            # phi3 in q, k, -q - k
            phi3=self.fc3.Interpolate(k,-q-k, asr = False)
            #print(phi3)
            # phi2 in k
            phi2_k = self.fc2.Interpolate(k, asr = False) 

            # phi2 in -q-k
            phi2_mq_mk = self.fc2.Interpolate(-q -k, asr = False)

            # dynamical matrices (divide by the masses)
            d2_k = phi2_k * mm_inv_mat
            d2_mq_mk = phi2_mq_mk * mm_inv_mat
            
            # Diagonalize the dynamical matrices
            w2_k, pols_k = np.linalg.eigh(d2_k)
            w2_mq_mk, pols_mq_mk = np.linalg.eigh(d2_mq_mk)
            
            
            is_k_gamma = CC.Methods.is_gamma(structure.unit_cell, k)
            is_mq_mk_gamma = CC.Methods.is_gamma(structure.unit_cell, -q-k)
            
            if is_k_gamma:
                w2_k[0:3]=0.0
            if not (w2_k >= 0.0).all():
                print('k= ',k, '    (2pi/A)')
                print('w(k)= ',np.sign(w2_k)*np.sqrt(np.abs(w2_k))*CC.Units.RY_TO_CM,'  (cm-1)')
                print('Cannot continue with SSCHA negative frequencies')
                exit()
            w_k=np.sqrt(w2_k)

            if is_mq_mk_gamma:
                w2_mq_mk[0:3]=0.0
            if not (w2_mq_mk >= 0.0).all():
                print('-q-k= ',-q-k, '    (2pi/A)')
                print('w(-q-k)= ',np.sign(w2_mq_mk)*np.sqrt(np.abs(w2_mq_mk))*CC.Units.RY_TO_CM,'  (cm-1)')
                print('Cannot continue with SSCHA negative frequencies')
                exit()
            w_mq_mk=np.sqrt(w2_mq_mk)
           
            # Dividing the phi3 by the sqare root of masses
            d3 = np.einsum("abc, a, b, c -> abc", phi3, 1/np.sqrt(m), 1/np.sqrt(m), 1/np.sqrt(m))
            #print(d3)
            # d3 in mode components
            #d3_pols = np.einsum("abc, ai, bj, ck -> ijk", d3, pols_q, pols_k, pols_mq_mk)
            d3_pols = np.einsum("abc, ai -> ibc", d3, pols_q)
            d3_pols = np.einsum("abc, bi -> aic", d3_pols, pols_k)
            d3_pols = np.einsum("abc, ci -> abi", d3_pols, pols_mq_mk)
            #print(d3_pols)

            n_mod=3*structure.N_atoms 
            # Fortran duty ====
            
            selfnrg  = thermal_conductivity.third_order_cond.compute_perturb_selfnrg_single(smear,T,
                                                                np.array([w_q,w_k,w_mq_mk]).T,
                                                                np.array([is_q_gamma,is_k_gamma,is_mq_mk_gamma]),
                                                                d3_pols,n_mod)
            return selfnrg*float(weight)    
        
        CC.Settings.SetupParallel()

        input_list = []
        for i in range(len(weights)):
            input_list.append([k_grid[i], weights[i]])
        selfnrg =CC.Settings.GoParallel(compute_k, input_list, reduce_op = "+")
        # divide by the N_k factor
        selfnrg /= float(sum(weights)) # (n_mod,nsigma)
        
        #w_q_ext=w_q[...,None]
            
        shift=np.divide(selfnrg.real, 2*w_q, out=np.zeros_like(selfnrg.real), where=w_q!=0)
        hwhm=np.divide(-selfnrg.imag, 2*w_q, out=np.zeros_like(selfnrg.imag), where=w_q!=0)

        return w_q, shift,hwhm

    ######################################################################################################################################

    def calculate_mode_gruneisen(self):

        uc_positions = self.dyn.structure.coords.copy()
        natom = len(uc_positions)
        masses = self.dyn.structure.get_masses_array()
        for ikpt in range(self.nirrkpt):
            for iband in range(self.nband):
                gruneisen = 0.0 + 0.0j
                for iuc in range(len(self.fc3.r_vector2[0])):
                    phase_factor = np.exp(complex(0.0, 2.0*np.pi*np.dot(self.k_points[ikpt], self.fc3.r_vector2[:,iuc])))
                    for iat in range(natom):
                        for jat in range(natom):
                            #rvec = self.fc3.r_vector2[:,iuc] + uc_positions[jat,:] - uc_positions[iat, :]
                            #phase_factor = np.exp(complex(0.0, 2.0*np.pi*np.dot(self.k_points[ikpt], rvec)))
                            mass_factor = 1.0/np.sqrt(masses[iat]*masses[jat])
                            for kat in range(natom):
                                #for i in range(3):
                                #    for j in range(3):
                                #        for k in range(3):
                                #            gruneisen += self.fc3.tensor[iuc,3*iat + i,3*jat + j,3*kat + k]*self.eigvecs[ikpt,iband,3*iat + i].conj()*self.eigvecs[ikpt,iband,3*jat + j]*(self.fc3.r_vector3[k,iuc] + uc_positions[kat,k])\
                                #                    *phase_factor*mass_factor
                                gruneisen += np.einsum('ijk, i, j, k', self.fc3.tensor[iuc,3*iat:3*(iat + 1),3*jat:3*(jat + 1),3*kat:3*(kat + 1)], self.eigvecs[ikpt,iband,3*iat:3*(iat+1)].conj(),
                                        self.eigvecs[ikpt,iband,3*jat:3*(jat+1)], self.fc3.r_vector3[:,iuc] + uc_positions[kat])*phase_factor*mass_factor
                if(self.freqs[self.qstar_list[ikpt][0], iband] > 0.0):
                    #print(gruneisen.real, gruneisen.imag)
                    for iqpt in range(len(self.qstar_list[ikpt])):
                        jqpt = self.qstar_list[ikpt][iqpt]
                        self.gruneisen[jqpt][iband] = -1.0*gruneisen.real/self.freqs[jqpt, iband]**2/6.0/BOHR_TO_ANGSTROM
                else:
                    for iqpt in range(len(self.qstar_list[ikpt])):
                        jqpt = self.qstar_list[ikpt][iqpt]
                        self.gruneisen[jqpt][iband] = 0.0

        with open('Gruneisen_parameter', 'w+') as outfile:
            outfile.write('#   ' + format('Frequencies (THz)', STR_FMT))
            outfile.write('    ' + format('Gruneisen_parameter', STR_FMT))
            outfile.write('\n')
            for ikpt in range(self.nkpt):
                for iband in range(self.nband):
                    outfile.write(3*' ' + format(self.freqs[ikpt][iband]*SSCHA_TO_THZ, '.12e'))
                    outfile.write(3*' ' + format(self.gruneisen[ikpt][iband], '.12e'))
                    outfile.write('\n')
