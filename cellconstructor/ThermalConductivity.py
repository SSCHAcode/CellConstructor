#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import os, sys
import scipy, scipy.optimize
from scipy import integrate

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

__SEEKPATH__ = False
try:
    import seekpath
    __SEEKPATH__ = True
except:
    __SEEKPATH__ = False

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
        'I' :[[1.0000, 126.9045]],\
        'Xe':[[0.0009,123.905896],  [0.0009,125.904269], [0.0192, 127.903530], [0.2644,128.904779], [0.0408,129.903508], [0.2118,130.905082], [0.2689,131.904154], [0.1044,133.905395], [0.0887,135.907220]],\
        'Cs':[[1.0000,132.905447]],\
        'Ba':[[0.00106,129.906310], [0.00101,131.905056], [0.02417,133.904503], [0.06592,134.905683], [0.07854,135.904570], [0.11232,136.905821], [0.71698,137.905241]],\
        'La':[[0.00090,137.907107], [0.99910,138.906348]],\
        'Ce':[[0.00185,135.907144], [0.00251,137.905986], [0.88450,139.905434], [0.11114,141.909240]],\
        'Pr':[[1.0000,140.907648]],\
        'Nd':[[0.272,141.907719], [0.122,142.909810], [0.238,143.910083], [0.083,144.912569], [0.172,145.913112], [0.057,147.916889], [0.056,149.920887]],\
        'Pm':[[1.000,144.912744]],\
        'Sm':[[0.0307,143.911995], [0.1499,146.914893], [0.1124,147.914818], [0.1382,148.917180], [0.0738,149.917271], [0.2675,151.919728], [0.2275,153.922205]],\
        'Eu':[[0.4781,150.919846], [0.5219,152.921226]],\
        'Gd':[[0.0020,151.919788], [0.0218,153.920862], [0.1480,154.922619], [0.2047,155.922120], [0.1565,156.923957], [0.2484,157.924101], [0.2186,159.927051]],\
        'Tb':[[1.0000,158.925343]],\
        'Dy':[[0.0006,155.924278], [0.0010,157.924405], [0.0234,159.925194], [0.1891,160.926930], [0.2551,161.926795], [0.2490,162.928728], [0.2818,163.929171]],\
        'Ho':[[1.0000,164.930319]],\
        'Er':[[0.0014,161.928775], [0.0161,163.929197], [0.3361,165.930290], [0.2293,166.932045], [0.2678,167.932368], [0.1493,169.935460]],\
        'Tm':[[1.0000,168.934211]],\
        'Yb':[[0.0013,167.933894], [0.0304,169.934759], [0.1428,170.936322], [0.2183,171.936378], [0.1613,172.938207], [0.3183,173.938858], [0.1276,175.942568]],\
        'Lu':[[0.9741,174.940768], [0.0259,175.942682]],\
        'Hf':[[0.0016,173.940040], [0.0526,175.941402], [0.1860,176.943220], [0.2728,177.943698], [0.1362,178.945815], [0.3508,179.946549]],\
        'Ta':[[0.00012,179.947466], [0.99988,180.947996]],\
        'W' :[[0.0012,179.946706], [0.2650,181.948206], [0.1431,182.950224], [0.3064,183.950933], [0.2843,185.954362]],\
        'Re':[[0.3740,184.952956], [0.6260,186.955751]],\
        'Os':[[0.0002,183.952491], [0.0159,185.953838], [0.0196,186.955748], [0.1324,187.955836], [0.1615,188.958145], [0.2626,189.958445], [0.4078,191.961479]],\
        'Ir':[[0.373,190.960591], [0.627,192.962924]],\
        'Pt':[[0.00014,189.959930], [0.00782,191.961035], [0.32967,193.962664], [0.33832,194.964774], [0.25242,195.964935], [0.07163,197.967876]],\
        'Au':[[1.0000,196.966552]],\
        'Hg':[[0.0015,195.965815], [0.0997,197.966752], [0.1687,198.968262], [0.2310,199.968309], [0.1318,200.970285], [0.2986,201.970626], [0.0687,203.973476]],\
        'Tl':[[0.29524,202.972329], [0.70476, 204.974412]],\
        'Pb':[[0.014,203.973029],[0.241,205.974449], [0.221,206.975881], [0.524,207.976636]],\
        'Bi':[[1.0000,208.980383]],}
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
                numbers[iat] = jat
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
            print(rot)
            raise RuntimeError('Not a rotation!')
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
        nkpts1 = int(np.floor(length/dl))
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


def centering_fc3(tensor3, check_for_symmetries = True, Far = 1):
    rprim = tensor3.unitcell_structure.unit_cell.copy()
    irprim = np.linalg.inv(rprim)
    rsup = tensor3.supercell_structure.unit_cell.copy()
    irsup = np.linalg.inv(rsup)
    positions = tensor3.unitcell_structure.coords.copy()
    xpos = np.dot(positions, np.linalg.inv(rprim))
    natom = len(xpos)
    symbols = tensor3.unitcell_structure.atoms
    unique_symbols = np.unique(symbols)
    unique_numbers = np.arange(len(unique_symbols), dtype=int) + 1
    numbers = np.zeros(len(symbols))
    for iat in range(len(symbols)):
        for jat in range(len(unique_symbols)):
            if(symbols[iat] == unique_symbols[jat]):
                numbers[iat] = unique_numbers[jat]
    cell = (rprim, xpos, numbers)
    if(tensor3.n_R == tensor3.n_sup**2):
        print('ForceTensor most likely not previously centered! ')
        if(check_for_symmetries):
            permutation = thermal_conductivity.third_order_cond_centering.check_permutation_symmetry(tensor3.tensor, tensor3.r_vector2.T, tensor3.r_vector3.T, tensor3.n_R, natom)
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
            maxtrip = thermal_conductivity.third_order_cond_centering.number_of_triplets(hfc3, hr_vector2, hr_vector3, tot_trip, natom, tensor3.n_R)
            fc3, r_vector2, r_vector3, ntrip = thermal_conductivity.third_order_cond_centering.distribute_fc3(hfc3, hr_vector2, hr_vector3, tot_trip, maxtrip, natom, tensor3.n_R)
            print('Final number of triplets: ', ntrip)
            tensor3.n_R = ntrip
            tensor3.r_vector2 = r_vector2[0:ntrip,:].T
            tensor3.r_vector3 = r_vector3[0:ntrip,:].T
            tensor3.x_r_vector2 = np.zeros_like(tensor3.r_vector2)
            tensor3.x_r_vector3 = np.zeros_like(tensor3.r_vector3)
            tensor3.tensor = fc3[0:ntrip]
            tensor3.x_r_vector2 = np.rint(np.dot(r_vector2[0:ntrip,:], irprim), dtype=float).T
            tensor3.x_r_vector3 = np.rint(np.dot(r_vector3[0:ntrip,:], irprim), dtype=float).T
            #write_fc3(tensor3)
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

def find_q_mq_pairs(kpoints):

    pairs = []
    found = [False for x in range(len(kpoints))]
    for iqpt in range(len(kpoints)):
        if(not found[iqpt]):
            kpt = kpoints[iqpt]
            for jqpt in range(len(kpoints)):
                if(not found[jqpt] and not found[iqpt] and iqpt != jqpt):
                    kpt1 = kpoints[jqpt]
                    if(np.linalg.norm(kpt + kpt1) < 1.0e-6):
                        found[iqpt] = True
                        found[jqpt] = True
                        pairs.append([iqpt, jqpt])
                        break
    return pairs

def apply_permutation_symmetry(tensor3, pairs):
        for ipair in range(len(pairs)):
            if(pairs[ipair][0] != pairs[ipair][1]):
                ip = pairs[ipair][0]
                jp = pairs[ipair][1]
                for i in range(3*tensor3.nat):
                    for iat in range(tensor3.nat):
                        for jat in range(tensor3.nat):
                            tensor3.tensor[ip,i,3*iat:3*(iat + 1), 3*jat:3*(jat+1)] = (tensor3.tensor[ip,i,3*iat:3*(iat + 1), 3*jat:3*(jat+1)] + tensor3.tensor[jp,i,3*jat:3*(jat + 1), 3*iat:3*(iat+1)].T)/2.0
                            tensor3.tensor[jp,i,3*jat:3*(jat + 1), 3*iat:3*(iat+1)] = tensor3.tensor[ip,i,3*iat:3*(iat + 1), 3*jat:3*(jat+1)].T

def find_minimum_vector(v1, m):
    rv1 = np.dot(v1, np.linalg.inv(m))
    rv2 = rv1 - np.rint(rv1)
    return np.dot(rv2, m)

def rotate_eigenvectors(ddm, eigs):

    _, eigvecs = np.linalg.eigh(np.dot(eigs.conj().T, np.dot(ddm, eigs)))
    rot_eigvecs = np.dot(eigvecs, eigs)

    return rot_eigvecs

def load_thermal_conductivity(filename = 'tc.pkl'):

    import pickle

    infile = open(filename, 'rb')
    tc = pickle.load(infile)
    infile.close()
    return tc

def get_mapping_of_q_points(star, points, rotations):

    mapping = []
    for istar in range(len(star)):
        star_mapping = []
        iqpt1 = star[istar][0]
        qpt1 = points[iqpt1]
        for iqpt in range(len(star[istar])):
            found = False
            qpt_mapping = []
            iqpt2 = star[istar][iqpt]
            qpt2 = points[iqpt2]
            if(np.linalg.norm(qpt2 + qpt1 - np.rint(qpt2 + qpt1)) < 1.0e-6):
                qpt_mapping.append((-1, True))
                found = True
            else:
                for irot in range(len(rotations)):
                    qpt21 = np.dot(rotations[irot].T, qpt1)
                    diffq = qpt21 - qpt2
                    addq = qpt21 + qpt2
                    if(np.linalg.norm(diffq - np.rint(diffq)) < 1.0e-6):
                        qpt_mapping.append((irot, False))
                        found = True
                    elif(np.linalg.norm(addq - np.rint(addq)) < 1.0e-6):
                        qpt_mapping.append((irot, True))
                        found = True
            if(found):
                star_mapping.append(qpt_mapping)
            else:
                print('QPT1: ', qpt1)
                print('QPT2: ', qpt2)
                raise RuntimeError('Could not find any mapping between qpoints!')
        mapping.append(star_mapping)

    return mapping

def construct_symmetry_matrix(rotation, translation, qvec, pos, atom_map, cell):
    matrix = np.zeros((len(pos)*3, len(pos)*3), dtype=complex)
    rx1 = []
    for iat in range(len(pos)):
        for jat in range(len(pos)):
            if(jat == atom_map[iat]):
                r = np.zeros_like(pos[iat])
                phase = 0.0
                r = pos[iat] - np.dot(rotation, pos[jat]) - translation
                rx = np.dot(r, np.linalg.inv(cell))
                rx1.append(rx)
                if(np.linalg.norm(rx - np.rint(rx))>1.0e-6):
                    print(rotation)
                    print(translation)
                    print(pos[iat], pos[jat])
                    print(atom_map)
                    print(rx)
                    raise RuntimeError('The atom is translated different than the translation vector!')
                #else:
                #    print(rx)
                #phase = np.dot(qvec, r)*2.0*np.pi
                phase = -1.0*np.dot(qvec, r)*2.0*np.pi
                #matrix[3*jat:3*(jat+1), 3*iat:3*(iat+1)] = rotation*np.exp(complex(0.0, phase))
                matrix[3*iat:3*(iat+1), 3*jat:3*(jat+1)] = rotation*np.exp(complex(0.0, phase))
    try:
        imatrix = np.linalg.inv(matrix)
        #print('Gamma is invertible!')
        if(not np.linalg.norm(imatrix - matrix.conj().T)/np.linalg.norm(matrix) < 1.0e-5):
            raise RuntimeError('The transformation matrix is not unitary!')
    except:
        print(np.matmul(matrix, matrix.conj().T))
        raise RuntimeError('Gamma is not invertible!')
    return matrix#, rx1, np.exp(complex(0.0, phase)), phase, qvec, np.dot(qvec, r)


class ThermalConductivity:

    def __init__(self, dyn, tensor3, kpoint_grid = 2, scattering_grid = None, smearing_scale = 1.0, smearing_type = 'adaptive', cp_mode = 'quantum', group_velocity_mode = 'analytical', off_diag = False, phase_conv = 'smooth'):

        """

        This class contains necesary routines to calculate lattice thermal conductivity using SSCHA auxiliary 2nd and 3rd order force constants.

        Parameters:

        Necesary:

            dyn                 : SSCHA dynamical matrix object
            tensor3             : SSCHA 3rd order force constants

            kpoint_grid         : Initializes the grid for Brillouin zone integration. It is used in the calculation of lattice thermal conductivity and
            the calculation of the phonon lifetimes. Default is 2.
            scattering_grid     : Grid size for the Brilouin integration in determining the phonon lifetimes. Default is to be the same as kpoint_grid
            smearing_scale      : Scale for the smearing constant if adaptive smearing is used. Default value is 2.0
            smearing_type       : Type of smearing used. Could be constant (same for all phonon modes) or adaptive (scaled by the phonon group velocity and the q point density).
            cp_mode             : Flag determining how phonon occupation factors are calculated (quantum/classical), default is quantum
            group_velocity_mode : How to calculate group velocities. 'analytical', 'finite_difference', 'wigner'
            off_diag            : Boolean parameter for the calculation of the off-diagonal elements of group velocity.
            phase_conv          : Phase convention for Fourier interpolation. Smooth (wrt atomic positions) or step (wrt to lattice vectors)

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
        if(smearing_scale == None):
            smearing_scale = 1.0
        self.smearing_scale = smearing_scale
        self.unitcell = self.dyn.structure.unit_cell
        #print('Primitive cell: ')
        #print(self.unitcell)
        self.supercell = self.dyn.structure.generate_supercell(dyn.GetSupercell()).unit_cell
        #print('Supercell: ')
        #print(self.supercell)
        self.phase_conv = phase_conv
        self.smearing_type = smearing_type
        self.cp_mode = cp_mode
        self.off_diag = off_diag
        self.volume = self.dyn.structure.get_volume()

        self.reciprocal_lattice = np.linalg.inv(self.unitcell).T
        self.force_constants = []
        self.ruc = []
        self.set_force_constants(dyn)
        self.nuc = len(self.ruc)

        self.symmetry = symmetries.QE_Symmetry(self.dyn.structure)
        self.symmetry.SetupQPoint()
        self.nsyms = self.symmetry.QE_nsym
        syms = np.array(self.symmetry.QE_s.copy()).transpose(2,0,1)

        self.set_kpoints_spglib()
        self.nband = 3*self.dyn.structure.N_atoms
        self.delta_omega = 0.0

        self.group_velocity_mode = group_velocity_mode
        self.freqs = np.zeros((self.nkpt, self.nband))
        self.gruneisen = np.zeros((self.nkpt, self.nband))
        self.eigvecs = np.zeros((self.nkpt, self.nband, self.nband), dtype=complex)
        self.dynmats = np.zeros((self.nkpt, self.nband, self.nband), dtype=complex)
        if(self.off_diag):
            self.gvels = np.zeros((self.nkpt, self.nband, self.nband, 3), dtype = complex)
        else:
            self.gvels = np.zeros((self.nkpt, self.nband, 3))
        self.ddms = np.zeros((self.nkpt, self.nband, self.nband, 3), dtype = complex)
        self.sigmas = np.zeros_like(self.freqs)
        # Lifetimes, frequency shifts, lineshapes, heat_capacities and thermal conductivities are stored in dictionaries
        # Dictionary key is the temperature at which property is calculated on
        self.lifetimes = {}
        self.freqs_shifts = {}
        self.lineshapes = {}
        self.cp = {}
        self.kappa = {}
        self.got_scattering_rates_isotopes = False

    ###################################################################################################################

    def save_pickle(self, filename = 'tc.pkl'):

        import pickle

        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    ###################################################################################################################################

    def set_kpoints_spglib(self):

        cell = get_spglib_cell(self.dyn)
        mapping, grid = spglib.get_ir_reciprocal_mesh(self.kpoint_grid, cell, is_shift=[0, 0, 0])
        symmetry_dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5, angle_tolerance=-1.0, hall_number=0)
        rotations = symmetry_dataset['rotations']
        translations = symmetry_dataset['translations']
        print('Spacegroup: ' + symmetry_dataset['international'] + str(symmetry_dataset['number']))
        self.atom_map = np.zeros((len(rotations), len(cell[1])), dtype=int)

        nrot = len(rotations)
        for irot in range(nrot):
            found = [False for jat in range(len(cell[1]))]
            new_pos = np.einsum('ij,jk->ik', cell[1], rotations[irot].T) + translations[irot]
            for iat in range(len(cell[1])):
                for jat in range(len(cell[1])):
                    diff = cell[1][jat] - new_pos[iat]
                    if(np.linalg.norm(diff - np.rint(diff)) < 1.0e-5):
                        self.atom_map[irot, iat] = jat
                        if(not found[iat]):
                            found[iat] = True
                        else:
                            print('Again found mapping to this atom!')
            if(not all(found)):
                raise RuntimeError('Could not find atom mapping for spacegroup symmetry: ' + str(irot + 1))

        self.irr_k_points = np.array(grid[np.unique(mapping)] / np.array(self.kpoint_grid, dtype=float))
        self.irr_k_points = np.dot(self.irr_k_points, self.reciprocal_lattice)
        self.qpoints = grid / np.array(self.kpoint_grid, dtype=float)
        self.k_points = np.dot(self.qpoints, self.reciprocal_lattice)
        self.rotations = np.array(rotations).copy()
        self.translations = np.array(translations).copy()

        self.qstar = []
        for i in np.unique(mapping):
            curr_star = []
            for j, j_map in enumerate(mapping):
                if(i == j_map):
                    curr_star.append(j)
            self.qstar.append(curr_star)

        self.little_group = [[] for x in range(len(self.k_points))]
        for istar in self.qstar:
            for iqpt in istar:
                curr_little_group = []
                for irot, rot in enumerate(rotations):
                    qpt1 = np.dot(rot.T, self.qpoints[iqpt])
                    diffq = qpt1 - self.qpoints[iqpt]
                    diffq -= np.rint(diffq)
                    if(np.linalg.norm(diffq) < 1.0e-6):
                        curr_little_group.append(irot)
                self.little_group[iqpt].extend(curr_little_group)
                #if(len(istar) * len(self.little_group[iqpt]) != len(rotations)):
                #    raise RuntimeError('Number of symmetry operation wrong!', len(istar), len(self.little_group[iqpt]), len(rotations))

        mapping1, grid1 = spglib.get_ir_reciprocal_mesh(self.scattering_grid, cell, is_shift=[0, 0, 0])
        self.scattering_qpoints = np.array(grid1 / np.array(self.scattering_grid, dtype=float))
        self.scattering_k_points = np.dot(self.scattering_qpoints, self.reciprocal_lattice)
        self.nkpt = np.shape(self.k_points)[0]
        self.scattering_nkpt = np.shape(self.scattering_k_points)[0]
        self.nirrkpt = np.shape(self.irr_k_points)[0]
        self.weights = np.zeros(self.nirrkpt, dtype = int)
        for iqpt in range(self.nirrkpt):
            self.weights[iqpt] = len(self.qstar[iqpt])
        self.set_up_scattering_grids = False

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
            q1 = self.qpoints[self.qstar[iqpt][0]]
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
                        if(same_vector(q3, curr_grid[i][0], np.eye(3)) and same_vector(q2, curr_grid[i][1], np.eye(3))):
                                pair_found = True
                                curr_w[i] += 1
                                break
                        else:
                            for isym in range(curr_nsym):
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
            if(np.all(tot_t[i] < 1.0e-6)):
                rotations.append(tot_r[i])
        rotations = np.asfortranarray(rotations)
        nsym = len(rotations)

        irrgrid = []
        for iqpt in range(self.nirrkpt):
            irrgrid.append(self.qpoints[self.qstar[iqpt][0]])
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
            self.scattering_grids.append(self.scattering_k_points)
            self.scattering_weights.append(np.ones(self.scattering_nkpt))
            if(sum(self.scattering_weights[-1]) != self.scattering_nkpt):
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
                if(np.all(self.freqs == 0.0)):
                    print('It is frequencies!')
                if(np.all(self.gvels == 0.0)):
                    print('It is group velocities!')
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
            min_smear = np.amax(self.sigmas)/10.0 # We can't have smearing zero for modes with 0 group velocity, so we set it to this number!
            self.sigmas[self.sigmas < min_smear] = min_smear
            if(np.all(self.sigmas == 0.0)):
                raise RuntimeError('All smearing values are zero!')
        elif(self.smearing_type == 'constant'):
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
        spec_kappa_off = np.zeros((3,3,self.lineshapes[ls_key].shape[-1]))
        energies = np.arange(spec_kappa.shape[-1], dtype=float)*self.delta_omega + self.delta_omega
        exponents_plus = np.exp(energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        integrands_plus = self.lineshapes[ls_key]**2*exponents_plus/(exponents_plus - 1.0)**2
        exponents_minus = np.exp(-1.0*energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        integrands_minus = self.lineshapes[ls_key]**2*exponents_minus/(exponents_minus - 1.0)**2
        integrands = (integrands_plus + integrands_minus)

        for istar in self.qstar:
            for iqpt in istar:
                for iband in range(self.nband):
                    if(self.freqs[iqpt, iband] != 0.0):
                        if(self.off_diag):
                            for jband in range(self.nband):
                                if(self.freqs[iqpt, jband] != 0.0):
                                    if(self.group_velocity_mode != 'wigner'):
                                        vel_fact = 1.0
                                    else:
                                        vel_fact = 2.0*np.sqrt(self.freqs[iqpt, jband]*self.freqs[iqpt, iband])/(self.freqs[iqpt, jband] + self.freqs[iqpt, iband]) # as per Eq.34 in Caldarelli et al
                                    if(iband == jband):
                                        gvel = np.zeros_like(self.gvels[iqpt, iband, iband])
                                        gvel_sum = np.zeros((3,3), dtype=complex)
                                        for r in self.rotations:
                                            rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                            gvel = np.dot(rot_q, self.gvels[iqpt, iband, iband])
                                            gvel_sum += np.outer(gvel.conj(), gvel)
                                        gvel_sum = gvel_sum.real*SSCHA_TO_MS**2/float(len(self.rotations))
                                        spec_kappa += np.einsum('ij,k->ijk',gvel_sum,integrands[iqpt, iband])*self.freqs[iqpt, iband]**2
                                    else:
                                        integrands_plus1 = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_plus/(exponents_plus - 1.0)**2
                                        integrands_minus1 = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_minus/(exponents_minus - 1.0)**2
                                        gvel = np.zeros_like(self.gvels[iqpt, iband, jband])
                                        gvel_sum = np.zeros((3,3), dtype=complex)
                                        for r in self.rotations:
                                            rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                            gvel = np.dot(rot_q, self.gvels[iqpt, iband, jband])
                                            gvel_sum += np.outer(gvel.conj(), gvel)
                                        gvel_sum = gvel_sum.real/vel_fact**2/float(len(self.rotations))
                                        spec_kappa_off += np.einsum('ij,k->ijk',gvel_sum,integrands_plus1 + integrands_minus1)*self.freqs[iqpt,iband]*self.freqs[iqpt,jband]*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/2.0
                        else:
                            gvel = np.zeros_like(self.gvels[iqpt, iband])
                            gvel_sum = np.zeros((3,3), dtype=complex)
                            for r in self.rotations:
                                rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                gvel = np.dot(rot_q, self.gvels[iqpt, iband])
                                gvel_sum += np.outer(gvel.conj(), gvel)
                            gvel_sum = gvel_sum.real*SSCHA_TO_MS**2/float(len(self.rotations))
                            spec_kappa += np.einsum('ij,k->ijk',gvel_sum,integrands[iqpt, iband])*self.freqs[iqpt, iband]**2
        for ie in range(np.shape(spec_kappa)[-1]):
            spec_kappa[:,:,ie] += spec_kappa[:,:,ie].T
            spec_kappa[:,:,ie] /= 2.0
            spec_kappa_off[:,:,ie] += spec_kappa_off[:,:,ie].T
            spec_kappa_off[:,:,ie] /= 2.0

        spec_kappa = spec_kappa*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi*(SSCHA_TO_THZ*2.0*np.pi)*1.0e12/2.0
        spec_kappa_off = spec_kappa_off*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi
        tot_kappa = np.sum(spec_kappa + spec_kappa_off, axis = len(spec_kappa) - 1)*self.delta_omega
        print('Total kappa is: ', np.diag(tot_kappa))

        return energies*SSCHA_TO_THZ, spec_kappa/SSCHA_TO_THZ, spec_kappa_off/SSCHA_TO_THZ

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
                        gvel = np.zeros_like(self.gvels[iqpt, iband, iband])
                        gvel_sum = np.zeros((3,3), dtype=complex)
                        for r in self.rotations:
                            rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                            gvel = np.dot(rot_q, self.gvels[iqpt, iband, iband])
                            gvel_sum += np.outer(gvel.conj(), gvel)
                        gvel_sum = gvel_sum.real/float(len(self.rotations))
                        spec_kappa[:,:,ien] += self.cp[key][iqpt,iband]*gvel_sum*self.lifetimes[key][iqpt][iband]*weight
                    else:
                        gvel = np.zeros_like(self.gvels[iqpt, iband])
                        gvel_sum = np.zeros((3,3), dtype=complex)
                        for r in self.rotations:
                            rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                            gvel = np.dot(rot_q, self.gvels[iqpt, iband])
                            gvel_sum += np.outer(gvel.conj(), gvel)
                        gvel_sum = gvel_sum.real/float(len(self.rotations))
                        spec_kappa[:,:,ien] += self.cp[key][iqpt,iband]*gvel_sum*self.lifetimes[key][iqpt][iband]*weight
        spec_kappa = spec_kappa*SSCHA_TO_MS**2/self.volume/float(self.nkpt)*1.0e30#*(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        tot_kappa = np.sum(spec_kappa, axis = len(spec_kappa) - 1)*delta_en
        print('Total kappa is: ', np.diag(tot_kappa))

        return energies*SSCHA_TO_THZ, spec_kappa/SSCHA_TO_THZ

   ####################################################################################################################################

    def calculate_kappa(self, temperatures = [300.0], write_lifetimes = True, mode = 'SRTA', offdiag_mode = 'wigner',  gauss_smearing = False, lf_method = 'fortran-LA', isotope_scattering = False, isotopes = None, \
            write_lineshapes=False, ne = 2000, mode_mixing = False, kappa_filename = 'Thermal_conductivity'):

        """
        Main function that calculates lattice thermal conductivity.

        temperatures     : list of temperatures to be calculated in K. Warning: Code does not recognize difference in temperatures smaller than 0.1 K.
        write_lifetimes  : Boolean parameter for writing lifetimes as they are being calculated.
        mode             : Method to calculate lattice thermal conductivity:
            SRTA         : Single relaxation time approximation (NOT selfconsistent solution) solution of Boltzmann transport equation
            GK           : Green-Kubo method (npj Computational Materials volume 7, Article number: 57 (2021))
        offdiag_mode     : How to treat the off diagonal terms. 'gk' - Isaeva et al. Nat. Comm., 'wigner' - Simoncelli et al. Nature
        gauss_smearing   : If true will use the Gaussian function to satisfy energy conservation insted of Lorentzian
        lf_method        : In case of mode == SRTA, specifies the way to calculate lifetimes. See method in get_lifetimes function.
        write_lineshapes : Boolean parameter to write phonon lineshapes as they are being calculated.
        ne               : Number of frequency points to calculate phonon lineshapes on in case of GK. \
                           Number of frequency points to solve self-consistent equation on in case of SRTA. \
                           Less anharmonic materials and lower temperatures will need more points (in case of GK).
        mode_mixing      : Calculate full self energy matrix
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
                    if(offdiag_mode == 'isaeva'):
                        kappa_diag, kappa_nondiag = self.calculate_kappa_srta_offdiag_isaeva(temperatures[itemp], ne, write_lifetimes, gauss_smearing = gauss_smearing, isotope_scattering=isotope_scattering, isotopes=isotopes, lf_method = lf_method)
                    elif(offdiag_mode == 'wigner'):
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
            elif(mode == 'GK' and not mode_mixing):
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
            elif(mode == 'GK' and mode_mixing):
                self.delta_omega = np.amax(self.freqs)*2.0/float(ne)
                energies = np.arange(ne, dtype=float)*self.delta_omega + self.delta_omega
                if(self.off_diag):
                    kappa = self.calculate_kappa_gk_offdiag_mode_mixing(temperatures[itemp], write_lineshapes, energies, gauss_smearing = gauss_smearing)
                    kappa_file.write(3*' ' + format(temperatures[itemp], '.12e'))
                    for icart in range(3):
                        kappa_file.write(3*' ' + format(kappa[icart][icart], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[0][1], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[1][2], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[2][0], '.12e'))
                    kappa_file.write('\n')
                    self.kappa[tc_key] = kappa
            elif(mode == 'AC'):
                self.delta_omega = np.amax(self.freqs)*2.0/float(ne)
                energies = np.arange(ne, dtype=float)*self.delta_omega + self.delta_omega
                if(self.off_diag):
                    kappa, kappa_nondiag, im_kappa, im_kappa_nondiag = self.calculate_kappa_gk_AC(temperatures[itemp], write_lineshapes, energies, gauss_smearing = gauss_smearing)
                    kappa_file.write(3*' ' + format(temperatures[itemp], '.12e'))
                    for icart in range(3):
                        kappa_file.write(3*' ' + format(kappa[icart][icart][0], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[0][1][0], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[1][2][0], '.12e'))
                    kappa_file.write(3*' ' + format(kappa[2][0][0], '.12e'))
                    for icart in range(3):
                        kappa_file.write(3*' ' + format(kappa_nondiag[icart][icart][0], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_nondiag[0][1][0], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_nondiag[1][2][0], '.12e'))
                    kappa_file.write(3*' ' + format(kappa_nondiag[2][0][0], '.12e'))
                    kappa_file.write('\n')
                    self.kappa[tc_key] = kappa
                    with open('AC_thermal_conductivity_' + format(temperatures[itemp], '.1f'), 'w+') as outfile:
                        outfile.write('# Temperature = ' + format(temperatures[itemp], '.1f') + '\n')
                        for ien in range(np.shape(kappa)[-1]):
                            if(ien == 0):
                                outfile.write(3*' ' + format(0.0, '.12e'))
                            else:
                                outfile.write(3*' ' + format(energies[ien - 1]*SSCHA_TO_THZ, '.12e'))
                            for icart in range(3):
                                outfile.write(3*' ' + format(kappa[icart][icart][ien], '.12e'))
                            outfile.write(3*' ' + format(kappa[0][1][ien], '.12e'))
                            outfile.write(3*' ' + format(kappa[1][2][ien], '.12e'))
                            outfile.write(3*' ' + format(kappa[2][0][ien], '.12e'))
                            for icart in range(3):
                                outfile.write(3*' ' + format(kappa_nondiag[icart][icart][ien], '.12e'))
                            outfile.write(3*' ' + format(kappa_nondiag[0][1][ien], '.12e'))
                            outfile.write(3*' ' + format(kappa_nondiag[1][2][ien], '.12e'))
                            outfile.write(3*' ' + format(kappa_nondiag[2][0][ien], '.12e'))
                            for icart in range(3):
                                outfile.write(3*' ' + format(im_kappa[icart][icart][ien], '.12e'))
                            outfile.write(3*' ' + format(im_kappa[0][1][ien], '.12e'))
                            outfile.write(3*' ' + format(im_kappa[1][2][ien], '.12e'))
                            outfile.write(3*' ' + format(im_kappa[2][0][ien], '.12e'))
                            for icart in range(3):
                                outfile.write(3*' ' + format(im_kappa_nondiag[icart][icart][ien], '.12e'))
                            outfile.write(3*' ' + format(im_kappa_nondiag[0][1][ien], '.12e'))
                            outfile.write(3*' ' + format(im_kappa_nondiag[1][2][ien], '.12e'))
                            outfile.write(3*' ' + format(im_kappa_nondiag[2][0][ien], '.12e'))
                            outfile.write('\n')
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
        #kappa = np.einsum('ijk,ijl,ij,ij,ij->kl', self.gvels.conj(), self.gvels, integrals, self.freqs, self.freqs).real*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        kappa = np.zeros((3,3))
        for istar in self.qstar:
            for iqpt in istar:
                for iband in range(self.nband):
                    if(self.freqs[iqpt, iband] != 0.0):
                        gvel = np.zeros_like(self.gvels[iqpt, iband])
                        gvel_sum = np.zeros_like(kappa, dtype=complex)
                        for r in self.rotations:
                            rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                            gvel = np.dot(rot_q, self.gvels[iqpt, iband])
                            gvel_sum += np.outer(gvel.conj(), gvel)
                        gvel_sum = gvel_sum.real*SSCHA_TO_MS**2/float(len(self.rotations))
                        kappa += gvel_sum*integrals[iqpt][iband]*self.freqs[iqpt][iband]**2

        kappa += kappa.T
        kappa = kappa/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi

        return kappa

    ##################################################################################################################################

    def calculate_kappa_gk_offdiag_mode_mixing(self, temperature, write_lineshapes, energies, gauss_smearing = False):

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
            self.get_lineshapes(temperature, write_lineshapes, energies, method = 'fortran', gauss_smearing = gauss_smearing, mode_mixing = 'mode_mixing')
        exponents_plus = np.exp(energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        exponents_minus = np.exp(-1.0*energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)

        kappa_diag = np.zeros((3,3), dtype = complex)
        kappa_nondiag = np.zeros((3,3), dtype = complex)
        for istar in self.qstar:
            for iqpt in istar:
                for iband in range(self.nband):
                    if(self.freqs[iqpt, iband] != 0.0):
                        for jband in range(self.nband):
                            if(self.freqs[iqpt, jband] != 0.0):
                                for kband in range(self.nband):
                                    if(self.freqs[iqpt, kband] != 0.0):
                                        for lband in range(self.nband):
                                            if(self.freqs[iqpt, lband] != 0.0):
                                                if(self.group_velocity_mode != 'wigner'):
                                                    vel_fact = 1.0
                                                else:
                                                    vel_fact = 2.0*np.sqrt(self.freqs[iqpt, jband]*self.freqs[iqpt, iband])/(self.freqs[iqpt, jband] + self.freqs[iqpt, iband]) # as per Eq.34 in Caldarelli et al
                                                #if(iband == jband and iband == kband and kband == lband):
                                                integrals = 0.0
                                                integrands1_plus = self.lineshapes[ls_key][iqpt, iband, lband]*np.conj(self.lineshapes[ls_key][iqpt, jband, kband])*exponents_plus/(exponents_plus - 1.0)**2
                                                integrands1_minus = self.lineshapes[ls_key][iqpt, iband, lband]*np.conj(self.lineshapes[ls_key][iqpt, jband, kband])*exponents_minus/(exponents_minus - 1.0)**2
                                                integrands2_plus = self.lineshapes[ls_key][iqpt, iband, kband]*np.conj(self.lineshapes[ls_key][iqpt, jband, lband])*exponents_plus/(exponents_plus - 1.0)**2
                                                integrands2_minus = self.lineshapes[ls_key][iqpt, iband, kband]*np.conj(self.lineshapes[ls_key][iqpt, jband, lband])*exponents_minus/(exponents_minus - 1.0)**2
                                                integrals = (np.sum(integrands1_plus, axis = len(integrands1_plus.shape) - 1) + np.sum(integrands1_minus, axis = len(integrands1_minus.shape) - 1)).real*self.delta_omega
                                                integrals += (np.sum(integrands2_plus, axis = len(integrands2_plus.shape) - 1) + np.sum(integrands2_minus, axis = len(integrands2_minus.shape) - 1)).real*self.delta_omega
                                                #if(np.abs(integrals.imag/integrals.real) > 1.0e-3 and np.abs(integrals.imag) > 1.0e-3):
                                                #    raise RuntimeError('Large imaginary part in integrals of spectral functions!', integrals)
                                                gvel1 = np.zeros_like(self.gvels[iqpt, iband, jband])
                                                gvel2 = np.zeros_like(self.gvels[iqpt, kband, lband])
                                                gvel_sum = np.zeros_like(kappa_diag, dtype=complex)
                                                for r in self.rotations:
                                                    rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                                    gvel1 = np.dot(rot_q, self.gvels[iqpt, iband, jband])
                                                    gvel2 = np.dot(rot_q, self.gvels[iqpt, kband, lband])
                                                    gvel_sum += np.outer(gvel1.conj(), gvel2)
                                                gvel_sum = gvel_sum.real/vel_fact**2/float(len(self.rotations))
                                                if(iband == jband and iband == kband and kband == lband):
                                                    kappa_diag += integrals*np.sqrt(self.freqs[iqpt,iband]*self.freqs[iqpt, kband]*self.freqs[iqpt,jband]*self.freqs[iqpt, lband])*\
                                                            gvel_sum*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/4.0
                                                else:
                                                    kappa_nondiag += integrals*np.sqrt(self.freqs[iqpt,iband]*self.freqs[iqpt, kband]*self.freqs[iqpt,jband]*self.freqs[iqpt, lband])*\
                                                            gvel_sum*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/4.0

        kappa_diag += kappa_diag.T
        kappa_diag = kappa_diag/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi
        kappa_nondiag += kappa_nondiag.T
        kappa_nondiag = kappa_nondiag/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi
        print(kappa_diag, kappa_nondiag)
        return kappa_diag.real + kappa_nondiag.real

    #################################################################################################################################

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
        #kappa_diag = np.zeros((3,3))
        exponents_plus = np.exp(energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        #integrands_plus = self.lineshapes[ls_key]**2*energies**2*exponents_plus/(exponents_plus - 1.0)**2
        #integrands_plus = self.lineshapes[ls_key]**2*exponents_plus/(exponents_plus - 1.0)**2
        exponents_minus = np.exp(-1.0*energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        #integrands_minus = self.lineshapes[ls_key]**2*energies**2*exponents_minus/(exponents_minus - 1.0)**2
        #integrands_minus = self.lineshapes[ls_key]**2*exponents_minus/(exponents_minus - 1.0)**2
        #integrals = (np.sum(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.sum(integrands_minus, axis = len(integrands_plus.shape) - 1))*self.delta_omega*(SSCHA_TO_THZ)*1.0e12/2.0*2.0*np.pi
        #kappa_diag = np.einsum('ijjk,ijjl,ij,ij,ij->kl', self.gvels.conj(), self.gvels, integrals, self.freqs, self.freqs).real*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        #kappa_diag += kappa_diag.T
        #kappa_diag = kappa_diag/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi

        #kappa_nondiag = np.zeros_like(kappa_diag)
        #for iqpt in range(self.nkpt):
        #    for iband in range(self.nband-1):
        #        if(self.freqs[iqpt, iband] != 0.0):
        #            for jband in range(iband, self.nband):
        #                if(iband != jband and self.freqs[iqpt, jband] != 0.0):
        #                    if(self.group_velocity_mode == 'wigner'):
        #                        vel_fact = 2.0*np.sqrt(self.freqs[iqpt, jband]*self.freqs[iqpt, iband])/(self.freqs[iqpt, jband] + self.freqs[iqpt, iband]) # as per Eq.34 in Caldarelli et al
        #                    else:
        #                        vel_fact = 1.0
        #                    integrands_plus = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_plus/(exponents_plus - 1.0)**2
        #                    integrands_minus = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_minus/(exponents_minus - 1.0)**2
        #                    integrals = (np.sum(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.sum(integrands_minus, axis = len(integrands_plus.shape) - 1))*self.delta_omega*(SSCHA_TO_THZ*2.0*np.pi)*1.0e12/4.0
        #                    kappa_nondiag += integrals*(self.freqs[iqpt, iband]**2 + self.freqs[iqpt, jband]**2)**2/self.freqs[iqpt][jband]/self.freqs[iqpt][iband]*\
        #                            np.outer(self.gvels[iqpt, iband, jband].conj(), self.gvels[iqpt, jband, iband]).real/vel_fact**2\
        #                            *SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2

        kappa_diag = np.zeros((3,3))
        kappa_nondiag = np.zeros_like(kappa_diag)
        for istar in self.qstar:
            for iqpt in istar:
                for iband in range(self.nband):
                    if(self.freqs[iqpt, iband] != 0.0):
                        for jband in range(self.nband):
                            if(self.group_velocity_mode != 'wigner'):
                                vel_fact = 1.0
                            else:
                                vel_fact = 2.0*np.sqrt(self.freqs[iqpt, jband]*self.freqs[iqpt, iband])/(self.freqs[iqpt, jband] + self.freqs[iqpt, iband]) # as per Eq.34 in Caldarelli et al
                            if(self.freqs[iqpt, jband] != 0.0 and iband != jband):
                                integrands_plus = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_plus/(exponents_plus - 1.0)**2
                                integrands_minus = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_minus/(exponents_minus - 1.0)**2
                                integrals = (np.sum(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.sum(integrands_minus, axis = len(integrands_plus.shape) - 1))*self.delta_omega
                                gvel = np.zeros_like(self.gvels[iqpt, iband, jband])
                                gvel_sum = np.zeros_like(kappa_diag, dtype=complex)
                                for r in self.rotations:
                                    rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                    gvel = np.dot(rot_q, self.gvels[iqpt, iband, jband])
                                    gvel_sum += np.outer(gvel.conj(), gvel)
                                gvel_sum = gvel_sum.real/vel_fact**2/float(len(self.rotations))
                                #kappa_nondiag += integrals*self.freqs[iqpt,iband]**2*(self.freqs[iqpt, iband]**2 + self.freqs[iqpt, jband]**2)/self.freqs[iqpt][jband]/self.freqs[iqpt][iband]*\
                                #        gvel_sum*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/4.0
                                kappa_nondiag += integrals*self.freqs[iqpt,iband]*self.freqs[iqpt,jband]*gvel_sum*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/2.0
                            elif(self.freqs[iqpt, jband] != 0.0 and iband == jband):
                                gvel = np.zeros_like(self.gvels[iqpt, iband, jband])
                                gvel_sum = np.zeros_like(kappa_diag, dtype=complex)
                                for r in self.rotations:
                                    rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                    gvel = np.dot(rot_q, self.gvels[iqpt, iband, iband])
                                    gvel_sum += np.outer(gvel.conj(), gvel)
                                gvel_sum = gvel_sum.real*vel_fact**2/float(len(self.rotations))
                                integrands_plus = self.lineshapes[ls_key][iqpt, iband]**2*exponents_plus/(exponents_plus - 1.0)**2
                                integrands_minus = self.lineshapes[ls_key][iqpt, iband]**2*exponents_minus/(exponents_minus - 1.0)**2
                                integrals = (np.sum(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.sum(integrands_minus, axis = len(integrands_plus.shape) - 1))*self.delta_omega
                                kappa_diag += gvel_sum*integrals*self.freqs[iqpt][iband]**2*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/2.0
                                # Factor of 1/2 comes from the fact that we multiplied the lineshapes with 2.0 after we calculated them!

        kappa_diag += kappa_diag.T
        kappa_diag = kappa_diag/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi

        kappa_nondiag += kappa_nondiag.T
        kappa_nondiag = kappa_nondiag/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi

        return kappa_diag, kappa_nondiag

    #################################################################################################################################

    def calculate_kappa_gk_AC(self, temperature, write_lineshapes, energies, gauss_smearing = False):

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

        ne = len(energies)
        exponents_plus = np.exp(energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        exponents_minus = np.exp(-1.0*energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature)
        x = energies*SSCHA_TO_THZ*1.0e12*HPLANCK/KB/temperature
        scale = (np.exp(x) - 1.0)/x
        scale = np.insert(scale, 0, 1.0)

        kappa_diag = np.zeros((3,3, ne + 1))
        kappa_diag1 = np.zeros((3,3))
        kappa_nondiag = np.zeros_like(kappa_diag)
        for istar in self.qstar:
            for iqpt in istar:
                for iband in range(self.nband):
                    if(self.freqs[iqpt, iband] != 0.0):
                        for jband in range(self.nband):
                            if(self.group_velocity_mode != 'wigner'):
                                vel_fact = 1.0
                            else:
                                vel_fact = 2.0*np.sqrt(self.freqs[iqpt, jband]*self.freqs[iqpt, iband])/(self.freqs[iqpt, jband] + self.freqs[iqpt, iband]) # as per Eq.34 in Caldarelli et al
                            if(self.freqs[iqpt, jband] != 0.0 and iband != jband):
                                i1 = np.append(np.append(np.flip(self.lineshapes[ls_key][iqpt, jband]*exponents_minus/(exponents_minus - 1.0)), np.zeros(1, dtype=float)), self.lineshapes[ls_key][iqpt, jband]*exponents_plus/(exponents_plus - 1.0))
                                i2 = np.append(np.append(np.flip(self.lineshapes[ls_key][iqpt, iband]/(exponents_minus - 1.0)), np.zeros(1, dtype=float)), self.lineshapes[ls_key][iqpt, iband]/(exponents_plus - 1.0))
                                integrals = np.correlate(i2, i1, mode = 'full')[len(i1) - 1:len(i1) + ne ]*self.delta_omega
                                i3 = np.append(np.append(np.flip(energies), np.zeros(1, dtype=float)), energies)
                                i4 = np.divide(i2, i3, out=np.zeros_like(i2), where=i3!=0.0)
                                integrals += np.append(np.zeros(1, dtype=float), energies)*np.correlate(i2, i1, mode = 'full')[len(i1) - 1:len(i1) + ne ]*self.delta_omega*0.5
                                #if(np.abs(np.sum(i1*i2)*self.delta_omega - integrals[0]) > np.abs(np.sum(i1*i2)*self.delta_omega)*1.0e-6):
                                #    print(np.sum(i1*i2)*self.delta_omega, integrals[0])
                                #    raise RuntimeError()
                                #integrands_plus = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_plus/(exponents_plus - 1.0)**2
                                #integrands_minus = self.lineshapes[ls_key][iqpt, iband]*self.lineshapes[ls_key][iqpt, jband]*exponents_minus/(exponents_minus - 1.0)**2
                                #integrals1 = (np.sum(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.sum(integrands_minus, axis = len(integrands_plus.shape) - 1))*self.delta_omega
                                #if(np.abs(integrals1 - integrals[0]) > np.abs(np.sum(i1*i2)*self.delta_omega)*1.0e-6):
                                #    print(integrals1, integrals[0])
                                #    raise RuntimeError()
                                gvel = np.zeros_like(self.gvels[iqpt, iband, jband])
                                gvel_sum = np.zeros_like(kappa_diag[:,:,0], dtype=complex)
                                for r in self.rotations:
                                    rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                    gvel = np.dot(rot_q, self.gvels[iqpt, iband, jband])
                                    gvel_sum += np.outer(gvel.conj(), gvel)
                                gvel_sum = gvel_sum.real*vel_fact**2/float(len(self.rotations))
                                #kappa_nondiag += integrals*self.freqs[iqpt,iband]*self.freqs[iqpt,jband]*gvel_sum*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/2.0
                                kappa_nondiag += np.einsum('ij,k->ijk', gvel_sum, integrals)*self.freqs[iqpt,iband]*self.freqs[iqpt,jband]*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/2.0
                            elif(self.freqs[iqpt, jband] != 0.0 and iband == jband):
                                gvel = np.zeros_like(self.gvels[iqpt, iband, jband])
                                gvel_sum = np.zeros_like(kappa_diag[:,:,0], dtype=complex)
                                for r in self.rotations:
                                    rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                    gvel = np.dot(rot_q, self.gvels[iqpt, iband, iband])
                                    gvel_sum += np.outer(gvel.conj(), gvel)
                                gvel_sum = gvel_sum.real*vel_fact**2/float(len(self.rotations))
                                i1 = np.append(np.append(np.flip(self.lineshapes[ls_key][iqpt, iband]*exponents_minus/(exponents_minus - 1.0)), np.zeros(1, dtype=float)), self.lineshapes[ls_key][iqpt, iband]*exponents_plus/(exponents_plus - 1.0))
                                i2 = np.append(np.append(np.flip(self.lineshapes[ls_key][iqpt, iband]/(exponents_minus - 1.0)), np.zeros(1, dtype=float)), self.lineshapes[ls_key][iqpt, iband]/(exponents_plus - 1.0))
                                integrals = np.correlate(i2, i1, mode = 'full')[len(i1)-1:len(i1) + ne]*self.delta_omega
                                i3 = np.append(np.append(np.flip(energies), np.zeros(1, dtype=float)), energies)
                                i4 = np.divide(i2, i3, out=np.zeros_like(i2), where=i3!=0.0)
                                integrals += np.append(np.zeros(1, dtype=float), energies)*np.correlate(i2, i1, mode = 'full')[len(i1) - 1:len(i1) + ne ]*self.delta_omega*0.5
                                #integrands_plus = self.lineshapes[ls_key][iqpt, iband]**2*exponents_plus/(exponents_plus - 1.0)**2
                                #integrands_minus = self.lineshapes[ls_key][iqpt, iband]**2*exponents_minus/(exponents_minus - 1.0)**2
                                #integrals1 = (np.sum(integrands_plus, axis = len(integrands_plus.shape) - 1) + np.sum(integrands_minus, axis = len(integrands_plus.shape) - 1))*self.delta_omega
                                #if(np.abs(integrals1 - integrals[0]) > np.abs(integrals1)*1.0e-6):
                                #    print(integrals1, integrals[0])
                                #    raise RuntimeError('1389')
                                #kappa_diag1 += gvel_sum*integrals1*self.freqs[iqpt][iband]**2*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/2.0
                                kappa_diag += np.einsum('ij,k->ijk', gvel_sum, integrals)*self.freqs[iqpt][iband]**2*SSCHA_TO_MS**2*(SSCHA_TO_THZ)*1.0e12*2.0*np.pi/2.0
                                #if(np.any(np.abs(kappa_diag[:,:,0] - kappa_diag1) > np.abs(kappa_diag1)*1.0e-6)):
                                #    print(kappa_diag1, kappa_diag[:,:,0])
                                #    raise RuntimeError('1394')
                                # Here freqs[iqpt,iband] is squared instead of power of 4 because imag self-energy in SSCHA is defined as 2*freqs[iqpt,iband]*Gamma[iqpt, iband]
                                # Factor of 1/2 comes from the fact that we multiplied the lineshapes with 2.0 after we calculated them!
        for ie in range(np.shape(kappa_diag)[-1]):
            kappa_diag[:,:,ie] += kappa_diag[:,:,ie].T
        kappa_diag = kappa_diag/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi

        for ie in range(np.shape(kappa_nondiag)[-1]):
            kappa_nondiag[:,:,ie] += kappa_nondiag[:,:,ie].T
        kappa_nondiag = kappa_nondiag/2.0*HBAR_JS**2/KB/temperature**2/self.volume/float(self.nkpt)*1.0e30*np.pi

        for i in range(3):
            for j in range(3):
                kappa_diag[i,j] *= scale
                kappa_nondiag[i,j] *= scale

        from scipy.signal import hilbert
        im_kappa_diag = np.zeros_like(kappa_diag)
        im_kappa_nondiag = np.zeros_like(kappa_nondiag)
        for i in range(3):
            for j in range(3):
                im_kappa_diag[i,j] = hilbert(kappa_diag[i,j]).imag
                im_kappa_nondiag[i,j] = hilbert(kappa_nondiag[i,j]).imag


        return kappa_diag, kappa_nondiag, im_kappa_diag, im_kappa_nondiag

    #################################################################################################################################

    def get_self_energy_at_q(self, iqpt, temperature, energies, mode_mixing = 'no', gauss_smearing = False, write_self_energy = False):

        if(self.delta_omega == 0.0 and energies is not None):
            self.delta_omega = energies[1] - energies[0]

        if(not self.set_up_scattering_grids):
            self.set_scattering_grids_simple()

        is_q_gamma = CC.Methods.is_gamma(self.fc2.unitcell_structure.unit_cell, self.k_points[iqpt])

        if(mode_mixing == 'mode_mixing'):
            self_energy = thermal_conductivity.get_lf.calculate_self_energy_full(self.freqs[iqpt], self.k_points[iqpt], self.eigvecs[iqpt], is_q_gamma, \
                    self.scattering_grids[iqpt].T, self.scattering_weights[iqpt], self.fc2.tensor, self.fc3.tensor, self.fc2.r_vector2, self.fc3.r_vector2, \
                    self.fc3.r_vector3, self.dyn.structure.coords.T, self.reciprocal_lattice, self.dyn.structure.get_masses_array(), self.sigmas[iqpt], \
                    temperature, energies, True, gauss_smearing, False, len(self.scattering_grids[iqpt]), self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), len(energies))
        elif(mode_mixing == 'no'):
            self_energy = thermal_conductivity.get_lf.calculate_self_energy_p(self.freqs[iqpt], self.k_points[iqpt], self.eigvecs[iqpt], is_q_gamma, \
                    self.scattering_grids[iqpt].T, self.scattering_weights[iqpt], self.fc2.tensor, self.fc3.tensor, self.fc2.r_vector2, self.fc3.r_vector2, \
                    self.fc3.r_vector3, self.dyn.structure.coords.T, self.reciprocal_lattice, self.dyn.structure.get_masses_array(), self.sigmas[iqpt], \
                    temperature, energies, True, gauss_smearing, False, len(self.scattering_grids[iqpt]), self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), len(energies))
        else:
            raise RuntimeError('The chosen option for mode_mixing(' + mode_mixing +  ') does not exist!')
        if(gauss_smearing):
            from scipy.signal import hilbert
            real_part = hilbert(self_energy.imag)
            self_energy.real = -1.0*real_part.imag

        self_energy /= float(self.nkpt)
        if(write_self_energy):
            with open('Self_energy_' + str(iqpt + 1), 'w+') as outfile:
                outfile.write('#   ' + format('Energy (THz)', STR_FMT))
                for iband in range(self.nband):
                    if(mode_mixing == 'mode_mixing'):
                        for jband in range(self.nband):
                            outfile.write('    ' + format('Self energy ' + str(iband) + ' - ' + str(jband) + ' (THz)', STR_FMT))
                    else:
                        outfile.write('    ' + format('Self energy ' + str(iband) +' (THz)', STR_FMT))
                outfile.write('\n')
                for ie in range(len(energies)):
                    outfile.write(3*' ' + format(energies[ie]*SSCHA_TO_THZ, '.12e'))
                    for iband in range(self.nband):
                        if(mode_mixing == 'mode_mixing'):
                            for jband in range(self.nband):
                                outfile.write(3*' ' + format(self_energy[ie, iband, jband].real*SSCHA_TO_THZ**2, '.12e') + ' ' + format(self_energy[ie, iband, jband].imag*SSCHA_TO_THZ**2, '.12e'))
                        else:
                            outfile.write(3*' ' + format(self_energy[ie, iband].real*SSCHA_TO_THZ**2, '.12e') + ' ' + format(self_energy[ie, iband].imag*SSCHA_TO_THZ**2, '.12e'))
                    outfile.write('\n')

        return self_energy

    #################################################################################################################################

    def get_lineshapes(self, temperature, write_lineshapes, energies, method = 'fortran', mode_mixing = 'no', gauss_smearing = False):

        """
        Calculate phonon lineshapes in full Brillouin zone.

        temperature      : temperature to calculate lineshapes on.
        write_lineshapes : Boolean parameter to write phonon lineshapes as they are being calculated.
        energies         : the list of frequencies for which lineshapes are calculated.
        method           : practically only determines how many times fortran routines are called. "fortran" should be much faster.
        mode_mixing      : Calculate full self-energy matrix ("mode_mixing" gives it in mode basis, "cartesian" gives it in cartesian basis)
        gauss_smearing   : are we using Gaussian smearing as approximation for energy conservation
        """

        start_time = time.time()
        if(self.delta_omega == 0.0 and energies is not None):
            self.delta_omega = energies[1] - energies[0]
        ls_key = format(temperature, '.1f')

        #if(mode_mixing != 'no'):
        #    print('WARNING! mode_mixing approach has been selected. Calculation of kappa in GK will not be possible!')

        if(method == 'python'):

            lineshapes = np.zeros((self.nkpt, self.nband, len(energies)))
            for ikpt in range(self.nirrkpt):
                jkpt = self.qstar[ikpt][0]
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

                for iqpt in range(len(self.qstar[ikpt])):
                    jqpt = self.qstar[ikpt][iqpt]
                    lineshapes[jqpt,:] = curr_ls*2.0
            print('Shape of lineshapes', lineshapes.shape)
            self.lineshapes[ls_key] = lineshapes

        elif(method == 'fortran'):

            if(mode_mixing != 'no'):
                lineshapes = np.zeros((self.nkpt, self.nband, self.nband, len(energies)), dtype=complex)
            else:
                lineshapes = np.zeros((self.nkpt, self.nband, len(energies)))
            if(not self.set_up_scattering_grids):
                self.set_scattering_grids_simple()

            irrqgrid = np.zeros((3, self.nirrkpt))
            scattering_events = np.zeros(self.nirrkpt, dtype=int)
            sigmas = np.zeros((self.nirrkpt, self.nband))
            for ikpt in range(self.nirrkpt):
                irrqgrid[:,ikpt] = self.k_points[self.qstar[ikpt][0]].copy()
                scattering_events[ikpt] = len(self.scattering_grids[ikpt])
                sigmas[ikpt] = self.sigmas[self.qstar[ikpt][0]]
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

            if(mode_mixing == 'mode_mixing'):
                curr_ls = thermal_conductivity.get_lf.calculate_lineshapes_mode_mixing(irrqgrid, scattering_grids, weights, scattering_events,\
                        self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, self.fc3.r_vector3, \
                        self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(),\
                        sigmas.T, np.zeros_like(sigmas.T, dtype=float), temperature, gauss_smearing, classical, energies, len(energies), self.nirrkpt, \
                        self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)
            elif(mode_mixing == 'cartesian'):
                curr_ls = thermal_conductivity.get_lf.calculate_lineshapes_cartesian(irrqgrid, scattering_grids, weights, scattering_events,\
                        self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, self.fc3.r_vector3, \
                        self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(),\
                        sigmas.T, np.zeros_like(sigmas.T, dtype=float), temperature, gauss_smearing, classical, energies, len(energies), self.nirrkpt, \
                        self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)
            elif(mode_mixing == 'no'):
                curr_ls = thermal_conductivity.get_lf.calculate_lineshapes(irrqgrid, scattering_grids, weights, scattering_events,\
                        self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, self.fc3.r_vector3, \
                        self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(),\
                        sigmas.T, np.zeros_like(sigmas.T, dtype=float), temperature, gauss_smearing, classical, energies, len(energies), self.nirrkpt, \
                        self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)
            else:
                print('Selected mode_mixing approach: ', mode_mixing)
                raise RuntimeError('Do not recognize the selected mode_mixing approach!')
            scaled_positions = np.dot(self.dyn.structure.coords, np.linalg.inv(self.unitcell))
            rotations, translations = self.get_sg_in_cartesian()
            mapping = get_mapping_of_q_points(self.qstar, self.qpoints, self.rotations)
            for ikpt in range(self.nirrkpt):
                jkpt = self.qstar[ikpt][0]
                if(mode_mixing == 'no'):
                    if(CC.Methods.is_gamma(self.dyn.structure.unit_cell, self.k_points[jkpt])):
                        for iband in range(self.nband):
                            if(self.freqs[jkpt, iband] < np.amax(self.freqs[jkpt])*1.0e-6):
                                curr_ls[ikpt, iband] = 0.0

                for iqpt in range(len(self.qstar[ikpt])):
                    jqpt = self.qstar[ikpt][iqpt]
                    found = False
                    if(mode_mixing != 'no'):
                        #for iband in range(self.nband):
                        #    for jband in range(self.nband):
                        #        curr_ls[ikpt, iband,jband,:] = curr_ls[ikpt, iband,jband,:]/np.sum(curr_ls[ikpt,iband,jband,:])/(energies[1]-energies[0]) # Forcing the normalization. Not sure if the best option!
                        if(iqpt == 0):
                            lineshapes[jqpt,:,:,:] = curr_ls[ikpt,:,:,:]
                            found = True
                        else:
                            qpt1 = self.qpoints[self.qstar[ikpt][0]]
                            qpt2 = self.qpoints[jqpt]
                            if(np.linalg.norm(qpt2 + qpt1 - np.rint(qpt2 + qpt1)) < 1.0e-6):
                                lineshapes[jqpt,:,:,:] = curr_ls[ikpt,:,:,:].conj()
                                found = True
                            else:
                                irot = mapping[ikpt][iqpt][0][0]
                                atom_map = self.atom_map[irot]
                                qpt21 = np.dot(self.rotations[irot].T, qpt1)
                                kpt21 = np.dot(qpt21, self.reciprocal_lattice)
                                gamma = construct_symmetry_matrix(rotations[irot], translations[irot], kpt21, self.dyn.structure.coords, atom_map, self.unitcell)
                                lineshapes[jqpt,:,:,:] = np.einsum('ij,jkl,km->iml', gamma, curr_ls[ikpt,:,:,:], gamma.conj().T)
                                if(mapping[ikpt][iqpt][0][1]):
                                    lineshapes[jqpt,:,:,:] = lineshapes[jqpt,:,:,:].conj()
                        #tot_const_diag = 0.0
                        #tot_const_nondiag = 0.0
                        #for iband in range(len(lineshapes[jqpt])):
                        #    tot_const_diag += np.sum(lineshapes[jqpt][iband][iband]).real*(energies[2] - energies[1])
                        #    for jband in range(len(lineshapes[jqpt][iband])):
                        #        print('Normalization constant (' + str(iband + 1) + ',' + str(jband + 1)+ '): ', np.sum(lineshapes[jqpt][iband][jband]).real*(energies[2] - energies[1]))
                        #        tot_const_nondiag += np.sum(lineshapes[jqpt][iband][jband]).real*(energies[2] - energies[1])
                        #print('Normalization constant diagonal: ', tot_const_diag)
                        #print('Normalization constant all elements: ', tot_const_diag)
                    else:
                        lineshapes[jqpt,:,:] = curr_ls[ikpt,:,:]*2.0
                if(write_lineshapes):
                    filename = 'Lineshape_irrkpt_' + str(jkpt) + '_T_' + format(temperature, '.1f')
                    self.write_lineshape(filename, lineshapes[jkpt], jkpt, energies, mode_mixing)
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
            self.set_scattering_grids_fortran()

        irrqgrid = np.zeros((3, self.nirrkpt))
        scattering_events = np.zeros(self.nirrkpt, dtype=int)
        sigmas = np.zeros((self.nirrkpt, self.nband))
        for ikpt in range(self.nirrkpt):
            irrqgrid[:,ikpt] = self.k_points[self.qstar[ikpt][0]].copy()
            scattering_events[ikpt] = len(self.scattering_grids[ikpt])
            sigmas[ikpt] = self.sigmas[self.qstar[ikpt][0]]
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
            for iqpt in range(len(self.qstar[ikpt])):
                jqpt = self.qstar[ikpt][iqpt]
                lifetimes[jqpt,:] = -1.0*np.divide(np.ones_like(selfengs[ikpt].imag, dtype=float), selfengs[ikpt].imag, out=np.zeros_like(selfengs[ikpt].imag), where=selfengs[ikpt].imag!=0.0)/2.0
                shifts[jqpt,:] = selfengs[ikpt].real

        self.lifetimes[lf_key] = lifetimes/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)
        self.freqs_shifts[lf_key] = shifts

        print('Calculated SSCHA lifetimes in: ', time.time() - start_time)

    ##################################################################################################################################

    def get_lineshapes_along_the_line(self, temperature, ne = 1000, filename = 'spectral_function_along_path', gauss_smearing = True, mode_mixing = 'no', kpoints = None, start_nkpts = 100, smear = None):

        """
        Calculate phonon lineshapes for specific k-points.

        temperature      : temperature to calculate lineshapes on.
        ne               : Number of frequency points for the lineshapes
        gauss_smearing   : are we using Gaussian smearing as approximation for energy conservation
        mode_mixing      : If true will calculate full phonon spectral function
        kpoints          : the list of kpoints in reduced coordinates to calculate lineshapes for.
                           If not provided generate them using seekpath
        start_nkpts      : Number of k points along the path. Will differ from the final number of points!
        smear            : Smearing used for energy conservation.

        """

        if(not __SEEKPATH__ and kpoints is None):
            raise RuntimeError('To automatically generated a line in reciprocal space one need seekpath. First do "pip install seekpath"!')

        start_time = time.time()

        tics = []
        distances = []
        segments = []
        if(kpoints is None):
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
            kpoints = np.array(kpoints)#np.dot(kpoints, self.reciprocal_lattice)
            distances = np.zeros(len(kpoints), dtype=float)
            for ikpt in range(1, len(kpoints)):
                distances[ikpt] = distances[ikpt - 1] + np.linalg.norm(kpoints[ikpt] - kpoints[ikpt - 1])
            if(len(distances) > 1):
                distances /= distances[-1]
        nkpts = len(kpoints)
        freqs = np.zeros((nkpts, self.nband))
        for ikpt in range(nkpts):
            freqs[ikpt], _, _ = self.get_frequency_at_q(kpoints[ikpt])

        maxfreq = np.amax(freqs)*2.1
        energies = (np.arange(ne, dtype=float) + 1.0)/float(ne)*maxfreq

        if(mode_mixing != 'no'):
            lineshapes = np.zeros((nkpts, self.nband, self.nband, ne), dtype = complex)
        else:
            lineshapes = np.zeros((nkpts, self.nband, ne))

        irrqgrid = kpoints.T
        scattering_events = np.zeros(nkpts, dtype=int)
        if(smear is None):
            # One has to provide smearing. Otherwise we are using the largest smearing from the full Brillouin zone!
            sigmas = np.zeros((nkpts, self.nband))
            sigmas[:,:] = np.amax(self.sigmas)
        else:
            if(nkpts > 1):
                if(np.shape(smear) == np.shape(freqs)):
                    sigmas = smear
                else:
                    print(np.shape(smear), np.shape(freqs))
                    raise RuntimeError('Smearing array does not match shape of the kpoints!')
            else:
                if(len(smear) == len(freqs[0])):
                    sigmas = smear
                else:
                    raise RuntimeError('Smearing array does not match shape of the kpoints!')
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

        if(mode_mixing == 'mode_mixing'):
            curr_ls = thermal_conductivity.get_lf.calculate_lineshapes_mode_mixing(irrqgrid, scattering_grids, weights, scattering_events,\
                    self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, self.fc3.r_vector3, \
                    self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(),\
                    sigmas.T, np.zeros_like(sigmas.T, dtype=float), temperature, gauss_smearing, classical, energies, len(energies), nkpts, \
                    self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)
        elif(mode_mixing == 'cartesian'):
            curr_ls = thermal_conductivity.get_lf.calculate_lineshapes_cartesian(irrqgrid, scattering_grids, weights, scattering_events,\
                    self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, self.fc3.r_vector3, \
                    self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(),\
                    sigmas.T, np.zeros_like(sigmas.T, dtype=float), temperature, gauss_smearing, classical, energies, len(energies), nkpts, \
                    self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)
        elif(mode_mixing == 'no'):
            curr_ls = thermal_conductivity.get_lf.calculate_lineshapes(irrqgrid, scattering_grids, weights, scattering_events,\
                    self.fc2.tensor, self.fc2.r_vector2, self.fc3.tensor, self.fc3.r_vector2, self.fc3.r_vector3, \
                    self.unitcell, self.dyn.structure.coords.T, self.dyn.structure.get_masses_array(),\
                    sigmas.T, np.zeros_like(sigmas.T, dtype=float), temperature, gauss_smearing, classical, energies, len(energies), nkpts, \
                    self.dyn.structure.N_atoms, len(self.fc2.tensor), len(self.fc3.tensor), num_scattering_events)
        else:
            raise RuntimeError('Unknown mode_mixing method! ')

        for ikpt in range(nkpts):
            if(mode_mixing == 'no'):
                if(CC.Methods.is_gamma(self.dyn.structure.unit_cell, kpoints[ikpt])):
                    for iband in range(self.nband):
                        if(freqs[ikpt, iband] < np.amax(freqs[ikpt])*1.0e-6):
                            curr_ls[ikpt, iband] = 0.0
            if(mode_mixing != 'no'):
                lineshapes[ikpt,:,:,:] = curr_ls[ikpt,:,:,:]*2.0
                tot_const_diag = 0.0
                tot_const_nondiag = 0.0
                for iband in range(len(lineshapes[ikpt])):
                    tot_const_diag += np.sum(lineshapes[ikpt][iband][iband]).real*(energies[2] - energies[1])
                    for jband in range(len(lineshapes[ikpt][iband])):
                        #print('Normalization constant (' + str(iband + 1) + ',' + str(jband + 1)+ '): ', np.sum(lineshapes[ikpt][iband][jband])*(energies[2] - energies[1]))
                        tot_const_nondiag += np.sum(lineshapes[ikpt][iband][jband]).real*(energies[2] - energies[1])
                print('Normalization constant diagonal: ', tot_const_diag)
                print('Normalization constant all elements: ', tot_const_diag)
            else:
                lineshapes[ikpt,:,:] = curr_ls[ikpt,:,:]*2.0
                for iband in range(len(lineshapes[ikpt])):
                    print('Normalization constant: ', np.sum(lineshapes[ikpt][iband])*(energies[2] - energies[1]))

        with open('Qpoints_along_line', 'w+') as outfile:
            for ikpt in range(nkpts):
                qpt = np.dot(kpoints[ikpt], np.linalg.inv(self.reciprocal_lattice))
                for i in range(3):
                    outfile.write(3*' ' + format(qpt[i], '.12f'))
                outfile.write('\n')

        with open(filename, 'w+') as outfile:
            if(len(tics) > 0 and len(segments) > 0):
                outfile.write('# Path and tics: \n')
                outfile.write('# ' + segments[0][0] + '  ' + format(tics[0], '.8f'))
                for i in range(len(segments) - 1):
                    if(segments[i][1] == segments[i + 1][0]):
                        outfile.write('  ' + segments[i][1] + '  ' + format(tics[i+1], '.8f'))
                    else:
                        outfile.write('  ' + segments[i][1] + ' | ' + segments[i+1][0] + '  ' + format(tics[i+1], '.8f'))
                outfile.write('  ' + segments[len(segments)-1][1] + '  ' + format(tics[len(segments)], '.8f') + '\n')
                outfile.write('# normalized distance       energy (THz)         lineshape (1/THz) \n')
            else:
                outfile.write('# User defined kpoint line ... \n')
            for ikpt in range(nkpts):
                for ie in range(ne):
                    outfile.write('  ' + format(distances[ikpt], '.12e'))
                    outfile.write('  ' + format(energies[ie]*SSCHA_TO_THZ, '.12e'))
                    diag = complex(0.0,0.0)
                    for iband in range(self.nband):
                        if(mode_mixing != 'no'):
                            for jband in range(self.nband):
                                outfile.write('  ' + format(lineshapes[ikpt,iband, jband, ie].real/SSCHA_TO_THZ, '.12e'))
                                outfile.write('  ' + format(lineshapes[ikpt,iband, jband, ie].imag/SSCHA_TO_THZ, '.12e'))
                                if(iband == jband):
                                    diag += lineshapes[ikpt,iband, jband, ie]
                        else:
                            outfile.write('  ' + format(lineshapes[ikpt,iband,ie]/SSCHA_TO_THZ, '.12e'))
                    if(mode_mixing != 'no'):
                        outfile.write(3*' ' + format(np.sum(lineshapes[ikpt, :, :, ie]).real/SSCHA_TO_THZ, '.12e'))
                        outfile.write(3*' ' + format(np.sum(lineshapes[ikpt, :, :, ie]).imag/SSCHA_TO_THZ, '.12e'))
                        outfile.write(3*' ' + format(diag.real/SSCHA_TO_THZ, '.12e'))
                        outfile.write(3*' ' + format(diag.imag/SSCHA_TO_THZ, '.12e'))
                    else:
                        outfile.write(3*' ' + format(np.sum(lineshapes[ikpt, :, ie])/SSCHA_TO_THZ, '.12e'))
                    outfile.write('\n')
                outfile.write('\n')

        with open(filename + '_phonons', 'w+') as outfile:
            if(len(tics) > 0 and len(segments) > 0):
                outfile.write('# Path and tics: \n')
                outfile.write('# ' + segments[0][0] + '  ' + format(tics[0], '.8f'))
                for i in range(len(segments) - 1):
                    if(segments[i][1] == segments[i + 1][0]):
                        outfile.write('  ' + segments[i][1] + '  ' + format(tics[i+1], '.8f'))
                    else:
                        outfile.write('  ' + segments[i][1] + ' | ' + segments[i+1][0] + '  ' + format(tics[i+1], '.8f'))
                outfile.write('  ' + segments[-1][1] + '  ' + format(tics[-1], '.8f') + '\n')
            else:
                outfile.write('# User defined kpoints ... \n')
            outfile.write('# normalized distance       frequency (THz)          \n')
            for ikpt in range(nkpts):
                outfile.write('  ' + format(distances[ikpt], '.12e'))
                for iband in range(self.nband):
                    outfile.write('  ' + format(freqs[ikpt,iband]*SSCHA_TO_THZ, '.12e'))
                outfile.write('\n')

        print('Calculated SSCHA lineshapes in: ', time.time() - start_time)
        return energies, lineshapes

    ##################################################################################################################################
    def write_lineshape(self, filename, curr_ls, jkpt, energies, mode_mixing):

        """

        Function to write phonon lineshapes onto a file.

        filename       : title of the file at which lineshape is to be written.
        curr_ls        : lineshape to be written
        jkpt           : the index of the k point for which lineshapes are to be written
        energies       : frequencies at which lineshapes have been calculated
        mode_mixing    : Defines the shape of input curr_ls. if true curr_ls = (nband, nband, ne)

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
                diag_lineshape = complex(0.0, 0.0)
                outfile.write(3*' ' + format(energies[ien]*SSCHA_TO_THZ, '.12e'))
                for iband in range(self.nband):
                    if(mode_mixing != 'no'):
                        for jband in range(self.nband):
                            outfile.write(3*' ' + format(curr_ls[iband, jband, ien].real/SSCHA_TO_THZ, '.12e'))
                            outfile.write(3*' ' + format(curr_ls[iband, jband, ien].imag/SSCHA_TO_THZ, '.12e'))
                            if(iband == jband):
                                diag_lineshape += curr_ls[iband, iband, ien]
                    else:
                        outfile.write(3*' ' + format(curr_ls[iband, ien]/SSCHA_TO_THZ, '.12e'))
                if(mode_mixing != 'no'):
                    outfile.write(3*' ' + format(np.sum(curr_ls[:, :, ien]).real/SSCHA_TO_THZ, '.12e'))
                    outfile.write(3*' ' + format(np.sum(curr_ls[:, :, ien]).imag/SSCHA_TO_THZ, '.12e'))
                    outfile.write(3*' ' + format(diag_lineshape.real/SSCHA_TO_THZ, '.12e'))
                    outfile.write(3*' ' + format(diag_lineshape.imag/SSCHA_TO_THZ, '.12e'))
                else:
                    outfile.write(3*' ' + format(np.sum(curr_ls[:, ien])/SSCHA_TO_THZ, '.12e'))
                outfile.write('\n')

    ##################################################################################################################################

    def get_heat_capacity(self, temperature, shifted = False):

        """
        Calculate phonon mode heat capacity at temperature.

        shifted :: Boolean to tell us whether we shift frequencies by the real part of the self-energy!

        """

        cp_key = format(temperature, '.1f')
        if(shifted):
            freqs = self.freqs + self.freqs_shifts[cp_key]
        else:
            freqs = self.freqs.copy()
        cp = np.zeros_like(self.freqs)
        for ikpt in range(self.nkpt):
            for iband in range(self.nband):
                cp[ikpt, iband] = heat_capacity(freqs[ikpt, iband]*SSCHA_TO_THZ*1.0e12, temperature, HPLANCK, KB, cp_mode = self.cp_mode)
        self.cp[cp_key] = cp


    ##################################################################################################################################

    def calculate_kappa_srta_diag(self, temperature, ne, write_lifetimes, gauss_smearing = False, isotope_scattering = False, isotopes = None, lf_method = 'fortran-LA'):

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
            if(lf_method == 'SC'):
                self.get_heat_capacity(temperature)
            else:
                self.get_heat_capacity(temperature)

        if(write_lifetimes):
            self.write_transport_properties_to_file(temperature, isotope_scattering)

        kappa = np.zeros((3,3))
        for istar in self.qstar:
            for iqpt in istar:
                for iband in range(self.nband):
                    if(self.freqs[iqpt, iband] != 0.0):
                        gvel = np.zeros_like(self.gvels[iqpt, iband])
                        gvel_sum = np.zeros_like(kappa, dtype=complex)
                        for r in self.rotations:
                            rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                            gvel = np.dot(rot_q, self.gvels[iqpt, iband])
                            gvel_sum += np.outer(gvel.conj(), gvel)
                        gvel_sum = gvel_sum.real/float(len(self.rotations))
                        kappa += self.cp[cp_key][iqpt][iband]*gvel_sum*self.lifetimes[lf_key][iqpt][iband]
        kappa += kappa.T
        kappa = kappa/2.0*SSCHA_TO_MS**2

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
            if(lf_method == 'SC'):
                self.get_heat_capacity(temperature)
            else:
                self.get_heat_capacity(temperature)
        scatt_rates = np.divide(np.ones_like(self.lifetimes[lf_key], dtype=float), self.lifetimes[lf_key], out=np.zeros_like(self.lifetimes[lf_key]), where=self.lifetimes[lf_key]!=0.0)/(SSCHA_TO_THZ*1.0e12*2.0*np.pi)
        if(write_lifetimes):
            self.write_transport_properties_to_file(temperature, isotope_scattering)

        kappa_diag = np.zeros((3,3))
        kappa_nondiag = np.zeros_like(kappa_diag)
        for istar in self.qstar:
            for iqpt in istar:
                for iband in range(self.nband):
                    if(self.freqs[iqpt, iband] != 0.0 and scatt_rates[iqpt, iband] != 0.0):
                        for jband in range(self.nband):
                            #if(self.freqs[iqpt, jband] != 0.0 and np.abs(self.freqs[iqpt, jband] - self.freqs[iqpt, iband]) > 1.0e-4/SSCHA_TO_THZ and iband != jband):
                            if(self.freqs[iqpt, jband] != 0.0 and scatt_rates[iqpt, jband] != 0.0 and iband != jband):
                                if(self.group_velocity_mode == 'wigner'):
                                    vel_fact = 1.0
                                else:
                                    vel_fact = 2.0*np.sqrt(self.freqs[iqpt, jband]*self.freqs[iqpt, iband])/(self.freqs[iqpt, jband] + self.freqs[iqpt, iband]) # as per Eq.34 in Caldarelli et al
                                gvel = np.zeros_like(self.gvels[iqpt, iband, jband])
                                gvel_sum = np.zeros_like(kappa_diag, dtype=complex)
                                for r in self.rotations:
                                    rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                    gvel = np.dot(rot_q, self.gvels[iqpt, iband, jband])
                                    gvel_sum += np.outer(gvel.conj(), gvel)
                                gvel_sum = gvel_sum.real*vel_fact**2
                                a1 = (self.freqs[iqpt, jband] + self.freqs[iqpt, iband])/4.0
                                a2 = self.cp[cp_key][iqpt, iband]/self.freqs[iqpt, iband] + self.cp[cp_key][iqpt, jband]/self.freqs[iqpt, jband]
                                a3 = 0.5*(scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])
                                a4 = ((self.freqs[iqpt,iband] - self.freqs[iqpt,jband]))**2 + (scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])**2/4.0
                                kappa_nondiag += a1*a2*a3/a4*gvel_sum/2.0/np.pi/float(len(self.rotations))#*float(len(istar))
                            elif(self.freqs[iqpt, jband] != 0.0 and iband == jband):
                                if(self.group_velocity_mode == 'wigner'):
                                    vel_fact = 1.0
                                else:
                                    vel_fact = 2.0*np.sqrt(self.freqs[iqpt, jband]*self.freqs[iqpt, iband])/(self.freqs[iqpt, jband] + self.freqs[iqpt, iband]) # as per Eq.34 in Caldarelli et al
                                gvel_sum = np.zeros_like(kappa_diag, dtype=complex)
                                gvel = np.zeros_like(self.gvels[iqpt, iband, jband])
                                for r in self.rotations:
                                    rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                    gvel = np.dot(rot_q, self.gvels[iqpt, iband, jband])
                                    gvel_sum += np.outer(gvel.conj(), gvel)
                                gvel_sum = gvel_sum.real*vel_fact**2
                                a1 = (self.freqs[iqpt, jband] + self.freqs[iqpt, iband])/4.0
                                a2 = self.cp[cp_key][iqpt, iband]/self.freqs[iqpt, iband] + self.cp[cp_key][iqpt, jband]/self.freqs[iqpt, jband]
                                a3 = 0.5*(scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])
                                a4 = (self.freqs[iqpt,iband] - self.freqs[iqpt,jband])**2 + (scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])**2/4.0
                                kappa_diag += a1*a2*a3/a4*gvel_sum/2.0/np.pi/float(len(self.rotations))#*float(len(istar))

        kappa_diag = kappa_diag/SSCHA_TO_THZ/1.0e12
        kappa_nondiag = kappa_nondiag/SSCHA_TO_THZ/1.0e12

        kappa_diag += kappa_diag.T
        kappa_nondiag += kappa_nondiag.T
        kappa_diag = kappa_diag/2.0*SSCHA_TO_MS**2#*(SSCHA_TO_THZ*100.0*2.0*np.pi)**2
        kappa_nondiag = kappa_nondiag/2.0*SSCHA_TO_MS**2#(SSCHA_TO_THZ*100.0*2.0*np.pi)**2


        return kappa_diag, kappa_nondiag

    ################################################################################################################################################################################

    def calculate_kappa_srta_offdiag_isaeva(self, temperature, ne, write_lifetimes, gauss_smearing = False, isotope_scattering = False, isotopes = None, lf_method = 'fortran-LA'):

        """
        Calculates both diagonal and off diagonal contribution to the lattice thermal conductivity (Nature Communications volume 10, Article number: 3853 (2019)).
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
        scatt_rates = np.divide(np.ones_like(self.lifetimes[lf_key], dtype=float), self.lifetimes[lf_key], out=np.zeros_like(self.lifetimes[lf_key]), where=self.lifetimes[lf_key]!=0.0)/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)/2.0
        if(write_lifetimes):
            self.write_transport_properties_to_file(temperature, isotope_scattering)
        #kappa_diag = np.einsum('ij,ijjk,ijjl,ij->kl',self.cp[cp_key],self.gvels.conj(),self.gvels,self.lifetimes[lf_key]).real

        pops = np.zeros_like(self.freqs)
        for iqpt in range(self.nkpt):
            for iband in range(self.nband):
                pops[iqpt, iband] = bose_einstein(self.freqs[iqpt, iband]*SSCHA_TO_THZ*1.0e12, temperature, HPLANCK, KB, cp_mode = self.cp_mode)

        kappa_diag = np.zeros((3,3))
        kappa_nondiag = np.zeros_like(kappa_diag)
        for iqpt in range(self.nkpt):
            for iband in range(self.nband):
                if(self.freqs[iqpt, iband] != 0.0):
                    for jband in range(self.nband):
                        if(self.group_velocity_mode == 'wigner'):
                            vel_fact = 2.0*np.sqrt(self.freqs[iqpt, jband]*self.freqs[iqpt, iband])/(self.freqs[iqpt, jband] + self.freqs[iqpt, iband]) # as per Eq.34 in Caldarelli et al
                        else:
                            vel_fact = 1.0
                        if(self.freqs[iqpt, jband] != 0.0 and self.freqs[iqpt, jband] - self.freqs[iqpt, iband] != 0.0 and iband != jband):
                            gvel_sum = np.zeros_like(kappa_diag, dtype=complex)
                            gvel = np.zeros_like(self.gvels[iqpt, iband, jband])
                            for r in self.rotations:
                                rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                gvel = np.dot(rot_q, self.gvels[iqpt, iband, jband])
                                gvel_sum += np.outer(gvel.conj(), gvel)
                            gvel_sum = gvel_sum.real/vel_fact**2/float(len(self.rotations))
                            kappa_nondiag += self.freqs[iqpt, iband]*self.freqs[iqpt, jband]*(pops[iqpt, iband] - pops[iqpt, jband])/(self.freqs[iqpt, jband] - self.freqs[iqpt, iband])*\
                                gvel_sum*(scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])/\
                                ((self.freqs[iqpt,iband] - self.freqs[iqpt,jband])**2 + (scatt_rates[iqpt, iband] + scatt_rates[iqpt, jband])**2)
                        elif(self.freqs[iqpt, jband] != 0.0 and iband == jband):
                            gvel_sum = np.zeros_like(kappa_diag, dtype=complex)
                            gvel = np.zeros_like(self.gvels[iqpt, iband, jband])
                            for r in self.rotations:
                                rot_q = np.dot(self.reciprocal_lattice.T, np.dot(r.T, np.linalg.inv(self.reciprocal_lattice.T)))
                                gvel = np.dot(rot_q, self.gvels[iqpt, iband, jband])
                                gvel_sum += np.outer(gvel.conj(), gvel)
                            gvel_sum = gvel_sum.real/vel_fact**2/float(len(self.rotations))
                            kappa_diag += self.cp[cp_key][iqpt][iband]*gvel_sum*self.lifetimes[lf_key][iqpt][iband]

        kappa_nondiag = kappa_nondiag*HPLANCK/temperature/2.0/np.pi

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

        """

        Still not implemented!

        """

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
            jkpt = self.qstar[ikpt][0]
            curr_scatt_rate = self.get_scattering_rates_isotope_at_q(jkpt, g_factor, av_mass)
            for iqpt in range(len(self.qstar[ikpt])):
                jqpt = self.qstar[ikpt][iqpt]
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
                        factor += g_factor[iat]*np.dot(self.eigvecs[iqpt,3*iat:3*(iat+1),iband].conj(), self.eigvecs[jqpt,3*iat:3*(iat+1),jband])**2
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
                jkpt = self.qstar[ikpt][0]
                print('Calculating lifetimes: ' + format(float(ikpt)/float(self.nirrkpt)*100.0, '.2f') + ' %')
                curr_freq, curr_shift, curr_lw = self.get_lifetimes_at_q(self.kpoint_grid, self.k_points[jkpt], self.sigmas[jkpt], temperature)
                curr_lf = np.divide(np.ones_like(curr_lw, dtype=float), curr_lw, out=np.zeros_like(curr_lw), where=curr_lw!=0.0)/2.0
                if(CC.Methods.is_gamma(self.dyn.structure.unit_cell, self.k_points[jkpt])):
                    for iband in range(self.nband):
                        if(self.freqs[jkpt, iband] < np.amax(self.freqs[jkpt])*1.0e-6):
                            curr_lf[iband] = 0.0
                            curr_shift[iband] = 0.0
                for iqpt in range(len(self.qstar[ikpt])):
                    jqpt = self.qstar[ikpt][iqpt]
                    lifetimes[jqpt,:] = curr_lf
                    shifts[jqpt,:] = curr_shift
            print('Shape of lifetimes', lifetimes.shape)
            self.lifetimes[lf_key] = lifetimes/(SSCHA_TO_THZ*2.0*np.pi*1.0e12)
            self.freqs_shifts[lf_key] = shifts

        elif(method == 'python-P'):

            lifetimes = np.zeros((self.nkpt, self.nband))
            shifts = np.zeros((self.nkpt, self.nband))
            for ikpt in range(self.nirrkpt):
                jkpt = self.qstar[ikpt][0]
                print('Calculating lifetimes: ' + format(float(ikpt)/float(self.nirrkpt)*100.0, '.2f') + ' %')
                selfnrg = np.diag(self.get_just_diag_dynamic_bubble(self.kpoint_grid, self.k_points[jkpt], self.sigmas[jkpt], self.freqs[jkpt], temperature))
                curr_lf = -1.0*np.divide(selfnrg.imag, self.freqs[jkpt], out=np.zeros_like(self.freqs[jkpt]), where=self.freqs[jkpt]!=0.0)/2.0
                curr_shifts = selfnrg.real
                if(CC.Methods.is_gamma(self.dyn.structure.unit_cell, self.k_points[jkpt])):
                    for iband in range(self.nband):
                        if(self.freqs[jkpt, iband] < np.amax(self.freqs[jkpt])*1.0e-6):
                            curr_lf[iband] = 0.0
                            curr_shifts[iband] = 0.0
                for iqpt in range(len(self.qstar[ikpt])):
                    jqpt = self.qstar[ikpt][iqpt]
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
                irrqgrid[:,ikpt] = self.k_points[self.qstar[ikpt][0]].copy()
                scattering_events[ikpt] = len(self.scattering_grids[ikpt])
                sigmas[ikpt] = self.sigmas[self.qstar[ikpt][0]]
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
                for iqpt in range(len(self.qstar[ikpt])):
                    jqpt = self.qstar[ikpt][iqpt]
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
                irrqgrid[:,ikpt] = self.k_points[self.qstar[ikpt][0]].copy()
                scattering_events[ikpt] = len(self.scattering_grids[ikpt])
                sigmas[ikpt] = self.sigmas[self.qstar[ikpt][0]]
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
                for iqpt in range(len(self.qstar[ikpt])):
                    jqpt = self.qstar[ikpt][iqpt]
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

    ###################################################################################################################################

    def set_lifetimes(self, temperatures, lifetimes, freqs_shifts):

        for temperature in temperatures:
            lf_key = format(temperature, '.1f')
            if(lf_key in self.lifetimes.keys()):
                print('WARNING! This will overwrite the previously set lifetimes!')
            self.lifetimes[lf_key] = lifetimes
            self.freqs_shifts[lf_key] = freqs_shifts


   ####################################################################################################################################

    def setup_harmonic_properties(self, smearing_value = 0.00005, symmetrize = True):

        """

        Sets up harmonic properties (calculates frequencies, group velocities and smearing parameters.)

        smearing_value : Value of the smearing in case smearing_method == "constant"
        """

        for ikpt, kpt in enumerate(self.k_points):
            self.freqs[ikpt], self.eigvecs[ikpt], self.dynmats[ikpt] = self.get_frequency_at_q(kpt)
            if(self.group_velocity_mode == 'wigner'):
                self.gvels[ikpt], self.ddms[ikpt] = self.get_group_velocity_wigner(kpt)
            elif(self.group_velocity_mode == 'analytical'):
                self.gvels[ikpt], self.ddms[ikpt] = self.get_group_velocity(kpt, self.freqs[ikpt], self.eigvecs[ikpt])
            elif(self.group_velocity_mode == 'finite_difference'):
                self.gvels[ikpt], self.ddms[ikpt] = self.get_group_velocity_finite_difference(kpt, self.freqs[ikpt], self.eigvecs[ikpt])
            else:
                raise RuntimeError('Can not recognize the group_velocity_mode!')
            if(symmetrize):
                self.gvels[ikpt] = self.symmetrize_group_velocity_by_index(self.gvels[ikpt], ikpt)

        #if(symmetrize):
        #    self.symmetrize_group_velocities_over_star()
        #self.check_group_velocities()
        #self.check_frequencies()
        #self.check_dynamical_matrices()
        self.setup_smearings(smearing_value)
        print('Harmonic properties are set up!')

    #################################################################################################################################

    def symmetrize_eigenvectors(self):

        pairs = find_q_mq_pairs(self.k_points)
        print('Found ' + str(len(pairs)) + ' q, -q pairs!')
        for ipair in range(len(pairs)):
            if(np.any(np.abs(self.eigvecs[pairs[0]] - self.eigvecs[pairs[1]].conj()) > 1.0e-6)):
                print('Eigenvector symmetry not satisfied!')
                self.eigvecs[pairs[0]] = (self.eigvecs[pairs[0]] + self.eigvecs[pairs[1]].conj())/2.0
                self.eigvecs[pairs[1]] = self.eigvecs[pairs[0]].conj()

    #################################################################################################################################

    def get_sg_in_cartesian(self):

        rotations = np.zeros_like(self.rotations, dtype=float)
        translations = np.zeros_like(self.translations, dtype=float)
        for irot in range(len(rotations)):
            rotations[irot] = np.matmul(self.reciprocal_lattice.T, np.matmul(self.rotations[irot].T, np.linalg.inv(self.reciprocal_lattice.T)))
        translations = np.dot(self.translations, self.unitcell)

        return rotations, translations

    #################################################################################################################################

    def check_dynamical_matrices(self):

        rotations, translations = self.get_sg_in_cartesian()
        #for irot in range(len(rotations)):
        #    print('SG' + str(irot + 1))
        #    print(rotations[irot])
        #    print(translations[irot])
        #    print(tc.atom_map[irot])
        mapping = get_mapping_of_q_points(self.qstar, self.qpoints, self.rotations)
        for istar in range(len(self.qstar)):
            iqpt1 = self.qstar[istar][0]
            qpt1 = self.qpoints[iqpt1]
            kpt1 = self.k_points[iqpt1]
            dyn1 = self.dynmats[iqpt1].copy()
            for iqpt in range(1, len(self.qstar[istar])):
                iqpt2 = self.qstar[istar][iqpt]
                qpt2 = self.qpoints[iqpt2]
                kpt2 = self.k_points[iqpt2]
                dyn2 = self.dynmats[iqpt2].copy()
                for imap in range(len(mapping[istar][iqpt])):
                    irot = mapping[istar][iqpt][imap][0]
                    atom_map = self.atom_map[irot]
                    conjugate = mapping[istar][iqpt][imap][1]
                    qpt21 = np.dot(self.rotations[irot].T, qpt1)
                    kpt22 = np.dot(rotations[irot], kpt1)
                    kpt21 = np.dot(self.reciprocal_lattice.T, qpt21)
                    if(np.linalg.norm(kpt21 - kpt22) > 1.0e-6):
                        print(np.linalg.norm(kpt21 - kpt22))
                        print('Rotation in cartesian and reduced coordinates gives different results!')
                    diffq = qpt21 - qpt2
                    addq = qpt21 + qpt2
                    if(irot == -1):
                        dyn21 = dyn1.conj()
                        if(np.any(np.abs(dyn21 - dyn2)/np.amax(np.abs(dyn2)) > 1.0e-2)):
                            print('Some differences between rotated and original dynamical matrices!')
                            print(np.abs(dyn21 - dyn2)/np.amax(np.abs(dyn2)) > 1.0e-2)
                            print(qpt1, qpt2)
                            for iband in range(len(dyn1)):
                                print(dyn2[iband])
                                print(dyn21[iband])
                                print(dyn2[iband] - dyn21[iband])
                                print('')
                            #raise RuntimeError('Mapping dynamical matrices from q to -q did not work!')
                    else:
                        if(np.linalg.norm(diffq - np.rint(diffq)) < 1.0e-6 and not conjugate):
                            gamma = construct_symmetry_matrix(rotations[irot], translations[irot], kpt21, self.dyn.structure.coords, atom_map, self.unitcell)
                            dyn21 = np.matmul(gamma, np.matmul(dyn1, gamma.conj().T))
                            if(np.any(np.abs(dyn21 - dyn2)/np.amax(np.abs(dyn2)) > 1.0e-2)):
                                print('Some differences between rotated and original dynamical matrices!')
                                print(rotations[irot])
                                print(np.abs(dyn21 - dyn2)/np.amax(np.abs(dyn2)) > 1.0e-2)
                                print(qpt1, qpt2)
                                for iband in range(len(dyn1)):
                                    print(dyn2[iband])
                                    print(dyn21[iband])
                                    print(dyn2[iband] - dyn21[iband])
                                    print('')
                                #raise RuntimeError('Mapping dynamical matrices from q to q star did not work!')
                        elif(np.linalg.norm(addq - np.rint(addq)) < 1.0e-6 and conjugate):
                            gamma = construct_symmetry_matrix(rotations[irot], translations[irot], kpt21, self.dyn.structure.coords, atom_map, self.unitcell)
                            dyn21 = np.matmul(gamma, np.matmul(dyn1, gamma.conj().T))
                            dyn21 = dyn21.conj()
                            if(np.any(np.abs(dyn21 - dyn2)/np.amax(np.abs(dyn2)) > 1.0e-2)):
                                print('Some differences between rotated and original dynamical matrices!')
                                print(rotations[irot])
                                print(np.abs(dyn21 - dyn2)/np.amax(np.abs(dyn2)) > 1.0e-2)
                                print(qpt1, qpt2)
                                for iband in range(len(dyn1)):
                                    print(dyn2[iband])
                                    print(dyn21[iband])
                                    print(dyn2[iband] - dyn21[iband])
                                    print('')
                                #raise RuntimeError('Mapping dynamical matrices from q to q star through -q did not work!')
                        else:
                            raise RuntimeError('The mapping was wrong! This rotation does not give expected q point!')

        print('Dynamical matrices satisfy symmetries!')

    #################################################################################################################################

    def check_frequencies(self):

        """

        Routine to check whether the frequencies in q star are all the same

        """

        for istar in range(self.nirrkpt):
            freqs0 = self.freqs[self.qstar[istar][0]]
            for jqpt in range(1, len(self.qstar[istar])):
                freqs1 = self.freqs[self.qstar[istar][jqpt]]
                if(np.any(np.abs(freqs0 - freqs1) > 1.0e-6*np.amax(freqs0))):
                    print('WARNING! Frequencies in star not the same. ', istar, jqpt)
                    print(freqs0)
                    print(freqs1)

    #################################################################################################################################

    def check_group_velocities(self):

        """

        Check if group velocities in q star are all related by symmetry operation !

        """

        for istar in self.qstar:
            q0 = self.qpoints[istar[0]]
            vel0 = self.gvels[istar[0]].copy()
            for iqpt in istar:
                found_rot = False
                q1 =  self.qpoints[iqpt]
                for irot in range(len(self.rotations)):
                    q2 = np.dot(self.rotations[irot].T, q1)
                    diffq = q2 - q0
                    diffq -= np.rint(diffq)
                    if(np.linalg.norm(diffq) < 1.0e-6):
                        #rotation = np.dot(self.reciprocal_lattice.T, np.dot(np.linalg.inv(self.rotations[irot].T), np.linalg.inv(self.reciprocal_lattice.T)))
                        rotation = np.dot(self.reciprocal_lattice.T, np.dot(self.rotations[irot].T, np.linalg.inv(self.reciprocal_lattice.T)))
                        found_rot = True
                        break
                if(found_rot):
                    if(self.off_diag):
                        vel1 = np.einsum('ij,klj->kli', rotation, self.gvels[iqpt])
                    else:
                        vel1 = np.einsum('ij,kj->ki', rotation, self.gvels[iqpt])
                    if(np.any(np.abs(vel0 - vel1) > 1.0e-4)):
                        print('Velocities in star not agreeing!', istar, iqpt)
                        print(vel0[np.where(vel0 - vel1 > 1.0e-4)])
                        print(vel1[np.where(vel0 - vel1 > 1.0e-4)])
                else:
                    print('Could not find rotation between vectors in star! ', self.qpoints[iqpt], self.qpoints[istar[0]])

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
            gvel = np.zeros((self.nband, self.nband, 3), dtype = complex)
        else:
            gvel = np.zeros((self.nband, 3), dtype = complex)
        m = np.tile(self.dyn.structure.get_masses_array(), (3,1)).T.ravel()
        mm_mat = np.sqrt(np.outer(m, m))
        mm_inv_mat = 1.0 / mm_mat

        degs = check_degeneracy(freqs, np.amax(freqs)*1.0e-8)

        ddynmat = []
        for icart in range(3):
            auxfc = np.zeros_like(self.force_constants[0], dtype = complex)
            for iuc in range(len(self.force_constants)):
                for iat in range(len(uc_positions)):
                    for jat in range(len(uc_positions)):
                        ruc = -self.ruc[iuc] + uc_positions[iat] - uc_positions[jat]
                        phase = np.dot(ruc, q)*2.0*np.pi
                        auxfc[3*iat:3*(iat+1),3*jat:3*(jat+1)] += complex(0.0,1.0)*ruc[icart]*self.force_constants[iuc,3*iat:3*(iat+1),3*jat:3*(jat+1)]*np.exp(1j*phase)
            ddynmat.append(auxfc * mm_inv_mat)
            ddynmat[-1] += ddynmat[-1].conj().T
            ddynmat[-1] /= 2.0
            if(icart == 0):
                dirdynmat = ddynmat.copy()

        new_eigvecs = np.zeros_like(eigvecs)
        for deg in degs:
            _, eigvecs1 = np.linalg.eigh(np.dot(eigvecs[:,deg].T.conj(), np.dot(np.sum(ddynmat,axis=0)/3.0, eigvecs[:,deg])))
            new_eigvecs[:,deg] = np.dot(eigvecs[:,deg], eigvecs1)

        ddynmat = np.array(ddynmat)
        tmp_gvel = []
        freqs_matrix = np.einsum('i,j->ij', freqs, freqs)
        freqs_matrix = np.divide(np.ones_like(freqs_matrix), freqs_matrix, out=np.zeros_like(freqs_matrix), where=freqs_matrix!=0.0)
        for icart in range(3):
            tmp_gvel.append(np.dot(new_eigvecs.T.conj(), np.dot(ddynmat[icart], new_eigvecs))/2.0*np.sqrt(freqs_matrix))
        tmp_gvel = np.array(tmp_gvel).transpose((1,2,0))

        if(not self.off_diag):
            gvel = np.einsum('iij->ij', tmp_gvel)
        else:
            for i in range(3):
                gvel[:,:,i] = (tmp_gvel[:,:,i].conj().T + tmp_gvel[:,:,i])/2.0

        if(np.any(np.isnan(tmp_gvel))):
            raise RuntimeError('NaN is group velocity matrix!')

        return gvel, ddynmat.transpose((1,2,0))

    #################################################################################################################################

    def get_group_velocity_wigner(self, q):

        """

        Calculate group velocities following definition in "Wigner formulation of thermal transport in solids" Simoncelli et al.

        """


        is_q_gamma = CC.Methods.is_gamma(self.fc2.unitcell_structure.unit_cell, q)
        if(self.off_diag):
            tmp_gvel = np.zeros((self.nband, self.nband, 3))
            gvel = np.zeros((self.nband, self.nband, 3))
        else:
            tmp_gvel = np.zeros((self.nband, 3))
            gvel = np.zeros((self.nband, 3))

        dynmat0 = self.get_dynamical_matrix(q)
        eigvals, eigvecs = np.linalg.eigh(dynmat0)

        degs = check_degeneracy(eigvals, np.amax(eigvals)*1.0e-8)
        ddynmat = []
        for icart in range(3):
            dq = np.zeros_like(q)
            dq[icart] = np.sum(np.amax(np.linalg.norm(self.reciprocal_lattice[:,:], axis = 1)))/100000.0
            q1 = q + dq
            dynmat1 = self.get_dynamical_matrix(q1)
            sqrt_dynmat1 = self.sqrt_dynamical_matrix(is_q_gamma, dynmat1)
            q2 = q - dq
            dynmat2 = self.get_dynamical_matrix(q2)
            sqrt_dynmat2 = self.sqrt_dynamical_matrix(is_q_gamma, dynmat2)
            ddynmat.append((sqrt_dynmat1 - sqrt_dynmat2)/np.linalg.norm(dq)/2.0/2.0/np.pi)
            if(icart == 0):
                dirdynmat = ddynmat[0].copy()

        #rot_eigvecs = np.zeros_like(eigvecs)
        #for ideg, deg in enumerate(degs):
        #    rot_eigvecs[deg, :] = rotate_eigenvectors(dirdynmat, eigvecs[deg, :])
        tmp_gvel = []
        for icart in range(3):
            tmp_gvel.append(np.dot(eigvecs.T.conj(), np.dot(ddynmat[icart], eigvecs)))
        tmp_gvel = np.array(tmp_gvel).transpose((1,2,0))

        if(not self.off_diag):
            gvel = np.einsum('iij->ij', tmp_gvel).real
        else:
            gvel = np.zeros_like(tmp_gvel)
            for i in range(3):
                gvel[:,:,i] = (tmp_gvel[:,:,i].conj().T + tmp_gvel[:,:,i])/2.0

        return gvel, np.array(ddynmat).transpose((1,2,0))

    ##################################################################################################################################

    def sqrt_dynamical_matrix(self, flag_gamma, dm):

        eigvals, eigvecs = np.linalg.eigh(dm)
        eigvals = eigvals.real
        freqs = np.sign(eigvals) * np.sqrt(abs(eigvals))

        if flag_gamma:
            freqs[0] = 0.0
            eigvecs[:, 0] = 0.0
            freqs[1] = 0.0
            eigvecs[:, 1] = 0.0
            freqs[2] = 0.0
            eigvecs[:, 2] = 0.0
        if any(f < 0.0 for f in freqs):
            print("ERROR: negative frequency=", freqs)

        omega_matrix = np.sqrt(np.matmul(eigvecs.T.conj(), np.matmul(dm, eigvecs)))
        omega_matrix = np.diag(np.diag(omega_matrix))
        sqrt_dm = np.matmul(eigvecs, np.matmul(omega_matrix, eigvecs.T.conj()))

        return sqrt_dm

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

        ddynmat = []
        for icart in range(3):
            dynmat0 = self.get_dynamical_matrix(q)
            dq = np.zeros_like(q)
            dq[icart] = np.sum(np.linalg.norm(self.reciprocal_lattice[:,icart]))/1000.0
            q1 = q + dq
            dynmat1 = self.get_dynamical_matrix(q1)
            q2 = q - dq
            dynmat2 = self.get_dynamical_matrix(q2)
            ddynmat.append((dynmat1 - dynmat2)/np.linalg.norm(dq)/2.0/2.0/np.pi)
            if(icart == 0):
                dirdynmat = ddynmat[0].copy()
        ddynmat = np.array(ddynmat)

        new_eigvecs = np.zeros_like(eigvecs)
        for deg in degs:
            _, eigvecs1 = np.linalg.eigh(np.dot(eigvecs[:,deg].T.conj(), np.dot(np.sum(ddynmat,axis=0)/3.0, eigvecs[:,deg])))
            new_eigvecs[:,deg] = np.dot(eigvecs[:,deg], eigvecs1)

        tmp_gvel = []
        freqs_matrix = np.einsum('i,j->ij', freqs, freqs)
        freqs_matrix = np.divide(np.ones_like(freqs_matrix), freqs_matrix, out=np.zeros_like(freqs_matrix), where=freqs_matrix!=0.0)
        for icart in range(3):
            tmp_gvel.append(np.dot(new_eigvecs.T.conj(), np.dot(ddynmat[icart], new_eigvecs))/2.0*np.sqrt(freqs_matrix))
        tmp_gvel = np.array(tmp_gvel).transpose((1,2,0))

        if(not self.off_diag):
            gvel = np.einsum('iij->ij', tmp_gvel).real
        else:
            gvel = np.zeros_like(tmp_gvel)
            for i in range(3):
                gvel[:,:,i] = (tmp_gvel[:,:,i].conj().T + tmp_gvel[:,:,i])/2.0

        return gvel, np.array(ddynmat).transpose((1,2,0))

    ##################################################################################################################################

    def symmetrize_group_velocity(self, vels, q):

        """

        Symmetrize group velocites according to the little group of wave vector.

        vels : group velocities at this wave vector
        q    : wave vector in question!

        """

        if(self.off_diag):
            for i in range(3):
                vels[:,:,i] = (vels[:,:,i] + vels[:,:,i].T)/2.0

        qred = np.dot(q, np.linalg.inv(self.reciprocal_lattice))
        qred -= np.rint(qred)
        cell = get_spglib_cell(self.dyn)
        tot_r = spglib.get_symmetry_dataset(cell)['rotations']
        nsym = len(tot_r)
        for i in range(self.nsyms):
            tot_r[i] = tot_r[i].T

        rot_q = []
        for i in range(nsym):
            diff = qred- np.dot(tot_r[i,:,:], qred)
            diff -= np.rint(diff)
            if (np.all(np.abs(diff) < 1.0e-6)):
                rot_q.append(tot_r[i,:,:])
        if(len(rot_q) > 0):
            print('Size of the small group is: ', len(rot_q))
            rot_vels = np.zeros_like(vels)
            for i in range(len(rot_q)):
                #rot_q[i] = np.dot(self.reciprocal_lattice.T, np.dot(np.linalg.inv(rot_q[i]), np.linalg.inv(self.reciprocal_lattice.T)))
                rot_q[i] = np.dot(self.reciprocal_lattice.T, np.dot(rot_q[i], np.linalg.inv(self.reciprocal_lattice.T)))
                if(self.off_diag):
                    rot_vels += np.einsum('ij,klj->kli', rot_q[i], vels)
                else:
                    rot_vels += np.einsum('ij,kj->ki', rot_q[i], vels)
            rot_vels /= float(len(rot_q))
        else:
            rot_vels = vels.copy()

        if(self.off_diag):
            for i in range(3):
                rot_vels[:,:,i] = (rot_vels[:,:,i] + rot_vels[:,:,i].T)/2.0

        return rot_vels

    ################################################################################################################################

    def symmetrize_group_velocity_by_index(self, vels, iqpt):

        """

        Symmetrize group velocites according to the little group of wave vector.

        vels : group velocities at this wave vector
        iqpt    : index of the wave vector in question!

        """

        rot_q = self.rotations[self.little_group[iqpt]].copy()

        if(self.off_diag):
            for i in range(3):
                vels[:,:,i] = (vels[:,:,i] + vels[:,:,i].T)/2.0

        rot_vels = np.zeros_like(vels)
        for rot in rot_q:
            #rot_c = np.dot(self.reciprocal_lattice.T, np.dot(np.linalg.inv(rot.T), np.linalg.inv(self.reciprocal_lattice.T)))
            rot_c = np.dot(self.reciprocal_lattice.T, np.dot(rot.T, np.linalg.inv(self.reciprocal_lattice.T)))
            if(self.off_diag):
                rot_vels += np.einsum('ij,klj->kli', rot_c, vels)
            else:
                rot_vels += np.einsum('ij,kj->ki', rot_c, vels)
        rot_vels /= float(len(rot_q))

        if(self.off_diag):
            for i in range(3):
                rot_vels[:,:,i] = (rot_vels[:,:,i] + rot_vels[:,:,i].T)/2.0

        return rot_vels

    #################################################################################################################################

    def symmetrize_group_velocities_over_star(self):

        for istar in self.qstar:
            q0 = self.qpoints[istar[0]]
            vel0 = np.zeros_like(self.gvels[istar[0]])
            rotations = []
            for iqpt in istar:
                found_rot = False
                q1 =  self.qpoints[iqpt]
                for irot in range(len(self.rotations)):
                    q2 = np.dot(self.rotations[irot].T, q1)
                    diffq = q2 - q0
                    diffq -= np.rint(diffq)
                    if(np.linalg.norm(diffq) < 1.0e-6):
                        #rotation = np.dot(self.reciprocal_lattice.T, np.dot(np.linalg.inv(self.rotations[irot].T), np.linalg.inv(self.reciprocal_lattice.T)))
                        rotation = np.dot(self.reciprocal_lattice.T, np.dot(self.rotations[irot].T, np.linalg.inv(self.reciprocal_lattice.T)))
                        found_rot = True
                        break
                if(found_rot):
                    rotations.append(rotation)
                    if(self.off_diag):
                        vel1 = np.einsum('ij,klj->kli', rotation, self.gvels[iqpt])
                    else:
                        vel1 = np.einsum('ij,kj->ki', rotation, self.gvels[iqpt])
                    vel0 += vel1
                else:
                    print('Could not find rotation between vectors in star! ')
                    print(q0)
                    print(q1)
            if(len(rotations) != len(istar)):
                print('Number of rotations does not match number of q points in the star: ', len(rotations), len(istar))
            vel0 = vel0/float(len(rotations))
            for irot, jqpt in enumerate(istar):
                if(self.off_diag):
                    self.gvels[jqpt] = np.einsum('ij,klj->kli', np.linalg.inv(rotations[irot]), vel0)
                else:
                    self.gvels[jqpt] = np.einsum('ij,kj->ki', np.linalg.inv(rotations[irot]), vel0)


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
        #if(np.any(w2_q < 0.0)):
        #    print('At q: ')
        #    print(q)
        #    print(w2_q)
        #    raise RuntimeError('SSCHA frequency imaginary. Stopping now.')
        #else:
        w_q = np.sqrt(w2_q)

        return w_q, pols_q, dynmat

    ###################################################################################################################################

    def get_dynamical_matrix(self, q, q_direct = None, lo_to_splitting = True):

        """

        Get dynamical matrix at wave vector.

        q : wave vector in cartesian coordinates without 2pi factor

        """



        uc_positions = self.dyn.structure.coords.copy()
        m = np.tile(self.dyn.structure.get_masses_array(), (3,1)).T.ravel()
        mm_mat = np.sqrt(np.outer(m, m))
        mm_inv_mat = 1.0 / mm_mat
        dynmat = np.zeros_like(self.force_constants[0], dtype = complex)
        for ir in range(len(self.ruc)):
            if(self.phase_conv == 'smooth'):
                for iat in range(len(uc_positions)):
                    for jat in range(len(uc_positions)):
                        r = -1.0*self.ruc[ir] + uc_positions[iat] - uc_positions[jat]
                        phase = np.dot(r, q)*2.0*np.pi
                        dynmat[3*iat:3*(iat+1),3*jat:3*(jat+1)] += self.force_constants[ir,3*iat:3*(iat+1),3*jat:3*(jat+1)]*np.exp(1j*phase)
            elif(self.phase_conv == 'step'):
                r = -1.0*self.ruc[ir]
                phase = np.dot(r, q)*2.0*np.pi
                dynmat += self.force_constants[ir]*np.exp(1j*phase)
            else:
                raise RuntimeError('Can not recognize phase convention!')

        #dynmat = dynmat#*mm_inv_mat
        #dynmat = (dynmat + dynmat.conj().T)/2.0

        if self.fc2.effective_charges is not None:
            self.add_long_range(dynmat, q, q_direct, lo_to_splitting)

        dynmat = dynmat*mm_inv_mat
        dynmat = (dynmat + dynmat.conj().T)/2.0

        return dynmat

    ####################################################################################################################################

    def add_long_range(self, dynmat, q, q_direct, lo_to_splitting):

        """

        Add long-range dipole-dipole dispersion to short-range force constant (dynmat) at wave vector (q (2.0*pi/A)).

        """

        dynq = np.zeros((3,3,self.fc2.nat, self.fc2.nat), dtype = np.complex128, order = "F")
        for i in range(self.fc2.nat):
            for j in range(self.fc2.nat):
                dynq[:,:, i, j] = dynmat[3*i : 3*i+3, 3*j:3*j+3]

        # Add the nonanalitic part back
        QE_q = q * self.fc2.QE_alat / A_TO_BOHR
        symph.rgd_blk_diff_phase_conv(0, 0, 0, dynq, QE_q, self.fc2.QE_tau, self.fc2.dielectric_tensor, self.fc2.QE_zeu, self.fc2.QE_bg, self.fc2.QE_omega, self.fc2.QE_alat, 0, +1.0, self.fc2.nat)

        # Check if the vector is gamma
        if np.max(np.abs(q)) < 1e-12:
            q_vect = np.zeros(3, dtype = np.double)
            compute_nonanal = lo_to_splitting
            if q_direct is not None:
                # the - to take into account the difference between QE convension and our
                if np.linalg.norm(q_direct) < 1e-8:
                    compute_nonanal = False
                else:
                    q_vect[:] = -q_direct / np.sqrt(q_direct.dot(q_direct))
            else:
                q_vect[:] = np.random.normal(size = 3)
                q_vect /= np.sqrt(q_vect.dot(q_vect))

            # Apply the nonanal contribution at gamma
            if compute_nonanal:
                QE_itau = np.arange(self.fc2.nat) + 1
                symph.nonanal(QE_itau, self.fc2.dielectric_tensor, q_vect, self.fc2.QE_zeu, self.fc2.QE_omega, dynq, self.fc2.nat, self.fc2.nat)

        # Copy in the final fc the result
        for i in range(self.fc2.nat):
            for j in range(self.fc2.nat):
                dynmat[3*i : 3*i+3, 3*j:3*j+3] = dynq[:,:, i, j]

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

    def calculate_mode_gruneisen_at_q(self, q):

        """

        Calculate Gruneisen parameter for specific q-point!


        q : Q point in cartesian coordinates


        """

        uc_positions = self.dyn.structure.coords.copy()
        natom = len(uc_positions)
        masses = self.dyn.structure.get_masses_array()
        m1 = np.zeros(3*natom, dtype = float)
        for iat in range(natom):
            m1[3*iat:3*iat + 3] = masses[iat]

        dynmat = self.get_dynamical_matrix(q)
        w2_q, pols_q = np.linalg.eigh(dynmat)
        w_q = np.sqrt(np.abs(w2_q))*np.sign(w2_q)
        gruneisens = np.zeros_like(w_q)

        for iband in range(len(w_q)):
            gruneisen = 0.0 + 1j*0.0
            for iuc in range(len(self.fc3.r_vector2[0])):
                for iat in range(natom):
                    for jat in range(natom):
                        r = -1.0*self.fc3.r_vector2[:,iuc] + uc_positions[iat] - uc_positions[jat]
                        phase_factor = np.exp(complex(0.0, 2.0*np.pi*np.dot(q, r)))
                        eig1 = pols_q[3*iat:3*iat + 3, iband].conj().copy()/np.sqrt(masses[iat])
                        eig2 = pols_q[3*jat:3*jat + 3, iband].copy()/np.sqrt(masses[jat])
                        for kat in range(natom):
                            gruneisen += np.einsum('ijk, i, j, k', self.fc3.tensor[iuc,3*iat:3*(iat + 1),3*jat:3*(jat + 1),3*kat:3*(kat + 1)], eig1,
                                    eig2, self.fc3.r_vector3[:,iuc] + uc_positions[kat])*phase_factor
            if(w_q[iband] > 0.0):
                gruneisens[iband] = -1.0*gruneisen.real/w2_q[iband]/6.0/BOHR_TO_ANGSTROM
            else:
                print('Negative frequency in calculation of Gruneisen parameter for q point: ', q)
                print(format(w_q[iband]*SSCHA_TO_THZ, '.3f'))
                gruneisens[iband] = 0.0
        return gruneisens

    #######################################################################################################################################'

    def calculate_mode_gruneisen_mesh(self):

        """

        Calculate Gruneisen parameter on a grid. Using equation from 10.1103/PhysRevLett.79.1885 !

        """

        print('Calculating Gruneisen parameters ...')
        uc_positions = self.dyn.structure.coords.copy()
        natom = len(uc_positions)
        masses = self.dyn.structure.get_masses_array()
        for ikpt in range(self.nirrkpt):
            ikpt0 = self.qstar[ikpt][0]
            for iband in range(self.nband):
                gruneisen = 0.0 + 0.0j
                for iuc in range(len(self.fc3.r_vector2[0])):
                    for iat in range(natom):
                        for jat in range(natom):
                            r = -1.0*self.fc3.r_vector2[:,iuc] + uc_positions[iat] - uc_positions[jat]
                            phase_factor = np.exp(complex(0.0, 2.0*np.pi*np.dot(self.k_points[ikpt0], r)))
                            eig1 = self.eigvecs[ikpt0,3*iat:3*iat + 3, iband].conj().copy()/np.sqrt(masses[iat])
                            eig2 = self.eigvecs[ikpt0,3*jat:3*jat + 3, iband].copy()/np.sqrt(masses[jat])
                            for kat in range(natom):
                                gruneisen += np.einsum('ijk, i, j, k', self.fc3.tensor[iuc,3*iat:3*(iat + 1),3*jat:3*(jat + 1),3*kat:3*(kat + 1)], eig1,
                                        eig2, self.fc3.r_vector3[:,iuc] + uc_positions[kat])*phase_factor
                if(self.freqs[ikpt0, iband] > 0.0):
                    for iqpt in range(len(self.qstar[ikpt])):
                        jqpt = self.qstar[ikpt][iqpt]
                        self.gruneisen[jqpt][iband] = -1.0*gruneisen.real/self.freqs[jqpt, iband]**2/6.0/BOHR_TO_ANGSTROM
                else:
                    for iqpt in range(len(self.qstar[ikpt])):
                        jqpt = self.qstar[ikpt][iqpt]
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

    ##############################################################################################################################################

    def calculate_thermal_expansion_quasiharmonic(self, temperatures, bulk):

        """

        Calculate thermal expansion assuming quasiharmonic approximation. Should work well only for cubic materials with no free internal degrees of freedom.

        temperatures : List of temperatures for which to calculate thermal expansion.
        bulk         : Bulk modulus of the material (GPa).

        """

        te = np.zeros(len(temperatures))
        volume = np.zeros(len(temperatures))
        for itemp in range(len(temperatures)):
            cp_key = format(temperatures[itemp], '.1f')
            if(cp_key in self.cp.keys()):
                print('Phonon mode heat capacities for this temperature have already been calculated. Continuing ...')
            else:
                self.get_heat_capacity(temperatures[itemp])
            te[itemp] = np.einsum('ij,ij', self.cp[cp_key], self.gruneisen)/float(self.nkpt)/self.volume/bulk*1.0e21
            volume[itemp] = self.volume*(1.0 + np.sum(te[:itemp])*(temperatures[1] - temperatures[0]))

        with open('Thermal_expansion', 'w+') as outfile:
            outfile.write('#   ' + format('Temperature (K)', STR_FMT))
            outfile.write('    ' + format('Thermal expansion (1/K)', STR_FMT))
            outfile.write('    ' + format('Volume (A^3)', STR_FMT))
            outfile.write('\n')
            for itemp in range(len(temperatures)):
                outfile.write(3*' ' + format(temperatures[itemp], '.12e'))
                outfile.write(3*' ' + format(te[itemp], '.12e'))
                outfile.write(3*' ' + format(volume[itemp], '.12e'))
                outfile.write('\n')

    ################################################################################################################################################

    def calculate_vibrational_part_of_bulk_modulus(self, temperatures):

        bulk = np.zeros(len(temperatures))

        for itemp in range(len(temperatures)):
            pops = np.zeros_like(self.freqs)
            for iqpt in range(len(pops)):
                for iband in range(len(pops[iqpt])):
                    pops[iqpt,iband] = bose_einstein(self.freqs[iqpt,iband]*SSCHA_TO_THZ*1.0e12, temperatures[itemp], HPLANCK, KB, cp_mode = self.cp_mode)
            exponents = HPLANCK*self.freqs*SSCHA_TO_THZ*1.0e12/KB/temperatures[itemp]
            term = np.einsum('ij,ij,ij,ij,ij', self.freqs, exponents, self.gruneisen**2, pops**2,np.exp(exponents))
            bulk[itemp] = -1.0*term*SSCHA_TO_THZ*1.0e12*HPLANCK/self.volume*1.0e21/float(self.nkpt)

        with open('Quasiharmonic_bulk_modulus', 'w+') as outfile:
            outfile.write('#   ' + format('Temperature (K)', STR_FMT))
            outfile.write('    ' + format('Bulk modulus (GPa)', STR_FMT))
            outfile.write('\n')
            for itemp in range(len(temperatures)):
                outfile.write(3*' ' + format(temperatures[itemp], '.12e'))
                outfile.write(3*' ' + format(bulk[itemp], '.12e'))
                outfile.write('\n')
