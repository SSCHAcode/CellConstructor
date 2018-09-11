# -*- coding: utf-8 -*-

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons


SUPER_DYN = "../TestPhononSupercell/dynmat"
NQIRR = 8
SUPERCELL = (3, 3, 2)


dyn = CC.Phonons.Phonons(SUPER_DYN, NQIRR)


fc = dyn.GetRealSpaceFC(SUPERCELL)
fc_new = fc.copy()


print "Real space:"
print fc[:6, :6]

print "First one:"
print dyn.dynmats[0]


print "Distances"
m = dyn.structure.get_masses_array()

nat = dyn.structure.N_atoms
_m_ = np.zeros(3*nat)
for i in range(nat):
    _m_[3 * i : 3*i + 3] = m[i]
    
m_mat = np.outer(1 / np.sqrt(_m_), 1 / np.sqrt(_m_))
print m_mat
nq = len(dyn.q_tot)

for i in range(nq):
    s1 = 3 * nat * i
    for j in range(nq):
        s2 = 3*nat * j
        
        fc[s1 : s1 + 3*nat, s2:s2 + 3*nat] *= m_mat

w_tot = np.sqrt(np.real(np.linalg.eigvals(fc)))
w_tot.sort()

w_old = np.zeros(len(w_tot))

for i in range(nq):
    w,p = dyn.DyagDinQ(i)
    w_old[ i * len(w) : (i+1) * len(w)] = w

w_old.sort()    
print "Freq:"
print "\n".join ( [" %.5f vs %.5f" % (w_tot[i] * CC.Phonons.RY_TO_CM, w_old[i] * CC.Phonons.RY_TO_CM) for i in range (len(w_tot))])


# Try to revert the code

dynmats_new = CC.Phonons.GetDynQFromFCSupercell(fc_new, np.array(dyn.q_tot), SUPERCELL, dyn.structure.unit_cell)
print np.sqrt(np.sum( (dynmats_new[2,:,:] - dyn.dynmats[2])**2 ))

#print "\n".join ( ["RATIO: %.5f " % (w_tot[i] / w_old[i] ) for i in range (len(w_tot))])