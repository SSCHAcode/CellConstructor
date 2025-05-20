import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.Structure
import cellconstructor.Units
import os
import numpy as np
import sys

if __name__ == "__main__":
    """
    THE MEANING OF THE ASR IN THE SCHA
    """
    struct = CC.Structure.Structure(2)
    struct.atoms = ['H', 'C']
    struct.coords = np.array([[-1, 0., 0.],
                              [+1, 0., 0.]])
    struct.masses = {'H' : 1837, 'C' : 1837 * 12}
    struct.unit_cell = np.eye(3) * 100
    dyn = CC.Phonons.Phonons(struct)
    dyn.dynmats[0] = np.random.uniform(size = (6,6))
    dyn.dynmats[0] += dyn.dynmats[0]
    dyn.ForcePositiveDefinite()
    dyn.ApplySumRule()
    Temp = 100 # Kelvin
    masses = np.tile(struct.get_masses_array(), (3, 1)).T.ravel()

    T = np.zeros((3, 6))
    for i in range(3):
        v = np.zeros(3)
        v[i] = 1
        T[i, :] = np.tile(v, 2)

    Y = dyn.GetUpsilonMatrix(Temp)
    
    # Multiply by a translation
    FC_T = np.einsum('ab, ib -> ia', dyn.dynmats[0], T)
    
    Y_T = np.einsum('ab, ib -> ia', Y, T)
    
    print('\n### TEST ###')
    print('Phi dot T')
    for i in range(3):
        print('trasl {}'.format(i))
        print(FC_T[i,:])
        if np.any(np.abs(FC_T[i, :]) > 1e-3):
            raise ValueError('The traslation of FC does not work')
        
    print('\n### TEST ###')
    print('Y dot T')
    for i in range(3):
        print('trasl {}'.format(i))
        print(Y_T[i,:])
        if np.any(np.abs(Y_T[i, :]) > 1e-3):
            raise ValueError('The traslation of Y does not work')
        

    # The matrix A of SCHA
    def get_A(T, w, pols, masses):
        A = np.zeros((6,6))
        if T < 1e-3:
            return A
        
        a_mu = np.zeros(6)
        mask = np.abs(w) < 1e-4
        a_mu[mask] = T * (CC.Units.RY_TO_KELVIN**-1) # beta^-1
        # The BE occupations
        n = (np.exp(w[~mask] * CC.Units.RY_TO_KELVIN /T) - 1)**-1
        # The other eigenvalues
        a_mu[~mask] = 2 * w[~mask] * n * (1 + n) /(1 + 2 * n)

        A = np.einsum('m, am, bm -> ab', a_mu, pols, pols)
        # Multiply by the masses
        return np.einsum('a, ab, b -> ab', np.sqrt(masses), A, np.sqrt(masses))
    
    # The matrix Y of SCHA
    def get_Y(T, w, pols, masses):
        Y = np.zeros((6,6))

        if T < 1e-3:
            return A
        
        y_mu = np.zeros(6)
        mask = np.abs(w) < 1e-4
        y_mu[mask] = 0.
        n = (np.exp(w[~mask] * CC.Units.RY_TO_KELVIN /T) - 1)**-1
        y_mu[~mask] = 2 * w[~mask] /(1 + 2 * n)

        Y = np.einsum('m, am, bm -> ab', y_mu, pols, pols)

        return np.einsum('a, ab, b -> ab', np.sqrt(masses), Y, np.sqrt(masses))
    
    
    
    w, pols = dyn.DyagDinQ(0)
    A = get_A(Temp, w, pols, masses)
    print('\nSCHA frequencies cm-1')
    print(w[np.abs(w) > 1e-3] * CC.Units.RY_TO_CM)
    
    print('\n\nTest of the COM kinetic energy @ {} K'.format(Temp))
    # Check the sum in the COM kinetic energy
    totA = 0.
    _A_ = A.reshape((2, 3, 2, 3))
    for i in range(2):
        for j in range(2):
            for alpha in range(3):
                totA += _A_[i, alpha, j, alpha]

    print('>The A contribution')
    print( totA)
    
    
    # Check the sum in the COM kinetic energy
    totY = 0.
    _Y_ = Y.reshape((2, 3, 2, 3))
    for i in range(2):
        for j in range(2):
            for alpha in range(3):
                totY += _Y_[i, alpha, j, alpha]

    print(' >The Y contribution')
    print( totY)
    
    print(np.einsum('am, bm -> ab', pols, pols))
    
    # print(np.abs(Y - get_Y(Temp, w, pols, masses)).max())
    # print(np.abs(Y - get_Y(Temp, w, pols, masses)).min())