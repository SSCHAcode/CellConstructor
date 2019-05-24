#ifndef LINALG_HEADER
#define LINALG_HEADER


/*
 * The orthonormalization following the Gram-Schmidt algorithm.
 * The vectors to be orthonormalized are taken and saved into the vectors array.
 * The j-th element of the i-th vector is at index j + L*i of the array
 *
 * If the gram-schmidth algorithm finds that the vecotrs are dependent the last N dependent 
 * vecotors are filled with zeros.
 * It returns the number of independent vectors
 */
int GramSchmidt(double * vectors, int L, int N_vectors);

#endif