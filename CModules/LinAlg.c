#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "LinAlg.h"
#define DEB 0
/*
 * Source in MM_LPCA.c
 * The orthonormalization following the Gram-Schmidt algorithm.
 * The vectors to be orthonormalized are taken and saved into the vectors array.
 * The j-th element of the i-th vector is at index j + L*i of the array
 *
 * If the gram-schmidth algorithm finds that the vecotrs are dependent the last N dependent 
 * vecotors are filled with zeros.
 * It returns the number of independent vectors
 */
 
int GramSchmidt(double * vectors, int L, int N_vectors){
   
  // Threshold for assess whether norm is equal to zeros.. 
  const double THRES = 1E-9;
   
  double braket = 0, norm = 0;
  int i,j, k;
 
  int pos = N_vectors;
   
  double *f_n;//  tmp vector */
  f_n=(double*)calloc(L, sizeof(double)); 

  if (DEB) {
    printf("GS first vector: ");
    for (i = 0; i < L; ++i) printf(" %.4f ", vectors[i]);
    printf("\n");
  }
 
  /* normalizing first basis vector |e> = |phi>/norm */
  norm = 0; 
  for(i=0; i<L; i++){
    norm += vectors[0*L + i]*vectors[0*L + i];
  }
     
  norm = sqrt(norm);
     
  for(i=0; i<L; i++){
    vectors[0*L + i] /= norm;
  }
  /**/
     
  if (DEB) {
    printf("GS first vector after norm: ");
    for (i = 0; i < L; ++i) printf(" %.4f ", vectors[i]);
    printf("\n");
  }
   
  k = 1;
  while(k<N_vectors && pos > k){
    //for(k=1; k<N_vectors; k++){
    if (DEB) printf("GS k=%d\n", k);
     
    /* creating copy of k-esime vector f_n */
    
    for(i=0; i<L; i++){
      f_n[i] = vectors[k*L + i];
    }
    
    /**/
   
    /* performing Gramh-Smith algoritm               */
    /* |phi_n> = |f_n> - sum_i^(n-1) <f_n|e_i>|e_i>  */
     
    for(i=0; i<k; i++){
       
      // computing braket = <f_n| e_i>
      braket = 0;
      for(j=0; j<L; j++){
    braket += f_n[j]*vectors[i*L + j];
      }
 
      if(braket <= THRES){
    //braket = 0;
      }
      for(j=0; j<L; j++){
     
    vectors[k*L + j] -= braket*vectors[i*L + j];
      }
    }
 
    //printf("ok\n");
    /* normalizing k-esime basis vector |e> = |phi>/norm 
       if k-esime vector is dependent, another vector (in position pos)  is 
       taken (backward from the end of list) and process is 
       repeated. ( |f_k> = |f_pos>, |f_pos> = 0, with pos = N_vec-1, N_vec-2 
       etc)
    */
     
    norm = 0; 
    for(i=0; i<L; i++){
      norm += vectors[(k)*L + i]*vectors[(k)*L + i];
    }
     
    norm = sqrt(norm);
 
    /*check whether k-esime vector is dependent (norm = 0) */
    if(norm > THRES){
      for(i=0; i<L; i++){
    vectors[(k)*L + i] /= norm;
      }
       
      k++;
    }else{
 
      /*k-esime vector is dependent so we repeat the k-esime step with 
    vector in position pos
      */
      pos--;
      if (DEB) {printf("Found a dependent k = %d, pos = %d\n", k, pos);}
        for(i = 0; i<L; i++){
      vectors[k*L + i] = vectors[pos*L + i];
      vectors[pos*L + i] = 0;
        }

     
    }
    /**/
  }

  return k;
}

