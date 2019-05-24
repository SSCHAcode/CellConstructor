#include <python2.7/Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "LinAlg.h"

// Function Prototypes
static PyObject * gram_schmidt(PyObject * self, PyObject * args);


static PyMethodDef Engine[] = {
    {"GramSchmidt", gram_schmidt, METH_VARARGS, "Apply the Gram-Schmidt orthonormalization on the given vectors"},
    {NULL, NULL, 0, NULL}
};

// Module initialization
PyMODINIT_FUNC initcc_linalg(void) {
    (void) Py_InitModule("cc_linalg", Engine);
}



static PyObject * gram_schmidt(PyObject* self, PyObject* args) {
    double * vectors;
    int N_vectors, N_dim;
    int NewDim;

    PyObject * npy_vectors;
    
    // Get the path dir
    if (!PyArg_ParseTuple(args, "Oii", &npy_vectors, &N_vectors, &N_dim))
        return NULL;

    // Get the C pointers to the data of the numpy ndarray
    vectors = (double*) PyArray_DATA(npy_vectors);

    NewDim = GramSchmidt(vectors, N_dim, N_vectors);

    return PyInt_FromLong(NewDim);
}
