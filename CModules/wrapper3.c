/*
 * This source takes care of the wrapping for the python 3
 * version
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "LinAlg.h"

// Function Prototypes
static PyObject * gram_schmidt(PyObject * self, PyObject * args);

// Packing functions all together
static PyMethodDef Engine[] = {
    {"GramSchmidt", gram_schmidt, METH_VARARGS, "Apply the Gram-Schmidt orthonormalization on the given vectors"},
    {NULL, NULL, 0, NULL}
};

// Module initialization
static struct PyModuleDef cc_linalgmodule = {
  PyModuleDef_HEAD_INIT,
  "cc_linalg",
  NULL, // The module documentation, for now null
  -1,
  Engine
};

// Now execute the initialization of the module
PyMODINIT_FUNC PyInit_cc_linalg(void) {
  return PyModule_Create(&cc_linalgmodule);
}

/* // Module initialization as it were in python2 */
/* PyMODINIT_FUNC initcc_linalg(void) { */
/*     (void) Py_InitModule("cc_linalg", Engine); */
/* } */



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

    return PyLong_FromLong(NewDim);
}
