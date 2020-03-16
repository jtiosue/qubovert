//   Copyright 2020 Joseph T. Iosue
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include "Python.h"
#include "simulate_quso.h"


/*
Here we wrap the source code in src/simulate_quso.c so that it can be called
directly from Python. This file creates the ``qubovert.sim._simulate_quso``
module with the function ``c_simulate_quso``.
*/

// Module info
static char _simulate_quso_name[] = "_simulate_quso";

static char _simulate_quso_docstring[] =
    "``qubovert.sim._simulate_quso`` is a module for simulating a QUSO\n"
    "with the C source code.";


// Define module functions; wrap the source code.

static char c_simulate_quso_docstring[] =
    "c_simulate_quso.\n\n"
    "Simulate a QUSO with the C source.\n\n"
    "Parameters\n"
    "----------\n"
    "state: list of ints.\n"
    "    ``state[i]`` is the value of the ith spin, either 1 or -1.\n"
    "h : list of floats.\n"
    "    ``h[i]`` is the field value on spin ``i``.\n"
    "num_neighbors : list of ints.\n"
    "    ``num_neighbors[i]`` is the number of neighbors that spin i has.\n"
    "neighbors : list of ints.\n"
    "    ``neighbors[i]`` is the jth neighbor of spin ``k``, where\n"
    "    ``j = i - num_neighbors[k-1] - num_neighbors[k-2] - ...``\n"
    "J : list of doubles.\n"
    "    ``J[i]`` is the coupling value between spin ``k`` and\n"
    "    ``neighbors[i]``.\n"
    "Ts : tuple of floats\n"
    "    Each ``T`` in ``Ts`` indicates a temperature to update the\n"
    "    simulation at.\n"
    "num_updates : tuple of ints\n"
    "    ``num_updates[i]`` indicates how many times to update the\n"
    "    simulation at temperature ``T[i]``."
    "in_order : int.\n"
    "    ``in_order`` indicates whether to iterate through the variables in\n"
    "    order ``in_order=1`` or randomly ``in_order=0`` during an\n"
    "    update step.\n"
    "seed : int.\n"
    "    seeds the random number generator If ``seed`` is a negative integer,\n"
    "    then we seed the random number generator with ``time(NULL)``.\n"
    "    Otherwise, we use ``seed``.\n\n"
    "Returns\n"
    "-------\n"
    "new_state : list of ints.\n\n"
    "Example\n"
    "-------\n"
    "``neighbors`` and ``J`` are basically flattened arrays.\n"
    "In other words, we flatten the arrays ``temp_neighbors`` and\n"
    "``temp_J``, where ``temp_neighbors`` points to an array where\n"
    "``temp_neighbors[i][j]`` is the jth neighbor of spin i,\n"
    "for j=0,...,num_neighbors[i]-1, and similarly, ``temp_J`` points to an\n"
    "array where ``temp_J[i][j]`` is the coupling value between\n"
    "spin i and spin ``neighbors[i][j]``, for j=0,...,num_neighbors[i]-1.\n\n"
    "A spin model such as\n"
    "    :math:`-z_0 z_1 + 2*z_1*z_2 + z_0`\n"
    "must be represented as\n"
    "    ``h = {1., 0, 0}``\n"
    "    ``num_neighbors = {1, 2, 1}``\n"
    "    ``temp_neighbors = {{1}, {0, 2}, {1}}``\n"
    "    ``temp_J = {{-1.},\n"
    "          {-1, 2},\n"
    "          {2}}``\n"
    "    ``neighbors = {1, 0, 2, 1}``\n"
    "    ``J = {-1.,\n"
    "          -1, 2,\n"
    "          2}``\n";


static PyObject* c_simulate_quso(PyObject* self, PyObject* args) {
    /*
    This is the function that we call from python with
    ``qubovert.sim._simulate_quso.c_simulate_quso``. See the docstring above
    for details on what ``args`` should be.
    */
    PyObject *state, *h, *num_neighbors, *neighbors, *J, *Ts, *num_updates;
    int in_order, seed;

    if (!PyArg_ParseTuple(args, "OOOOOOOii",
                          &state, &h, &num_neighbors, &neighbors,
                          &J, &Ts, &num_updates, &in_order, &seed)) {
        return NULL;
    }

    int len_state = (int)PyList_Size(state);
    int len_J = (int)PyList_Size(J);
    int len_Ts = (int)PyTuple_Size(Ts);
    int *c_state = (int*)malloc(len_state * sizeof(int));
    double *c_h = (double*)malloc(len_state * sizeof(double));
    int *c_num_neighbors = (int*)malloc(len_state * sizeof(int));
    int *c_neighbors = (int*)malloc(len_J * sizeof(int));
    double *c_J = (double*)malloc(len_J * sizeof(double));
    double *c_Ts = (double*)malloc(len_Ts * sizeof(double));
    int *c_num_updates = (int*)malloc(len_Ts * sizeof(int));

    int i;  // iterator

    for(i=0; i<len_state; i++) {
        c_state[i] = (int)PyLong_AsLong(PyList_GetItem(state, i));
        c_h[i] = PyFloat_AsDouble(PyList_GetItem(h, i));
        c_num_neighbors[i] = (int)PyLong_AsLong(PyList_GetItem(num_neighbors, i));
    }
    for(i=0; i<len_J; i++) {
        c_neighbors[i] = (int)PyLong_AsLong(PyList_GetItem(neighbors, i));
        c_J[i] = PyFloat_AsDouble(PyList_GetItem(J, i));
    }
    for(i=0; i<len_Ts; i++) {
        c_Ts[i] = PyFloat_AsDouble(PyTuple_GetItem(Ts, i));
        c_num_updates[i] = (int)PyLong_AsLong(PyTuple_GetItem(num_updates, i));
    }

    // call C source code in src/ directory
    simulate_quso(  // updates c_state in place
        len_state, c_state, c_h,
        c_num_neighbors, c_neighbors, c_J,
        len_Ts, c_Ts, c_num_updates,
        in_order, seed
    );

    PyObject *new_state = PyList_New(len_state);
    for(i=0; i<len_state; i++) {
        PyList_SetItem(new_state, i, PyFloat_FromDouble(c_state[i]));
    }

    free(c_state);
    free(c_h);
    free(c_num_neighbors);
    free(c_neighbors);
    free(c_J);
    free(c_Ts);
    free(c_num_updates);

    return new_state;
}


// Create the module.

static PyMethodDef CSimulateQUSOMethods[] = {
    {
        "c_simulate_quso",
        c_simulate_quso,
        METH_VARARGS,
        c_simulate_quso_docstring
    },
    {NULL, NULL, 0, NULL}  // Sentinel
};


static struct PyModuleDef CSimulateQUSOModule = {
    PyModuleDef_HEAD_INIT,
    _simulate_quso_name,  // name of module
    _simulate_quso_docstring,  // module documentation, may be NULL
    -1,  // size of per-interpreter state of the module,
         // or -1 if the module keeps state in global variables.
    CSimulateQUSOMethods
};


PyMODINIT_FUNC PyInit__simulate_quso(void) {
    // MUST BE PyInit_modulename.
    return PyModule_Create(&CSimulateQUSOModule);
}
