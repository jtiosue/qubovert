#include "Python.h"
#include "anneal_quso.h"
#include "anneal_puso.h"


/*
Here we wrap the source code in the src folder so that it can be called
directly from Python. This file creates the ``qubovert.sim._canneal`` module.
*/

// Module info
static char _canneal_name[] = "_canneal";

static char _canneal_docstring[] =
    "``qubovert.sim._canneal`` is a module for annealing\n"
    "with the C source code.";


// helper code for the module functions below.
PyObject *build_py_states_values(
    int num_anneals, int len_state, int *states, double *values
) {
    /*
    Build a Python tuple of ``py_states, py_values``, where ``py_states`` is
    a list of lists, where each list represents a state, and ``py_values`` is
    a list of floats, where each float is the energy of that state. We build
    these from the states in values in ``states``, and ``values``. ``states``
    is an array of length ``len_state * num_anneals``. So
    ``states[i * len_state + j]`` is the sign that the jth spin took on the
    ith anneal. ``values`` is an array of length ``num_anneals``.

    Parameters
    ----------
    num_anneals : int.
    len_state : int.
    states : points to an int array of size ``num_anneals * len_state``.
    values : points to a double array of size ``num_anneals``.

    Returns
    -------
    res : a Python tuple.
        The first element of the tuple is a list of lists of ints, and the
        second element is a list of floats.

    */
    PyObject *py_states = PyList_New(num_anneals);
    PyObject *py_values = PyList_New(num_anneals);
    PyObject *py_state; int i, j;
    for(i=0; i<num_anneals; i++) {
        py_state = PyList_New(len_state);
        for(j=0; j<len_state; j++) {
            PyList_SetItem(py_state, j, PyLong_FromLong(states[i * len_state + j]));
        }
        PyList_SetItem(py_states, i, py_state);
        PyList_SetItem(py_values, i, PyFloat_FromDouble(values[i]));
    }
    return Py_BuildValue("OO", py_states, py_values);
}


// Define module functions; wrap the source code.

static char c_anneal_quso_docstring[] =
    "c_anneal_quso.\n\n"
    "Anneal a QUSO with the C source.\n\n"
    "Parameters\n"
    "----------\n"
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
    "Ts : list of floats\n"
    "    Each ``T`` in ``Ts`` indicates a temperature to update the\n"
    "    simulation at.\n"
    "num_anneals : int\n"
    "    How many times to run the simulated annealing algorithm.\n"
    "in_order : int.\n"
    "    ``in_order`` indicates whether to iterate through the variables in\n"
    "    order ``in_order=1`` or randomly ``in_order=0`` during an\n"
    "    update step.\n"
    "initial_state : list of ints.\n"
    "    If an initial state is not provided, then ``initial_state`` should\n"
    "    be ``[]``.\n"
    "seed : int.\n"
    "    seeds the random number generator If ``seed`` is a negative integer,\n"
    "    then we seed the random number generator with ``time(NULL)``.\n"
    "    Otherwise, we use ``seed``.\n\n"
    "Returns\n"
    "-------\n"
    "tuple : (states, values).\n"
    "    ``states`` is a list of lists, where each list represents a state.\n"
    "    ``values`` is a list of floats, where each float is the QUSO value\n"
    "    for the corresponding state.\n\n"
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
    "    ``h = [1., 0, 0]``\n"
    "    ``num_neighbors = [1, 2, 1]``\n"
    "    ``temp_neighbors = [[1], [0, 2], [1]]``\n"
    "    ``temp_J = [[-1.],\n"
    "          [-1, 2],\n"
    "          [2]]``\n"
    "    ``neighbors = [1, 0, 2, 1]``\n"
    "    ``J = [-1.,\n"
    "          -1, 2,\n"
    "          2]``\n";


static PyObject* c_anneal_quso(PyObject* self, PyObject* args) {
    /*
    This is the function that we call from python with
    ``sanneal._canneal.c_anneal_quso``. See the docstring above
    for details on what ``args`` should be.
    */
    PyObject *py_h, *py_num_neighbors, *py_neighbors,
             *py_J, *py_Ts, *py_initial_state;
    int num_anneals, in_order, seed;

    if (!PyArg_ParseTuple(args, "OOOOOiiOi",
                          &py_h, &py_num_neighbors, &py_neighbors, &py_J,
                          &py_Ts, &num_anneals, &in_order, 
                          &py_initial_state, &seed)) {
        return NULL;
    }

    int len_state = (int)PyList_Size(py_h);
    int len_J = (int)PyList_Size(py_J);
    int len_Ts = (int)PyList_Size(py_Ts);
    double *h = (double*)malloc(len_state * sizeof(double));
    int *num_neighbors = (int*)malloc(len_state * sizeof(int));
    int *neighbors = (int*)malloc(len_J * sizeof(int));
    double *J = (double*)malloc(len_J * sizeof(double));
    double *Ts = (double*)malloc(len_Ts * sizeof(double));

    int i, j;
    for(i=0; i<len_state; i++) {
        h[i] = PyFloat_AsDouble(PyList_GetItem(py_h, i));
        num_neighbors[i] = (int)PyLong_AsLong(PyList_GetItem(py_num_neighbors, i));
    }
    for(i=0; i<len_J; i++) {
        neighbors[i] = (int)PyLong_AsLong(PyList_GetItem(py_neighbors, i));
        J[i] = PyFloat_AsDouble(PyList_GetItem(py_J, i));
    }
    for(i=0; i<len_Ts; i++) {
        Ts[i] = PyFloat_AsDouble(PyList_GetItem(py_Ts, i));
    }

    // create buffers for the states and values of anneal_quso.
    double *values = (double*)malloc(num_anneals * sizeof(double));
    int *states = (int*)malloc(num_anneals * len_state * sizeof(int));

    int initial_state_provided = (int)PyList_Size(py_initial_state);
    if(initial_state_provided) {
        // encode the initial state into the buffers.
        for(i=0; i<num_anneals; i++) {
            for(j=0; j<len_state; j++) {
                states[i * len_state + j] = (int)PyLong_AsLong(
                    PyList_GetItem(py_initial_state, j)
                );
            }
        }
    }

    // call C source code in src/ directory
    anneal_quso(  // updates states and values in place
    	num_anneals, states, values,
        len_state, h, num_neighbors, neighbors, J,
        len_Ts, Ts, in_order, initial_state_provided, seed
    );

    PyObject *py_states_values = build_py_states_values(
        num_anneals, len_state, states, values
    );

    free(h); free(num_neighbors); free(neighbors); free(J);
    free(Ts); free(states); free(values);

    return py_states_values;
}


static char c_anneal_puso_docstring[] =
    "c_anneal_puso.\n\n"
    "Anneal a PUSO with the C source.\n\n"
    "Parameters\n"
    "----------\n"
    "len_state : int.\n"
    "    The number of spin variables in the problem.\n"
    "num_couplings : list of ints.\n"
    "    To see how ``num_couplings`` works, see the Example below, or look\n"
    "    at the ``puso_value`` function in the sanneal/src/anneal_puso.c\n"
    "    file. Or see the ``sanneal.anneal_puso`` function source code.\n"
    "terms : list of ints.\n"
    "    To see how ``terms`` works, see the Example below, or look at the\n"
    "    ``puso_value`` function in the sanneal/src/anneal_puso.c file.\n"
    "    Or see the ``sanneal.anneal_puso`` function source code.\n"
    "couplings : list of floats.\n"
    "    To see how ``couplings`` works, see the Example below, or look at\n"
    "    the ``puso_value`` function in the sanneal/src/anneal_puso.c file.\n"
    "    Or see the ``sanneal.anneal_puso`` function source code.\n"
    "Ts : list of floats\n"
    "    Each ``T`` in ``Ts`` indicates a temperature to update the\n"
    "    simulation at.\n"
    "num_anneals : int\n"
    "    How many times to run the simulated annealing algorithm.\n"
    "initial_state : list of ints.\n"
    "    If an initial state is not provided, then ``initial_state`` should\n"
    "    be ``[]``.\n"
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
    "tuple : (states, values).\n"
    "    ``states`` is a list of lists, where each list represents a state.\n"
    "    ``values`` is a list of floats, where each float is the PUSO value\n"
    "    for the corresponding state.\n\n"
    "Example\n"
    "-------\n"
    "Consider a PUSO\n"
    "    ``z_0 z_1 - z_1 z_2 z_3 + 3 z_2``.\n"
    "Then we would have the following arguments:\n"
    "    ``num_terms = 3``.\n"
    "    ``terms = {0, 1,   1, 2, 3,   2}``.\n"
    "    ``num_couplings = {2, 3, 1}``.\n"
    "    ``couplings = {1, -1, 3}``.\n"
    "Note that I add those big spaces in the ``terms`` array for the\n"
    "visuals so that you can see how these terms are separated.\n";


static PyObject* c_anneal_puso(PyObject* self, PyObject* args) {
    /*
    This is the function that we call from python with
    ``sanneal._canneal.c_anneal_puso``. See the docstring above
    for details on what ``args`` should be.
    */
    PyObject *py_num_couplings, *py_terms, *py_couplings,
             *py_Ts, *py_initial_state;
    int num_anneals, in_order, seed, len_state;

    if (!PyArg_ParseTuple(args, "iOOOOiiOi",
                          &len_state, &py_num_couplings, &py_terms,
                          &py_couplings, &py_Ts, &num_anneals,
                          &in_order, &py_initial_state, &seed)) {
        return NULL;
    }

    int len_Ts = (int)PyList_Size(py_Ts);
    double *Ts = (double*)malloc(len_Ts * sizeof(double));
    long num_terms = (long int)PyList_Size(py_couplings);
    int *num_couplings = (int*)malloc(num_terms * sizeof(int));
    long len_terms = (long int)PyList_Size(py_terms);
    int *terms = (int*)malloc(len_terms * sizeof(int));
    double *couplings = (double*)malloc(num_terms * sizeof(double));

    long i; int j;
    for(i=0; i<len_terms; i++) {
        terms[i] = (int)PyLong_AsLong(PyList_GetItem(py_terms, i));
    }
    for(i=0; i<num_terms; i++) {
        num_couplings[i] = (int)PyLong_AsLong(PyList_GetItem(py_num_couplings, i));
        couplings[i] = PyFloat_AsDouble(PyList_GetItem(py_couplings, i));
    }
    for(i=0; i<len_Ts; i++) {
        Ts[i] = PyFloat_AsDouble(PyList_GetItem(py_Ts, i));
    }

    // create buffers for the states and values of anneal_quso.
    double *values = (double*)malloc(num_anneals * sizeof(double));
    int *states = (int*)malloc(num_anneals * len_state * sizeof(int));

    int initial_state_provided = (int)PyList_Size(py_initial_state);
    if(initial_state_provided) {
        // encode the initial state into the buffers.
        for(i=0; i<num_anneals; i++) {
            for(j=0; j<len_state; j++) {
                states[i * len_state + j] = (int)PyLong_AsLong(
                    PyList_GetItem(py_initial_state, j)
                );
            }
        }
    }

    // call C source code in src/ directory
    anneal_puso(  // updates states and values in place
        num_anneals, states, values, len_state,
        num_terms, num_couplings, terms, couplings,
        len_Ts, Ts, in_order, initial_state_provided, seed
    );

    PyObject *py_states_values = build_py_states_values(
        num_anneals, len_state, states, values
    );

    free(num_couplings); free(terms); free(couplings);
    free(Ts); free(states); free(values);

    return py_states_values;
}


// Create the module.

static PyMethodDef CAnnealMethods[] = {
    {
        "c_anneal_quso",
        c_anneal_quso,
        METH_VARARGS,
        c_anneal_quso_docstring
    },
    {
        "c_anneal_puso",
        c_anneal_puso,
        METH_VARARGS,
        c_anneal_puso_docstring
    },
    {NULL, NULL, 0, NULL}  // Sentinel
};


static struct PyModuleDef CAnnealModule = {
    PyModuleDef_HEAD_INIT,
    _canneal_name,  // name of module
    _canneal_docstring,  // module documentation, may be NULL
    -1,  // size of per-interpreter state of the module,
         // or -1 if the module keeps state in global variables.
    CAnnealMethods
};


PyMODINIT_FUNC PyInit__canneal(void) {
    // MUST BE PyInit_modulename.
    return PyModule_Create(&CAnnealModule);
}
