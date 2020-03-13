# distutils: language=c
# cython: language_level=3
#   Copyright 2020 Joseph T. Iosue
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from libc.stdlib cimport malloc, free


cdef extern from "simulate_quso.h":
    void simulate_quso(
        int len_state, int *state, double *h,
        int *num_neighbors, int *neighbors, double *J,
        int len_Ts, double *Ts, int *num_updates,
        int in_order, int seed
    ) nogil


def c_simulate_quso(state, h, num_neighbors,
                    neighbors, J, schedule, in_order, seed):
    """c_simulate_quso.

    Simulate a QUSO with the C source.

    Parameters
    ----------
    state: list of ints.
        `state[i]` is the value of the ith spin, either 1 or -1.
    h : list of floats.
        `h[i]` is the field value on spin `i`.
    num_neighbors : list of ints.
        `num_neighbors[i]` is the number of neighbors that spin i has.
    neighbors : list of ints.
        ``neighbors[i]`` is the jth neighbor of spin ``k``, where
        ``j = i - num_neighbors[k-1] - num_neighbors[k-2] - ...``
    J : list of doubles.
        ``J[i]`` is the coupling value between spin ``k`` and
        ``neighbors[i]``.
    schedule : iterable of tuples.
        Each tuple is a ``T, n`` pairs, where ``n`` is the number of time
        steps to update the simulation at temperature ``T``.
    in_order : bool.
        ``in_order`` indicates whether to iterate through the variables in
        order ``in_order=true`` or randomly ``in_order=false`` during an
        update step.
    seed : int.
        seeds the random number generator If `seed` is a negative integer,
        then we seed the random number generator with `time(NULL)`.
        Otherwise, we use `seed`.

    Returns
    -------
    new_state : list of ints.

    Example
    -------
    `neighbors` and `J` are basically flattened arrays.
    In other words, we flatten the arrays `temp_neighbors` and
    `temp_J`, where `temp_neighbors` points to an array where
    `temp_neighbors[i][j]` is the jth neighbor of spin i,
    for j=0,...,num_neighbors[i]-1, and similarly, `temp_J` points to an array
    where `temp_J[i][j]` is the coupling value between
    spin i and spin `neighbors[i][j]`, for j=0,...,num_neighbors[i]-1.

    A spin model such as
        -z_0 z_1 + 2*z_1*z_2 + z_0
    must be represented as
        `h = {1., 0, 0}`
        `num_neighbors = {1, 2, 1}`
        `temp_neighbors = {{1}, {0, 2}, {1}}`
        `temp_J = {{-1.},
              {-1, 2},
              {2}}`
        `neighbors = {1, 0, 2, 1}`
        `J = {-1.,
              -1, 2,
              2}`
    """
    # convert all Python types to C
    cdef int c_len_state = len(state)
    cdef int *c_state = <int*>malloc(len(state) * sizeof(int))
    cdef double *c_h = <double*>malloc(len(state) * sizeof(double))
    cdef int *c_num_neighbors = <int*>malloc(len(state) * sizeof(int))
    cdef int *c_neighbors = <int*>malloc(len(neighbors) * sizeof(int))
    cdef double *c_J = <double*>malloc(len(J) * sizeof(double))
    cdef int c_len_Ts = len(schedule)
    cdef double *c_Ts = <double*>malloc(len(schedule) * sizeof(double))
    cdef int *c_num_updates = <int*>malloc(len(schedule) * sizeof(int))
    cdef int c_in_order = int(in_order)
    cdef int c_seed = seed

    for i in range(len(state)):
        c_state[i] = state[i]
        c_h[i] = h[i]
        c_num_neighbors[i] = num_neighbors[i]
    for i in range(len(J)):
        c_neighbors[i] = neighbors[i]
        c_J[i] = J[i]
    for i, (T, n) in enumerate(schedule):
        c_Ts[i] = T
        c_num_updates[i] = n
        if n < 0:
            raise ValueError("Cannot update a negative number of times")

    with nogil:
        simulate_quso(
            c_len_state, c_state, c_h,
            c_num_neighbors, c_neighbors, c_J,
            c_len_Ts, c_Ts, c_num_updates,
            c_in_order, c_seed
        )

    final_state = [c_state[i] for i in range(len(state))]
    free(c_state)
    free(c_h)
    free(c_num_neighbors)
    free(c_neighbors)
    free(c_J)
    free(c_Ts)
    free(c_num_updates)
    return final_state
