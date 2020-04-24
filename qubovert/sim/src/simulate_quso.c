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

#include "pcg_basic.h"
#include "simulate_quso.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
a few slight modifications and one major modification from
    https://github.com/dwavesystems/dwave-neal/blob/master/neal/src/cpu_sa.cpp
We use the minimal C random number generator from
    http://www.pcg-random.org.
*/


double rand_double(pcg32_random_t *rng) {
    // random double in [0, 1)
    return ldexp((double)pcg32_random_r(rng), -32);
}


int rand_int(pcg32_random_t *rng, int stop) {
    //random integer in [0, top)
    return (int)pcg32_boundedrand_r(rng, stop);
}


void compute_flip_dE(
    double *flip_spin_dE,
    int len_state, int *state, double *h,
    int *num_neighbors, int *neighbors, double *J,
    long *index
) {
    /*
    Determine the energy of the subgraph around a certain spin.

    Parameters
    ----------
    `flip_spin_dE` points to an array with memory allocated for
        `len_state` doubles.
    `state` points to an array where `state[i]` is the value
        of the ith spin, either 1 or -1.
    `h` points to an array such that `h[i]` is the field value
        for spin i.
    `num_neighbors` points to an array such that `num_neighbors[i]` is
        the length of the array `neighbors[i]`.
    `neighbors` points to an array where `neighbors[index[i]+j]` is
        one of the neighboring spins of spin i, for j
        in {0, 1, ..., num_neighbors[i]-1}.
    `J` points to an array where `J[index[i]+j]` is the coupling value
        between the spin i and spin `neighbors[index[i]+j]`, for j
        in {0, 1, ..., num_neighbors[i]-1}.
    `index` points to an array such that `index[i]` is the index of
        `neighbors` and `J` where the information for spin `i` starts.
        See the `simulate_quso` function for more info.

    */
    int i, j, neighbor;
    double subgraph_energy;
    for(i=0; i<len_state; i++) {
        // subgraph energy is the h + sum_neighbors J * state[neighbor]
        subgraph_energy = h[i];
        for(j=0; j<num_neighbors[i]; j++) {
            neighbor = neighbors[index[i] + j];
            subgraph_energy += J[index[i] + j] * state[neighbor];
        }

        // flipping spin i results in a delta energy of
        // -2*(value of spin i)*(the subgraph energy)
        flip_spin_dE[i] = -2. * state[i] * subgraph_energy;
    }
}

void recompute_flip_dE(
    int spin, double *flip_spin_dE, int *state,
    int *num_neighbors, int *neighbors, double *J,
    long *index
) {
    /*
    `flip_spin_dE` points to an array such that `flip_spin_dE[i]` is
    the amount that the energy would change if we flipped spin `i`.	Now
    suppose we decide that we are going to flip spin `spin`. Then
    `flip_spin_dE[j]` will change for each `j` that is a neighbor of `spin`.
    Here we adjust each of these `j`s.

    Parameters
    ----------
    `spin` is the spin that we are going to flip.
    `flip_spin_dE` points to an array such that `flip_spin_dE[i]` is
        the amount that the energy would change if we flipped spin `i`.
    `state` points to an array where `state[i]` is the value
        of the ith spin, either 1 or -1.
    `num_neighbors` points to an array such that `num_neighbors[i]` is
        the length of the array `neighbors[i]`.
    `neighbors` points to an array where `neighbors[index[i]+j]` is
        one of the neighboring spins of spin i, for j
        in {0, 1, ..., num_neighbors[i]-1}.
    `J` points to an array where `J[index[i]+j]` is the coupling value
        between the spin i and spin `neighbors[index[i]+j]`, for j
        in {0, 1, ..., num_neighbors[i]-1}.
    `index` points to an array such that `index[i]` is the index of
        `neighbors` and `J` where the information for spin `i` starts.
        See the `simulate_quso` function for more info.

    */
    int j, n;

    // flipping spin `spin` means that the next time we flip `spin` the
    // delta energy will be negative.
    flip_spin_dE[spin] *= -1;

    // go through each of the neighbors of spin `spin`.
    for(j=0; j<num_neighbors[spin]; j++) {
        n = neighbors[index[spin] + j];
        // previously, the delta_energy for spin `n` was
        // -2*state[n] * (
        //     sum(state[x] * (couping between x and n)
        //         for x is a neighbor of n
        //     ) + h[n]
        // ).
        // One of those neighbors is `spin`. Therefore, we need to flip the
        // effect of -2*state[n]*state[spin]*(coupoing between n and spin),
        // meaning that we adjust the delta energy by
        // 4*state[n]*state[spin]*(coupling between n and spin)
        flip_spin_dE[n] += 4. * state[spin] * state[n] * J[index[spin] + j];
    }
}


void simulate_quso(
    int len_state, int *state, double *h,
    int *num_neighbors, int *neighbors, double *J,
    int len_Ts, double *Ts, int *num_updates,
    int in_order, int seed
) {
    /*
    Simulate a QUSO.

    Parameters
    ----------
    `len_state` is the length of `state`, ie the number of spin.
    `state` points to an array where `state[i]` is the value of the ith
        spin, either 1 or -1.
    `h` points to an array where `h[i]` is the field value on spin `i`.
    `num_neighbors` points to an array where `num_neighbors[i]` is
        number of neighbors that spin i has.
    `neighbors` points to an array where `neighbors[i]` is the jth neighbor of
        spin `k`, where `j = i - num_neighbors[k-1] - num_neighbors[k-2] - ...`
    `J` points to an array where `J[i]` is the coupling value between
        spin `k` and `neighbors[i]`.
    `len_Ts` is the length of `Ts` and the length of `num_updates`.
    `Ts` points to an array where `Ts[j]` is the jth temperature to simulate
        the QUSO at.
    `num_updates` points to an array where `num_updates[j]` is the number of
        times steps to simulate the QUSO at temperature `Ts[j]`.
    `in_order` indicates whether to iterate through the variables in order
        `in_order=1` or randomly `in_order=0` during an update step.
    `seed` seeds the random number generator. If `seed` is a negative integer,
        then we seed the random number generator with `time(NULL)`. Otherwise
        we will use `seed`.

    Returns
    -------
    None.
    This function updates `state` and does not return anything.

    Example
    -------
    `neighbors` and `J` are basically flattened arrays.
    In other words, we flatten the arrays `temp_neighbors` and
    `temp_J`, where `temp_neighbors` points to an array where `temp_neighbors[i][j]`
    is the jth neighbor of spin i, for j=0,...,num_neighbors[i]-1, and similarly,
    `temp_J` points to an array where `temp_J[i][j]` is the coupling value between
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
    */
    pcg32_random_t rng;
    if(seed < 0) {
        pcg32_srandom_r(&rng, (unsigned int)time(NULL), (intptr_t)&rng);
    } else {
        pcg32_srandom_r(&rng, (unsigned)seed, 54u);
    }

    double T, dE;
    int t, i, j, _;

    // index[i] points to where the information for spin i starts
    // in the J and neighbor arrays.
    long *index = (long*)malloc(len_state * sizeof(long));
    index[0] = 0;
    for(i=1; i<len_state; i++) {
        index[i] = index[i-1] + num_neighbors[i-1];
    }

    double *flip_spin_dE;
    flip_spin_dE = (double*)malloc(len_state * sizeof(double));
    compute_flip_dE(
        flip_spin_dE,
        len_state, state, h,
        num_neighbors, neighbors, J,
        index
    );

    for(t=0; t<len_Ts; t++) {
        T = Ts[t];
        for(_=0; _<num_updates[t]; _++) {
            for(j=0; j<len_state; j++) {
                i = in_order ? j : rand_int(&rng, len_state);
                dE = flip_spin_dE[i];
                if(dE <= 0 || (T > 0 && rand_double(&rng) < exp(-dE / T))) {
                    recompute_flip_dE(
                        i, flip_spin_dE, state,
                        num_neighbors, neighbors, J,
                        index
                    );
                    state[i] *= -1;
                }
            }
        }
    }
    free(flip_spin_dE); free(index);
}
