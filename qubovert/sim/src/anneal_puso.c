#include "anneal_puso.h"
#include "random.h"
#include <math.h>
#include <stdlib.h>


double puso_subgraph_value(
    int *state, int spin,
    int *num_couplings, int *terms, double *couplings,
    long *index, long **subgraphs
) {
    /*
    Find the value of the PUSO defined by the subgraph of terms containing
    spin ``spin``.

    Parameters
    ----------
    state : points to an integer array.
        `state[i]` is either 1 or -1 representing the state of spin ``i``.
    spin : int.
        The spin defining the subgraph.
    num_couplings : points to an int array.
        ``num_couplings[i]`` is the number of spins in the ith term.
    terms : points to an int array.
        ``terms`` contains all the terms in the PUSO.
    couplings : points to a double array.
        The coupling value for each term.
    index : points to a long array.
        ``index[i]`` points to where the ith term starts in the
        ``terms`` array.
    subgraphs : a bunch of pointers to long arrays.
        ``subgraphs[i][0]`` is the number of terms that spin ``i`` is involved
        in. ``subgraphs[i][j + 1]`` is the jth term that spin ``i`` is
        involved in, for ``j = 0`` to ``j = subgraphs[i][0] - 1``.

    Return
    ------
    value : double.

    Example
    -------
    Consider a PUSO
        ``z_0 z_1 - z_1 z_2 z_3 + 3 z_2``.
    Then we would have the following arguments:
        ``terms = {0, 1,   1, 2, 3,   2}``.
        ``num_couplings = {2, 3, 1}``.
        ``couplings = {1, -1, 3}``.
    Note that I add those big spaces in the ``terms`` array for the visuals
    so that you can see how these terms are separated.

    In the example above:
        ``index = {0, 2, 5}``
        ``subgraphs = {
            {1, 0},  // spin 0 is involved in term 0
            {2, 0, 1},  // spin 1 is involved in term 0 and 1
            {2, 1, 2},  // spin 2 is involved in term 1 and 2
            {1, 1}  // spin 3 is involved in term 1
        }``

    Notice how the zeroth element of the array ``subgraphs[i]``
    is the number of terms that spin ``i`` is involved in.

    */
    double value = 0.; int product; long i, term; int j;
    for(i=1; i<=subgraphs[spin][0]; i++) {
        term = subgraphs[spin][i];
        product = 1;
        for(j=0; j<num_couplings[term]; j++) {
            // index[term] maps the term to where it starts in `terms`.
            product *= state[terms[index[term] + j]];
        }
        value += couplings[term] * (double)product;
    }
    return value;
}


void single_anneal_puso(
    int len_state, int *state,
    int *num_couplings, int *terms, double *couplings,
    long *index, long **subgraphs,
    int len_Ts, double *Ts, int in_order, rng_t *rng
) {
    /*
    Run one simulated annealing algorithm. Updates ``state`` in place.

    Parameters
    ----------
    len_state : int.
        The number of spins.
    state : points to an integer array.
        `state[i]` is either 1 or -1 representing the state of spin ``i``.
    num_couplings : points to an int array.
        ``num_couplings[i]`` is the number of spins in the ith term.
    terms : points to an int array.
        ``terms`` contains all the terms in the PUSO.
    couplings : points to a double array.
        The coupling value for each term.
    index : points to a long array.
        ``index[i]`` points to where the ith term starts in the
        ``terms`` array.
    subgraphs : a bunch of pointers to long arrays.
        ``subgraphs[i][0]`` is the number of terms that spin ``i`` is involved
        in. ``subgraphs[i][j + 1]`` is the jth term that spin ``i`` is
        involved in, for ``j = 0`` to ``j = subgraphs[i][0] - 1``.
    len_Ts : int.
        The duration of the temperature schedule.
    Ts : points to a double array.
        `Ts[i]` is the temperature to update the state at time step ``i``.
    in_order : bool.
        Indicates whether to iterate through the variables in order
        `in_order=1` or randomly `in_order=0` during an update step.
    rng : rng_t (from random.h). 
        The random number generator's state.

    Example
    -------
    Consider a PUSO
        ``z_0 z_1 - z_1 z_2 z_3 + 3 z_2``.
    Then we would have the following arguments:
        ``terms = {0, 1,   1, 2, 3,   2}``.
        ``num_couplings = {2, 3, 1}``.
        ``couplings = {1, -1, 3}``.
    Note that I add those big spaces in the ``terms`` array for the visuals
    so that you can see how these terms are separated.

    In the example above:
        ``index = {0, 2, 5}``
        ``subgraphs = {
            {1, 0},  // spin 0 is involved in term 0
            {2, 0, 1},  // spin 1 is involved in term 0 and 1
            {2, 1, 2},  // spin 2 is involved in term 1 and 2
            {1, 1}  // spin 3 is involved in term 1
        }``

    Notice how the zeroth element of the array ``subgraphs[i]``
    is the number of terms that spin ``i`` is involved in.

    */
    double T, dE;
    int t, i, j;

    for(t=0; t<len_Ts; t++) {
        T = Ts[t];
        for(j=0; j<len_state; j++) {
            i = in_order ? j : rand_int(rng, len_state);

            // We take the value of the PUSO of all spins in the subgraph
            // containing spin i. Call this value E. Then if we flip spin i,
            // the new energy will be -E. Therefore, the change in energy from
            // flipping spin i is dE = -2 E.
            dE = -2 * puso_subgraph_value(
                state, i, num_couplings, terms, couplings,
                index, subgraphs
            );

            if(dE <= 0 || (T > 0 && rand_double(rng) < exp(-dE / T))) {
                state[i] *= -1;
            }
        }
    }
}


double puso_value(
    int *state,
    long num_terms, int *num_couplings, int *terms, double *couplings
) {
    /*
    Find the PUSOs value with the spin state `state`.

    Parameters
    ----------
    state : points to an integer array.
        `state[i]` is either 1 or -1 representing the state of spin ``i``.
    num_terms : long int.
        The number of terms in the PUSO.
    num_couplings : points to an int array.
        ``num_couplings[i]`` is the number of spins in the ith term.
    terms : points to an int array.
        ``terms`` contains all the terms in the PUSO.
    couplings : points to a double array.
        The coupling value for each term.

    Returns
    -------
    value : double

    Example
    -------
    Consider a PUSO
        ``z_0 z_1 - z_1 z_2 z_3 + 3 z_2``.
    Then we would have the following arguments:
        ``num_terms = 3``.
        ``terms = {0, 1,   1, 2, 3,   2}``.
        ``num_couplings = {2, 3, 1}``.
        ``couplings = {1, -1, 3}``.
    Note that I add those big spaces in the ``terms`` array for the visuals
    so that you can see how these terms are separated.

    */
    long index = 0; double value = 0.; int _, product;
    for(long term=0; term<num_terms; term++) {
        product = 1;
        for(_=0; _<num_couplings[term]; _++) {
            product *= state[terms[index]];
            index++;
        }
        value += couplings[term] * (double)product;
    }
    return value;
}


void anneal_puso(  // updates states and values in place
    int num_anneals, int *states, double *values, int len_state,
    long num_terms, int *num_couplings, int *terms, double *couplings,
    int len_Ts, double *Ts, int in_order, int initial_state_provided, int seed
) {
    /*
    Run many rounds of simulated annealing.
    Updates ``states`` and ``values`` in place.

    Parameters
    ----------
    num_anneals : int.
        The number of times to run simulated annealing.
    states : points to a buffer array to build the resulting states.
        It will be of dimension `states[num_anneals * len_state]`.
        The jth spin in the ith state can be accessed with 
        `states[i * len_state + j]`.
    values : points to a buffer array to store the resulting values.
        It will be of dimension `values[num_anneals]`.
    len_state : int.
        The number of spins.
    state : points to an integer array.
        `state[i]` is either 1 or -1 representing the state of spin ``i``.
    num_terms : long int.
        The number of terms in the PUSO.
    num_couplings : points to an int array.
        ``num_couplings[i]`` is the number of spins in the ith term.
    terms : points to an int array.
        ``terms`` contains all the terms in the PUSO.
    couplings : points to a double array.
        The coupling value for each term.
    len_Ts : int.
        The duration of the temperature schedule.
    Ts : points to a double array.
        `Ts[i]` is the temperature to update the state at time step ``i``.
    in_order : bool.
        Indicates whether to iterate through the variables in order
        `in_order=1` or randomly `in_order=0` during an update step.
    initial_state_provided : bool.
        If ``initial_state_provided == 0``, then we randomly initiate each
        initial state for each anneal. Otherwise, we assume that the desired
        starting states are encoded in the buffer ``states``.
    seed : int.
        The value to seed the random number generator.
        If `seed < 0`, then the random number generator will be seeded with
        the internal clock.

    Example
    -------
    Consider a PUSO
        ``z_0 z_1 - z_1 z_2 z_3 + 3 z_2``.
    Then we would have the following arguments:
        ``num_terms = 3``.
        ``terms = {0, 1,   1, 2, 3,   2}``.
        ``num_couplings = {2, 3, 1}``.
        ``couplings = {1, -1, 3}``.
    Note that I add those big spaces in the ``terms`` array for the visuals
    so that you can see how these terms are separated.

    */
    int i, j, k;

    int *state = (int*)malloc(len_state * sizeof(int));
    rng_t rng = rand_init(seed);

    // create subgraphs and index. please see the Parameters and Example
    // sections in the comments of the `single_anneal_puso` function for info
    // on what these array are. But basically, `index[term]` maps a term
    // number `term` to where that term starts in the `terms` array.
    // `subgraphs[spin][0]` is the number of terms that `spin` is involved
    // in. And `subgraphs[spin][j]` is the jth term that `spin` is involved
    // in.

    long **subgraphs = (long**)malloc(len_state * sizeof(long*));
    for(i=0; i<len_state; i++) {
        subgraphs[i] = (long*)malloc(sizeof(long));
        subgraphs[i][0] = 0;  // number of terms that spin i is involved in.
    }

    long *index = (long*)malloc(num_terms * sizeof(long));
    index[0] = 0;
    for(long term=0; term<num_terms; term++) {
        if(term) {
            index[term] = index[term-1] + num_couplings[term-1];
        }
        for(i=0; i<num_couplings[term]; i++) {
            j = terms[index[term] + i];  // spin j is involved in term `term`.
            subgraphs[j][0]++; k = subgraphs[j][0];
            subgraphs[j] = (long*)realloc(subgraphs[j], (k+1) * sizeof(long));
            subgraphs[j][k] = term;
        }
    }

    // run simulated annealing `num_anneals` times.
    for(i=0; i<num_anneals; i++) {
        // generate random initial state
        for(j=0; j<len_state; j++) {
            if(initial_state_provided) {
                state[j] = states[i * len_state + j];
            } else {
                state[j] = rand_double(&rng) < 0.5 ? 1 : -1;
            }
        }

        // run simulated annealing, updates `state` in place.
        single_anneal_puso(
            len_state, state,
            num_couplings, terms, couplings,
            index, subgraphs,
            len_Ts, Ts, in_order, &rng
        );

        // add the new state and the new value to the buffers
        values[i] = puso_value(state, num_terms, num_couplings, terms, couplings);
        for(j=0; j<len_state; j++) {
            states[i * len_state + j] = state[j];
        }
    }

    // free the arrays we created.
    free(state); free(index);
    for(i=0; i<len_state; i++) {
        free(subgraphs[i]);
    }
    free(subgraphs);
}
