#ifndef ANNEAL_PUSO_H_INCLUDED
#define ANNEAL_PUSO_H_INCLUDED

void anneal_puso(  // updates states and values in place
    int num_anneals, int *states, double *values, int len_state,
    long num_terms, int *num_couplings, int *terms, double *couplings,
    int len_Ts, double *Ts, int in_order, int initial_state_provided, int seed
);

#endif
