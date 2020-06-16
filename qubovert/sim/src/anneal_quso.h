#ifndef ANNEAL_QUSO_H_INCLUDED
#define ANNEAL_QUSO_H_INCLUDED

void anneal_quso(  // updates states and values in place
    int num_anneals, int *states, double *values, int len_state,
    double *h, int *num_neighbors, int *neighbors, double *J,
    int len_Ts, double *Ts, int in_order, int initial_state_provided, int seed
);

#endif
