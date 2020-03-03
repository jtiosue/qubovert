#ifndef simulate_quso_INCLUDED
#define simulate_quso_INCLUDED
void simulate_quso(
	int len_state, int *state, double *h,
	int *num_neighbors, int *neighbors, double *J,
	int len_Ts, double *Ts, int *num_updates,
	int seed
);
#endif
