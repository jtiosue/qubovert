#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED

// We use the minimal C random number generator from
//     http://www.pcg-random.org.
// Here we just wrap the pcg_basic.c and pcg_basic.h code with what we
// need it for.

#include "pcg_basic.h"

#define rng_t pcg32_random_t

void rand_seed(rng_t *rng, int seed);
rng_t rand_init(int seed);
double rand_double(rng_t *rng);
int rand_int(rng_t *rng, int stop);

#endif
