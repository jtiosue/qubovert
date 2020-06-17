#include "random.h"
#include <math.h>
#include <time.h>


// We use the minimal C random number generator from
//     http://www.pcg-random.org.
// Here we just wrap the pcg_basic.c and pcg_basic.h code with what we
// need it for.


void rand_seed(rng_t *rng, int seed) {
    // If seed is negative, then we will seed with time, otherwise,
    // seed with seed.
    if(seed < 0) {
        pcg32_srandom_r(rng, (unsigned int)time(NULL), (intptr_t)rng);
    } else {
        pcg32_srandom_r(rng, (unsigned)seed, 54u);
    }
}


rng_t rand_init(int seed) {
    // If seed is negative, then we will seed with time, otherwise,
    // seed with seed.
    rng_t rng;
    rand_seed(&rng, seed);
    return rng;
}


double rand_double(rng_t *rng) {
    // random double in [0, 1)
    return ldexp((double)pcg32_random_r(rng), -32);
}


int rand_int(rng_t *rng, int stop) {
    //random integer in [0, top)
    return (int)pcg32_boundedrand_r(rng, stop);
}
