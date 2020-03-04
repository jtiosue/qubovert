#include "random.h"
#include <stdlib.h>

// TODO; implement better random number generator!

void rand_seed(unsigned int seed) {
	srand(seed);
}


double rand_double() {
    // Compute a random double number with standard library.
    return (double)rand() / ((double)RAND_MAX + 1.);
}


int rand_int(int start, int stop) {
    // random int in [start, stop)
    return (int)floor(
        start + rand_double() * (stop - start)
    );
}
