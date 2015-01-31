// Implement your solutions in this file
#include "hamming.h"

unsigned int hamming(uint64_t x, uint64_t y)
{
    /*
     * Hamming distance between x and y is calculated
     * by first obtaining XOR of the two numbers and
     * then counting the number of 1s in the resulting
     * number by right shifting 64 times (maximum
     * possible size) and checking the least significant
     * bit.
     *
     * TODO: Is there a better/more efficient way
     *       of counting 1s in bit representation of a number?
     */
    unsigned int d = 0;
    uint64_t z = x ^ y;
    for (unsigned short i = 0; i < 64; ++i) {
      d += z & 1;
      z = z >> 1;
    }
    return d;
}
