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
     */
    uint64_t z = x ^ y;

    /*
     * Number of set bits is calculated based on the method described at
     * https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSet64
     */
    z = (z & (0x5555555555555555)) + ((z >> 1) & (0x5555555555555555));
    z = (z & (0x3333333333333333)) + ((z >> 2) & (0x3333333333333333));
    z = (z & (0x0f0f0f0f0f0f0f0f)) + ((z >> 4) & (0x0f0f0f0f0f0f0f0f));
    z = (z & (0x00ff00ff00ff00ff)) + ((z >> 8) & (0x00ff00ff00ff00ff));
    z = (z & (0x0000ffff0000ffff)) + ((z >> 16) & (0x0000ffff0000ffff));
    z = (z & (0x00000000ffffffff)) + ((z >> 32) & (0x00000000ffffffff));

    return z;
}
