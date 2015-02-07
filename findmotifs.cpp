// Implement your solutions in this file
#include "findmotifs.h"
#include "hamming.h"

#include <stdexcept>

// implements the sequential findmotifs function
std::vector<bits_t> findmotifs(unsigned int n, unsigned int l,
                               unsigned int d, const bits_t* input)
{
    // If you are not familiar with C++ (using std::vector):
    // For the output (return value) `result`:
    //                  The function asks you to return all values which are
    //                  of a hamming distance `d` from all input values. You
    //                  should return all these values in the return value
    //                  `result`, which is a std::vector.
    //                  For each valid value that you find (i.e., each output
    //                  value) you add it to the output by doing:
    //                      result.push_back(value);
    //                  Note: No other functionality of std::vector is needed.
    // You can get the size of a vector (number of elements) using:
    //                      result.size()

    // create an empty vector
    std::vector<bits_t> result;

    std::vector<bits_t> combinations(1, input[0]); // Stores all combinations obtained by flipping bits in the reference number.
    // Since distance from a particular number can be anything less than d,
    // we iterate over all such possibilities in the following for loop.
    bits_t flipper = 1; // Used for flipping bits, one bit a time from LSB to MSB.
    for (unsigned int i = 0; i < l; ++i, flipper *= 2) {
      uint64_t currentSize = combinations.size(); // Number of combinations to be flipped by one bit in this iteration.
      for (uint64_t j = 0; j < currentSize; ++j) {
        bits_t flipped = combinations[j] ^ flipper; // Store the flipped number after flipping one bit.
        unsigned int ham = hamming(flipped, input[0]); // Hamming distance of the base from the flipped number.
        if (ham <= d) {
          bool isResult = true; // If the flipped number can be reported as result.
          unsigned int idx = 1;
          while (idx < n) {
            // This is a solution only if its Hamming distance is less than or equal to d from all the inputs.
            unsigned int h = hamming(flipped, input[idx]);
            isResult = isResult && (h <= d);
            // If Hamming distance is more than the number of inversions left (2d - ham) then it can never lead to a result.
            if (h > ((2 * d) - ham)) {
              break;
            }
            ++idx;
          }
          if (isResult) {
            result.push_back(flipped);
          }
          // Store the number only if it is a potential solution and we can flip the bits further.
          if ((idx == n) && (ham < d)) {
            combinations.push_back(flipped);
          }
        }
      }
    }

    return result;
}
