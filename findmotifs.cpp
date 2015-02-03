// Implement your solutions in this file
#include "findmotifs.h"
#include "hamming.h"

// Returns all l bit number which contain d 1s in the binary representation.
std::vector<bits_t> getcombinations(unsigned int l, unsigned int d,
                                    unsigned int currentidx = 0, unsigned int currentd = 0,
                                    bits_t number = 0)
{
    std::vector<bits_t> combinations;
    if (d > 0) {
      // We push 1s to the number recursively at every level until we have pushed maximum number
      // of allowed 1s in the number. We consider all remaining possibilities at this level in
      // the following for loop.
      bits_t base = 1;
      for (unsigned int idx = currentidx; idx <= l - (d - currentd); ++idx) {
        bits_t shifted = base << idx;
        if ((currentd + 1) < d) {
          std::vector<bits_t> idxCombinations(getcombinations(l, d, idx + 1, currentd + 1, number | shifted));
          combinations.insert(combinations.end(), idxCombinations.begin(), idxCombinations.end());
        }
        else {
          combinations.push_back(number | shifted);
        }
      }
    }
    else {
      // There is only one possibility, 0, if d is 0.
      combinations.push_back(number);
    }
    return combinations;
}

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

    // Since distance from a particular number can be anything less than d,
    // we iterate over all such possibilities in the following for loop.
    for (unsigned int i = 0; i <= d; ++i) {
      // Get all l digit numbers with exactly i '1's.
      std::vector<bits_t> combinations(getcombinations(l, i));
      // For getting all combinations of numbers, we take the combinations we got and XOR it with
      // the first input (first input is treated as reference).
      for (std::vector<bits_t>::const_iterator c = combinations.begin(); c != combinations.end(); ++c) {
        bits_t flipped = input[0] ^ *c;
        unsigned int j = 1;
        // Compare the newly obtained number with every other input number.
        while ((j < n) && (hamming(input[j], flipped) <= d)) {
          ++j;
        }
        // Store the number if all the input numbers have a hamming distance of less than equal to d.
        if (j == n) {
          result.push_back(flipped);
        }
      }
    }
    return result;
}
