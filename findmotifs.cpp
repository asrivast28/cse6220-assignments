// Implement your solutions in this file
#include "findmotifs.h"
#include "hamming.h"

#include <stdexcept>

// Returns all l bit combinations of number with d bits flipped in the binary representation.
void getcombinations(const unsigned int l, const unsigned int d,
                     const bits_t number, std::vector<bits_t>::iterator& cit,
                     unsigned int currentidx = 0, unsigned int currentd = 0)
{
    if (d > 0) {
      // We flip bits in the original number recursively at every level until we have flipped maximum number
      // of allowed bits in the number. We consider all remaining possibilities at this level in
      // the following for loop.
      bits_t base = 1;
      for (unsigned int idx = currentidx; idx <= l - (d - currentd); ++idx) {
        // flip the bit by XOR-ing with appropriate number
        bits_t flipped = number ^ (base << idx);
        if ((currentd + 1) < d) {
          getcombinations(l, d, flipped, cit, idx + 1, currentd + 1);
        }
        else {
          *cit = flipped;
          ++cit;
        }
      }
    }
    else {
      // There is only one possibility, 0, if d is 0.
      *cit = number;
      ++cit;
    }
}

// Returns the number of ways in which i items can be chosen from l items.
uint64_t lChoosei(unsigned int l, unsigned int i)
{
    if (i == 0) {
      return 1;
    }

    if ((i * 2) > l) {
      i = l - i;
    }
    uint64_t result = l;
    for (unsigned int j = 2; j <= i; ++j) {
      result *= (l - j + 1);
      result /= j;
    }
    return result;
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
    std::vector<bits_t> combinations;
    // Iterate in reverse order so that std::vector doesn't have to reallocate memory.
    for (int i = d; i >= 0; --i) {
      // Get all l digit numbers which differs from the first input (first input is treated as reference)
      // at exactly i bits.
      combinations.resize(lChoosei(l, i));
      std::vector<bits_t>::iterator cit = combinations.begin();
      getcombinations(l, i, input[0], cit);
      if (cit != combinations.end()) {
        throw std::runtime_error("Not all combinations were found!");
      }
      for (std::vector<bits_t>::const_iterator c = combinations.begin(); c != combinations.end(); ++c) {
        unsigned int j = 1;
        // Compare the newly obtained number with every other input number.
        while ((j < n) && (hamming(input[j], *c) <= d)) {
          ++j;
        }
        // Store the number if all the input numbers have a hamming distance of less than equal to d.
        if (j == n) {
          result.push_back(*c);
        }
      }
    }
    return result;
}
