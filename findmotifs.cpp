// Implement your solutions in this file
#include "findmotifs.h"
#include "hamming.h"

// checks if the given number is a solution
void check_solution(const unsigned int n, const unsigned int d,
                    const bits_t* input, const bits_t number,
                    std::vector<bits_t>& result)
{
    unsigned int j = 1;
    while ((j < n) && (hamming(input[j], number) <= d)) {
      ++j;
    }
    if (j == n) {
      result.push_back(number);
    }
}

// Explores the solution space obtained by flipping b bits in the l bit number
void explore_solutions(const unsigned int n, const unsigned int l,
                       const unsigned int d, const bits_t* input,
                       const unsigned int b, const bits_t number,
                       std::vector<bits_t>& result,
                       unsigned int currentidx, unsigned int currentd)
{
    if (b > 0) {
      // We flip bits in the original number recursively at every level until we have flipped maximum number
      // of allowed bits in the number. We consider all remaining possibilities at this level in
      // the following for loop.
      bits_t base = 1;
      for (unsigned int idx = currentidx; idx <= l - (b - currentd); ++idx) {
        // flip the bit by XOR-ing with appropriate number
        bits_t flipped = number ^ (base << idx);
        if ((currentd + 1) < b) {
          explore_solutions(n, l, d, input, b, flipped, result, idx + 1, currentd + 1);
        }
        else {
          check_solution(n, d, input, flipped, result);
        }
      }
    }
    else {
      // There is only one possibility, 0, if b is 0.
      check_solution(n, d, input, number, result);
    }
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
    for (unsigned int b = 0; b <= d; ++b) {
      // Search the solution space obtained by flipping b bits in input[0]
      explore_solutions(n, l, d, input, b, input[0], result);
    }
    return result;
}
