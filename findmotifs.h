/**
 * @file    findmotifs.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Declares the sequential findmotifs function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
// You DO NOT need to change anything in this file.
#ifndef FINDMOTIFS_H
#define FINDMOTIFS_H

#include <vector>
#include <stdint.h>

/// The datatype used for representing a (up to) 64 bit sequence.
typedef uint64_t bits_t;

/**
 * @brief   Solves the motif finding problem _sequentially_.
 *
 * @param n         The number of input sequences.
 * @param l         The length (in bits) of each input sequence.
 * @param d         The number of bits that are allowed to differ between
 *                  the answers and all input sequences.
 * @param input     An array of the `n` input sequences. Represented as
 *                  64 bit integers.
 *
 * @return          A std::vector containing all answers.
 */
std::vector<bits_t> findmotifs(unsigned int n, unsigned int l, unsigned int d,
                               const bits_t* input);

#endif // FINDMOTIFS_H
