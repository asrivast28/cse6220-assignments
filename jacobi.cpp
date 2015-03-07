/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include <limits>
#include <vector>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    for (int i = 0; i < n; ++i) {
      y[i] = 0.0;
      for (int j = 0; j < n; ++j) {
        y[i] += (A[n*i + j] * x[j]);
      }
    }
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    for (int i = 0; i < n; ++i) {
      y[i] = 0.0;
      for (int j = 0; j < m; ++j) {
        y[i] += (A[n*i + j] * x[j]);
      }
    }
}

// calculates L2-norm of (Ax - b)
double calculate_l2_norm(const int n, const double* A, const double* b, const double* x)
{
  double l2_norm = 0.0;
  std::vector<double> y(n);
  matrix_vector_mult(n, A, x, &y[0]); 
  for (int i = 0; i < n; ++i) {
    l2_norm += pow(y[i] - b[i], 2.0);
  }
  return sqrt(l2_norm);
}

// sets x to (b - Rx) / D
void update_x(const int n, const double* b, double* x, const double* D, const double* R)
{
  std::vector<double> y(n);
  matrix_vector_mult(n, R, x, &y[0]); 
  for (int i = 0; i < n; ++i) {
    x[i] = (b[i] - y[i]) / D[i];
  }
}

// epsilon used for comparing double elements
const double DOUBLE_EPSILON = std::numeric_limits<double>::epsilon();

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    // vector for storing diagonal elements of A
    std::vector<double> D(n);
    // vector for storing A - D
    std::vector<double> R(A, A + n*n);
    // fill D and remove corresponding elements from R
    for (int i = 0; i < n; ++i) {
      D[i] = A[i*n + i];
      R[i*n + i] = 0.0;
    }

    // iterate until a maximum number of iterations has been reached
    for (int iter = 0; iter < max_iter; ++iter) {
      // calculate ||Ax - b||
      double l2_norm = calculate_l2_norm(n, A, b, x);
      // check if ||Ax - b|| > l
      if ((l2_norm - l2_termination) < DOUBLE_EPSILON) {
        break;
      }
      // update x otherwise
      update_x(n, b, x, &D[0], &R[0]);
    }
}
