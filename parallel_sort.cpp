/**
 * @file    parallel_sort.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the parallel, distributed sorting function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "parallel_sort.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <vector>

//uncomment for pretty printing
//#include "prettyprint.hpp"

// count the number of times this rank has been used and use the count as a seed
static int counter = 1;


// implementation of your parallel sorting
void parallel_sort(int * begin, int* end, MPI_Comm comm) {
// use the following line for indented printing
//void parallel_sort(int * begin, int* end, MPI_Comm comm, int level) {
  // communicator size
  int q = 0;
  MPI_Comm_size(comm, &q);

  // rank in the communicator
  int r = -1;
  MPI_Comm_rank(comm, &r);

  int local_size = end - begin;

  //int global_r = -1;
  //MPI_Comm_rank(MPI_COMM_WORLD, &global_r);

  //std::cout << std::string(level, '\t') << global_r << ": initial = " << pretty_print_array(begin, local_size) << std::endl;
  //fflush(stdout);

  // sort sequentially if the communicator consists of only one processor
  if (q == 1) {
    // use C++ std::sort for now
    // TODO: review this later
    std::sort(begin, end);
    //std::cout << std::string(level, '\t') << global_r << ": sequentially sorted = " << pretty_print_array(begin, local_size) << std::endl;
    //fflush(stdout);
    return;
  }

  // determine minimum index in this processor
  int m = 0;
  MPI_Allreduce(&local_size, &m, 1, MPI_INT, MPI_SUM, comm);
  int min_index = 0;
  for (int p = 0; p < r; ++p) {
    min_index += block_decompose(m, q, p);
  }

  // seed the random number generator now
  srand(counter);
  ++counter;
  // pick an index in the range [0, m-1]
  int k = random_at_max(m - 1);
  // set pivot to actual value in the processor that has the pivot
  int pivot = 0;
  if ((k >= min_index) && (k < min_index + local_size)) {
    pivot = begin[k - min_index];
  }
  // now do an all-reduce using sum so that all the processors get pivot
  // notice that the complexity of this operation is same as broadcast
  MPI_Allreduce(MPI_IN_PLACE, &pivot, 1, MPI_INT, MPI_SUM, comm);

  // partition the local array into two subarrays
  // one containing elements greater than the pivot
  // the other with elements less than or equal to the pivot
  int boundary = partition(begin, local_size, pivot);

  int partition_sizes[2];
  partition_sizes[0] = boundary + 1;
  partition_sizes[1] = (local_size - partition_sizes[0]);

  //std::cout << std::string(level, '\t') << global_r << ": pivot = " << pivot << std::endl;
  //std::cout << std::string(level, '\t') << global_r << ": " << pretty_print_array(begin, partition_sizes[0]) << ", " <<
                                                               //pretty_print_array(&begin[boundary + 1], partition_sizes[1]) << std::endl;
  //fflush(stdout);
  //std::cout << r << ": partition_sizes = " << partition_sizes << std::endl;
  //fflush(stdout);

  int current_sizes[2 * q];
  MPI_Allgather(&partition_sizes[0], 2, MPI_INT, &current_sizes[0], 2, MPI_INT, comm);

  int sum_sizes[2] = {};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < q; ++j) {
      sum_sizes[i] += current_sizes[j * 2 + i];
    }
  }
  //std::cout << r << ": " << sum_sizes << std::endl;
  //fflush(stdout);

  // partition the number of processors based on the sizes of subarrays
  int num_processors[2] = {};
  for (int i = 0; i < 2; ++i) {
    num_processors[i] = ceil((sum_sizes[i] * q) / static_cast<double>(m));
    if (num_processors[i] == 0) {
      num_processors[i] = 1;
    }
  }
  while (num_processors[0] + num_processors[1] > q) {
    int i = sum_sizes[0] > sum_sizes[1] ? 1 : 0;
    if (num_processors[i] == 1) {
      i = (i + 1) % 2;
    }
    num_processors[i] -= 1;
  }

  //std::cout << std::string(level, '\t') << global_r << ": num_processors = " << num_processors << std::endl;
  //fflush(stdout);

  // now split the communicator
  MPI_Comm newcomm;
  int color = (r < num_processors[0]) ? 0 : 1;
  MPI_Comm_split(comm, color, r, &newcomm);

  // determine the final size in this processor after the distribution
  int final_size = block_decompose(sum_sizes[color], newcomm);
  // create a new buffer of the appropriate size
  std::vector<int> final_buf(final_size);
  int* final_ptr = final_size > 0 ? &final_buf[0] : NULL;

  // now distribute_data the data, once for each subarray
  int offset = 0;
  int data_offset = 0;
  for (int i = 0; i < 2; ++i) {
    std::vector<int> final_sizes(q);
    for (int j = 0; j < num_processors[i]; ++j) {
      final_sizes[j + offset] = block_decompose(sum_sizes[i], num_processors[i], j);
    }
    //std::cout << r << ", " << i << ": final_sizes = " << final_sizes << std::endl;
    //fflush(stdout);
    //std::cout << r << ", " << i << ": current_sizes = " << pretty_print_array(current_sizes, 2 * q) << std::endl;
    //fflush(stdout);
    //std::cout << r << ", " << i << ": begin[data_offset] = " << pretty_print_array(&begin[data_offset], current_sizes[r * 2 + i]) << std::endl;
    //fflush(stdout);
    int* sendbuf = (begin != NULL) ? &begin[data_offset] : NULL;
    if (i == color) {
      distribute_data(sendbuf, current_sizes, final_ptr, &final_sizes[0], i, comm);
    }
    else {
      distribute_data(sendbuf, current_sizes, NULL, &final_sizes[0], i, comm);
    }
    offset += num_processors[i];
    data_offset += (boundary + 1);
  }
  //std::cout << std::string(level, '\t') << global_r << ": final_buf = " << final_buf << std::endl;
  //fflush(stdout);

  // again call parallel sort, only if there is something to sort though
  if (sum_sizes[color] > 0) {
    parallel_sort(final_ptr, final_ptr + final_size, newcomm);
    // uncomment the following line for indented printing
    //parallel_sort(final_ptr, final_pt + final_size, newcomm, level + 1);
  }

  for (int i = 0, p = 0; i < 2; ++i) {
    for (int j = 0; j < num_processors[i]; ++j) {
      current_sizes[p++] = block_decompose(sum_sizes[i], num_processors[i], j);
    }
  }

  std::vector<int> final_sizes(q);
  for (int p = 0; p < q; ++p) {
    final_sizes[p] = block_decompose(m, q, p);
  }

  // collect back the data from both the subcommunicators
  collect_data(final_ptr, &current_sizes[0], begin, &final_sizes[0], comm);

  //std::cout << std::string(level, '\t') << global_r << ": parallel sorted = " << pretty_print_array(begin, local_size) << std::endl;
  //fflush(stdout);
}


/*********************************************************************
 *             Implement your own helper functions here:             *
 *********************************************************************/

// slightly modified version of http://stackoverflow.com/a/6852396
// Assumes 0 <= max <= RAND_MAX
// Returns in the half-open interval [0, max]
int random_at_max(int max) {
  unsigned num_bins = static_cast<unsigned>(max) + 1;
  // max <= RAND_MAX < ULONG_MAX, so this is okay.
  unsigned num_rand = static_cast<unsigned>(RAND_MAX) + 1;
  unsigned bin_size = num_rand / num_bins;
  unsigned defect = num_rand % num_bins;

  int x;
  do {
    x = rand();
  }
  // This is carefully written not to overflow
  while ((num_rand - defect) <= static_cast<unsigned>(x));

  // Truncated division is intentional
  return x / static_cast<int>(bin_size);
}

// partitions the given array using the given pivot
// returns the index of the largest element <= pivot after partition
int partition(int* a, int size, int pivot)
{
  int left = 0, right = size - 1;
  while (left < right) {
    if ((a[left] > pivot) && (a[right] <= pivot)) {
      int temp = a[left];
      a[left] = a[right];
      a[right] = temp;
    }
    if (a[left] <= pivot) {
      ++left;
    }
    if (a[right] > pivot) {
      --right;
    }
  }
  while (left >= size) {
    --left;
  }
  while ((left >= 0) && (a[left] > pivot)) {
    --left;
  }
  return left;
}

void calculate_displacements(const int* const count, int* const displs, int size)
{
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    displs[i] = offset;
    offset += count[i];
  }
}

void distribute_data(int* sendbuf, int* current_sizes, int* recvbuf, int* final_sizes, int color, MPI_Comm comm)
{
  int r = -1;
  MPI_Comm_rank(comm, &r);

  int q = 0;
  MPI_Comm_size(comm, &q);

  std::vector<int> extra_sizes(q);
  std::vector<int> need_sizes(q);
  for (int p = 0; p < q; ++p) {
    if (final_sizes[p] < current_sizes[p * 2 + color]) {
      extra_sizes[p] = current_sizes[p * 2 + color] - final_sizes[p];
    }
    else if (final_sizes[p] > current_sizes[p * 2 + color]) {
      need_sizes[p] = final_sizes[p] - current_sizes[p * 2 + color];
    }
  }
  //std::cout << r << ", " << color << ": current_sizes = " << current_sizes << std::endl;
  //std::cout << r << ", " << color << ": need_sizes = " << need_sizes << std::endl;
  //std::cout << r << ", " << color << ": extra_sizes = " << extra_sizes << std::endl;
  //fflush(stdout);

  std::vector<int> cumulative_extra(q);
  std::partial_sum(extra_sizes.begin(), extra_sizes.end(), cumulative_extra.begin());
  std::vector<int> cumulative_need(q);
  std::partial_sum(need_sizes.begin(), need_sizes.end(), cumulative_need.begin());

  std::vector<int> sendcounts(q);
  std::vector<int> recvcounts(q);

  std::vector<int> sdispls(q);
  std::vector<int> rdispls(q);

  int extra_min = cumulative_extra[r] - extra_sizes[r];
  int extra_max = cumulative_extra[r] - 1;

  int need_min = cumulative_need[r] - need_sizes[r];
  int need_max = cumulative_need[r] - 1;

  for (int p = 0; p < q; ++p) {
    if (r != p) {
      if ((cumulative_extra[p] > need_min) && (cumulative_extra[p] - extra_sizes[p] <= need_max)) {
        recvcounts[p] = std::min(cumulative_extra[p] - 1, need_max) - std::max(cumulative_extra[p] - extra_sizes[p], need_min) + 1;
      }
      if ((extra_max + 1 > cumulative_need[p] - need_sizes[p]) && (extra_min <= cumulative_need[p] - 1)) {
        sendcounts[p] = std::min(extra_max, cumulative_need[p] - 1) - std::max(extra_min, cumulative_need[p] - need_sizes[p]) + 1;
      }
    }
  }

  if (extra_sizes[r] > 0) {
    sendcounts[r] = final_sizes[r];
    recvcounts[r] = final_sizes[r];
  }
  else {
    sendcounts[r] = current_sizes[r * 2 + color];
    recvcounts[r] = current_sizes[r * 2 + color];
  }
  calculate_displacements(&sendcounts[0], &sdispls[0], q);
  calculate_displacements(&recvcounts[0], &rdispls[0], q);


  //std::cout << r << ", " << color << ": sendcounts = " << sendcounts << std::endl;
  //std::cout << r << ", " << color << ": sdispls = " << sdispls << std::endl;
  //std::cout << r << ", " << color << ": recvcounts = " << recvcounts << std::endl;
  //std::cout << r << ", " << color << ": rdispls = " << rdispls << std::endl;
  //fflush(stdout);
  MPI_Alltoallv(sendbuf, &sendcounts[0], &sdispls[0], MPI_INT, recvbuf, &recvcounts[0], &rdispls[0], MPI_INT, comm);
}

void collect_data(int* sendbuf, int* current_sizes, int* recvbuf, int* final_sizes, MPI_Comm comm)
{
  int r = -1;
  MPI_Comm_rank(comm, &r);

  int q = 0;
  MPI_Comm_size(comm, &q);

  std::vector<int> cumulative_current(current_sizes, current_sizes + q);
  std::partial_sum(cumulative_current.begin(), cumulative_current.end(), cumulative_current.begin());
  std::vector<int> cumulative_final(final_sizes, final_sizes + q);
  std::partial_sum(cumulative_final.begin(), cumulative_final.end(), cumulative_final.begin());

  std::vector<int> sendcounts(q);
  std::vector<int> recvcounts(q);

  std::vector<int> sdispls(q);
  std::vector<int> rdispls(q);

  int current_min = cumulative_current[r] - current_sizes[r];
  int current_max = cumulative_current[r] - 1;

  int final_min = cumulative_final[r] - final_sizes[r];
  int final_max = cumulative_final[r] - 1;

  for (int p = 0; p < q; ++p) {
    if ((cumulative_current[p] > final_min) && (cumulative_current[p] - current_sizes[p] <= final_max)) {
      recvcounts[p] = std::min(cumulative_current[p] - 1, final_max) - std::max(cumulative_current[p] - current_sizes[p], final_min) + 1;
    }
    if ((current_max + 1 > cumulative_final[p] - final_sizes[p]) && (current_min <= cumulative_final[p] - 1)) {
      sendcounts[p] = std::min(current_max, cumulative_final[p] - 1) - std::max(current_min, cumulative_final[p] - final_sizes[p]) + 1;
    }
  }
  calculate_displacements(&sendcounts[0], &sdispls[0], q);
  calculate_displacements(&recvcounts[0], &rdispls[0], q);

  //std::cout << r << ": sendcounts = " << sendcounts << std::endl;
  //std::cout << r << ": sdispls = " << sdispls << std::endl;
  //std::cout << r << ": recvcounts = " << recvcounts << std::endl;
  //std::cout << r << ": rdispls = " << rdispls << std::endl;
  //fflush(stdout);
  MPI_Alltoallv(sendbuf, &sendcounts[0], &sdispls[0], MPI_INT, recvbuf, &recvcounts[0], &rdispls[0], MPI_INT, comm);
}
