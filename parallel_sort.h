/**
 * @file    parallel_sort.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Declares the parallel sorting function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#ifndef PARALLEL_SORT_H
#define PARALLEL_SORT_H

#include <mpi.h>

/**
 * @brief   Parallel, distributed sorting over all processors in `comm`. Each
 *          processor has the local input [begin, end).
 *
 * Note that `end` is given one element beyond the input. This corresponds to
 * the API of C++ std::sort! You can get the size of the local input with:
 * int local_size = end - begin;
 *
 * @param begin Pointer to the first element in the input sequence.
 * @param end   Pointer to one element past the input sequence. Don't access this!
 * @param comm  The MPI communicator with the processors participating in the
 *              sorting.
 */
//void parallel_sort(int * begin, int* end, MPI_Comm comm);
// uncomment the following line for indented printing
void parallel_sort(int * begin, int* end, MPI_Comm comm, int level = 0);


/*********************************************************************
 *              Declare your own helper functions here               *
 *********************************************************************/

int random_at_max(int max);

int partition(int* a, int size, int pivot); 

void distribute_data(int* sendbuf, int* proc_sizes, int* recvbuf, int* final_sizes, int color, MPI_Comm comm);

void collect_data(int* sendbuf, int* current_sizes, int* recvbuf, int* final_sizes, MPI_Comm comm);

#endif // PARALLEL_SORT_H
