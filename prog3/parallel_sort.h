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
void parallel_sort(int * begin, int* end, MPI_Comm comm);
// uncomment the following line for indented printing
//void parallel_sort(int * begin, int* end, MPI_Comm comm, int level = 0);


/*********************************************************************
 *              Declare your own helper functions here               *
 *********************************************************************/

/**
 * @brief Returns a random number in the range [0, max].
 *
 * @param max Maximum possible random number.
 *
 * @return The random number generated.
 */
int random_at_max(int max);

/**
 * @brief Partitions the given array about the pivot, in place.
 *
 * @param a     Pointer to the first element in the array.
 * @param size  Total size of the array.
 * @param pivot The pivot element.
 *
 * @return Returns the index of the last element less than or equal to the pivot,
 *         in the partitioned array.
 */
int partition(int* a, int size, int pivot);

/**
 * @brief Distributes the partitioned data in each processor to the processors assigned to the partition.
 *
 * @param sendbuf     Pointer to the buffer which contains elements in this partition.
 * @param proc_sizes  Number of elements belonging to both the partitions in all the processors.
 * @param recvbuf     Pointer to the buffer which will store the distributed elements.
 * @param final_sizes Expected final number of elements in all the processors.
 * @param color       Indicates the partition for which we are distributing.
 * @param comm        The MPI communicator for all the processors in which data is to be distributed.
 */
void distribute_data(int* sendbuf, int* proc_sizes, int* recvbuf, int* final_sizes, int color, MPI_Comm comm);

/**
 * @brief Distributes the sorted data in both the partitions such that each processor gets
 *        same number of elements as it originally had.
 *
 * @param sendbuf       Pointer to the buffer which contains elements in this partition.
 * @param current_sizes Current number of elements in all the processors.
 * @param recvbuf       Pointer to the buffer which will store the distributed elements.
 * @param final_sizes   Expected final number of elements in all the processors.
 * @param comm          The MPI communicator for all the processors in which data is to be collected.
 */
void collect_data(int* sendbuf, int* current_sizes, int* recvbuf, int* final_sizes, MPI_Comm comm);

#endif // PARALLEL_SORT_H
