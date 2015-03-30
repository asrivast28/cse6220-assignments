/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);

    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    MPI_Comm col_comm;
    int col_rank;

   if(grid_rank % q == 0)
   {
       col_comm = col_subcomm(comm);
       MPI_Comm_rank(col_comm, &col_rank);
   }

    //not the first column
    //compute the column subcommunicator then return
    else
    {
        col_comm = col_subcomm(comm);
        return;
    }


    int *scounts = NULL, *displs = NULL;


    int local_size = block_decompose(n, q, col_rank);
    *local_vector = (double *)malloc(local_size * sizeof(double));


    if(col_rank == 0)
    {
        int offset = 0;

        displs = (int *)malloc(q * sizeof(int));
        scounts = (int *)malloc(q * sizeof(int));

        for (int i = 0; i < q; i++)
        {
            displs[i] = offset;
            scounts[i] = block_decompose(n, q, i);
            offset += scounts[i];
        }

    }

    MPI_Scatterv(input_vector, &scounts[0], &displs[0], MPI_DOUBLE, *local_vector, local_size, MPI_DOUBLE, 0, col_comm);

    free(displs);
    free(scounts);

    //for(int i = 0; i < block_decompose(n, q, col_rank); i++)
        //std::cout <<"processor: " << grid_rank << " local_x["<< i << "] = " << local_vector[i]<< std::endl;


//    =              n: 4processor: 0 ccol rank: 0 local_vector[0] = 6
//    =              n: 4processor: 0 ccol rank: 0 local_vector[1] = 25
//    =              n: 4processor: 3 ccol rank: 1 local_vector[0] = -11
//    =              n: 4processor: 6 ccol rank: 2 local_vector[0] = 15
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);

    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    MPI_Comm col_comm;
    int col_rank;

    if(grid_rank%q == 0)
    {
        col_comm = col_subcomm(comm);
        MPI_Comm_rank(col_comm, &col_rank);
    }

    //not the first column
    //compute the column subcommunicator then return
    else
    {
        col_comm = col_subcomm(comm);
        return;
    }


    int *scounts = NULL, *displs = NULL;

    int local_size = block_decompose(n, q, col_rank);
    //local_vector = (double *)malloc(local_size*sizeof(double));


    if(col_rank == 0)
    {
        int offset = 0;

        displs = (int *)malloc(q * sizeof(int));
        scounts = (int *)malloc(q * sizeof(int));

        for (int i = 0; i < q; i++)
        {
            displs[i] = offset;
            scounts[i] = block_decompose(n, q, i);
            offset += scounts[i];
        }
    }

    MPI_Gatherv(local_vector, local_size, MPI_DOUBLE, output_vector, scounts, displs, MPI_DOUBLE, 0, col_comm);

    free(displs);
    free(scounts);

}

void local_transpose(const int row, const int col, double* matrix)
{
  double* temp = (double *)malloc(row * col * sizeof(double));
  memcpy(temp, matrix, row * col * sizeof(double));
  for(int i = 0; i < row; ++i)
  {
    for(int j = 0; j < col; ++j)
    {
      //std::cout << (j * row + i) << ":" << (i * col + j) << std::endl;
      matrix[j * row + i] = temp[i * col + j];
    }
  }
  free(temp);
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    int rank00, grid_rank;
    int coords[2] = {0, 0};
    MPI_Cart_rank(comm, coords, &rank00);

    MPI_Comm_rank(comm, &grid_rank);
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    MPI_Comm row_comm;
    row_comm = row_subcomm(comm);

    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    MPI_Comm col_comm;
    col_comm = col_subcomm(comm);

    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);

    int row_count = block_decompose_by_dim(n, comm, 0);
    int col_count = block_decompose_by_dim(n, comm, 1);


    int local_size = row_count * col_count;
    *local_matrix = (double *)malloc(local_size * sizeof(double));

    //create sub_matrices for every processor in the first row, and distribute the matrix within these processors.
    int *scounts = NULL, *displs = NULL;

    if(grid_rank == rank00)
    {
        int offset = 0;

        displs = (int *)malloc(q * sizeof(int));
        scounts = (int *)malloc(q * sizeof(int));

        for (int i = 0; i < q; ++i)
        {
            displs[i] = offset;
            scounts[i] = block_decompose(n, q, i) * n;
            offset += scounts[i];
        }
    }

    double *scatter_matrix = NULL;

    if(row_rank == 0)
    {
        int scatter_size = row_count * n;
        scatter_matrix = (double *)malloc(scatter_size * sizeof(double));
        MPI_Scatterv(input_matrix, scounts, displs, MPI_DOUBLE, scatter_matrix, scatter_size, MPI_DOUBLE, 0, col_comm);
        //std::cout << col_rank << ":" << scatter_matrix[0] << "," << scatter_matrix[1] << "," << scatter_matrix[2] << std::endl;


        //copy 1 dimensional sub_matrix to 2d sub_matrix as rows and columns for distribution

        //std::cout << col_rank << std::endl;
        //for (int i = 0; i < scatter_size; ++i)
          //std::cout << i << "=" << scatter_matrix[i] << " ";
        //std::cout << std::endl;

        local_transpose(row_count, n, scatter_matrix);

        //std::cout << col_rank << std::endl;
        //for (int i = 0; i < scatter_size; ++i)
          //std::cout << i << "=" << scatter_matrix[i] << " ";
        //std::cout << std::endl;


        int offset = 0;

        if (displs == NULL)
            displs = (int*)malloc(q * sizeof(int));
        if (scounts == NULL)
            scounts = (int*)malloc(q * sizeof(int));

        for (int i = 0; i < q; i++)
        {
            displs[i] = offset;
            //scounts[i] = block_decompose(n, q, i);
            scounts[i] = block_decompose(n, q, i) * row_count;
            offset += scounts[i];
        }

    }


    //distribute the sub_matrices within their columns to finish the matrix distribution.

    MPI_Scatterv(scatter_matrix, scounts, displs, MPI_DOUBLE, *local_matrix, local_size, MPI_DOUBLE, 0 , row_comm);
    free(scatter_matrix);

    //std::cout << col_rank << "," << row_rank << ": ";
    //for (int i = 0; i < local_size; ++i)
      //std::cout << i << "=" << (*local_matrix)[i] << " ";
    //std::cout << std::endl;

    local_transpose(col_count, row_count, *local_matrix);
    //std::cout << col_rank << "," << row_rank << ": ";
    //for (int i = 0; i < local_size; ++i)
      //std::cout << i << "=" << (*local_matrix)[i] << " ";
    //std::cout << std::endl;

    free(displs);
    free(scounts);

}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    MPI_Comm row_comm, col_comm;
    col_comm = col_subcomm(comm);
    row_comm = row_subcomm(comm);

    int col_rank, row_rank;
    int coords[2];
    MPI_Cart_coords(comm, grid_rank, 2, coords);
    col_rank = coords[0];
    row_rank = coords[1];

//    *  To accomplish this, first, each proccessor (i,0) sends it's local vector
//    *  to the diagonal processor (i,i). Then the diagonal processor (i,i)
//    *  broadcasts the message among it's column using a column sub-communicator.

    if(grid_rank == 0)
    {
        int count = block_decompose(n, q, grid_rank);
        memcpy (row_vector, col_vector, count*sizeof(double));
    }
    else if(row_rank == 0) // sending processor
    {
        int scount = block_decompose(n, q, col_rank);
        MPI_Send(col_vector, scount, MPI_DOUBLE, col_rank, 0, row_comm);
    }
    else if(col_rank == row_rank)//receiving processor
    {
        int rcount = block_decompose(n, q, col_rank);
        MPI_Recv(row_vector, rcount, MPI_DOUBLE, 0, 0, row_comm, MPI_STATUS_IGNORE);
    }

    int bcount = block_decompose(n, q, row_rank);
    MPI_Bcast(row_vector, bcount, MPI_DOUBLE, row_rank, col_comm);

}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    MPI_Comm row_comm;
    row_comm = row_subcomm(comm);

    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    int scount = block_decompose(n, q, row_rank);
    double *local_vector = (double *)malloc(scount*sizeof(double));

    //transpose the vector in local_vectors
    transpose_bcast_vector(n, local_x, local_vector, comm);


    //locally multiply local_A with local_vector into local_y
    int row_count = block_decompose_by_dim(n, comm, 0);

    int r = 0;
    for(int i = 0; i < row_count; i++)
    {
        local_y[i] = 0;
        for(int j = 0; j < scount; j++)
        {

//            std::cout << "  rank: " << grid_rank << " local_A=[ " <<r << "]: "<< local_A[r] <<" c and local_vector[" << j << "] :" << local_vector[j] << std::endl;
            local_y[i] += local_A[r]*local_vector[j];
            r++;

        }

//            std::cout << "      rank: " << grid_rank << "   y[" << i << "] = " << local_y[i] << std::endl;
    }

    free(local_vector);


    //reduce the local_y's to column
    if(row_rank == 0)
        MPI_Reduce(MPI_IN_PLACE, local_y, row_count, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    else
        MPI_Reduce(local_y, local_y, row_count, MPI_DOUBLE, MPI_SUM, 0, row_comm);


//    if(row_rank == 0)
//    {
//        for(int a=0; a<row_count; a++)
//            std::cout << "      rank: " << grid_rank << "scount = " << scount<< "   y[" << a << "] = " << local_y[a] << std::endl;
//
//    }

}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);

    MPI_Comm row_comm;
    row_comm = row_subcomm(comm);

    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    MPI_Comm col_comm;
    col_comm = col_subcomm(comm);

    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);

    int row_count = block_decompose_by_dim(n, comm, 0);
    int col_count = block_decompose_by_dim(n, comm, 1);

    // pointer to local diagonal matrix
    double *local_D = NULL;

    int local_size = row_count * col_count;
    double *local_R = (double *)malloc(local_size * sizeof(double));
    // assign local A matrix to local R matrix
    memcpy(local_R, local_A, local_size * sizeof(double));

    //std::cout << row_rank << "," << col_rank << ":" << local_A[0] << std::endl;
    //std::cout.flush();

    // check if the current processor is a diagonal processor or the first column
    if ((row_rank == col_rank) || (row_rank == 0))
    {
        // assign memory for diagonal elements now
        local_D = (double *)malloc(row_count * sizeof(double));
        if (row_rank == col_rank)
        {
            // fill diagonal elements and set the corresponding elements in R to 0.0
            for (int i = 0; i < row_count; ++i)
            {
                local_D[i] = local_A[i * row_count + i];
                local_R[i * row_count + i] = 0.0;
            }
        }

        if (row_rank != 0)
        {
            // send all local D from diagonal processors to the first column of processors
            MPI_Send(local_D, row_count, MPI_DOUBLE, 0, 0, row_comm);
            // now free the memory
            free(local_D);
            local_D = NULL;
        }
        else if (col_rank != 0)
        {
            MPI_Recv(local_D, row_count, MPI_DOUBLE, col_rank, 0, row_comm, MPI_STATUS_IGNORE);
        }
    }

    //if (row_rank == 0)
        //for(int i = 0; i < row_count; i++)
            //std::cout << "          RANK: " << grid_rank << " local_D[" << i << "] :"  <<local_D[i] << " local_R[" << i << "] : " << local_R[i] << " local_A[" << i << "] : " << local_A[i] <<std::endl;

    //init x to 0
    if(row_rank == 0)
    {
        for(int i = 0; i < row_count; i++)
        {
            local_x[i] = 0;
        }
    }

    //Jacobi Method

    // temporary buffer
    double temp[row_count];
    for (int iter = 0; iter < max_iter; ++iter)
    {
        // calculate A*x
        distributed_matrix_vector_mult(n, local_A, local_x, temp, comm);

        // l2 norm calculations
        double l2_norm = 0.0;
        // only first column participates in the calculations
        if (row_rank == 0)
        {
            for (int i = 0; i < row_count; ++i)
            {
                l2_norm += pow(temp[i] - local_b[i], 2.0);
                //std::cout << iter << ":" << local_x[i] << "," << temp[i] << std::endl;
                //std::cout << local_A[0] << std::endl;
                //std::cout.flush();
                //std::cout << col_rank << ":" << temp[i] << "," << local_b[i] << std::endl;
            }
        }
        // the value of l2 norm is required in all the processors
        MPI_Allreduce(MPI_IN_PLACE, &l2_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
        l2_norm = sqrt(l2_norm);
        //if (grid_rank == 0)
        //{
          //std::cout << "parallel:" << l2_norm << std::endl;
          //std::cout.flush();
        //}
        // check the termination condition
        if ((l2_norm - l2_termination) < DOUBLE_EPSILON)
        {
            break;
        }

        // now update x
        distributed_matrix_vector_mult(n, local_R, local_x, temp, comm);
        // update x in the first column
        if (row_rank == 0)
        {
            for (int i = 0; i < row_count; ++i)
            {
                local_x[i] = (local_b[i] - temp[i]) / local_D[i];
            }
        }
    }

    // free locally allocated memory
    free(local_D);
    free(local_R);
}




// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{

    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);


//    int i;
//    if(grid_rank % q == 0)
//        for(i = 0; i < block_decompose(n, q, grid_rank/q); i++)
//            std::cout <<"processor: " << grid_rank << " local_x["<< i << "] = " << local_x[i] << std::endl;
//
//    for(i = 0; i < block_decompose_by_dim(n, comm, 0) * block_decompose_by_dim(n, comm, 1); i++)
//        std::cout <<"processor: " << grid_rank << " local_ARRAY["<< i << "] = " << local_A[i] << std::endl;

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);


//    int rank_root = (grid_rank%q);
//    int scount = block_decompose(n, q, rank_root);
//    double* row_vector = new double[scount];
//    transpose_bcast_vector(n, local_x, row_vector, comm);
//
//    for(int i = 0; i < scount; i++)
//        std::cout <<"       processor: " << grid_rank << " size: " << scount <<  " y["<< i << "] = " << row_vector[i] << std::endl;

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);

    free(local_A);
    free(local_x);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{

    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
