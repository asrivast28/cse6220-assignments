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

/*
 * TODO: Implement your solutions here
 */


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    // TODO
    int rank00, grid_rank;
    int coords00[2] = {0, 0};
    MPI_Cart_rank(comm, coords00, &rank00);
  
    
    MPI_Comm_rank(comm, &grid_rank);
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);
  
    MPI_Comm col_comm;
    int col_rank;
    int coords[2];
    
   if(grid_rank % q == 0)
   {
       col_comm = col_subcomm(comm, grid_rank, q);
       MPI_Cart_coords(comm, grid_rank, 2, coords);
       col_rank = coords[0];
   }

    //not the first column
    //compute the column subcommunicator then return
    else
    {
        col_comm = col_subcomm(comm, grid_rank, q);
        return;
    }
    

    int scounts[q], displs[q];
    
    
    int local_size = block_decompose(n, q, col_rank);
    *local_vector = (double *)malloc(local_size*sizeof(double));

    
    if(grid_rank == rank00)
    {
        int offset = 0;
        for (int i = 0; i < q; i++)
        {
            displs[i] = offset;
            scounts[i] = block_decompose(n, q, i);
            offset += scounts[i];
        }
        
    }
    
    //wait for all the processors in first column finish.
    MPI_Barrier(col_comm);

    MPI_Scatterv(input_vector, &scounts[0], &displs[0], MPI_DOUBLE, *local_vector, local_size, MPI_DOUBLE, rank00, col_comm);

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
    // TODO
    int rank00, grid_rank;
    int coords00[2] = {0, 0};
    MPI_Cart_rank(comm, coords00, &rank00);
    
    
    MPI_Comm_rank(comm, &grid_rank);
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);
    
    MPI_Comm col_comm, col_comm1;
    int col_rank;
    int coords[2];
    
    if(grid_rank%q == 0)
    {
        col_comm = col_subcomm(comm, grid_rank, q);
        MPI_Cart_coords(comm, grid_rank, 2, coords);
        col_rank = coords[0];
    }
    
    //not the first column
    //compute the column subcommunicator then return
    else
    {
        col_comm1 = col_subcomm(comm, grid_rank, q);
        return;
    }
    
    
    int scounts[q], displs[q];
    
    int local_size = block_decompose(n, q, col_rank);
    //local_vector = (double *)malloc(local_size*sizeof(double));
    
    
    if(grid_rank == rank00)
    {
        int offset = 0;
        for (int i = 0; i < q; i++)
        {
            displs[i] = offset;
            scounts[i] = block_decompose(n, q, i);
            offset += scounts[i];
        }
    }
    
    //wait for all the processors in first column finish.
    MPI_Barrier(col_comm);
    
    
    MPI_Gatherv(local_vector, local_size, MPI_DOUBLE, output_vector, scounts, displs, MPI_DOUBLE, rank00, col_comm);
    


}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // TODO
    int rank00, grid_rank;
    int coords[2] = {0, 0};
    MPI_Cart_rank(comm, coords, &rank00);
    
    MPI_Comm_rank(comm, &grid_rank);
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);
    
    MPI_Comm row_comm;
    row_comm = row_subcomm(comm, grid_rank, q);
    
    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    
    MPI_Comm col_comm;
    col_comm = col_subcomm(comm, grid_rank, q);
    
    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);
   
    int row_count = block_decompose_by_dim(n, comm, 0);
    int col_count = block_decompose_by_dim(n, comm, 1);
    
    
    int local_size = row_count * col_count;
    *local_matrix = (double *)malloc(local_size * sizeof(double));
   
    //create sub_matrices for every processor in the first row, and distribute the matrix within these processors.
    int scounts[q], displs[q];
    int offset = 0;
    
    if(grid_rank == rank00)
    {
        
        for (int i = 0; i < q; i++)
        {
            displs[i] = offset;
            scounts[i] = block_decompose(n, q, i) * n;
            offset += scounts[i];
        }
    }

    double * sub_matrix1d;
    double ** sub_matrix2d;
    
    int rank_root;
    
    rank_root = grid_rank%q;

    MPI_Barrier(comm);
    
    
    
    
    if(col_rank == 0)
    {
        sub_matrix1d = (double *)malloc((n * col_count) * sizeof(double));
        MPI_Scatterv(input_matrix, scounts, displs, MPI_DOUBLE, sub_matrix1d, scounts[row_rank], MPI_DOUBLE, rank00, row_comm);
       
        
        //copy 1 dimensional sub_matrix to 2d sub_matrix as rows and columns for distribution
        
        sub_matrix2d = (double **)malloc(col_count * sizeof(double*));
        for(int i = 0; i < col_count; i++)
            sub_matrix2d[i] = (double *)malloc(n*sizeof(double));
        
        
        
        int index = 0;
        for(int i = 0; i < col_count; i++)
            for(int j = 0; j < n; j++)
            {
                sub_matrix2d[i][j] = sub_matrix1d[index];
                index++;
            }
        
        //now rewrite sub_matrix1d with sub_matrix2d (transpose)
        index = 0;
        for(int i = 0; i < n; i++)
            for(int j = 0; j < col_count; j++)
            {
                sub_matrix1d[index] = sub_matrix2d[j][i];
                index++;
            }

    }
    
    
    //distribute the sub_matrices within their columns to finish the matrix distribution.
    
        offset = 0;

    if(col_rank == 0)
    {
            for (int i = 0; i < q; i++)
            {
                displs[i] = offset;
                scounts[i] = block_decompose(n, q, i) * col_count;
                offset += scounts[i];
            }
   
        
    }
    
    
    MPI_Barrier(col_comm);
    
    
    MPI_Scatterv(sub_matrix1d, scounts, displs, MPI_DOUBLE, *local_matrix, scounts[col_rank], MPI_DOUBLE, 0 , col_comm);
    
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    // TODO
    
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);
    
    MPI_Comm row_comm, col_comm;
    col_comm = col_subcomm(comm, grid_rank, q);
    row_comm = row_subcomm(comm, grid_rank, q);
    
    int col_rank, row_rank;
    int coords[2];
    MPI_Cart_coords(comm, grid_rank, 2, coords);
    col_rank = coords[0];
    row_rank = coords[1];
    MPI_Status status;

//    *  To accomplish this, first, each proccessor (i,0) sends it's local vector
//    *  to the diagonal processor (i,i). Then the diagonal processor (i,i)
//    *  broadcasts the message among it's column using a column sub-communicator.
    
    int rank_root = (grid_rank%q);
    int scount = block_decompose(n, q, rank_root);

    if(grid_rank == 0)
        memcpy ( row_vector, col_vector, scount*sizeof(double) );
    
    else if(row_rank == 0) // sending processor
    {
        MPI_Send(col_vector, scount, MPI_DOUBLE, col_rank, 0, row_comm);
    }
    else if(col_rank == row_rank)//receiving processor
    {
        MPI_Recv(row_vector, scount, MPI_DOUBLE, 0, 0, row_comm, &status);
    }

    MPI_Bcast(row_vector, scount, MPI_DOUBLE, row_rank, col_comm);


    
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // TODO
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // TODO
}




// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    int p, q, row_rank, col_rank, grid_rank;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);
    MPI_Comm_rank(comm,&grid_rank);


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
//    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);


    int rank_root = (grid_rank%q);
    int scount = block_decompose(n, q, rank_root);
    double* row_vector = new double[scount];
    transpose_bcast_vector(n, local_x, row_vector, comm);
    
    for(int i = 0; i < scount; i++)
        std::cout <<"       processor: " << grid_rank << " size: " << scount <<  " y["<< i << "] = " << row_vector[i] << std::endl;
    
    // gather results back to rank 0
//    gather_vector(n, local_y, y, comm);
    
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
