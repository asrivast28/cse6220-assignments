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
    //get the rank in the grid
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);

    //calculate the number of processors in each row and column in the grid
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    MPI_Comm col_comm;
    int col_rank;

    //get the column subcommunicator for the first column
    if(grid_rank % q == 0)
    {
        col_comm = col_subcomm(comm);
        MPI_Comm_rank(col_comm, &col_rank);
    }
    //if not in first column, return
    else
    {
        col_comm = col_subcomm(comm);
        return;
    }

    int *scounts = NULL, *displs = NULL;
    int local_size = block_decompose(n, q, col_rank);
    *local_vector = (double *)malloc(local_size * sizeof(double));

    //the first processor in first column (and in the grid) calculates the send counts and displacements to scatter
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

    //scatter the vector from the first processor, within the first column
    MPI_Scatterv(input_vector, &scounts[0], &displs[0], MPI_DOUBLE, *local_vector, local_size, MPI_DOUBLE, 0, col_comm);

    //free displs and scounts memory
    free(displs);
    free(scounts);
}

void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    //get the rank in the grid
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);

    //calculate the number of processors in each row and column in the grid
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    MPI_Comm col_comm;
    int col_rank;

    //get the column communicator for the first column
    if(grid_rank%q == 0)
    {
        col_comm = col_subcomm(comm);
        MPI_Comm_rank(col_comm, &col_rank);
    }
    //if not in first column, return
    else
    {
        col_comm = col_subcomm(comm);
        return;
    }


    int *scounts = NULL, *displs = NULL;
    int local_size = block_decompose(n, q, col_rank);
    
    //the first processor in first column (and in the grid) calculates the send counts and displacements to gather
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

    //gather the vector from the first column, to the first column
    MPI_Gatherv(local_vector, local_size, MPI_DOUBLE, output_vector, scounts, displs, MPI_DOUBLE, 0, col_comm);

    //free displs and scounts memory
    free(displs);
    free(scounts);
}

//Locally transpose the matrix.
void local_transpose(const int row, const int col, double* matrix)
{
    //copy the matrix in temp
    double* temp = (double *)malloc(row * col * sizeof(double));
    memcpy(temp, matrix, row * col * sizeof(double));
    //transpose temp and fill matrix
    for(int i = 0; i < row; ++i)
    {
        for(int j = 0; j < col; ++j)
        {
            matrix[j * row + i] = temp[i * col + j];
        }
    }
    
    //free the temp memory
    free(temp);
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    //get {0,0}'s rank in the grid
    int rank00, grid_rank;
    int coords[2] = {0, 0};
    MPI_Cart_rank(comm, coords, &rank00);

    //get the rank in the grid
    MPI_Comm_rank(comm, &grid_rank);
    
    //calculate the number of processors in each row and column in the grid
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    //get row subcommunicator
    MPI_Comm row_comm;
    row_comm = row_subcomm(comm);
    
    //get the rank in the row
    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    //get column communicator
    MPI_Comm col_comm;
    col_comm = col_subcomm(comm);

    //get the rank in the column
    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);

    //get row and column counts that the processor should have
    int row_count = block_decompose_by_dim(n, comm, 0);
    int col_count = block_decompose_by_dim(n, comm, 1);

    int local_size = row_count * col_count;
    *local_matrix = (double *)malloc(local_size * sizeof(double));

    //create sub_matrices for every processor in the first row, and distribute the matrix within these processors
    int *scounts = NULL, *displs = NULL;

    //the first processor in the grid) calculates the send counts and displacements to distribute
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

    //distribute the matrix within the column
    if(row_rank == 0)
    {
        int scatter_size = row_count * n;
        scatter_matrix = (double *)malloc(scatter_size * sizeof(double));
        
        MPI_Scatterv(input_matrix, scounts, displs, MPI_DOUBLE, scatter_matrix, scatter_size, MPI_DOUBLE, 0, col_comm);

        local_transpose(row_count, n, scatter_matrix);

        int offset = 0;

        if (displs == NULL)
            displs = (int*)malloc(q * sizeof(int));
        if (scounts == NULL)
            scounts = (int*)malloc(q * sizeof(int));

        for (int i = 0; i < q; i++)
        {
            displs[i] = offset;
            scounts[i] = block_decompose(n, q, i) * row_count;
            offset += scounts[i];
        }

    }


    //distribute the sub_matrices within their rows to finish the matrix distribution
    MPI_Scatterv(scatter_matrix, scounts, displs, MPI_DOUBLE, *local_matrix, local_size, MPI_DOUBLE, 0 , row_comm);

    //free the sub_matrix memory
    free(scatter_matrix);

    //transpose each local matrix in every processor
    local_transpose(col_count, row_count, *local_matrix);

    //free scounts and displs memory
    free(displs);
    free(scounts);

}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    //get the rank in the grid
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);
    
    //calculate the number of processors in each row and column in the grid
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    //get row and column subcommunicators
    MPI_Comm row_comm, col_comm;
    col_comm = col_subcomm(comm);
    row_comm = row_subcomm(comm);

    //get rank in the column and in the row
    int col_rank, row_rank;
    int coords[2];
    MPI_Cart_coords(comm, grid_rank, 2, coords);
    col_rank = coords[0];
    row_rank = coords[1];

    //send the local vector from the first processor in the row to the diagonal processor
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

    //broadcast the vector within the column
    int bcount = block_decompose(n, q, row_rank);
    MPI_Bcast(row_vector, bcount, MPI_DOUBLE, row_rank, col_comm);

}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    
    //get the rank in the grid
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);
    
    //calculate the number of processors in each row and column in the grid
    int p, q;
    MPI_Comm_size(comm, &p);
    q = (int)sqrt(p);

    //get row subcommunicator
    MPI_Comm row_comm;
    row_comm = row_subcomm(comm);

    //get the rank in the row
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
            local_y[i] += local_A[r]*local_vector[j];
            r++;
        }
    }

    //free the local_vector memory
    free(local_vector);

    //reduce the local_y's to column
    if(row_rank == 0)
        MPI_Reduce(MPI_IN_PLACE, local_y, row_count, MPI_DOUBLE, MPI_SUM, 0, row_comm);
    else
        MPI_Reduce(local_y, local_y, row_count, MPI_DOUBLE, MPI_SUM, 0, row_comm);

}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    //get the rank in the grid
    int grid_rank;
    MPI_Comm_rank(comm, &grid_rank);

    //get the row subcommunicator
    MPI_Comm row_comm;
    row_comm = row_subcomm(comm);

    //get the rank in the row
    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    //get the column subcommunicator
    MPI_Comm col_comm;
    col_comm = col_subcomm(comm);

    //get the rank in the column
    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);

    //get row and column counts that the processor should have
    int row_count = block_decompose_by_dim(n, comm, 0);
    int col_count = block_decompose_by_dim(n, comm, 1);

    // pointer to local diagonal matrix
    double *local_D = NULL;

    int local_size = row_count * col_count;
    double *local_R = (double *)malloc(local_size * sizeof(double));
    // assign local A matrix to local R matrix
    memcpy(local_R, local_A, local_size * sizeof(double));


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
            }
        }
        // the value of l2 norm is required in all the processors
        MPI_Allreduce(MPI_IN_PLACE, &l2_norm, 1, MPI_DOUBLE, MPI_SUM, comm);
        l2_norm = sqrt(l2_norm);

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

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

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
