// Implement your solutions in this file
#include "findmotifs.h"
#include "hamming.h"
#include <mpi.h>

#include <stdexcept>

std::vector<bits_t> findmotifs_worker(const unsigned int n,
                       const unsigned int l,
                       const unsigned int d,
                       const bits_t* input,
                       const unsigned int startbitpos,
                       bits_t start_value)
{
    std::vector<bits_t> results;

    // TODO: implement your solution here

    return results;
}

void worker_main()
{
    MPI_Comm comm = MPI_COMM_WORLD;

    int p, my_rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    // Hardcoding master rank as per problem description.
    const int master_rank = 0;

    if (my_rank == master_rank) {
      // Master shouldn't have been in this function.
      // Something went wrong. Abort the mission!
      throw std::runtime_error("Master process detected in worker_main!");
    }

    // TODO:
    // 1.) receive input from master (including n, l, d, input, master-depth)
    unsigned int n, l, d, master_depth;
    MPI_Bcast(&n, 1, MPI_UNSIGNED, master_rank, comm);
    MPI_Bcast(&l, 1, MPI_UNSIGNED, master_rank, comm);
    MPI_Bcast(&d, 1, MPI_UNSIGNED, master_rank, comm);
    MPI_Bcast(&master_depth, 1, MPI_UNSIGNED, master_rank, comm);

    // 2.) while the master is sending work:
    //      a) receive subproblems
    //      b) locally solve (using findmotifs_worker(...))
    //      c) send results to master

    // 3.) you have to figure out a communication protocoll:
    //     - how does the master tell the workers that no more work will come?
    //       (i.e., when to break loop 2)
    // 4.) clean up: e.g. free allocated space
}



std::vector<bits_t> findmotifs_master(const unsigned int n,
                                      const unsigned int l,
                                      const unsigned int d,
                                      const bits_t* input,
                                      const unsigned int till_depth)
{
    std::vector<bits_t> results;


    // TODO: implement your solution here

    return results;
}

std::vector<bits_t> master_main(unsigned int n, unsigned int l, unsigned int d,
                                const bits_t* input, unsigned int master_depth)
{
    MPI_Comm comm = MPI_COMM_WORLD;

    int p, my_rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    if (my_rank != 0) {
      // Worker shouldn't have been in this function.
      // Something went wrong. Abort the mission!
      throw std::runtime_error("Worker process detected in master_main!");
    }

    // TODO
    // 1.) send input to all workers (including n, l, d, input, depth)
    MPI_Bcast(&n, 1, MPI_UNSIGNED, my_rank, comm);
    MPI_Bcast(&l, 1, MPI_UNSIGNED, my_rank, comm);
    MPI_Bcast(&d, 1, MPI_UNSIGNED, my_rank, comm);
    MPI_Bcast(&master_depth, 1, MPI_UNSIGNED, my_rank, comm);

    // 2.) solve problem till depth `master_depth` and then send subproblems
    //     to the workers and receive solutions in each communication
    //     Use your implementation of `findmotifs_master(...)` here.
    std::vector<bits_t> results;


    // 3.) receive last round of solutions
    // 4.) terminate (and let the workers know)

    return results;
}

