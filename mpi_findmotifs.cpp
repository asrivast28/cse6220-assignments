// Implement your solutions in this file
#include "findmotifs.h"
#include "hamming.h"
#include <mpi.h>

#include <stdexcept>

extern void check_solution(unsigned int n, unsigned int d,
                          const bits_t* input, bits_t flipped,
                          std::vector<bits_t>& result, std::vector<bits_t>& candidate);

// Returns all l bit candidate of number with d bits flipped in the binary representation.
void get_combinations(const unsigned int l, const unsigned int d,
                      const bits_t number, std::vector<bits_t>& combinations,
                      unsigned int currentidx = 0, unsigned int currentd = 0)
{
    if (d > 0) {
      // We flip bits in the original number recursively at every level until we have flipped maximum number
      // of allowed bits in the number. We consider all remaining possibilities at this level in
      // the following for loop.
      bits_t base = 1;
      for (unsigned int idx = currentidx; idx <= l - (d - currentd); ++idx) {
        // flip the bit by XOR-ing with appropriate number
        bits_t flipped = number ^ (base << idx);
        if ((currentd + 1) < d) {
          get_combinations(l, d, flipped, combinations, idx + 1, currentd + 1);
        }
        else {
          combinations.push_back(flipped);
        }
      }
    }
    else {
      // There is only one possibility, 0, if d is 0.
      combinations.push_back(number);
    }
}

std::vector<bits_t> findmotifs_worker(const unsigned int n,
                       const unsigned int l,
                       const unsigned int d,
                       const bits_t* input,
                       const unsigned int startbitpos,
                       bits_t start_value)
{
    std::vector<bits_t> result;

    std::vector<bits_t> candidate;
    check_solution(n, d, input, start_value, result, candidate);

    bits_t flipper = 1;
    flipper = flipper << startbitpos;
    for (unsigned int i = startbitpos; i < l; ++i, flipper *= 2) {
      uint64_t currentSize = candidate.size();
      for (uint64_t j = 0; j < currentSize; ++j) {
        bits_t flipped = candidate[j] ^ flipper;
        // check if the solution is a result or a potential result
        check_solution(n, d, input, flipped, result, candidate);
      }
    }

    return result;
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

    // 1.) receive input from master (including n, l, d, input, master-depth)
    unsigned int n, l, d, master_depth;
    MPI_Bcast(&n, 1, MPI_UNSIGNED, master_rank, comm);
    MPI_Bcast(&l, 1, MPI_UNSIGNED, master_rank, comm);
    MPI_Bcast(&d, 1, MPI_UNSIGNED, master_rank, comm);

    std::vector<bits_t> inputdata(n);
    bits_t* input = &inputdata[0];
    MPI_Bcast(input, n, MPI_UNSIGNED_LONG_LONG, master_rank, comm);

    MPI_Bcast(&master_depth, 1, MPI_UNSIGNED, master_rank, comm);

    // 2.) while the master is sending work:
    //      a) receive subproblems
    //      b) locally solve (using findmotifs_worker(...))
    //      c) send results to master
    //
    while (true) {
      bits_t start_value;
      MPI_Status status;
      MPI_Recv(&start_value, 1, MPI_UNSIGNED_LONG_LONG, master_rank, MPI_ANY_TAG, comm, &status);
      if (status.MPI_TAG != master_rank) {
        return;
      }

      std::vector<bits_t> result(findmotifs_worker(n, l, d, input, master_depth, start_value));
      unsigned int size = result.size();

      MPI_Send(&size, 1, MPI_UNSIGNED, master_rank, master_rank, comm);
      if (size > 0) {
        MPI_Request request;
        MPI_Isend(&result[0], size, MPI_UNSIGNED_LONG_LONG, master_rank, master_rank, comm, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
      }
    }

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
    std::vector<bits_t> result;

    MPI_Comm comm = MPI_COMM_WORLD;
    int p, my_rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);


    std::vector<unsigned int> resultSize(p, 0);
    std::vector<bits_t> combinations(1, input[0]);
    bits_t flipper = 1;
    for (unsigned int i = 0; i < till_depth; ++i, flipper *= 2) {
      uint64_t currentSize = combinations.size();
      for (uint64_t j = 0; j < currentSize; ++j) {
        bits_t flipped = combinations[j] ^ flipper;
        if (hamming(flipped, input[0]) <= d) {
          // Farm this subproblem out.
          MPI_Send(&flipped, 1, MPI_UNSIGNED_LONG_LONG, 1, my_rank, comm);

          unsigned int size = 0;
          MPI_Status status;
          MPI_Recv(&resultSize[0], 1, MPI_UNSIGNED, 1, my_rank, comm, &status);
          if (size > 0) {
            unsigned int prevSize = result.size();
            result.resize(prevSize + size);
            MPI_Recv(&result[prevSize], size, MPI_UNSIGNED_LONG_LONG, 1, my_rank, comm, &status);
          }
          combinations.push_back(flipped);
        }
      }
    }

    return result;
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

    // 1.) send input to all workers (including n, l, d, input, depth)
    MPI_Bcast(&n, 1, MPI_UNSIGNED, my_rank, comm);
    MPI_Bcast(&l, 1, MPI_UNSIGNED, my_rank, comm);
    MPI_Bcast(&d, 1, MPI_UNSIGNED, my_rank, comm);

    MPI_Bcast((void*)input, n, MPI_UNSIGNED_LONG_LONG, my_rank, comm);

    MPI_Bcast(&master_depth, 1, MPI_UNSIGNED, my_rank, comm);

    // 2.) solve problem till depth `master_depth` and then send subproblems
    //     to the workers and receive solutions in each communication
    //     Use your implementation of `findmotifs_master(...)` here.
    std::vector<bits_t> result(findmotifs_master(n, l, d, input, master_depth));



    // 3.) receive last round of solutions
    // 4.) terminate (and let the workers know)
    for (int i = 1; i < p; ++i) {
      bits_t num = 0;
      MPI_Send(&num, 1, MPI_UNSIGNED_LONG_LONG, i, 666, comm);
    }

    return result;
}
