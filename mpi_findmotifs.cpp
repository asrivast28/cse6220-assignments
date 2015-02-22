// Implement your solutions in this file
#include "findmotifs.h"
#include "hamming.h"
#include <mpi.h>

#include <numeric>
#include <stdexcept>

// Hi, from Cansu!

// special tag for signalling workers to exit
const int EXIT_TAG = 666;

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
    //bool pending_send = false;
    while (true) {
      bits_t start_value;
      MPI_Status status;
      MPI_Recv(&start_value, 1, MPI_UNSIGNED_LONG_LONG, master_rank, MPI_ANY_TAG, comm, &status);
      if (status.MPI_TAG != master_rank) {
        return;
      }

      std::vector<bits_t> result(findmotifs_worker(n, l, d, input, master_depth, start_value));
      unsigned int size = result.size();

      //std::cout << my_rank << ": " << size << std::endl;
      //std::cout << "solution size is " << size << " in " << my_rank << std::endl;
      MPI_Send(&size, 1, MPI_UNSIGNED, master_rank, master_rank, comm);
      if (size > 0) {
        //for (unsigned int i = 0; i < size; ++i) {
          //std::cout << result[i] << " ";
        //}
        //std::cout << std::endl;
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


void receive_results(const int busy_count, std::vector<MPI_Request>& request,
                     std::vector<unsigned int>& result_size, std::vector<bits_t>& result,
                     MPI_Comm& comm, const int my_rank)
{
    MPI_Waitall(busy_count, &request[0], MPI_STATUSES_IGNORE);
    std::partial_sum(result_size.begin(), result_size.end(), result_size.begin());
    size_t prev_size = result.size();
    result.resize(prev_size + result_size[busy_count]);
    unsigned int num_recv = 0;
    for (int id = 1; id <= busy_count; ++id) {
      unsigned int this_size = result_size[id] - result_size[id-1];
      if (this_size > 0) {
        unsigned int offset = prev_size + result_size[id-1];
        //std::cout << "receiving " << this_size << " from " << id << " with offset " << offset << std::endl;
        //std::cout << "result size is " << result.size() << std::endl;
        MPI_Irecv(&result[offset], this_size, MPI_UNSIGNED_LONG_LONG, id, my_rank, comm, &request[num_recv++]);
      }
    }
    MPI_Waitall(num_recv, &request[0], MPI_STATUSES_IGNORE);
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


    std::vector<unsigned int> result_size(p, 0);
    std::vector<MPI_Request> request(p);

    int worker_id = 1;
    int busy_count = 0;
    for (unsigned int i = 0; i <= till_depth; ++i) {
      std::vector<bits_t> combinations;
      get_combinations(till_depth, i, input[0], combinations);

      for (unsigned int c = 0; c < combinations.size(); ++c) {
        MPI_Send(&combinations[c], 1, MPI_UNSIGNED_LONG_LONG, worker_id, my_rank, comm);
        MPI_Irecv(&result_size[worker_id], 1, MPI_UNSIGNED, worker_id, my_rank, comm, &request[busy_count++]);
        if (busy_count == (p - 1)) {
          receive_results(busy_count, request, result_size, result, comm, my_rank);
          busy_count = 0;
        }
        worker_id = (worker_id == (p - 1)) ? 1 : (worker_id + 1);
      }
    }

    if (busy_count > 0) {
      receive_results(busy_count, request, result_size, result, comm, my_rank);
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
      MPI_Send(&num, 1, MPI_UNSIGNED_LONG_LONG, i, EXIT_TAG, comm);
    }

    return result;
}
