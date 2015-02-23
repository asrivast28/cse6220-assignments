// Implement your solutions in this file
#include "findmotifs.h"
#include "hamming.h"
#include <mpi.h>

#include <numeric>
#include <stdexcept>

// Hi, from Cansu!

// special tag for signalling workers to exit
const int EXIT_TAG = 666;

// checks if the given flipped number is a solution

extern void explore_solutions(const unsigned int n, const unsigned int l,
                       const unsigned int d, const bits_t* input,
                       const unsigned int b, const bits_t number,
                       std::vector<bits_t>& result,
                       unsigned int currentidx, unsigned int currentd);

std::vector<bits_t> findmotifs_worker(const unsigned int n,
                       const unsigned int l,
                       const unsigned int d,
                       const bits_t* input,
                       const unsigned int startbitpos,
                       bits_t start_value)
{
    std::vector<bits_t> result;

    unsigned int ham = hamming(start_value, input[0]);

    for (unsigned int b = ham; b <= d; ++b) {
      explore_solutions(n, l, d, input, b, start_value, result, startbitpos, ham);
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

void receive_results(const int busy_count, int first_worker, std::vector<MPI_Request>& request,
                     std::vector<unsigned int>& result_size, std::vector<bits_t>& result,
                     MPI_Comm& comm, const int my_rank)
{
    int p = request.size();
    if (first_worker > p) {
      first_worker = 1;
      MPI_Waitall(busy_count, &request[first_worker], MPI_STATUSES_IGNORE);
    }
    else {
      if (p < (first_worker + busy_count)) {
        int wait_count = (p - first_worker);
        MPI_Waitall(wait_count, &request[first_worker], MPI_STATUSES_IGNORE);
        wait_count = busy_count - wait_count;
        MPI_Waitall(wait_count, &request[1], MPI_STATUSES_IGNORE);
      }
      else {
        MPI_Waitall(busy_count, &request[first_worker], MPI_STATUSES_IGNORE);
      }
    }
    std::vector<std::vector<bits_t> > this_result;
    int worker_id = first_worker;
    for (int wait_count = 0; wait_count < busy_count; ++wait_count) {
      unsigned int this_size = result_size[worker_id];
      if (this_size > 0) {
        unsigned int offset = result.size();
        result.resize(offset + this_size);
        MPI_Recv(&result[offset], this_size, MPI_UNSIGNED_LONG_LONG, worker_id, my_rank, comm, MPI_STATUS_IGNORE);
      }
      worker_id = (worker_id == (p - 1)) ? 1 : (worker_id + 1);
    }
}

void send_partial(bits_t partial, MPI_Comm& comm, const int p, const int my_rank,
                  std::vector<bits_t>& result, std::vector<unsigned int>& result_size,
                  int& busy_count, int& worker_id, std::vector<MPI_Request>& request)
{
    MPI_Send(&partial, 1, MPI_UNSIGNED_LONG_LONG, worker_id, my_rank, comm);
    int request_size = request.size();
    if (worker_id >= request_size) {
      request.resize(worker_id + 1);
    }
    MPI_Irecv(&result_size[worker_id], 1, MPI_UNSIGNED, worker_id, my_rank, comm, &request[worker_id]);
    ++busy_count;
    worker_id = (worker_id == (p - 1)) ? 1 : (worker_id + 1);
    if (busy_count == (p - 1)) {
      receive_results(1, worker_id, request, result_size, result, comm, my_rank);
      --busy_count;
    }
}

// Explores the solution space obtained by flipping b bits in the l bit number
void explore_master(const unsigned int l, const unsigned int b,
                       const bits_t number, MPI_Comm& comm,
                       const int p, const int my_rank,
                       std::vector<bits_t>& result, std::vector<unsigned int>& result_size,
                       int& busy_count, int& worker_id, std::vector<MPI_Request>& request,
                       unsigned int currentidx = 0, unsigned int currentd = 0)
{
    if (b > 0) {
      // We flip bits in the original number recursively at every level until we have flipped maximum number
      // of allowed bits in the number. We consider all remaining possibilities at this level in
      // the following for loop.
      bits_t base = 1;
      for (unsigned int idx = currentidx; idx <= l - (b - currentd); ++idx) {
        // flip the bit by XOR-ing with appropriate number
        bits_t flipped = number ^ (base << idx);
        if ((currentd + 1) < b) {
          explore_master(l, b, flipped, comm, p, my_rank, result, result_size, busy_count, worker_id, request, idx + 1, currentd + 1);
        }
        else {
          send_partial(flipped, comm, p, my_rank, result, result_size, busy_count, worker_id, request);
        }
      }
    }
    else {
      // There is only one possibility, 0, if b is 0.
      send_partial(number, comm, p, my_rank, result, result_size, busy_count, worker_id, request);
    }
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
    std::vector<MPI_Request> request;

    int worker_id = 1;
    int busy_count = 0;
    unsigned int max_depth = (till_depth < d) ? till_depth : d;
    for (unsigned int b = 0; b <= max_depth; ++b) {
      explore_master(till_depth, b, input[0], comm, p, my_rank, result, result_size, busy_count, worker_id, request);
    }

    if (busy_count > 0) {
      worker_id = (worker_id == (p - 1)) ? 1 : (worker_id + 1);
      receive_results(busy_count, worker_id, request, result_size, result, comm, my_rank);
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
