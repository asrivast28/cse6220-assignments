// Implement your solutions in this file
#include "findmotifs.h"
#include "hamming.h"
#include <mpi.h>

#include <limits>
#include <stdexcept>

// Hi, from Cansu!

// special tag for signalling workers to exit
const int PARTIAL_TAG = 111;
const int SIZE_TAG = 222;
const int RESULT_TAG = 333;
const int EXIT_TAG = 666;

// checks if the given flipped number is a solution

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
    std::vector<std::vector<bits_t> > result(2);
    std::vector<unsigned int> result_size(2, std::numeric_limits<unsigned int>::max());
    std::vector<bits_t> start(2);
    std::vector<MPI_Request> recv_request(2);
    std::vector<std::vector<MPI_Request> > send_request(2, std::vector<MPI_Request>(2));
    unsigned int iter = 0;

    MPI_Irecv(&start[iter], 1, MPI_UNSIGNED_LONG_LONG, master_rank, MPI_ANY_TAG, comm, &recv_request[iter]);
    while (true) {
      if (result_size[iter] < std::numeric_limits<unsigned int>::max()) {
        int request_size = (result_size[iter] == 0) ? 1 : 2;
        MPI_Waitall(request_size, &send_request[iter][0], MPI_STATUSES_IGNORE);
      }

      MPI_Status status;
      MPI_Wait(&recv_request[iter], &status);

      if (status.MPI_TAG == EXIT_TAG) {
        iter = (iter + 1) % 2;
        if (result_size[iter] < std::numeric_limits<unsigned int>::max()) {
          int request_size = (result_size[iter] == 0) ? 1 : 2;
          MPI_Waitall(request_size, &send_request[iter][0], MPI_STATUSES_IGNORE);
        }
        return;
      }

      iter = (iter + 1) % 2;
      MPI_Irecv(&start[iter], 1, MPI_UNSIGNED_LONG_LONG, master_rank, MPI_ANY_TAG, comm, &recv_request[iter]);
      iter = (iter + 1) % 2;

      result[iter] = findmotifs_worker(n, l, d, input, master_depth, start[iter]);
      result_size[iter] = result[iter].size();

      MPI_Isend(&result_size[iter], 1, MPI_UNSIGNED, master_rank, SIZE_TAG, comm, &(send_request[iter][0]));
      if (result_size[iter] > 0) {
        MPI_Isend(&result[iter][0], result_size[iter], MPI_UNSIGNED_LONG_LONG, master_rank, RESULT_TAG, comm, &(send_request[iter][1]));
      }

      iter = (iter + 1) % 2;
    }

    // 3.) you have to figure out a communication protocoll:
    //     - how does the master tell the workers that no more work will come?
    //       (i.e., when to break loop 2)
    // 4.) clean up: e.g. free allocated space
}

void receive_results(const int busy_count, int first_worker, std::vector<MPI_Request>& request,
                     std::vector<unsigned int>& result_size, std::vector<bits_t>& result,
                     MPI_Comm& comm, const int p)
{
    int request_size = request.size();
    if (first_worker > request_size) {
      first_worker = 1;
      MPI_Waitall(busy_count, &request[first_worker], MPI_STATUSES_IGNORE);
    }
    else {
      if (request_size < (first_worker + busy_count)) {
        int wait_count = (request_size - first_worker);
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
        int proc_id = (worker_id >= p) ? (worker_id + 1 - p) : worker_id;
        MPI_Recv(&result[offset], this_size, MPI_UNSIGNED_LONG_LONG, proc_id, RESULT_TAG, comm, MPI_STATUS_IGNORE);
      }
      worker_id = (worker_id == (2 * p - 2)) ? 1 : (worker_id + 1);
    }
}

void send_partial(bits_t partial, MPI_Comm& comm, const int p,
                  std::vector<bits_t>& result, std::vector<unsigned int>& result_size,
                  int& busy_count, int& worker_id, std::vector<MPI_Request>& request)
{
    int proc_id = (worker_id >= p) ? (worker_id + 1 - p) : worker_id;
    MPI_Send(&partial, 1, MPI_UNSIGNED_LONG_LONG, proc_id, PARTIAL_TAG, comm);
    int request_size = request.size();
    if (worker_id >= request_size) {
      request.resize(worker_id + 1);
    }
    MPI_Irecv(&result_size[worker_id], 1, MPI_UNSIGNED, proc_id, SIZE_TAG, comm, &request[worker_id]);
    ++busy_count;
    worker_id = (worker_id == (2 * p - 2)) ? 1 : (worker_id + 1);
    if (busy_count == (2 * p - 2)) {
      receive_results(1, worker_id, request, result_size, result, comm, p);
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
          send_partial(flipped, comm, p, result, result_size, busy_count, worker_id, request);
        }
      }
    }
    else {
      // There is only one possibility, 0, if b is 0.
      send_partial(number, comm, p, result, result_size, busy_count, worker_id, request);
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


    std::vector<unsigned int> result_size(2 * p - 1, 0);
    std::vector<MPI_Request> request(1);

    int worker_id = 1;
    int busy_count = 0;
    unsigned int max_depth = (till_depth < d) ? till_depth : d;
    for (unsigned int b = 0; b <= max_depth; ++b) {
      explore_master(till_depth, b, input[0], comm, p, my_rank, result, result_size, busy_count, worker_id, request);
    }

    if (busy_count > 0) {
      worker_id = (worker_id == (2 * p - 2)) ? 1 : (worker_id + 1);
      receive_results(busy_count, worker_id, request, result_size, result, comm, p);
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
