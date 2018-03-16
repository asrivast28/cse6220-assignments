CSE 6220 Programming Assignment 3
=================================

## Code structure

All the code is located at the root level of the project.
The `gtest` folder contains the Google Test Unit Testing framework version 1.7.

There are multiple header and .cpp files, your implementation will go
into the following files:

- `parallel_sort.h`: Declares the API for the parallel sorting implementation.
  Declare your own helper functions in this file.
- `parallel_sort.cpp`: Your implementation should go in here.
- `mpi_tests.cpp`: Unit tests for the parallel MPI code. Implement your own
  test cases for all declared functions in here.


Other files containing code that you should not change are:

- `main.cpp`: Implements code for the main executable `sort`. This does
  input/output reading and calling of the actual functions.
- `utils.h`/`utils.cpp`: common utility functions
- `io.h`/`io.cpp`: implements IO functions and random input generation, and MPI
  gather/scatter functions
- `mpi_gtest.cpp`: MPI wrapper for the GTest framework


Utility scripts (you may play around with these to generate your own custom
input):

- `input.py`: Python script to generate random inputs. Feel free to modify this
  script in order to generate different types of input.

## Compiling

In order to compile everything, simply run
```sh
make all
```

## Running the executable

After compilation, you'll have an executable `sort`. You can use this program as
follows:

```
Usage: ./sort [options] [input_file]
      Optional arguments:
          -o <file>    Output all solutions to the given file.
          -o -         Output all solutions to stdout.
          -t           Runs global tests on the sorting algorithm. NO input file!
          -r           Run random number tests, random numbers are generated only locally, no bottleneck at startup.
          -n <n>       Sets the number of generated input integers (mandatory with option -r and -t)
      Example:
          ./sort -o - input.txt
                  Sorts the numbers given by input.txt and outputs the sorted result to the terminal
```

## Running Tests

For running all tests do:
```sh
make test
```

You can also run the tests separately by:
```sh
mpirun -np 4 ./mpi_tests
```
