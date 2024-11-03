# Parallel Matrix Multiplication using MPI

This repository contains three implementations of a parallel matrix multiplication program using MPI (Message Passing Interface). Each implementation employs a different communication method:

1. **Blocking Point-to-Point Communications** (`MPI_Send` and `MPI_Recv`)
2. **Collective Communications** (`MPI_Bcast`, `MPI_Scatterv`, and `MPI_Gatherv`)
3. **Non-Blocking Point-to-Point Communications** (`MPI_Isend` and `MPI_Irecv`)

The programs perform matrix multiplication of two randomly generated matrices and verify the correctness of the parallel computation by comparing it with a serial computation.

---

## Table of Contents

- [Tasks](#tasks)
- [Implementations](#implementations)
  - [Blocking Point-to-Point Communications](#1-blocking-point-to-point-communications)
  - [Collective Communications](#2-collective-communications)
  - [Non-Blocking Point-to-Point Communications](#3-non-blocking-point-to-point-communications)
- [Compiling and Running the Programs](#compiling-and-running-the-programs)
- [Elaboration of the Parallel Algorithms](#elaboration-of-the-parallel-algorithms)
- [Differences Among Implementations](#differences-among-implementations)
- [Restrictions on Matrix Size (N)](#restrictions-on-matrix-size-n)
- [Running Times and Observations](#running-times-and-observations)
- [Files in the Repository](#files-in-the-repository)

---

## Tasks

1. **Create a Serial Matrix Multiplication Function**: Implement `Multiply_serial()` to perform matrix multiplication without parallelism.

   ```c
   void Matrix_Multiply(float *A, float *B, float *C, int m, int n, int p) {
       int i, j, k;
       for (i = 0; i < m; i++)
           for (j = 0; j < p; j++) {
               C[i * p + j] = 0;
               for (k = 0; k < n; k++)
                   C[i * p + j] += A[i * n + k] * B[k * p + j];
           }
   }
   ```

2. **Create a Matrix Comparison Function**: Implement `IsEqual()` to check if two matrices are exactly the same.

   ```c
   int IsEqual(float *A, float *B, int m, int n) {
       for (int i = 0; i < m * n; i++) {
           if (fabs(A[i] - B[i]) > 1e-6)
               return 0; // Matrices are not equal
       }
       return 1; // Matrices are equal
   }
   ```

3. **Implement the Parallel Algorithm in `main()`**: Use MPI to parallelize the matrix multiplication.

   - Initialize and finalize the MPI environment.
   - Let Process #0 generate matrices **A** (size `N x 32`) and **B** (size `32 x N`) with random numbers in `[0, 1]`.
   - Implement communications between Process #0 and other processes.
   - Compute **C = A * B** using parallel programming.
   - Let Process #0 compute **C_serial = A * B** using `Multiply_serial()`.
   - Verify the correctness by checking if **C** and **C_serial** are equal using `IsEqual()`.
   - Measure the running time of both computations.

4. **Compile and Run the Programs**:

   - Use `mpicc` to compile the programs.
   - Use `mpirun` or `mpiexec` to run the programs.
   - Test with different numbers of processes.

5. **Implement Other Communication Methods**:

   - Copy the code and implement parallelism with collective communications.
   - Copy the code again and implement parallelism with non-blocking communications.
   - Repeat steps 3 and 4 for each implementation.

---

## Implementations

### 1. Blocking Point-to-Point Communications

**Principle**:

- **Process 0** generates matrices **A** and **B**.
- **Process 0** sends portions of **A** and **B** to other processes using `MPI_Send`.
- Each process receives its portion using `MPI_Recv`.
- Each process computes its assigned portion of the result matrix **C**.
- Processes send their computed portions back to **Process 0**.
- **Process 0** assembles the final result and verifies correctness.

**Implementation Details**:

- Uses `MPI_Send` and `MPI_Recv` for communication.
- Communication is synchronous; a process waits until the send or receive is complete.

### 2. Collective Communications

**Principle**:

- **Process 0** generates matrices **A** and **B**.
- Uses `MPI_Bcast` to broadcast matrix **B** to all processes.
- Uses `MPI_Scatterv` to distribute portions of **A** to processes.
- Each process computes its assigned portion of **C**.
- Uses `MPI_Gatherv` to gather the computed portions back to **Process 0**.
- **Process 0** assembles the final result and verifies correctness.

**Implementation Details**:

- Simplifies communication code using collective operations.
- Collective operations are optimized for performance on many systems.

### 3. Non-Blocking Point-to-Point Communications

**Principle**:

- **Process 0** initiates non-blocking sends of data to other processes using `MPI_Isend`.
- Other processes initiate non-blocking receives using `MPI_Irecv`.
- While communication is in progress, processes can perform computations that do not depend on the incoming data.
- Each process computes its assigned portion of **C** after receiving data.
- Processes send their results back to **Process 0** using non-blocking sends.
- **Process 0** receives results using non-blocking receives and assembles the final result.
- Verification is performed as before.

**Implementation Details**:

- Uses `MPI_Isend` and `MPI_Irecv` for non-blocking communications.
- Requires synchronization using `MPI_Wait` or `MPI_Waitall`.
- Can overlap communication and computation for potential performance gains.

---

## Compiling and Running the Programs

### Prerequisites

- MPI library installed (e.g., OpenMPI or MPICH).
- C compiler with MPI support (`mpicc`).

### Compilation

Compile each program using `mpicc`:

```bash
# Compile Blocking P2P Communications Program
mpicc -o mpi_matrix_multiply_blocking mpi_matrix_multiply_blocking.c -lm

# Compile Collective Communications Program
mpicc -o mpi_matrix_multiply_collective mpi_matrix_multiply_collective.c -lm

# Compile Non-Blocking P2P Communications Program
mpicc -o mpi_matrix_multiply_nonblocking mpi_matrix_multiply_nonblocking.c -lm
```

### Execution

Run each program using `mpirun`:

```bash
# Run Blocking P2P Communications Program
mpirun -np 4 ./mpi_matrix_multiply_blocking

# Run Collective Communications Program
mpirun -np 4 ./mpi_matrix_multiply_collective

# Run Non-Blocking P2P Communications Program
mpirun -np 4 ./mpi_matrix_multiply_nonblocking
```

Replace `4` with the desired number of processes.

---

## Elaboration of the Parallel Algorithms

### **Design Principle**

The core idea is to divide the matrix multiplication task among multiple processes to leverage parallel computing capabilities. The matrices are partitioned so that each process handles a subset of the data, performs computations independently, and then combines the results.

### **Implementation Steps**

1. **Initialization**:

   - Initialize MPI environment.
   - Determine the rank and size of processes.

2. **Data Generation** (Process 0):

   - Generate matrices **A** and **B** with random values.
   - Decide how to partition the data among processes.

3. **Data Distribution**:

   - **Blocking P2P**: Use `MPI_Send` and `MPI_Recv` to distribute data.
   - **Collective**: Use `MPI_Scatterv` and `MPI_Bcast`.
   - **Non-Blocking P2P**: Use `MPI_Isend` and `MPI_Irecv`.

4. **Local Computation**:

   - Each process computes its assigned portion of the result matrix **C**.

5. **Result Gathering**:

   - **Blocking P2P**: Processes send results back using `MPI_Send`.
   - **Collective**: Use `MPI_Gatherv`.
   - **Non-Blocking P2P**: Use `MPI_Isend` and `MPI_Irecv`.

6. **Verification and Timing** (Process 0):

   - Compute the serial result using `Multiply_serial()`.
   - Verify correctness using `IsEqual()`.
   - Measure and output the running time.

---

## Differences Among Implementations

### **Principle Differences**

- **Blocking P2P Communications**:

  - Direct communication between processes using send and receive operations.
  - Processes block until the communication operation completes.

- **Collective Communications**:

  - Communication is performed using collective operations that involve all processes.
  - Simplifies code and may offer performance benefits due to optimizations.

- **Non-Blocking P2P Communications**:

  - Communication operations return immediately, allowing computation and communication to overlap.
  - Requires explicit synchronization to ensure data integrity.

### **Implementation Differences**

- **Code Structure**:

  - Collective communications reduce the amount of communication code needed.
  - Non-blocking communications introduce complexity with `MPI_Request` and synchronization.

- **Synchronization**:

  - Blocking operations inherently synchronize processes.
  - Non-blocking operations require `MPI_Wait` or `MPI_Waitall` for synchronization.

- **Performance Considerations**:

  - Non-blocking communications can improve performance by overlapping communication and computation.
  - Collective operations are often optimized for the underlying hardware.

---

## Restrictions on Matrix Size (N)

- The matrix size `N` should be chosen carefully based on the number of processes.

### **Possible Restrictions**

- **Divisibility**: For simplicity, the number of rows (`m`) may need to be divisible by the number of processes to ensure equal workload distribution.
- **Memory Constraints**: Large values of `N` increase memory usage and may exceed available memory on a single machine.
- **Performance**: Extremely large or small values of `N` may not effectively demonstrate performance benefits.

### **Explanation**

- In cases where `N` is not divisible by the number of processes, additional logic is required to handle the distribution of remaining rows (e.g., using `MPI_Scatterv` or adjusting the loop ranges).
- Unequal workload distribution can lead to idle processes and reduced performance gains.

---

## Running Times and Observations

### **Testing Procedure**

- Run each program with varying values of `N` (e.g., 100, 500, 1000).
- Use the same number of processes for each test (e.g., 2, 4).
- Record the running time for both serial and parallel computations.

### **Sample Running Times**

| N    | Serial Time (s) | Blocking P2P Time (s) | Collective Time (s) | Non-Blocking P2P Time (s) |
|------|-----------------|-----------------------|---------------------|---------------------------|
| 100  |      0.01       |         0.008         |        0.007        |           0.006           |
| 500  |      1.25       |         0.65          |        0.60         |           0.58            |
| 1000 |      10.0       |         5.2           |        5.0          |           4.8             |

*Note: These times are illustrative examples. Actual times will vary based on hardware and system load.*

### **Observations**

- **Performance Improvement**: The parallel implementations show significant reductions in computation time compared to the serial version.
- **Scalability**: As `N` increases, the benefits of parallelization become more pronounced.
- **Non-Blocking vs. Blocking**: Non-blocking communications may offer slight performance improvements due to overlapping communication and computation.
- **Collective Communications**: Collective operations are efficient and simplify the codebase.

### **Explanation**

- **Parallel Efficiency**: Dividing the workload reduces the computation time by leveraging multiple processors.
- **Communication Overhead**: Communication between processes introduces overhead, which can impact performance for smaller matrices.
- **Optimization**: Non-blocking and collective communications can optimize data transfer, leading to better performance.

---

## Files in the Repository

- `mpi_matrix_multiply_blocking.c`: Implementation using blocking point-to-point communications.
- `mpi_matrix_multiply_collective.c`: Implementation using collective communications.
- `mpi_matrix_multiply_nonblocking.c`: Implementation using non-blocking point-to-point communications.
- `README.md`: This document.

---

## How to Use the Code

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Zer0F8th/mpi_matrix_multiply.git
   cd mpi_matrix_multiply
   mkdir bin/
   ```

2. **Compile the Programs**:

   ```bash
   mpicc -o bin/mpi_matrix_multiply_blocking mpi_matrix_multiply_blocking.c -lm
   mpicc -o bin/mpi_matrix_multiply_collective mpi_matrix_multiply_collective.c -lm
   mpicc -o bin/mpi_matrix_multiply_nonblocking mpi_matrix_multiply_nonblocking.c -lm
   ```

3. **Run the Programs**:

   ```bash
   mpirun -np 4 ./mpi_matrix_multiply_blocking
   mpirun -np 4 ./mpi_matrix_multiply_collective
   mpirun -np 4 ./mpi_matrix_multiply_nonblocking
   ```

4. **Adjust Matrix Size (Optional)**:

   - Modify the `#define N` line in each source file to change the matrix size.

---

## Conclusion

This project demonstrates how different MPI communication methods can be applied to parallelize matrix multiplication. By comparing the implementations, we observe the trade-offs between code complexity, performance, and communication overhead. Understanding these differences is crucial for optimizing parallel applications in high-performance computing environments.

