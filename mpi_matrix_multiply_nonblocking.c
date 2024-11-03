/**
 * @file mpi_matrix_multiply_nonblocking.c
 * @brief MPI-based parallel matrix multiplication using non-blocking communications.
 *
 * This program performs matrix multiplication in parallel using MPI with non-blocking
 * point-to-point communications (MPI_Isend and MPI_Irecv). It generates two matrices with
 * random values, distributes the computation across multiple processes, and verifies the
 * correctness of the parallel computation by comparing it with a serial computation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h> // For fabs()

#define N 4 ///< Adjust N to control the size of the matrices

void Matrix_Multiply(float *A, float *B, float *C, int m, int n, int p);
void Multiply_serial(float *A, float *B, float *C, int m, int n, int p);
int IsEqual(float *C1, float *C2, int m, int p);

/**
 * @brief Main function to perform parallel matrix multiplication using MPI with non-blocking communications.
 *
 * This function initializes the MPI environment, generates random matrices A and B,
 * distributes parts of matrix A among processes using non-blocking communications,
 * performs parallel matrix multiplication, gathers the results using non-blocking communications,
 * performs serial multiplication for verification, and outputs the results.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line argument strings.
 * @return int Exit status of the program.
 */
int main(int argc, char** argv) {
    int rank, size;
    float *A = NULL;
    float *B = NULL;
    float *C = NULL;
    float *C_serial = NULL;
    int m = N; ///< Number of rows in matrices A and C.
    int n = N; ///< Number of columns in A and rows in B.
    int p = N; ///< Number of columns in matrices B and C.

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes and the rank of this process
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Determine the number of rows each process will handle
    int rows_per_proc = m / size;
    int remainder = m % size;

    // Calculate local_m and offset for each process
    int local_m;
    int offset;
    if (rank < remainder) {
        local_m = rows_per_proc + 1;
        offset = rank * local_m;
    } else {
        local_m = rows_per_proc;
        offset = rank * local_m + remainder;
    }

    // Allocate memory for local matrices
    float *local_A = (float*)malloc(local_m * n * sizeof(float));
    float *local_C = (float*)malloc(local_m * p * sizeof(float));

    MPI_Request *requests;
    MPI_Status *statuses;

    // Non-blocking communication for data distribution
    if (rank == 0) {
        // Allocate memory for A, B, C, and C_serial
        A = (float*)malloc(m * n * sizeof(float));
        B = (float*)malloc(n * p * sizeof(float));
        C = (float*)malloc(m * p * sizeof(float));
        C_serial = (float*)malloc(m * p * sizeof(float));

        if (A == NULL || B == NULL || C == NULL || C_serial == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Seed the random number generator
        srand(time(NULL));

        // Fill matrix A with random numbers between 0 and 1
        for (int i = 0; i < m * n; i++) {
            A[i] = ((float)rand()) / RAND_MAX;
        }

        // Fill matrix B with random numbers between 0 and 1
        for (int i = 0; i < n * p; i++) {
            B[i] = ((float)rand()) / RAND_MAX;
        }

        printf("Process %d generated matrices A and B.\n", rank);

        // Non-blocking sends of portions of A to other processes
        int total_requests = 0;
        requests = (MPI_Request*)malloc((2 * (size - 1) + (size - 1)) * sizeof(MPI_Request));
        statuses = (MPI_Status*)malloc((2 * (size - 1) + (size - 1)) * sizeof(MPI_Status));

        // Send data to other processes
        for (int dest = 1; dest < size; dest++) {
            int dest_local_m = (dest < remainder) ? rows_per_proc + 1 : rows_per_proc;
            int dest_offset = (dest < remainder) ? dest * dest_local_m : dest * dest_local_m + remainder;

            // Non-blocking send of the number of rows
            MPI_Isend(&dest_local_m, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, &requests[total_requests++]);

            // Non-blocking send of the corresponding rows of A
            MPI_Isend(&A[dest_offset * n], dest_local_m * n, MPI_FLOAT, dest, 1, MPI_COMM_WORLD, &requests[total_requests++]);

            // Non-blocking send of B to each process
            MPI_Isend(B, n * p, MPI_FLOAT, dest, 2, MPI_COMM_WORLD, &requests[total_requests++]);
        }

        // Copy the corresponding rows of A to local_A
        for (int i = 0; i < local_m * n; i++) {
            local_A[i] = A[offset * n + i];
        }

        // No need to send B to itself
    } else {
        MPI_Request recv_requests[3];

        // Non-blocking receive of the number of rows
        MPI_Irecv(&local_m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &recv_requests[0]);

        // Wait for the number of rows to be received
        MPI_Wait(&recv_requests[0], MPI_STATUS_IGNORE);

        // Adjust allocations based on received local_m
        free(local_A); // Free previous allocation
        local_A = (float*)malloc(local_m * n * sizeof(float));
        free(local_C);
        local_C = (float*)malloc(local_m * p * sizeof(float));

        // Non-blocking receive of the corresponding rows of A
        MPI_Irecv(local_A, local_m * n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &recv_requests[1]);

        // Allocate memory for B
        B = (float*)malloc(n * p * sizeof(float));

        // Non-blocking receive of B
        MPI_Irecv(B, n * p, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &recv_requests[2]);

        // Wait for all non-blocking receives to complete
        MPI_Waitall(2, &recv_requests[1], MPI_STATUSES_IGNORE);
    }

    // All processes have received their data; proceed to computation

    // Each process performs local matrix multiplication
    Matrix_Multiply(local_A, B, local_C, local_m, n, p);

    // Non-blocking communication to send results back to Process 0
    if (rank != 0) {
        MPI_Request send_requests[2];

        // Non-blocking send of the number of rows
        MPI_Isend(&local_m, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &send_requests[0]);

        // Non-blocking send of the computed portion of C
        MPI_Isend(local_C, local_m * p, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, &send_requests[1]);

        // Wait for sends to complete
        MPI_Waitall(2, send_requests, MPI_STATUSES_IGNORE);

        // Free allocated memory
        free(local_A);
        free(local_C);
        free(B);
    } else {
        // Process 0 gathers the results
        // Copy local_C into C
        for (int i = 0; i < local_m * p; i++) {
            C[offset * p + i] = local_C[i];
        }

        // Non-blocking receives from other processes
        int total_requests = 0;
        int expected_messages = 2 * (size - 1);
        requests = (MPI_Request*)malloc(expected_messages * sizeof(MPI_Request));
        statuses = (MPI_Status*)malloc(expected_messages * sizeof(MPI_Status));

        for (int src = 1; src < size; src++) {
            int src_local_m;
            int src_offset;
            if (src < remainder) {
                src_local_m = rows_per_proc + 1;
                src_offset = src * src_local_m;
            } else {
                src_local_m = rows_per_proc;
                src_offset = src * src_local_m + remainder;
            }

            // Non-blocking receive of the number of rows
            MPI_Irecv(&src_local_m, 1, MPI_INT, src, 3, MPI_COMM_WORLD, &requests[total_requests++]);

            // Wait for the number of rows to be received
            MPI_Wait(&requests[total_requests - 1], MPI_STATUS_IGNORE);

            // Allocate space for the incoming data
            float *recv_buffer = &C[src_offset * p];

            // Non-blocking receive of the computed portion of C
            MPI_Irecv(recv_buffer, src_local_m * p, MPI_FLOAT, src, 4, MPI_COMM_WORLD, &requests[total_requests++]);
        }

        // Wait for all non-blocking receives to complete
        MPI_Waitall(total_requests, requests, statuses);

        // Free request and status arrays
        free(requests);
        free(statuses);

        // Process 0 performs serial multiplication and compares results
        double serial_start = MPI_Wtime();
        Multiply_serial(A, B, C_serial, m, n, p);
        double serial_end = MPI_Wtime();

        // Verify the results
        if (IsEqual(C, C_serial, m, p)) {
            printf("Parallel and serial results are equal.\n");
        } else {
            printf("Parallel and serial results are NOT equal.\n");
        }

        printf("Serial computation time: %f seconds\n", serial_end - serial_start);

        // Print the result matrices
        printf("Result matrix C (Parallel):\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                printf("%f ", C[i * p + j]);
            }
            printf("\n");
        }

        printf("Result matrix C_serial (Serial):\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                printf("%f ", C_serial[i * p + j]);
            }
            printf("\n");
        }

        // Free allocated memory
        free(A);
        free(B);
        free(C);
        free(C_serial);
        free(local_A);
        free(local_C);
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

/**
 * @brief Performs matrix multiplication on local data.
 *
 * Multiplies matrix A of size m x n with matrix B of size n x p,
 * resulting in matrix C of size m x p.
 *
 * @param A Pointer to the first element of matrix A.
 * @param B Pointer to the first element of matrix B.
 * @param C Pointer to the first element of matrix C.
 * @param m Number of rows in matrices A and C.
 * @param n Number of columns in A and rows in B.
 * @param p Number of columns in matrices B and C.
 */
void Matrix_Multiply(float *A, float *B, float *C, int m, int n, int p) {
    int i, j, k;

    // Perform local matrix multiplication
    for (i = 0; i < m; i++) {
        for (j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

/**
 * @brief Performs serial matrix multiplication for verification.
 *
 * Multiplies matrix A of size m x n with matrix B of size n x p,
 * resulting in matrix C of size m x p.
 *
 * @param A Pointer to the first element of matrix A.
 * @param B Pointer to the first element of matrix B.
 * @param C Pointer to the first element of matrix C.
 * @param m Number of rows in matrices A and C.
 * @param n Number of columns in A and rows in B.
 * @param p Number of columns in matrices B and C.
 */
void Multiply_serial(float *A, float *B, float *C, int m, int n, int p) {
    int i, j, k;

    // Perform serial matrix multiplication
    for (i = 0; i < m; i++) {
        for (j = 0; j < p; j++) {
            C[i * p + j] = 0;
            for (k = 0; k < n; k++) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
}

/**
 * @brief Compares two matrices for equality within a tolerance.
 *
 * Checks if all corresponding elements of matrices C1 and C2 are equal
 * within a specified tolerance.
 *
 * @param C1 Pointer to the first element of matrix C1.
 * @param C2 Pointer to the first element of matrix C2.
 * @param m Number of rows in the matrices.
 * @param p Number of columns in the matrices.
 * @return int Returns 1 if matrices are equal, 0 otherwise.
 */
int IsEqual(float *C1, float *C2, int m, int p) {
    for (int i = 0; i < m * p; i++) {
        if (fabs(C1[i] - C2[i]) > 1e-6) {
            return 0; // Matrices are not equal
        }
    }
    return 1; // Matrices are equal
}
