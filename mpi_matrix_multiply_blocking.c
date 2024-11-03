#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define N 4 // Adjust N to control the size of the matrices

void Matrix_Multiply(float *A, float *B, float *C, int m, int n, int p);
void Multiply_serial(float *A, float *B, float *C, int m, int n, int p);
int IsEqual(float *C1, float *C2, int m, int p);

int main(int argc, char** argv) {
    int rank, size;
    float *A = NULL;
    float *B = NULL;
    float *C = NULL;
    float *C_serial = NULL;
    int m = N; // Number of rows in matrices A and C.
    int n = N; // Number of columns in A and rows in B.
    int p = N; // Number of columns in matrices B and C.

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

        // Send portions of A and B to other processes
        for (int dest = 1; dest < size; dest++) {
            int dest_local_m = (dest < remainder) ? rows_per_proc + 1 : rows_per_proc;
            int dest_offset = (dest < remainder) ? dest * dest_local_m : dest * dest_local_m + remainder;

            // Send the number of rows to each process
            MPI_Send(&dest_local_m, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

            // Send the corresponding rows of A
            MPI_Send(&A[dest_offset * n], dest_local_m * n, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);

            // Send B to each process
            MPI_Send(B, n * p, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        }

        // Process 0's local data
        local_m = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
        offset = (rank < remainder) ? rank * local_m : rank * local_m + remainder;

        // Copy the corresponding rows of A to local_A
        for (int i = 0; i < local_m * n; i++) {
            local_A[i] = A[offset * n + i];
        }

        // Process 0 already has B
    } else {
        // Receive the number of rows
        MPI_Recv(&local_m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Receive the corresponding rows of A
        MPI_Recv(local_A, local_m * n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Allocate memory for B
        B = (float*)malloc(n * p * sizeof(float));

        // Receive B
        MPI_Recv(B, n * p, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Each process performs local matrix multiplication
    Matrix_Multiply(local_A, B, local_C, local_m, n, p);

    // Send the results back to Process 0
    if (rank != 0) {
        MPI_Send(&local_m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_C, local_m * p, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    } else {
        // Process 0 gathers the results
        // Copy local_C into C
        for (int i = 0; i < local_m * p; i++) {
            C[offset * p + i] = local_C[i];
        }

        // Receive results from other processes
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

            // Receive the number of rows
            MPI_Recv(&src_local_m, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Receive the computed portion of C
            MPI_Recv(&C[src_offset * p], src_local_m * p, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

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

        // Free memory
        free(A);
        free(B);
        free(C);
        free(C_serial);
    }

    // All processes free their local memory
    free(local_A);
    free(local_C);
    if (rank != 0) {
        free(B);
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

// Implement Matrix_Multiply, Multiply_serial, and IsEqual functions

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

int IsEqual(float *C1, float *C2, int m, int p) {
    for (int i = 0; i < m * p; i++) {
        if (fabs(C1[i] - C2[i]) > 1e-6) {
            return 0; // Matrices are not equal
        }
    }
    return 1; // Matrices are equal
}
