#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

#define ITERATIONS 10e6

int main(int argc, char** argv) {
    int rank, size;
    float start_time, end_time, total_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "2 processes needed" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int n = 0; n <= 10; ++n) {
        int buffer_size = std::pow(2, n);
        std::vector<void*> buffer(buffer_size);

        total_time = 0;

        for (int i = 0; i < ITERATIONS; ++i) {
            if (rank == 0) {
                start_time = MPI_Wtime();

                MPI_Send(buffer.data(), buffer_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                end_time = MPI_Wtime();
                total_time += (end_time - start_time);
            } else if (rank == 1) {
                MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.data(), buffer_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        if (rank == 0) {
            float avg_time = (total_time / ITERATIONS) * 1e6;
            std::cout << "buffer_size (bytes) = " << buffer_size << ", duration (us) = " << avg_time << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
