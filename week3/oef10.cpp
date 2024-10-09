#include <mpi.h>
#include <iostream>
#include <vector>

#define ARRAY_SIZE 1048576

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            printf("only 2 processes allowed :D\n");
        }
        MPI_Finalize();
        return 0;
    }
    std::vector<float> send(ARRAY_SIZE, rank);
    std::vector<float> receive(ARRAY_SIZE, rank);

    int otherProcess = (rank == 0) ? 1 : 0;

    MPI_Sendrecv(send.data(), ARRAY_SIZE, MPI_FLOAT, otherProcess, 0, receive.data(), ARRAY_SIZE, MPI_FLOAT, otherProcess, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("I am process %d and I have received b(0) = %f\n", rank, receive[0]);

    MPI_Finalize();
}
