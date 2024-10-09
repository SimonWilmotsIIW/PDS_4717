#include <iostream>
#include <stdio.h>
#include "readfloats.cpp"
#include <limits>
#include <cmath>
#include <mpi.h>

std::pair<float, float> findMinMax(std::vector<float> data) {
    float max = std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();

    for (size_t i=0 ; i < data.size() ; i++){
        if (data[i] > max) {
            max = data[i];
        }

        if (data[i] < min) {
            min = data[i];
        }
    }
    return std::pair<float, float>(min,max);
}

std::vector<int> getHistogram(const std::vector<float>& data, int N, float min, float max) {
    std::vector<int> histogram(N, 0);
    float binWidth = (max - min) / N;

    for (size_t i = 0; i < data.size(); i++) {
        int binIndex = std::floor((data[i] - min) / binWidth);
        
        if (binIndex == N) {
            binIndex--;
        }
        histogram[binIndex]++;
    }
    return histogram;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Gebruik: " << argv[0] << " <aantal bins>" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1); // Beëindig met foutcode
    }
    size_t inputN = std::atoi(argv[1]);


    std::vector<float> data;
    int numValues;

    if (rank == 0) {
        data = readFloats("histvalues.dat");
        numValues = data.size();

        if (numValues % size != 0) {
            std::cerr << numValues << " % " << size << " != 0" << std::endl;
            std::cerr << "#waarden niet deelbaar met #processen" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&numValues, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int valuesPerProcess = numValues / size;

    std::vector<float> dataPerProcess(valuesPerProcess);

    MPI_Scatter(data.data(), valuesPerProcess, MPI_FLOAT, dataPerProcess.data(), valuesPerProcess, MPI_FLOAT, 0, MPI_COMM_WORLD);

    std::pair<float, float> localMinMax = findMinMax(dataPerProcess);

    float globalMin, globalMax;
    MPI_Reduce(&localMinMax.first, &globalMin, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localMinMax.second, &globalMax, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);


    MPI_Bcast(&inputN, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    std::vector<int> localHistogram = getHistogram(dataPerProcess, inputN, globalMin, globalMax);

    std::vector<int> globalHistogram(inputN);
    MPI_Reduce(localHistogram.data(), globalHistogram.data(), inputN, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Histogram\n");
        for (size_t i = 0; i < globalHistogram.size(); i++) {
            printf("Bin %lu: %d\n", i, globalHistogram[i]);
        }
    }
        
    MPI_Finalize();
    return 0;
}