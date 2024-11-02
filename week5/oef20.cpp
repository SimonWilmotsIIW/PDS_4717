#include <iostream>
#include <stdio.h>
#include <limits>
#include <cmath>
#include <vector>
#include <omp.h>
#include "readfloats.cpp"

std::pair<float, float> findMinMax(const std::vector<float>& data) {
    float global_max = std::numeric_limits<float>::min();
    float global_min = std::numeric_limits<float>::max();

    #pragma omp parallel
    {
        float local_max = std::numeric_limits<float>::min();
        float local_min = std::numeric_limits<float>::max();

        #pragma omp for
        for (int i = 0; i < data.size(); i++) {
            if (data[i] > local_max) {
                local_max = data[i];
            }
            if (data[i] < local_min) {
                local_min = data[i];
            }
        }

        #pragma omp critical
        {
            if (local_max > global_max) {
                global_max = local_max;
            }
            if (local_min < global_min) {
                global_min = local_min;
            }
        }
    }

    return std::pair<float, float>(global_min, global_max);
}

std::vector<int> getHistogram(const std::vector<float>& data, int N, float min, float max) {
    std::vector<int> histogram(N, 0);
    float binWidth = (max - min) / N;

    #pragma omp parallel
    {
        std::vector<int> local_histogram(N, 0);

        #pragma omp for
        for (size_t i = 0; i < data.size(); i++) {
            int binIndex = std::floor((data[i] - min) / binWidth);

            if (binIndex == N) {
                binIndex--;
            }
            local_histogram[binIndex]++;
        }

        #pragma omp critical
        {
            for (int j = 0; j < N; j++) {
                histogram[j] += local_histogram[j];
            }
        }
    }
    return histogram;
}

int main() {
    std::vector<float> data = readFloats("histvalues.dat");

    std::pair<float, float> minMax = findMinMax(data);

    printf("\nMin: %f Max: %f", minMax.first, minMax.second);

    size_t inputN;
    printf("\nGive number of bins (N): ");
    std::cin >> inputN;

    printf("Histogram\n");
    auto histogram = getHistogram(data, inputN, minMax.first, minMax.second);
    for (size_t i = 0; i < histogram.size(); i++) {
        printf("Bin %lu: %d\n", i, histogram[i]);
    }

    return 0;
}