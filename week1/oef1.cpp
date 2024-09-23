#include <iostream>
#include <stdio.h>
#include "readfloats.cpp"
#include <limits>
#include <cmath>

std::pair<float, float> findMinMax(std::vector<float> data) {
    float max = std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();

    for (int i=0 ; i < data.size() ; i++){
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

int main() {
    std::vector<float> data = readFloats("histvalues.dat");
    // for(int i=0 ; i<data.size() ; i++) {
    //     printf("%f\n", data[i]);
    // }

    std::pair<float, float> minMax = findMinMax(data);

    printf("\nMin: %f Max: %f", minMax.first, minMax.second);
    
    size_t inputN;
    printf("\nGive number of bins (N): ");
    std::cin >> inputN;

    printf("Histogram\n");
    auto histogram = getHistogram(data, inputN, minMax.first, minMax.second);
    for (size_t i = 0; i < histogram.size(); i++) {
        printf("Bin %i: %d\n", i, histogram[i]);
    }
    
    return 0;
}