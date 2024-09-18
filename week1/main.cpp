#include <iostream>
#include <stdio.h>
#include "readfloats.cpp"
#include <limits>

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

int main() {
    std::vector<float> data = readFloats("histvalues.dat");
    // for(int i=0 ; i<data.size() ; i++) {
    //     printf("%f\n", data[i]);
    // }

    std::pair<float, float> minMax = findMinMax(data);

    printf("\nMin: %f Max: %f", minMax.first, minMax.second);
    
    size_t input;
    printf("\nGive number of bins (N): ");
    std::cin >> input;
    
    return 0;
}