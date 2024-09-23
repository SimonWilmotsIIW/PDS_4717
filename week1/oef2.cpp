#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "timer.h"

constexpr int N = 20000;

size_t rowMajor(const std::vector<int>& array) {
    size_t sum = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum += array[i * N + j];
        }
    }
    return sum;
}

size_t columnMajor(const std::vector<int>& array) {
    size_t sum = 0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            sum += array[i * N + j];
        }
    }
    return sum;
}

int main() {
    std::vector<int> array(N * N);

    int count = 100;

    
    AutoAverageTimer timer1("Row major");
    for (int i = 0; i < count; ++i) {
        timer1.start();
        rowMajor(array);
        timer1.stop();
    }



    AutoAverageTimer timer2("Column major");
    for (int i = 0; i < count; ++i) {
        timer2.start();
        columnMajor(array);
        timer2.stop();
    }

    timer1.report(std::cout);
    timer2.report(std::cout);
    

    return 0;
}
