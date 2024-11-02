#include <iostream>
#include <thread>
#include <vector>
#include "timer.h"

const int64_t g_numLoops = 1 << 27;

void f(uint8_t *pBuffer, int offset) {
    for (int64_t i = 0; i < g_numLoops; i++) {
        pBuffer[offset] += 1;
    }
}

int main(int argc, char *argv[]) {
    int multiplier = std::stoi(argv[1]);
    const int numThreads = 4;
    uint8_t *buffer = new uint8_t[multiplier * numThreads]();
    
    AutoAverageTimer timer("oef18");

    std::vector<std::thread> threads;
    timer.start();
    for (int id = 0; id < numThreads; ++id) {
        threads.emplace_back([&, id]() {
            int offset = multiplier * id;
            f(buffer, offset);
        });
    }

    for (auto &t : threads) {
        t.join();
    }
    timer.stop();
    timer.report();

    delete[] buffer;
    return 0;
}