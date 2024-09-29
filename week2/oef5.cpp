#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "timer.h"

typedef struct Entry
{
    Entry* next;
    uint64_t padding[127];
} Entry;

int main(int argc, char *argv[]) {
    size_t arraySize = std::stoul(argv[1]);
    std::vector<Entry> entries(arraySize);
    printf("Iterating %d times\n", arraySize);
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i].next = &entries[(i + 1) % entries.size()]; 
    }

    AutoAverageTimer timer("Iterating circular list");

    const size_t steps = 2e7;

    timer.start();

    Entry* current = &entries[0];
    for (size_t i = 0; i < steps; ++i) {
        current = current->next;
    }
    
    timer.stop();

    timer.report(std::cout);
    
    return 0;
}