#include <stdio.h>
#include <omp.h>

int main() {
    // Stel het aantal threads in als gewenst, of laat OpenMP dit bepalen
    #pragma omp parallel
    {
        // Verkrijg de thread ID
        int thread_id = omp_get_thread_num();
        
        // Print een boodschap voor elke thread
        printf("Hello World from thread %d\n", thread_id);
    }
    
    return 0;
}
