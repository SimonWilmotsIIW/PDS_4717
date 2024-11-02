#include <iostream>
#include <cmath>
#include <omp.h>
#include "timer.h"

bool is_prime(size_t number) {
    size_t to_check = std::sqrt(number) + 1;
    for (size_t i = 2; i < to_check; ++i) {
        if (number % i == 0)
            return false;
    }
    return true;
}

int count_primes(size_t max_num, omp_sched_t schedule_type, int chunk_size) {
    int prime_count = 0;

    #pragma omp parallel for schedule(runtime) reduction(+:prime_count)
    for (size_t i = 2; i <= max_num; ++i) {
        if (is_prime(i)) {
            prime_count++;
        }
    }
    return prime_count;
}

int main() {
    size_t max_num = 108;
    int chunk_size = 10;
    int averageAmount = 100;
    int prime_count = 0;
    omp_sched_t schedules[] = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
    std::string schedule_names[] = {"static", "dynamic", "guided"};

    for (int i = 0; i < 3; ++i) {
        omp_set_schedule(schedules[i], chunk_size);

        AutoAverageTimer timer(schedule_names[i]);
        for (int i = 0 ; i < averageAmount ; ++i) {
            timer.start();
            prime_count = count_primes(max_num, schedules[i], chunk_size);
            timer.stop();
        }
        timer.report();

        printf(">> found %d primes\n", prime_count);
    }

    return 0;
}
