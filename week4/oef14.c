#include <stdio.h>
#include <math.h>
#include <pthread.h>
#define THREAD_COUNT 4

double total_result = 0.0;
pthread_mutex_t mutex;

double function(double x) {
    return sqrt(1 - x * x);
}

void* integrate(void* arg) {
    // all these variables can maybe be collected in a struct and pass this to the thread
    int thread_id = *((int*)arg);
    double start = -1.0 + thread_id * (2.0 / THREAD_COUNT);
    double end = start + (2.0 / THREAD_COUNT);
    int intervals = 1000000 / THREAD_COUNT;
    double width = (end - start) / intervals;
    double result = 0.0;

    for (int i = 0; i < intervals; ++i) {
        double xi = start + i * width;
        result += function(xi) * width;
    }

    pthread_mutex_lock(&mutex);
    total_result += result;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main() {
    pthread_t threads[THREAD_COUNT];
    int thread_ids[THREAD_COUNT];
    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < THREAD_COUNT; ++i) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, integrate, &thread_ids[i]);
    }

    for (int i = 0; i < THREAD_COUNT; ++i) {
        pthread_join(threads[i], NULL);
    }

    // TODO TIJDMETING

    printf("benadering: %.12f\n", total_result);
    printf("exacte waarde: %.12f\n", M_PI / 2);
    //printf("verschil: %f\n", ((M_PI / 2)-total_result)/total_result);

    pthread_mutex_destroy(&mutex);

    return 0;
}