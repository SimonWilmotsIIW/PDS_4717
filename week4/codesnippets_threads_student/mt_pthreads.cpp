#include <stdio.h>
#include <pthread.h>

static const int num_threads =10;

void* threadmain(void* arg)
{
  long id = (long)arg;    /* argument moet altijd void* */
  fprintf(stderr, "Thread %ld loopt.\n", id);
  return (void*)id;     /* void* return value : zie pthread_join */
}

int main(int argc, char **argv)
{
  pthread_t t[num_threads];

  for (int i=0; i<num_threads; i++) {
    long argument = i;
    int rc = pthread_create(&t[i], NULL, threadmain, (void*)argument);
    if (rc != 0) { /* fout */ }
  }

  fprintf(stderr, "Main loopt.\n");

  for (int i=0; i<num_threads; i++) {
    void* resultaat;
    int rc = pthread_join(t[i], &resultaat);    /* wacht tot thread is afgelopen */
    if (rc != 0) { /* fout */ }
    fprintf(stderr, "Thread %d is afgelopen met resultaat %ld.\n", i, (long)resultaat);
  }

  return 0;
}
