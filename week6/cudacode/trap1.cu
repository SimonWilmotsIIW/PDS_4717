#include <iostream>
#include <stdexcept>
#include <vector>
#include "timer.h"

using namespace std;

void checkDevice()
{
  int numDevs = 0;
  cudaGetDeviceCount(&numDevs);
  if (numDevs != 1)
    throw runtime_error("Expecting one CUDA device, but got " + to_string(numDevs));
}

__host__ __device__ float f(float x)
{
  return x*x + 1.0f;
}

__global__ void Trap(float a, float b, float h, int n, float *result)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id > 0 && id < n)
  {
    float x = a + h*id;
    float fx = f(x);
    
    atomicAdd(result, fx);
  }
}

int main(int argc, char *argv[])
{
  checkDevice();
  
  if (argc != 5)
    throw runtime_error("Specify number of threads, a, b and n");

  int numThreads = stoi(argv[1]);
  float a = stof(argv[2]);
  float b = stof(argv[3]);
  int n = stoi(argv[4]);
  float h = (b-a)/n;

  int numBlocks = n/numThreads;
  if (n%numThreads != 0)
    numBlocks++;

  float *result;
  cudaMallocManaged(&result, sizeof(float));

  AutoAverageTimer t("trap1-always_atomicadd");
  for (int r = 0 ; r < 100 ; r++)
  {
    t.start();
    *result = (f(a) + f(b)) * 0.5f;
  
    Trap<<<numBlocks, numThreads>>>(a, b, h, n, result);
    cudaDeviceSynchronize();
  
    *result *= h;
    t.stop();

    if (r == 0)
      cout << "Result is:" << *result << endl;
  }

  cudaFree(result);
  t.report(cout, 1e-6, "ms");
  return 0;
}