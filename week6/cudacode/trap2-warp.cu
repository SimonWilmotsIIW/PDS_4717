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

__device__ float Calc_sum(float *buffer, int num)
{
  int me = threadIdx.x;
  int next_offset = 1;

  while (next_offset < num)
  {
    float s = buffer[me] + buffer[(me + next_offset)%num];
    //__syncthreads(); // not needed for only one warp
    buffer[me] = s;
    //__syncthreads(); // not needed for only one warp
    next_offset *= 2;
  }
  return buffer[me];
}

__global__ void Trap(float a, float b, float h, int n, float *result)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  __shared__ float buffer[32]; // Warp size
  if (id > 0 && id < n)
  {
    float x = a + h*id;
    buffer[threadIdx.x] = f(x);
  }
  else
    buffer[threadIdx.x] = 0.0f;
  
  //__syncthreads(); // not needed for only one warp

  float s = Calc_sum(buffer, blockDim.x);
  if (threadIdx.x == 0)
    atomicAdd(result, s);
}

int main(int argc, char *argv[])
{
  checkDevice();
  
  if (argc != 4)
    throw runtime_error("Specify a, b and n");

  int numThreads = 32; // warp size
  float a = stof(argv[1]);
  float b = stof(argv[2]);
  int n = stoi(argv[3]);
  float h = (b-a)/n;

  int numBlocks = n/numThreads;
  if (n%numThreads != 0)
    numBlocks++;

  float *result;
  cudaMallocManaged(&result, sizeof(float));

  AutoAverageTimer t("trap2-1warp");
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