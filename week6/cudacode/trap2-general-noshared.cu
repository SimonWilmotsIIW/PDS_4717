#include <iostream>
#include <stdexcept>
#include <vector>
#include <bitset>
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
    __syncthreads();
    buffer[me] = s;
    __syncthreads();
    next_offset *= 2;
  }
  return buffer[me];
}

#define MAXTHREADS 1024

__global__ void Trap(float a, float b, float h, int n, float *result, float *totalBuffer)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  float *buffer = totalBuffer + blockDim.x*blockIdx.x;
  
  if (id > 0 && id < n)
  {
    float x = a + h*id;
    buffer[threadIdx.x] = f(x);
  }
  else
    buffer[threadIdx.x] = 0.0f;
  
  __syncthreads();

  float s = Calc_sum(buffer, blockDim.x);
  if (threadIdx.x == 0)
    atomicAdd(result, s);
}

bool IsPowerOfTwo(int x)
{
  return std::bitset<32>(x).count() == 1;
}

int main(int argc, char *argv[])
{
  checkDevice();
  
  if (argc != 5)
    throw runtime_error("Specify number of threads, a, b and n");

  int numThreads = stoi(argv[1]);
  if (!IsPowerOfTwo(numThreads) || numThreads > MAXTHREADS)
    throw runtime_error("Invalid number of threads: must be a power of two and max " + to_string(MAXTHREADS));

  float a = stof(argv[2]);
  float b = stof(argv[3]);
  int n = stoi(argv[4]);
  float h = (b-a)/n;

  int numBlocks = n/numThreads;
  if (n%numThreads != 0)
    numBlocks++;

  float *result, *buffer;
  cudaMallocManaged(&result, sizeof(float));
  cudaMallocManaged(&buffer, sizeof(float)*(numBlocks*numThreads));

  AutoAverageTimer t("trap2-general-noshared");
  for (int r = 0 ; r < 100 ; r++)
  {
    t.start();
    *result = (f(a) + f(b)) * 0.5f;
  
    Trap<<<numBlocks, numThreads>>>(a, b, h, n, result, buffer);
    cudaDeviceSynchronize();
  
    *result *= h;
    t.stop();

    if (r == 0)
      cout << "Result is:" << *result << endl;
  }

  cudaFree(result);
  cudaFree(buffer);

  t.report(cout, 1e-6, "ms");
  return 0;
}