#include <iostream>
#include <stdexcept>

using namespace std;

void checkDevice()
{
  int numDevs = 0;
  cudaGetDeviceCount(&numDevs);
  if (numDevs != 1)
    throw runtime_error("Expecting one CUDA device, but got " + to_string(numDevs));
}

__global__ void Hello()
{
  // printf works, not cout!!
  printf("Hello from thread %d in block %d\n", threadIdx.x, blockIdx.x);
}


int main(int argc, char *argv[])
{
  checkDevice();
  
  if (argc != 3)
    throw runtime_error("Specify number of blocks and number of threads");

  int numBlocks = stoi(argv[1]);
  int numThreads = stoi(argv[2]);
  
  Hello<<<numBlocks, numThreads>>>();
  cudaDeviceSynchronize();
  
  cudaError_t r = cudaGetLastError();
  if (r != cudaSuccess)
    throw runtime_error("Kernel failed, code is " + to_string(r) + ", message: " + cudaGetErrorString(r));
  return 0;
}