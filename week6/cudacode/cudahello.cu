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
  printf("Hello from thread %d\n", threadIdx.x);
}


int main(int argc, char *argv[])
{
  checkDevice();
  
  if (argc != 2)
    throw runtime_error("Specify number of threads");

  int numThreads = stoi(argv[1]);
  
  Hello<<<1, numThreads>>>();
  cudaDeviceSynchronize();
  
  cudaError_t r = cudaGetLastError();
  if (r != cudaSuccess)
    throw runtime_error("Kernel failed, code is " + to_string(r) + ", message: " + cudaGetErrorString(r));
  return 0;
}