#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

void checkDevice()
{
  int numDevs = 0;
  cudaGetDeviceCount(&numDevs);
  if (numDevs != 1)
    throw runtime_error("Expecting one CUDA device, but got " + to_string(numDevs));
}

__global__ void Vec_add(const float *x, const float *y, float *z, int n)
{
  // Typische constructie om ID te bepalen!
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < n) // Belangrijke check!
    z[id] = x[id] + y[id];
}


int main(int argc, char *argv[])
{
  checkDevice();
  
  if (argc != 3)
    throw runtime_error("Specify number of threads and vector size");

  int numThreads = stoi(argv[1]);
  int vecSize = stoi(argv[2]);

  int numBlocks = vecSize/numThreads;
  if (vecSize%numThreads != 0)
    numBlocks++;

  float *a, *b, *c;
  cudaMallocManaged(&a, sizeof(float)*vecSize);
  cudaMallocManaged(&b, sizeof(float)*vecSize);
  cudaMallocManaged(&c, sizeof(float)*vecSize);

  for (int i = 0 ; i < vecSize ; i++)
  {
    a[i] = i;
    b[i] = vecSize-i;
  }
  
  Vec_add<<<numBlocks, numThreads>>>(a, b, c, vecSize);
  
  // Wait until kernel is done!
  cudaDeviceSynchronize();

  // Check if it worked, this will automatically fecth GPU memory to CPU
  for (int i = 0 ; i < vecSize ; i++)
    if (c[i] != vecSize) // this should be the result for each element
      throw runtime_error("Got value " + to_string(c[i]) + " but was expecting " + to_string(vecSize));

  cout << "Seems to have worked, every element is " << c[0] << endl;

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return 0;
}