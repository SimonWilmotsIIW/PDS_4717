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

  vector<float> a(vecSize), b(vecSize), c(vecSize);
  for (int i = 0 ; i < vecSize ; i++)
  {
    a[i] = i;
    b[i] = vecSize-i;
  }

  float *a_dev, *b_dev, *c_dev;
  cudaMalloc(&a_dev, sizeof(float)*vecSize);
  cudaMalloc(&b_dev, sizeof(float)*vecSize);
  cudaMalloc(&c_dev, sizeof(float)*vecSize);

  cudaMemcpy(a_dev, a.data(), a.size()*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice);
  
  Vec_add<<<numBlocks, numThreads>>>(a_dev, b_dev, c_dev, vecSize);

  cudaMemcpy(c.data(), c_dev, c.size()*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);

  // Check if it worked
  for (auto x : c)
    if (x != vecSize) // this should be the result for each element
      throw runtime_error("Got value " + to_string(x) + " but was expecting " + to_string(vecSize));

/*
  cudaError_t r = cudaGetLastError();
  if (r != cudaSuccess)
    throw runtime_error("Kernel failed, code is " + to_string(r) + ", message: " + cudaGetErrorString(r));
  */
  cout << "Seems to have worked, every element is " << c[0] << endl;
  return 0;
}