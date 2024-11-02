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

int main(void)
{
  checkDevice();
  cout << "Found CUDA device!" << endl;
  return 0;
}