#include <iostream>
#include <cuda_runtime.h>

#define ARRAY_SIZE 500;
#define BLOCK_COUNT 16;
#define THREADS_PER_BLOCK (ARRAY_SIZE + BLOCK_COUNT - 1) / BLOCK_COUNT;

__global__ void writeBlockIndex(int *blockIndexArray) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ARRAY_SIZE) {
        blockIndexArray[idx] = blockIdx.x;
    }
}

__global__ void writeThreadIndex(int *threadIndexArray) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ARRAY_SIZE) {
        threadIndexArray[idx] = threadIdx.x;
    }
}

int main(int argc, char *argv[]) {
    int *device_blockIndexArray, *device_threadIndexArray;
    int host_blockIndexArray[ARRAY_SIZE];
    int host_threadIndexArray[ARRAY_SIZE];

    cudaMalloc((void**)&device_blockIndexArray, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void**)&device_threadIndexArray, ARRAY_SIZE * sizeof(int));

    writeBlockIndex<<<BLOCK_COUNT, THREADS_PER_BLOCK>>>(device_blockIndexArray);
    writeThreadIndex<<<BLOCK_COUNT, THREADS_PER_BLOCK>>>(device_threadIndexArray);

    cudaMemcpy(host_blockIndexArray, device_blockIndexArray, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_threadIndexArray, device_threadIndexArray, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "block Indices:\n";
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << host_blockIndexArray[i] << " ";
    }
    std::cout << "\n\nthread indices:\n";
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        std::cout << host_threadIndexArray[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(device_blockIndexArray);
    cudaFree(device_threadIndexArray);

    return 0;
}
