#include "image2d.h"
#include <cuda.h>
#include <stdio.h>
#include <string>
#include <iostream>

#define BLOCKSIZE 16

__global__ void CUDAKernel(int iterations, float xmin, float xmax, float ymin, float ymax, 
                           float *pOutput, int outputW, int outputH)
{
	// TODO: your code
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < outputW && y < outputH) {
        //mapping pixel to complex number
        float real = xmin + (x / (float)outputW) * (xmax - xmin);
        float imag = ymin + (y / (float)outputH) * (ymax - ymin);
        
		//init complex number
		float zReal = 0.0f;
		float zImag = 0.0f;
		int Q = 0;

		while (Q < iterations) {
			float zReal2 = zReal * zReal;
			float zImag2 = zImag * zImag;

			if (zReal2 + zImag2 > 4.0f) {
				break;
			}

			// zn+1 = zn * zn + c
			// formulas used form: https://www.cuemath.com/algebra/square-root-of-complex-number/
			zImag = 2.0f * zReal * zImag + imag;
			zReal = zReal2 - zImag2 + real;
			Q++;
		}

		float color;
		if (Q == iterations) {
			color = 0;
		} else {
			color = 255.0f * sqrt(Q / (float) iterations);
		}
        // float grayscale = (float)x / outputW * 255;
        pOutput[y * outputW + x] = color;
    }
}

// If an error occurs, return false and set a description in 'errStr'
bool cudaFractal(int iterations, float xmin, float xmax, float ymin, float ymax, 
                 Image2D &output, std::string &errStr)
{
	// We'll use an image of 512 pixels wide
	int ho = 512;
	int wo = ho * 3 / 2;
	output.resize(wo, ho);

	// And divide this in a number of blocks
	size_t xBlockSize = BLOCKSIZE;
	size_t yBlockSize = BLOCKSIZE;
	size_t numXBlocks = (wo/xBlockSize) + (((wo%xBlockSize) != 0)?1:0);
	size_t numYBlocks = (ho/yBlockSize) + (((ho%yBlockSize) != 0)?1:0);

	cudaError_t err;
	float *pDevOutput;

	// TODO: allocate memory on GPU
	cudaMalloc((void**)&pDevOutput, wo * ho * sizeof(float));
	// found out there are dimX variables to init dimentions
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize((wo + BLOCKSIZE - 1) / BLOCKSIZE, (ho + BLOCKSIZE - 1) / BLOCKSIZE);


	cudaEvent_t startEvt, stopEvt; // We'll use cuda events to time everything
	cudaEventCreate(&startEvt);
	cudaEventCreate(&stopEvt);

	cudaEventRecord(startEvt);

	// TODO: call kernel
	CUDAKernel<<<gridSize, blockSize>>>(iterations, xmin, xmax, ymin, ymax, pDevOutput, wo, ho);

	cudaEventRecord(stopEvt);
	
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "CUDA convolution kernel execution error code: " << err << std::endl;
	}

	// TODO: copy data back and free memory
	//       Note that output.getBufferPointer() gives direct access to the
	//       floating point array in the image
	cudaMemcpy(output.getBufferPointer(), pDevOutput, wo * ho * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(pDevOutput);

	float elapsed;
	cudaEventElapsedTime(&elapsed, startEvt, stopEvt);

	std::cout << "CUDA time elapsed: " << elapsed << " milliseconds" << std::endl;

	cudaEventDestroy(startEvt);
	cudaEventDestroy(stopEvt);

	return true;
}
