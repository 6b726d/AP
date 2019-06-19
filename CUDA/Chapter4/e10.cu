#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>

/*1-20*/
#define BLOCK_WIDTH 2
#define BLOCK_SIZE 4

using namespace std;

/*
//BlockTranspose
__global__
void BlockTranspose(float *A_elements, int A_width, int A_height) {
	__shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
	int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;
	blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
	A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
*/

/*BlockTranspose_Kernel*/
__global__
void BlockTranspose_Kernel(float *A_elements, int A_width, int A_height) {
	__shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];
	int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;
	blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
	__syncthreads();
	A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}

/*BlockTranspose_GPU*/
void BlockTranspose_GPU(float* h_A, int A_width, int A_height) {
	int size = A_width * A_height * sizeof(float);
	float *d_A;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 gridDim(A_width / blockDim.x, A_height / blockDim.y);
	BlockTranspose_Kernel <<< gridDim, blockDim >>> (h_A, A_width, A_height);
	cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
}

int main() {
	//Host
	float *h_A;

	int A_width = 8;
	int A_height = 8;

	h_A = (float*)malloc(A_width*A_height * sizeof(float));

	//Create
	for (int i = 0; i < A_width*A_height; i++) {
		h_A[i] = i + 1.0f;
	}

	//Print BlockTranspose
	for (int i = 0; i < A_height; i++) {
		for (int j = 0; j < A_width; j++) {
			cout << h_A[i*A_width + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	//BlockTranspose (Main)
	chrono::time_point<chrono::system_clock> BlockTranspose_GPU_Start, BlockTranspose_GPU_End;
	BlockTranspose_GPU_Start = chrono::system_clock::now();
	BlockTranspose_GPU(h_A, A_width, A_height);
	BlockTranspose_GPU_End = chrono::system_clock::now();

	cout << "BlockTranspose_GPU: " << chrono::duration_cast<chrono::nanoseconds>(BlockTranspose_GPU_End - BlockTranspose_GPU_Start).count() << "ns." << endl;

	//Print BlockTranspose
	for (int i = 0; i < A_height; i++) {
		for (int j = 0; j < A_width; j++) {
			cout << h_A[i*A_width + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	//Free
	free(h_A);

	return 0;
}
