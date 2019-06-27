#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>

//#define SIZE 32

using namespace std;
/*
//F 5.13
__global__
void Sum1_Kernel(float* X, float *Y) {
	__shared__ float partialSum[SIZE];
	partialSum[threadIdx.x] = X[blockIdx.x*blockDim.x + threadIdx.x];
	unsigned int t = threadIdx.x;
	for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		if (t % (2 * stride) == 0) {
			partialSum[t] += partialSum[t + stride];
		}
	}
	if (t == 0) {
		Y[blockIdx.x] = partialSum[0];
	}
}

//F 5.15
__global__
void Sum2_Kernel(float* X, float* Y) {
	__shared__ float partialSum[SIZE];
	partialSum[threadIdx.x] = X[blockIdx.x*blockDim.x + threadIdx.x];
	unsigned int t = threadIdx.x;
	for (unsigned int stride = blockDim.x / 2; stride >= 1; stride = stride >> 1) {
		__syncthreads();
		if (t < stride) {
			partialSum[t] += partialSum[t + stride];
		}
	}
	if (t == 0) {
		Y[blockIdx.x] = partialSum[0];
	}
}

//E 5.12
__global__
void Sum3_Kernel(float* X, float* Y) {
	__shared__ float partialSum[SIZE];
	partialSum[threadIdx.x] = X[blockIdx.x*blockDim.x + threadIdx.x];
	unsigned int tid = threadIdx.x;
	for (unsigned int stride = n>>1; stride >=32; stride >>= 1) {
		__syncthreads();
		if (tid == stride) {
			partialSum[tid] += partialSum[tid + stride];
		}
	}
	__syncthreads();
	if (tid < 32) { // unroll last 5 predicated steps
		partialSum[tid] += partialSum[tid + 16];
		partialSum[tid] += partialSum[tid + 8];
		partialSum[tid] += partialSum[tid + 4];
		partialSum[tid] += partialSum[tid + 2];
		partialSum[tid] += partialSum[tid + 1];
	}
	if (tid == 0) {
		Y[blockIdx.x] = partialSum[0];
	}
}
*/

// E 5.1
__global__
void Sum1_Kernel(float* X, float *Y, int size) {
	extern __shared__ float partialSum[];
	unsigned int t = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	partialSum[t] = 0;
	if (i < size) {
		partialSum[t] = X[i];
	}
	__syncthreads();
	for (unsigned int stride = 1; stride < 2048; stride <<= 2) {
		if (t % (2 * stride) == 0) {
			partialSum[t] += partialSum[t + stride];
		}
		__syncthreads();
	}
	if (t == 0) {
		Y[blockIdx.x] = partialSum[0];
	}
}

// E 5.1
__global__
void Sum2_Kernel(float* X, float* Y, int size) {
	extern __shared__ float partialSum[];
	unsigned int t = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	partialSum[t] = 0;
	if (i < size) {
		partialSum[t] = X[i];
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
		if (t < stride) {
			partialSum[t] += partialSum[t + stride];
		}
		__syncthreads();
	}
	if (t == 0) {
		Y[blockIdx.x] = partialSum[0];
	}
}

// E 5.3
__global__
void Sum3_Kernel(float* X, float* Y, int size) {
	extern __shared__ float partialSum[];
	unsigned int t = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	partialSum[t] = 0;
	if (i < size) {
		partialSum[t] = X[i] + X[i + blockDim.x];
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
		if (t < stride) {
			partialSum[t] += partialSum[t + stride];
		}
		__syncthreads();
	}
	if (t == 0) {
		Y[blockIdx.x] = partialSum[0];
	}
}

//E 5.12
__global__
void Sum4_Kernel(float* X, float* Y, int size) {
	extern __shared__ float partialSum[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	partialSum[tid] = 0;
	if (i < size) {
		partialSum[tid] = X[i] + X[i + blockDim.x];
	}
	__syncthreads();
	for (unsigned int stride = blockDim.x/2; stride > 32; stride >>= 1) {
		if (tid == stride) {
			partialSum[tid] += partialSum[tid + stride];
		}
		__syncthreads();
	}
	if (tid < 32) {
		partialSum[tid] += partialSum[tid + 32];
		partialSum[tid] += partialSum[tid + 16];
		partialSum[tid] += partialSum[tid + 8];
		partialSum[tid] += partialSum[tid + 4];
		partialSum[tid] += partialSum[tid + 2];
		partialSum[tid] += partialSum[tid + 1];
	}
	if (tid == 0) {
		Y[blockIdx.x] = partialSum[0];
	}
}

float Sum_GPU(float* x, int x_sz) {
	float total_sum = 0;
	int block_sz = 1024;
	int max_sz = block_sz;
	int grid_sz = 0;
	if (x_sz <= max_sz) {
		grid_sz = (int)ceil(float(x_sz) / float(max_sz));
	}
	else {
		grid_sz = x_sz / max_sz;
		if ((x_sz % max_sz) != 0) {
			grid_sz++;
		}
	}
	float *d_block_sums;
	cudaMalloc(&d_block_sums, sizeof(float)*grid_sz);
	cudaMemset(d_block_sums, 0, sizeof(float)*grid_sz);
	Sum4_Kernel <<< grid_sz, block_sz, sizeof(float)*max_sz >>> (x, d_block_sums, x_sz);
	if (grid_sz <= max_sz) {
		float* d_total_sum;
		cudaMalloc(&d_total_sum, sizeof(float));
		cudaMemset(d_total_sum, 0, sizeof(float));
		Sum4_Kernel <<< 1, block_sz, sizeof(float)*max_sz >>> (d_total_sum, d_block_sums, grid_sz);
		cudaMemcpy(&total_sum, d_total_sum, sizeof(float), cudaMemcpyDeviceToHost);
		cudaFree(d_total_sum);
	}
	else {
		float* d_in_block_sums;
		cudaMalloc(&d_in_block_sums, sizeof(float)*grid_sz);
		cudaMemcpy(&d_in_block_sums, d_block_sums, sizeof(float)*grid_sz, cudaMemcpyDeviceToDevice);
		total_sum = Sum_GPU(d_in_block_sums, grid_sz);
		cudaFree(d_in_block_sums);
	}
	cudaFree(d_block_sums);
	return total_sum;
}

int main() {
	//Host
	float *h_X;

	int size = 1024;

	h_X = (float*)malloc(size * sizeof(float));

	//Create
	for (int i = 0; i < size; i++) {
		h_X[i] = i + 1.0f;
	}

	float* d_X;
	cudaMalloc(&d_X, sizeof(unsigned int) * size);
	cudaMemcpy(d_X, h_X, sizeof(unsigned int) * size, cudaMemcpyHostToDevice);

	//Sum (Main)
	chrono::time_point<chrono::system_clock> Sum_GPU_Start, Sum_GPU_End;
	Sum_GPU_Start = chrono::system_clock::now();
	float gpu_total_sum = Sum_GPU(d_X, size);
	Sum_GPU_End = chrono::system_clock::now();

	cout << "Sum_GPU: " << chrono::duration_cast<chrono::nanoseconds>(Sum_GPU_End - Sum_GPU_Start).count() << "ns." << endl;

	cout << "Result: " << gpu_total_sum << endl;

	//Free
	cudaFree(d_X);
	free(h_X);

	return 0;
}
