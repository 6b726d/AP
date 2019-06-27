#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>

#define TILE_WIDTH 2
#define WIDTH 4

using namespace std;

/*
//MatrixMul_Kernel Algorithm GPU
__global__
void MatrixMulKernel(float* M, float* N, float* P, int Width) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the d_P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;
	// Loop over the M and N tiles required to compute P element
	for (int ph = 0; ph < ceil(Width / (float)TILE_WIDTH); ++ph) {
		// Collaborative loading of M and N tiles into shared memory
		if ((Row < Width) && ((ph*TILE_WIDTH + tx) < Width))
			Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
		if (((ph*TILE_WIDTH + ty) < Width) && (Col < Width))
			Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	if ((Row < Width) && (Col < Width))
		P[Row*Width + Col] = Pvalue;
}
*/

//MatrixMul_Kernel Algorithm GPU
__global__
void MatrixMulKernel(float* M, float* N, float* P, int Width) {
	__shared__ float Mds[TILE_WIDTH][WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the d_P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;
	// Loop over the M and N tiles required to compute P element
	for (int ph = 0; ph < ceil(Width / (float)TILE_WIDTH); ++ph) {
		// Collaborative loading of M and N tiles into shared memory
		if ((Row < Width) && ((ph*WIDTH + tx) < Width))
			Mds[ty][tx] = M[Row*Width + ph*WIDTH + tx];
		if (((ph*TILE_WIDTH + ty) < Width) && (Col < Width))
			Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	if ((Row < Width) && (Col < Width))
		P[Row*Width + Col] = Pvalue;
}

/*MatrixMul_GPU*/
void MatrixMulGPU(float* h_M, float* h_N, float* h_P, int width) {
	int size = width * width * sizeof(float);
	float *d_M;
	float *d_N;
	float *d_P;
	cudaMalloc(&d_M, size);
	cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_N, size);
	cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_P, size);
	cudaMemcpy(d_P, h_P, size, cudaMemcpyHostToDevice);
	dim3 dimGrid(ceil(width / 2.0), ceil(width / 2.0), 1);
	dim3 dimBlock(2.0, 2.0, 1);
	MatrixMulKernel << < dimGrid, dimBlock >> > (d_M, d_N, d_P, width);
	cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
}

int main() {
	float *h_M, *h_N, *h_P;

	int width = 4;

	h_M = (float*)malloc(width*width * sizeof(float));
	h_N = (float*)malloc(width*width * sizeof(float));
	h_P = (float*)malloc(width*width * sizeof(float));

	for (int i = 0; i < width*width; i++) {
		h_M[i] = 2.0f;
		h_N[i] = 3.0f;
		h_P[i] = 0.0f;
	}

	chrono::time_point<chrono::system_clock> MatrixMulGPU_Start, MatrixMulGPU_End;
	MatrixMulGPU_Start = chrono::system_clock::now();
	MatrixMulGPU(h_M, h_N, h_P, width);
	MatrixMulGPU_End = chrono::system_clock::now();

	cout << "MatrixMul_GPU: " << chrono::duration_cast<chrono::nanoseconds>(MatrixMulGPU_End - MatrixMulGPU_Start).count() << "ns." << endl;

	/*Print*/
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			cout << h_P[i*width + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	free(h_M);
	free(h_N);
	free(h_P);

	return 0;
}
