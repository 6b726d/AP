#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>

using namespace std;

/*MatVecMul_Kernel*/
__global__
void MatVecMul_Kernel(float* A, float* B, float* C, int n) {
	int i = threadIdx.x;
	int offset;
	float sum = 0;
	if (i < n) {
		for (int j = 0; j < n; j++) {
			offset = i*n + j;
			sum += A[offset] * B[j];
		}
		C[i] = sum;
	}
}

/*MatVecMul_GPU*/
void MatVecMul_GPU(float* h_A, float* h_B, float* h_C, int n) {
	int sizeM = n*n * sizeof(float);
	int sizeV = n * sizeof(float);
	float *d_A;
	float *d_B;
	float *d_C;
	cudaMalloc(&d_A, sizeM);
	cudaMemcpy(d_A, h_A, sizeM, cudaMemcpyHostToDevice);
	cudaMalloc(&d_B, sizeV);
	cudaMemcpy(d_B, h_B, sizeV, cudaMemcpyHostToDevice);
	cudaMalloc(&d_C, sizeV);
	cudaMemcpy(d_C, h_C, sizeV, cudaMemcpyHostToDevice);
	//dim3 dimGrid(ceil(n / 32.0), 1, 1);
	//dim3 dimBlock(32.0, 1, 1);
	MatVecMul_Kernel <<< 1, 10 >>> (d_A, d_B, d_C, n);
	cudaMemcpy(h_C, d_C, sizeV, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main() {
	//Host Matrix
	float *h_A, *h_B, *h_C;

	int n = 10;

	h_A = (float*)malloc(n*n * sizeof(float));
	h_B = (float*)malloc(n * sizeof(float));
	h_C = (float*)malloc(n * sizeof(float));

	//Create Matrix
	for (int i = 0; i < n*n; i++) {
		h_A[i] = 1.0;
	}

	//Create Vector
	for (int i = 0; i < n; i++) {
		h_B[i] = 1.0;
		h_C[i] = 1.0;
	}

	//MatVecMul (Main)
	chrono::time_point<chrono::system_clock> MatVecMul_GPU_Start, MatVecMul_GPU_End;
	MatVecMul_GPU_Start = chrono::system_clock::now();
	MatVecMul_GPU(h_A, h_B, h_C, n);
	MatVecMul_GPU_End = chrono::system_clock::now();

	cout << "MatVecMul_GPU: " << chrono::duration_cast<chrono::nanoseconds>(MatVecMul_GPU_End - MatVecMul_GPU_Start).count() << "ns." << endl;

	//Print MatVecMul
	for (int i = 0; i < n; i++) {
			cout << h_C[i] << " ";
	}
	cout << endl;

	//Free
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
