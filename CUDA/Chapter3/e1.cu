#include <cuda.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <chrono>

using namespace std;

/*B*/
__global__
void MatrixAddB(float* A, float* B, float* C, int n) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i < n*n) {
		C[i] = A[i] + B[i];
	}
}

/*C=>Row*/
__global__
void MatrixAddC(float* A, float* B, float* C, int n) {
	int i = threadIdx.x;
	int offset;
	if (i < n) {
		for (int j = 0; j < n; j++) {
			offset = i*n + j;
			C[offset] = A[offset] + B[offset];
		}
	}
}

/*D=>Column*/
__global__
void MatrixAddD(float* A, float* B, float* C, int n) {
	int i = threadIdx.x;
	int offset;
	if (i < n) {
		for (int j = 0; j < n; j++) {
			offset = j*n + i;
			C[offset] = A[offset] + B[offset];
		}
	}
}

/*AB*/
void MatrixAddAB(float* h_A, float* h_B, float* h_C, int n) {
	int size = n*n * sizeof(float);
	float *d_A;
	float *d_B;
	float *d_C;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_B, size);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_C, size);
	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
	//dim3 dimGrid(ceil(n / 32.0), 1, 1);
	//dim3 dimBlock(32.0, 1, 1);
	MatrixAddB <<< 10, 10>>> (d_A, d_B, d_C, n);
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

/*AC*/
void MatrixAddAC(float* h_A, float* h_B, float* h_C, int n) {
	int size = n*n * sizeof(float);
	float *d_A;
	float *d_B;
	float *d_C;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_B, size);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_C, size);
	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
	//dim3 dimGrid(ceil(n / 32.0), 1, 1);
	//dim3 dimBlock(32.0, 1, 1);
	MatrixAddC <<< 1, 10 >>> (d_A, d_B, d_C, n);
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

/*AD*/
void MatrixAddAD(float* h_A, float* h_B, float* h_C, int n) {
	int size = n*n * sizeof(float);
	float *d_A;
	float *d_B;
	float *d_C;
	cudaMalloc(&d_A, size);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_B, size);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	cudaMalloc(&d_C, size);
	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
	//dim3 dimGrid(ceil(n / 32.0), 1, 1);
	//dim3 dimBlock(32.0, 1, 1);
	MatrixAddD <<< 1, 10 >>> (d_A, d_B, d_C, n);
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main() {
	//Host Matrix
	float *h_A0, *h_B0, *h_C0;
	float *h_A1, *h_B1, *h_C1;
	float *h_A2, *h_B2, *h_C2;

	int n = 10;

	h_A0 = (float*)malloc(n*n * sizeof(float));
	h_B0 = (float*)malloc(n*n * sizeof(float));
	h_C0 = (float*)malloc(n*n * sizeof(float));

	h_A1 = (float*)malloc(n*n * sizeof(float));
	h_B1 = (float*)malloc(n*n * sizeof(float));
	h_C1 = (float*)malloc(n*n * sizeof(float));

	h_A2 = (float*)malloc(n*n * sizeof(float));
	h_B2 = (float*)malloc(n*n * sizeof(float));
	h_C2 = (float*)malloc(n*n * sizeof(float));

	//Create Matrix
	for (int i = 0; i < n*n; i++) {
		h_A0[i] = 1.0;
		h_B0[i] = 1.0;
		h_C0[i] = 1.0;
		h_A1[i] = 1.0;
		h_B1[i] = 1.0;
		h_C1[i] = 1.0;
		h_A2[i] = 1.0;
		h_B2[i] = 1.0;
		h_C2[i] = 1.0;
	}

	//B (Main)
	chrono::time_point<chrono::system_clock> B_GPU_Start, B_GPU_End;
	B_GPU_Start = chrono::system_clock::now();
	MatrixAddAB(h_A0, h_B0, h_C0, n);
	B_GPU_End = chrono::system_clock::now();

	cout << "B_GPU: " << chrono::duration_cast<chrono::nanoseconds>(B_GPU_End - B_GPU_Start).count() << "ns." << endl;

	//Print B
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << h_C0[i*n + j] << " ";
		}
		cout << endl;
	}

	//C (Main)
	chrono::time_point<chrono::system_clock> C_GPU_Start, C_GPU_End;
	C_GPU_Start = chrono::system_clock::now();
	MatrixAddAC(h_A1, h_B1, h_C1, n);
	C_GPU_End = chrono::system_clock::now();

	cout << "C_GPU: " << chrono::duration_cast<chrono::nanoseconds>(C_GPU_End - C_GPU_Start).count() << "ns." << endl;

	//Print C
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << h_C1[i*n + j] << " ";
		}
		cout << endl;
	}

	//D (Main)
	chrono::time_point<chrono::system_clock> D_GPU_Start, D_GPU_End;
	D_GPU_Start = chrono::system_clock::now();
	MatrixAddAD(h_A2, h_B2, h_C2, n);
	D_GPU_End = chrono::system_clock::now();

	cout << "D_GPU: " << chrono::duration_cast<chrono::nanoseconds>(D_GPU_End - D_GPU_Start).count() << "ns." << endl;

	//Print D
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << h_C2[i*n + j] << " ";
		}
		cout << endl;
	}

	//Free
	free(h_A0);
	free(h_B0);
	free(h_C0);
	free(h_A1);
	free(h_B1);
	free(h_C1);
	free(h_A2);
	free(h_B2);
	free(h_C2);

	return 0;
}
