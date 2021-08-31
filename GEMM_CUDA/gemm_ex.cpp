#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include "math.h"
#include <cuda_runtime.h>
#include "cublas_v2.h" 
using namespace std;

typedef char stype;
typedef int dtype;

void printArrayS(dtype *ptr, int rows, int cols, char mode, char *name)
{
	printf("%s\n", name);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (mode == 'N') /* Normal mode */
			{
				if (ptr[i * cols + j] >= 0)
					printf(" %5.f ", (float)ptr[i * cols + j]);
				else
					printf("%5.f ", (float)ptr[i * cols + j]);
			}
			else /* Transpose mode */
			{
				if (ptr[j * rows + i] >= 0)
					printf("%5.f ", (float)ptr[j * rows + i]);
				else
					printf("%5.f ", (float)ptr[j * rows + i]);
			}
		}
		printf("\n");
	}
}
int main(int argc, char* argv[])
{
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Device %d : \"%s\"\n", 0, deviceProp.name);
	int b_idx = 2;

	// Linear dimension of matrices
	int M = 4, N = 2, K = 3;
	int batch_count = 10;
	// Allocate host storage for batch_count A,B,C square matrices

	vector<stype> A = { 7, 8, 9, 10, 
						11, 12, 13, 14, 
						15, 16, 17, 18 };
	vector<stype> B = { 1, 2, 
						3, 4, 
						5, 6 };
	vector<dtype> C(M*N); // init 0

	stype *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, sizeof(stype) * M * K * batch_count);
	cudaMalloc(&d_B, sizeof(stype) * K * N * batch_count);
	cudaMalloc(&d_C, sizeof(dtype) * M * N * batch_count);
	cudaMemcpy(d_A + b_idx * M*K * sizeof(stype), A.data(), sizeof(stype) * M * K, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B + b_idx * K*N * sizeof(stype), B.data(), sizeof(stype) * K * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C + b_idx * M*N * sizeof(dtype), C.data(), sizeof(dtype) * M * N, cudaMemcpyHostToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);
	// CUBLAS_COMPUTE_16F CUBLAS_COMPUTE_32F CUBLAS_COMPUTE_32I
	// CUDA_R_8I CUDA_R_8U CUDA_R_32I CUDA_R_32F CUDA_R_16F CUDA_R_16BF
	dtype alpha = 1, beta = 0;

	cublasStatus_t ret = cublasGemmStridedBatchedEx(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		N, M, K,
		&alpha,
		d_B, CUDA_R_8I, N, K*N,
		d_A, CUDA_R_8I, K, M*K,
		&beta,
		d_C, CUDA_R_32I, N, N*M,
		batch_count,
		CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);

	printf("cublasStatus_t : %d\n", ret);

	cudaMemcpy(C.data(), d_C + b_idx * M*N * sizeof(dtype), sizeof(dtype) * M * N, cudaMemcpyDeviceToHost);
	// Destroy the handle
	cublasDestroy(handle);

	//printArrayS(A.data(), M, K, 'N', "A");
	//printArrayS(B.data(), K, N, 'N', "B");
	printArrayS(C.data(), M, N, 'N', "C");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
