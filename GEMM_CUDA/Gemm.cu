#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <stdio.h>

cublasStatus_t Sgemm(
	cublasHandle_t Blas,
	cublasOperation_t AOp, cublasOperation_t BOp,
	const float* dev_A, int WidthA, int HeightA,
	const float* dev_B, int WidthB, int HeightB,
	float *dev_C,
	float Alpha = 1.0f, float Beta = 0.0f)
{
	int lda = WidthA;
	int ldb = WidthB;

	if (AOp != CUBLAS_OP_N) {
		int tmp = WidthA;
		WidthA = HeightA;
		HeightA = tmp;
	}
	if (BOp != CUBLAS_OP_N) {
		int tmp = WidthB;
		WidthB = HeightB;
		HeightB = tmp;
	}
	int m = WidthB;
	int n = HeightA;
	int k = WidthA;

	return cublasSgemm(Blas, BOp, AOp, m, n, k, &Alpha, dev_B, ldb, dev_A, lda, &Beta, dev_C, m);
}


__global__ void kernel_col2im_gpu(float* __restrict__ output, float* __restrict__ input, int N, int K, int P, int Q, int tcount)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= tcount) return;

	int q_idx = tid % Q;// Q 
	int idx = tid / Q;
	int p_idx = idx % P;// P 
	idx /= P;
	int k_idx = idx % K;// K
	int b_idx = idx / K;// N

	int s_idx = b_idx * K * P * Q + k_idx * P * Q + p_idx * Q + q_idx;
	int n_idx = k_idx * N * P * Q + b_idx * P * Q + p_idx * Q + q_idx;

	output[s_idx] = input[n_idx];
}

void conv2d_col2im_gpu(float* output, float* input, int N, int K, int P, int Q, cudaStream_t stream)
{
	int tcount3 = N * K * P * Q;
	int threadX = 512;
	int blockX = (tcount3 + threadX - 1) / threadX;

	dim3 grid(blockX, 1, 1);
	dim3 block(threadX, 1, 1);

	kernel_col2im_gpu << <grid, block, 0, stream >> > (output, input, N, K, P, Q, tcount3);
}


__global__ void kernel_im2col_gpu2(float* __restrict__ output, float* __restrict__ input, int N, int P, int Q, int C, int H, int W, int KH, int KW, int SH, int SW, int left, int top, size_t tcount)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= tcount) return;

	int q_idx = tid % Q;
	int idx = tid / Q;
	int p_idx = idx % P;
	idx /= P;
	int b_idx = idx % N;
	idx /= N;
	int kw_idx = idx % KW;
	idx /= KW;
	int kh_idx = idx % KH;
	idx /= KH;
	int k_idx = kw_idx + kh_idx * KW;
	int c_idx = idx % C;
	int w_idx = q_idx * SW - left + kw_idx;
	int h_idx = p_idx * SH - top + kh_idx;

	int n_index2 = c_idx * P * Q * N * KH * KW
		+ k_idx * P * Q * N
		+ b_idx * P * Q
		+ p_idx * Q
		+ q_idx;

	if (w_idx < 0 || w_idx >= W || h_idx < 0 || h_idx >= H) {
		output[tid] = 0.f;
	}
	else {
		int s_idx = b_idx * C * H * W + c_idx * H * W + h_idx * W + w_idx;
		output[n_index2] = input[s_idx];
	}
}

void conv2d_im2col_gpu2(float* output, float* input, int N, int K, int P, int Q, int C, int H, int W, int KH, int KW, int SH, int SW, int left, int top, cudaStream_t stream)
{
	int tcount2 = C * KH * KW * N * P * Q;
	int threadX2 = 512;
	int blockX2 = (tcount2 + threadX2 - 1) / threadX2;

	dim3 grid(blockX2, 1, 1);
	dim3 block(threadX2, 1, 1);

	kernel_im2col_gpu2 << <grid, block, 0, stream >> > (output, input, N, P, Q, C, H, W, KH, KW, SH, SW, left, top, tcount2);
}


void conv2d_gemm(float* f_output, float* output, float * sgemmout, const float* weight, float* input, int N, int K, int P, int Q, int C, int H, int W, int KH, int KW, int SH, int SW, int left, int top, cudaStream_t stream,
	cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp)
{
	conv2d_im2col_gpu2(output, input, N, K, P, Q, C, H, W, KH, KW, SH, SW, left, top, stream);

	Sgemm(Blas, AOp, BOp, weight, KW * KH * C, K, output, P * Q * N, KH * KW * C, sgemmout);

	conv2d_col2im_gpu(f_output, sgemmout, N, K, P, Q, stream);
}

extern "C" void tt()
{
	int a = 0;
}