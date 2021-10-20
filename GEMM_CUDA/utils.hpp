#pragma once

#if defined(NDEBUG)
#define CUDA_CHECK(x) (x) // release mode
#else //debug mode
#define CUDA_CHECK(x) do{\
		(x); \
cudaError_t e = cudaGetLastError();\
if(cudaSuccess != e){\
	printf("cuda failure \n *** %s ***\n at %s :: line_  %d\n",\
			cudaGetErrorString(e), \
			__FILE__, __LINE__); \
		exit(1);\
	}\
}while(0)
#endif

cublasStatus_t Sgemm(cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp, const float* dev_A, int WidthA, int HeightA, const float* dev_B, int WidthB, int HeightB, float *dev_C, float Alpha = 1.0f, float Beta = 0.0f);

void conv2d_im2col_gpu2(float* output, float* input, int N, int K, int P, int Q, int C, int H, int W, int KH, int KW, int SH, int SW, int left, int top, cudaStream_t stream);

void conv2d_col2im_gpu(float* output, float* input, int N, int K, int P, int Q, cudaStream_t stream);

void conv2d_gemm(float* f_output, float* output, float * sgemmout, const float* weight, float* input, int N, int K, int P, int Q, int C, int H, int W, int KH, int KW, int SH, int SW, int left, int top, cudaStream_t stream,
	cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp);

void convolution(std::vector<float>& convOutput, std::vector<float>& convInput, std::vector<float>& kernel, int kernelSize, int stride, int input_n, int input_c, int input_h, int input_w, int ouput_c);

void zeroPadding(std::vector<float>& zeroPaddingOutput, std::vector<float>& zeroPaddingInput, int input_n, int input_c, int input_h, int input_w, int leftPadingSize, int rightPadingSize, int topPadingSize, int bottomPadingSize);

void deviceQuery();

void valueCheck(std::vector<float>& valueCheckInput, int input_n, int input_c, int input_h, int input_w, int offset =0);

void inititalizedData(std::vector<float>& container);

void inititalizedDataOne(std::vector<float>& container);


