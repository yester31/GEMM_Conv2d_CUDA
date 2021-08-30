#pragma once
#include <vector>
#include <iostream>
#include <helper_cuda.h> 
#include <iomanip>

using namespace std;
using namespace chrono;

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
	cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp, float Alpha = 1.0f, float Beta = 0.0f);


void convolution(vector<float>& convOutput, vector<float>& convInput, vector<float>& kernel, int kernelSize, int stride, int input_n, int input_c, int input_h, int input_w, int ouput_c) {
	int outputHeightSize = ((input_h - kernelSize) / stride) + 1;
	int outputWidthSize = ((input_w - kernelSize) / stride) + 1;
	//Conv_output.resize(input_n * Ouput_C * outputHeightSize * outputHeightSize);
	//cout << "===== Convolution ===== \n";

	int temp1i = input_h * input_w * input_c;
	int temp1o = outputHeightSize * outputWidthSize * ouput_c;
	int temp1k = kernelSize * kernelSize * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2i = ⁠n_idx * temp1i;
		int temp2o = ⁠n_idx * temp1o;
		for (int k_idx = 0; k_idx < ouput_c; k_idx++)
		{
			int temp2k = k_idx * temp1k;
			int temp3o = k_idx * outputHeightSize * outputWidthSize + temp2o;
			for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
			{
				int temp3i = ⁠c_idx * input_w * input_h + temp2i;
				int temp3k = ⁠c_idx * kernelSize * kernelSize + temp2k;
				for (int rowStride = 0; rowStride < outputHeightSize; rowStride++) {
					int temp4o = rowStride * outputWidthSize + temp3o;
					for (int colStride = 0; colStride < outputWidthSize; colStride++) {

						float sum = 0;
						int g_idx_o = colStride + temp4o;

						for (int x = rowStride * stride; x < rowStride * stride + kernelSize; x++) {
							int temp4i = x * input_w + temp3i;
							int temp4k = (x - rowStride * stride) * kernelSize + temp3k;
							for (int y = colStride * stride; y < colStride * stride + kernelSize; y++) {
								int ⁠g_idx_i = y + temp4i;
								int g_idx_k = (y - colStride * stride) + temp4k;
								sum += convInput[⁠g_idx_i] * kernel[g_idx_k];
							}
						}

						convOutput[g_idx_o] += sum;
					}
				}
			}
		}
	}
}

void zeroPadding(vector<float>& zeroPaddingOutput, vector<float>& zeroPaddingInput, int input_n, int input_c, int input_h, int input_w, int leftPadingSize, int rightPadingSize, int topPadingSize, int bottomPadingSize) {

	int temp1 = input_w * input_h * input_c;
	int temp1o = (input_h + topPadingSize + bottomPadingSize) * (input_w + leftPadingSize + rightPadingSize) * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		int temp2o = ⁠n_idx * temp1o;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			int temp3o = ⁠c_idx * (input_w + leftPadingSize + rightPadingSize) * (input_h + topPadingSize + bottomPadingSize) + temp2o;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				int temp4o = (⁠h_idx + topPadingSize) * (input_w + leftPadingSize + rightPadingSize) + leftPadingSize + temp3o;

				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int ⁠g_idx = w_idx + temp4;
					int g_idx_Output = w_idx + temp4o;
					zeroPaddingOutput[g_idx_Output] = zeroPaddingInput[⁠g_idx];
				}
			}
		}
	}
}

void deviceQuery()
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}
	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}
	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("Device %d : \"%s\"\n", dev, deviceProp.name);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("  Multiprocessors (MP) :                         %d\n", deviceProp.multiProcessorCount);
		printf("  CUDA Cores/MP :                                %d\n", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
		printf("  CUDA Cores :                                   %d\n", _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		printf("  Total amount of shared memory per block:       %llu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
	}
	printf("\n");
}

void valueCheck(vector<float>& valueCheckInput, int input_n, int input_c, int input_h, int input_w, int offset = 0) {
	if (offset == 1) { input_n = 1; }

	int temp1 = input_w * input_h * input_c;
	for (int ⁠n_idx = 0; ⁠n_idx < input_n; ⁠n_idx++)
	{
		int temp2 = ⁠n_idx * temp1;
		for (int ⁠c_idx = 0; ⁠c_idx < input_c; ⁠c_idx++)
		{
			int temp3 = ⁠c_idx * input_w * input_h + temp2;
			for (int ⁠h_idx = 0; ⁠h_idx < input_h; ⁠h_idx++)
			{
				int temp4 = ⁠h_idx * input_w + temp3;
				std::cout << "  ";
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int g_idx = w_idx + temp4;

					//cout.setf(ios::fixed);
					//cout.precision(6);
					std::cout << std::setw(8) << valueCheckInput[g_idx] << " ";
				}std::cout << std::endl;
			}std::cout << std::endl << std::endl;
		}std::cout << std::endl;
	}std::cout << std::endl;
}

void inititalizedData(vector<float>& container)
{
	int count = 1;
	for (vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
		count++;
	}
}

void inititalizedDataOne(vector<float>& container)
{
	int count = 1;
	for (vector<int>::size_type i = 0; i < container.size(); i++) {
		container[i] = count;
	}
}


