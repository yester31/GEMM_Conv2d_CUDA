#include <vector>
#include <fstream>
#include <chrono>
#include <random>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <helper_cuda.h> 
using namespace std;
using namespace chrono;

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



cublasStatus_t Sgemm( cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp, const float* dev_A, int WidthA, int HeightA, const float* dev_B, int WidthB, int HeightB, float *dev_C, float Alpha = 1.0f, float Beta = 0.0f);

void conv2d_im2col_gpu2(float* output, float* input, int N, int K, int P, int Q, int C, int H, int W, int KH, int KW, int SH, int SW, int left, int top, cudaStream_t stream);

void conv2d_col2im_gpu(float* output, float* input, int N, int K, int P, int Q, cudaStream_t stream);

void conv2d_gemm(float* f_output, float* output, float * sgemmout, const float* weight, float* input, int N, int K, int P, int Q, int C, int H, int W, int KH, int KW, int SH, int SW, int left, int top, cudaStream_t stream,
	cublasHandle_t Blas, cublasOperation_t AOp, cublasOperation_t BOp, float Alpha = 1.0f, float Beta = 0.0f);


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
				cout << "  ";
				for (int w_idx = 0; w_idx < input_w; w_idx++)
				{
					int g_idx = w_idx + temp4;

					//cout.setf(ios::fixed);
					//cout.precision(6);
					cout << setw(8) << valueCheckInput[g_idx] << " ";
				}cout << endl;
			}cout << endl; cout << endl;
		}cout << endl;
	}cout << endl;
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


typedef struct {
	int N;
	int C, H, W; // data [N,C,H,W]
	int K, P, Q; // output [N,K,P,Q]
	int KH, KW; // weight height, width
	int SH, SW;
	int left, right, top, bottom; // pad left, right, top, bottom
} Config;
vector<Config> config{
	//{10,10,10,10,    10, 8, 10,		3, 3,  1,1,		0,0,0,0 } 
	{3,128,240,240,  64, 240, 240,	3, 3,  1,1,		1,1,1,1}
	//{1,8,23,23,  4,23,23,  5,5, 1,1,  2,2,2,2 }
	//{1,128,240,240,  64, 240, 240,  5, 5, 1, 1,   2,2,2,2}
	//{1,128,240,240,  64, 236, 236,  5, 5, 1, 1,   0,0,0,0}
	//{1,2,6,6,         2, 6, 6,		3, 3,  1,1,		1,1,1,1 }
	//{3,3,6,6,         4, 6, 6,		3, 3,  1,1,		1,1,0,0 }
	//{3,2,16,16,       2, 15, 15,		3, 3,  1,1,		1,0,1,0 }
	//{2,3,4,4,         4, 2, 2,		3, 3,  1,1,		0,0,0,0 }
	//{3,3,6,6,	        4, 4, 4,		3, 3,  1,1,		0,0,0,0 }
	//{3,2,16,16,       2, 14, 14,		3, 3,  1,1,		1,1,1,1 }
	//{2,3,40,40,       2, 40, 40,		3, 3,  1,1,		0,0,0,0 }
	// {3, 3, 40, 40,  4, 40, 40,  2, 2,  1,1,  2,2,2,2 }//(o)
	//{3, 3, 40, 40,  4, 40, 40,  2, 2,  1,1,  0,0,0,0 }//(o)
	//{3, 3, 40, 40,  4, 40, 40,  3, 3,  1,1,  0,0,0,0 }//(o)

	//{3, 3, 40, 40,  4, 40, 40,  4, 4,  2,2,  2,2,2,2 }//(x)
	//{3, 3, 96, 96,  4, 46, 46,  9, 9,  3,3,  3,0,3,0 }//(o)

	//{3,2,6,6,  2, 5, 5,  3, 3,  1,1,  1,0,1,1 }

	//c.P = ((c.H + c.top + c.bottom - c.KH) / c.SH) + 1;

};


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


int main(int argc, char** argv) {

	//deviceQuery();
	cudaSetDevice(0);
	cudaStream_t stream;
	cublasHandle_t cublasHandle;

	int status;
	status = (int)cudaStreamCreate(&stream);
	status |= (int)cublasCreate(&cublasHandle);
	status |= (int)cublasSetStream(cublasHandle, stream);
	printf("status=%d CUDA Setup Done!\n\n", status);

	Config c = config[0];

	c.P = ((c.H + c.top + c.bottom - c.KH) / c.SH) + 1;
	c.Q = ((c.W + c.left + c.right - c.KW) / c.SW) + 1;

	printf(" input[%4d,%4d,%4d,%4d] kernel[%4d,%4d,%4d,%4d] output[%4d,%4d,%4d,%4d]\n", c.N, c.C, c.H, c.W, c.K, c.C, c.KH, c.KW, c.N, c.K, c.P, c.Q);


	vector<float> data(c.N*c.C*c.H*c.W);     // input data [N,C,H,W]
	vector<float> weight(c.K*c.C*c.KH*c.KW); // weight [K,C,3,3]


	inititalizedData(data);			//  1씩 증가하는 등차수열 
	inititalizedData(weight);		//  1씩 증가하는 등차수열 

	//cout << "Data(Input)" << endl;
	//valueCheck(data, c.N, c.C, c.H, c.W);		//입력값 확인
	//cout << "kernel" << endl;
	//valueCheck(weight, c.K, c.C, c.KH, c.KW);	// 가중치 확인
	//valueCheck(weight, 1, 1, c.K, c.C * c.KH * c.KW);	// 가중치 확인
	

	float* d_weight;	// device input data
	status |= (int)cudaMalloc(&d_weight, weight.size() * sizeof(float));
	status |= (int)cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice);

	float* d_data;	// device input data
	status |= (int)cudaMalloc(&d_data, data.size() * sizeof(float));
	status |= (int)cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);


	vector<float> data_output(c.C * c.KH * c.KW * c.N * c.P * c.Q); // im2col 결과 가져오기
	//cout << "data out size :: " << data_output.size() << endl;

	float* d_data_output;	// device output data
	//status |= (int)cudaMalloc(&d_data_output, data_output.size() * sizeof(float));
	CUDA_CHECK(cudaMalloc(&d_data_output, data_output.size() * sizeof(float)));


	vector<float> m_output(c.K * c.P * c.Q * c.N); // sgemm 결과 
	float* d_m_output;	// device output data
	status |= (int)cudaMalloc(&d_m_output, m_output.size() * sizeof(float));

	//vector<float> m_output2(c.K * c.P * c.Q * c.N); // sgemm 결과 
	//float* d_m_output2;	// device output data
	//status |= (int)cudaMalloc(&d_m_output2, m_output2.size() * sizeof(float));
	//cout << status << endl;

	float* d_f_m_output;	// device output data
	status |= (int)cudaMalloc(&d_f_m_output, m_output.size() * sizeof(float));

	//cout << status << endl;



	int ITER = 200;
	uint64_t total_time = 0;
	for (int iIdx = 0; iIdx < ITER; iIdx++) {
		uint64_t start_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();

		// 2. im2col 
		//conv2d_im2col_gpu2(d_data_output, d_data, c.N, c.K, c.P, c.Q, c.C, c.H, c.W, c.KH, c.KW, c.SH, c.SW, c.left, c.top, stream);
		// 3. sgemm
		//Sgemm(cublasHandle, (cublasOperation_t)0, (cublasOperation_t)0, d_weight, c.KW * c.KH * c.C, c.K, d_data_output, c.P * c.Q * c.N, c.KH * c.KW * c.C, d_m_output);
		// 4. col2im
		//conv2d_col2im_gpu(d_f_m_output, d_m_output, c.N, c.K, c.P, c.Q, stream);

		// Gemm_func(im2col + sgemm + col2im) 
		conv2d_gemm(d_f_m_output, d_data_output, d_m_output, d_weight, d_data, c.N, c.K, c.P, c.Q, c.C, c.H, c.W, c.KH, c.KW, c.SH, c.SW, c.left, c.top, stream, cublasHandle, (cublasOperation_t)0, (cublasOperation_t)0);

		status |= (int)cudaStreamSynchronize(stream);

		total_time += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;

	}
	
	status |= (int)cudaMemcpy(m_output.data(), d_f_m_output, m_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

	//cout << status << endl;

	double checksum = 0;
	for (auto d : m_output) checksum += fabs((double)d);
	printf("   Gemm       status=%d avg_dur_time=%6.3f[msec] checksum=%.6f\n", status, total_time / 1000.f / ITER, checksum);
	
	
	//status |= (int)cudaMemcpy(m_output2.data(), d_m_output, m_output2.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//double checksum2 = 0;
	//for (auto d : m_output2) checksum2 += fabs((double)d);
	//printf("   Gemm       status=%d avg_dur_time=%6.3f[msec] checksum=%.6f\n", status, total_time / 1000.f / ITER, checksum2);
	

	//status |= (int)cudaMemcpy(weight.data(), d_weight, weight.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//cout << "weight 재확인" << endl;
	//valueCheck(weight, 1, 1, c.K , c.KH * c.KW * c.C);		//d_weight 값 확인

	//status |= (int)cudaMemcpy(data_output.data(), d_data_output, data_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//cout << "Im2col output" << endl;
	//valueCheck(data_output, 1, 1, c.KH * c.KW * c.C, c.P * c.Q *c.N);		//im2col 결과값 확인

	//status |= (int)cudaMemcpy(m_output.data(), d_f_m_output, m_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
	//cout << "sgemm output" << endl;
	//valueCheck(m_output, 1, 1, c.K, c.P * c.Q * c.N);		    // 결과값 확인
	//valueCheck(m_output, c.N, c.K, c.P, c.Q, 1);					// 결과값 확인



	cudaFree(d_data);
	cudaFree(d_weight);

	cudaFree(d_data_output);
	cudaFree(d_m_output);
	cudaFree(d_f_m_output);

	/////////////////////
	

	
	return 0;
}