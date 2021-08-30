#include <fstream>
#include <chrono>
#include <random>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <helper_cuda.h> 
#include "util.h"

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
};

int main(int argc, char** argv) {

	deviceQuery();
	cudaSetDevice(0);
	cudaStream_t stream;
	cublasHandle_t cublasHandle;

	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cublasCreate(&cublasHandle));
	CUDA_CHECK(cublasSetStream(cublasHandle, stream));

	Config c = config[0];

	c.P = ((c.H + c.top + c.bottom - c.KH) / c.SH) + 1;
	c.Q = ((c.W + c.left + c.right - c.KW) / c.SW) + 1;

	printf(" input[%4d,%4d,%4d,%4d] kernel[%4d,%4d,%4d,%4d] output[%4d,%4d,%4d,%4d]\n\n", c.N, c.C, c.H, c.W, c.K, c.C, c.KH, c.KW, c.N, c.K, c.P, c.Q);

	vector<float> data(c.N*c.C*c.H*c.W);     // input data [N,C,H,W]
	vector<float> weight(c.K*c.C*c.KH*c.KW); // weight [K,C,3,3]

	inititalizedData(data);			//  1씩 증가하는 등차수열 
	inititalizedData(weight);		//  1씩 증가하는 등차수열 

	//cout << "Data(Input)" << endl;
	//valueCheck(data, c.N, c.C, c.H, c.W);		//입력값 확인
	//cout << "kernel" << endl;
	//valueCheck(weight, c.K, c.C, c.KH, c.KW);	// 가중치 확인
	//valueCheck(weight, 1, 1, c.K, c.C * c.KH * c.KW);	// 가중치 확인
	
	float* d_weight; // device input data
	CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));

	float* d_data;	// device input data
	CUDA_CHECK(cudaMalloc(&d_data, data.size() * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));

	vector<float> data_output(c.C * c.KH * c.KW * c.N * c.P * c.Q); // 결과 가져오기
	//cout << "data out size :: " << data_output.size() << endl;

	float* d_data_output;	// device output data
	//CUDA_CHECK(cudaMalloc(&d_data_output, data_output.size() * sizeof(float));
	CUDA_CHECK(cudaMalloc(&d_data_output, data_output.size() * sizeof(float)));

	vector<float> m_output(c.K * c.P * c.Q * c.N); // sgemm 결과 
	float* d_m_output;	// device output data
	CUDA_CHECK(cudaMalloc(&d_m_output, m_output.size() * sizeof(float)));

	float* d_f_m_output;	// device output data
	CUDA_CHECK(cudaMalloc(&d_f_m_output, m_output.size() * sizeof(float)));


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

		CUDA_CHECK(cudaStreamSynchronize(stream));

		total_time += duration_cast<microseconds>(system_clock::now().time_since_epoch()).count() - start_time;

	}
	
	CUDA_CHECK(cudaMemcpy(m_output.data(), d_f_m_output, m_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

	double checksum = 0;
	for (auto d : m_output) checksum += fabs((double)d);
	printf("   Gemm       status=%d avg_dur_time=%6.3f[msec] checksum=%.6f\n", status, total_time / 1000.f / ITER, checksum);
	
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