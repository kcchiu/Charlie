
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"

#define N 100

__global__ void kernel(int *a, int *b, int *c) {
	int tid = blockIdx.x;	// tid = 0,1,2,...,N-1. Perform the tid th copy (block) of the function.
	printf("tid:%d\r\n", tid);
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	int count;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&count);
	printf("Device count:%d\n", count);
	cudaGetDeviceProperties(&prop, 0);
	printf("name: %s\n", prop.name);
	printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
	printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
	printf("canMapHostMemory: %d\n", prop.canMapHostMemory);
	printf("compute capability: %d.%d\n", prop.major, prop.minor);

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	for (int i = 0; i < N; i++) {
		a[i] = i;
		b[i] = i*i;
	}

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

	// Parallize into N blocks
	kernel << <N, 1 >> > (dev_a, dev_b, dev_c); // N copies (blocks) of the function "kernel" are executed in parallel on the GPU

	HANDLE_ERROR(cudaMemcpy(
		c,
		dev_c,
		N * sizeof(int),
		cudaMemcpyDeviceToHost));

	for (int i = 0; i < N; i++) {
		printf("Ans:%d, a=%d, b=%d, c=%d\r\n", c[i], a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}
