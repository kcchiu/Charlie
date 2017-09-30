
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "book.h"


__global__ void kernel(int a, int b, int *c) {
	*c = a + b;
}

int main(void) {
	int a, b;
	int c;
	int *dev_c;
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

	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

	a = 2; b = 6;

	kernel << <1, 1 >> > (a, b, dev_c);

	HANDLE_ERROR(cudaMemcpy(
		&c,
		dev_c,
		sizeof(int),
		cudaMemcpyDeviceToHost));

	printf("2+6=%d\n", c);
	cudaFree(dev_c);

	printf("Hello World\n");
	return 0;
}
