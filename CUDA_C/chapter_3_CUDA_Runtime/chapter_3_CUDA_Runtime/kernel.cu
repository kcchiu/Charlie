
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../common/book.h"
#include <stdio.h>
#include <string.h>
#include <ctime>

#define N 5120
#define THREADS_PER_BLOCK 512

//  Windows
#ifdef _WIN32
#include <Windows.h>
double get_wall_time() {
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)) {
		//  Handle error
		return 0;
	}
	if (!QueryPerformanceCounter(&time)) {
		//  Handle error
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time() {
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
		//  Returns total user time.
		//  Can be tweaked to include kernel times as well.
		return
			(double)(d.dwLowDateTime |
			((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	else {
		//  Handle error
		return 0;
	}
}

//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time() {
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		//  Handle error
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time() {
	return (double)clock() / CLOCKS_PER_SEC;
}
#endif

__global__ void add(double *a, double *b, double *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] * b[index];
	//c[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
}

struct {
	char name[40];
	int age;
}person, person_copy;

	int main(void){
		double a[N], b[N], c[N], c_cpu[N];
		double *d_a, *d_b, *d_c;
		int size = N * sizeof(double);
		int i;

		//a = (int *)malloc(size);
		//b = (int *)malloc(size);

		for (i = 0; i < N; i++)
		{
			a[i] = (rand() % 100) / 3.14;
			b[i] = (rand() % 100) / 3.14;
			//printf("%d\t%d\n",i, a[i]);
		}

		//allocate memory in device for copy of a, b, c
		cudaMalloc((void **)&d_a, size);
		cudaMalloc((void **)&d_b, size);
		cudaMalloc((void **)&d_c, size);

		//copy a, b in host to d_a, d_b in device
		cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

		float elapsed = 0;
		cudaEvent_t start, stop;

		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));

		HANDLE_ERROR(cudaEventRecord(start, 0));

		//execute GPU computation
		add <<< N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (d_a, d_b, d_c);
		//add << < N, 1 >> > (d_a, d_b, d_c);

		HANDLE_ERROR(cudaEventRecord(stop, 0));
		HANDLE_ERROR(cudaEventSynchronize(stop));

		HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

		HANDLE_ERROR(cudaEventDestroy(start));
		HANDLE_ERROR(cudaEventDestroy(stop));

		//copy the computed result from device to host
		cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

		printf("GPU time: %.5f us\n", elapsed * 1000);

		double cpu0 = get_cpu_time();

		for (i = 0; i < N; i++)
		{
			c_cpu[i] = a[i] * b[i];
		}

		double cpu1 = get_cpu_time();
		
		printf("CPU time: %.8f s\n", (double)(cpu1-cpu0));
		printf("%f\t%f\n", cpu0*1000, cpu1*1000);
		printf("%d\t%d\t%d\n", c_cpu, &c_cpu[0], &c_cpu[1]);
		//free the allocated device memory
		cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

		/*
		int c;
		int *dev_c;
		char myname[30] = "Kuan-Cheng Chiu";
		HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

		add << <1,1 >> > (2, 7, dev_c);

		HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

		printf("2+7=%d\n", c);
		cudaFree(dev_c);

		printf("Hello World\n");

		memcpy(person.name, myname, strlen(myname) + 1);
		person.age = 40;
		printf("%s\n", person.name);
		memcpy(&person_copy, &person, sizeof(person));
		printf("%s, %d\n", person_copy.name, person_copy.age);
		*/
		return 0;
	}
