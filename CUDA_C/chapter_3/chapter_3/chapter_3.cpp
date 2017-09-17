
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void kernel(void) {}

int main(void) {
	kernel << <1, 1 >> > ();
	printf("Hello World\n");
	return 0;
}
