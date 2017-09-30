#include <stdio.h>


__global__ void kernel(int a, int b, int *c) {
	*c = a + b;
}

int main(void) {
	int a, b;
	int c;
	int *dev_c;

	a = 2; b = 5;

	kernel << <1, 1 >> > (a,b,dev_c);

	HANDLE_ERROR(cudaMemcpy(
		&c,
		dev_c,
		sizeof(int),
		cudaMemcpyDeviceToHost));

	printf("2+7=%d\n", c);
	cudaFree(dev_c);

	printf("Hello World\n");
	return 0;
}
