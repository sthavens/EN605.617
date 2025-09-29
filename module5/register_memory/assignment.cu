#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <curand.h>
#include <curand_kernel.h>

#ifndef ITEMS
#define ITEMS 1 << 10
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif


__constant__ unsigned int constant_data[ITEMS];

__host__ void populate_data(unsigned int * data, int size) {
	srand(time(NULL));
	for (int i = 0; i < size; i++) {
		unsigned int temp = rand();
		data[i] = temp;
	}
	
}

__host__ void run_cpu_test(unsigned int * data, unsigned int * result, int size) {
	for (int i = 0; i < size; i++) {
		result[i] = data[i];
		result[i] *= 2;
		result[i] += 1;
		result[i] -= 4;
		result[i] = result[i] << 3;
		result[i] = result[i] == 0 ? 10 : result[i] / 4;
	}
}

__global__ void run_register_test(unsigned int * data, unsigned int * result, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		unsigned int reg = data[idx];
		reg *= 2;
		reg += 1;
		reg -= 4;
		reg = reg << 3;
		reg = reg == 0 ? 10 : reg / 4;
		result[idx] = reg;
	}
}

__global__ void run_global_test(unsigned int * data, unsigned int * result, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		result[idx] = data[idx];
		result[idx] *= 2;
		result[idx] += 1;
		result[idx] -= 4;
		result[idx] = result[idx] << 3;
		result[idx] = result[idx] == 0 ? 10 : result[idx] / 4;	
	}
}

__global__ void run_shared_test(unsigned int * data, unsigned int * result, int size) {
	if (size > 8192 * 256) {
		return;
	}
	__shared__ unsigned int shared_data[ITEMS];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		shared_data[idx] = data[idx];
		shared_data[idx] *= 2;
		shared_data[idx] += 1;
		shared_data[idx] -= 4;
		shared_data[idx] <<= 3;
		shared_data[idx] = shared_data[idx] == 0 ? 10 : shared_data[idx] / 4;
		__syncthreads();
		result[idx] = shared_data[idx];
	}
}

__global__ void run_constant_test(unsigned int * result, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int temp = constant_data[idx];
	temp *= 2;
	temp += 1;
	temp -= 4;
	temp = temp << 3;
	temp = temp == 0 ? 10 : temp / 4;
	result[idx] = temp;
}

__host__ void run_gpu() {
	unsigned int * data;
	unsigned int * gpu_data;
	unsigned int * gpu_register_result;
	unsigned int * register_result;
	unsigned int * global_result;
	unsigned int * gpu_global_result;
	unsigned int * shared_result;
	unsigned int * gpu_shared_result;
	unsigned int * constant_result;
	unsigned int * gpu_constant_result;

	unsigned int num_items = ITEMS;
	unsigned int byte_size = sizeof(unsigned int) * num_items;
	clock_t start;

	// allocate memory on device
	cudaMalloc((void**)&gpu_data, byte_size);
	cudaMalloc((void**)&gpu_register_result, byte_size);
	cudaMalloc((void**)&gpu_global_result, byte_size);
	cudaMalloc((void**)&gpu_shared_result, byte_size);
	cudaMalloc((void**)&gpu_constant_result, byte_size);

	// allocate memory on host
	data = (unsigned int *) malloc(byte_size);
	register_result = (unsigned int *) malloc(byte_size);
	global_result = (unsigned int *) malloc(byte_size);
	shared_result = (unsigned int *) malloc(byte_size);
	constant_result = (unsigned int *) malloc(byte_size);

	// populate data
	populate_data(data, num_items);

	//move data to device
	cudaMemcpy(gpu_data, data, byte_size, cudaMemcpyHostToDevice);

	//deliver constants set
	printf("Number of items: %d\n", ITEMS);
	printf("Block size: %d\n", BLOCK_SIZE);

	//run tests
	start = clock();
	unsigned int * cpu_result = (unsigned int *) malloc(byte_size);
	run_cpu_test(data, cpu_result, num_items);
	printf("CPU time: %f\n", (double)(clock() - start) / CLOCKS_PER_SEC);

	start = clock();
	run_register_test<<<num_items/BLOCK_SIZE, BLOCK_SIZE>>>(gpu_data, gpu_register_result, num_items);
	printf("Register GPU time: %f\n", (double)(clock() - start) / CLOCKS_PER_SEC);
	cudaDeviceSynchronize();
	cudaMemcpy(register_result, gpu_register_result, byte_size, cudaMemcpyDeviceToHost);
	cudaFree(gpu_register_result);

	start = clock();
	run_global_test<<<num_items/BLOCK_SIZE, BLOCK_SIZE>>>(gpu_data, gpu_global_result, num_items);
	printf("Global GPU time: %f\n", (double)(clock() - start) / CLOCKS_PER_SEC);
	cudaDeviceSynchronize();
	cudaMemcpy(global_result, gpu_global_result, byte_size, cudaMemcpyDeviceToHost);
	cudaFree(gpu_global_result);

	start = clock();
	run_shared_test<<<num_items/BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>(gpu_data, gpu_shared_result, num_items);
	printf("Shared GPU time: %f\n", (double)(clock() - start) / CLOCKS_PER_SEC);
	cudaDeviceSynchronize();
	cudaMemcpy(shared_result, gpu_shared_result, byte_size, cudaMemcpyDeviceToHost);
	cudaFree(gpu_shared_result);

	start = clock();
	cudaMemcpyToSymbol(constant_data, data, byte_size);
	run_constant_test<<<num_items/BLOCK_SIZE, BLOCK_SIZE>>>(gpu_constant_result, num_items);
	printf("Constant GPU time: %f\n", (double)(clock() - start) / CLOCKS_PER_SEC);
	cudaDeviceSynchronize();
	cudaMemcpy(constant_result, gpu_constant_result, byte_size, cudaMemcpyDeviceToHost);
	cudaFree(constant_result);
	



	// verify results
	for (int i = 0; i < num_items; i++) {
		if (cpu_result[i] != register_result[i]) {
			printf("Error: Mismatch at index %d, CPU result = %u, Register result = %u\n", i, cpu_result[i], register_result[i]);
			break;
		}
		if (cpu_result[i] != global_result[i]) {
			printf("Error: Mismatch at index %d, CPU result = %u, Global result = %u\n", i, cpu_result[i], global_result[i]);
			break;
		}
		if (cpu_result[i] != shared_result[i]) {
			printf("Error: Mismatch at index %d, CPU result = %u, Shared result = %u\n", i, cpu_result[i], shared_result[i]);
			printf("Input data = %u\n", data[i]);
			break;
		}
		if (cpu_result[i] != constant_result[i]) {
			printf("Error: Mismatch at index %d, CPU result = %u, Constant result = %u\n", i, cpu_result[i], constant_result[i]);
			printf("Input data = %u\n", data[i]);
			break;
		}
	}

	// free memory
	free(data);
	free(cpu_result);
	free(register_result);
	free(global_result);
	free(shared_result);
	free(constant_result);
	cudaFree(gpu_data);
}

int main(int argc, char** argv)
{

	run_gpu();
}
