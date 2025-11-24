#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define M 1024UL
#define N 1024UL
#define K 1024UL

#define TILE_M 32UL
#define TILE_N 32UL
#define TILE_K 32UL

cl_context CreateContext(cl_device_id *deviceID);
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id deviceID);
cl_program CreateProgram(cl_context context, cl_device_id deviceID, const char* fileName);
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, float* A, float* B, float* C, cl_kernel kernel);
void RandomFill(float* matrix, int rows, int cols);
void WriteMatrixToFile(const char* filename, float* matrix, int rows, int cols);
void TransposeMatrix(float* input, float* output, int rows, int cols);

int main(int argc, char** argv)
{
	float* A = (float*) malloc(sizeof(float) * M * K);
	float* B = (float*) malloc(sizeof(float) * K * N);
	float* B_T = (float*) malloc(sizeof(float) * N * K);
	float* C = (float*) malloc(sizeof(float) * M * N);
	
	RandomFill(A, M, K);
	RandomFill(B, K, N);
	TransposeMatrix(B, B_T, K, N);

	cl_device_id deviceID;
	cl_program program;
	cl_command_queue commandQueue;
	cl_int err;
	cl_kernel kernel;

	cl_context context = CreateContext(&deviceID);
	if (context == NULL) {
		free(A);
		free(B);
		free(C);
		return -1;
	}

	commandQueue = CreateCommandQueue(context, deviceID);
	if (commandQueue == NULL) {
		free(A);
		free(B);
		free(C);
		clReleaseContext(context);
		return -1;
	}

	cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M * K, NULL, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to create buffer for A." << std::endl;
		free(A);
		free(B);
		free(C);
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
		return -1;
	}

	err = clEnqueueWriteBuffer(commandQueue, bufferA, CL_TRUE, 0, sizeof(float) * M * K, A, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		std::cerr << "Failed to write data to buffer A." << std::endl;
		free(A);
		free(B);
		free(C);
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
		clReleaseMemObject(bufferA);
		return -1;
	}

	cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * K * N, NULL, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to create buffer for B." << std::endl;
		free(A);
		free(B);
		free(C);
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
		clReleaseMemObject(bufferA);
		return -1;
	}

	err = clEnqueueWriteBuffer(commandQueue, bufferB, CL_TRUE, 0, sizeof(float) * K * N, B_T, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		std::cerr << "Failed to write data to buffer B." << std::endl;
		free(A);
		free(B);
		free(C);
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
		clReleaseMemObject(bufferA);
		clReleaseMemObject(bufferB);
		return -1;
	}

	cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &err);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to create buffer for C." << std::endl;
		free(A);
		free(B);
		free(C);
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
		clReleaseMemObject(bufferA);
		return -1;
	}


	program = CreateProgram(context, deviceID, "assignment.cl");
	if (program == NULL) {
		free(A);
		free(B);
		free(C);
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
		return -1;
	}

	kernel = clCreateKernel(program, "matrixMultiplyTiled", &err);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to create kernel." << std::endl;
		free(A);
		free(B);
		free(C);
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
		clReleaseProgram(program);
		return -1;
	}

	// Print matrix dimensions
	std::cout << "Matrix A: " << M << " x " << K << std::endl;
	std::cout << "Matrix B: " << K << " x " << N << std::endl;
	std::cout << "Matrix C: " << M << " x " << N << std::endl;

	double totalKernelTimeMs = 0.0;

	// In this case, we're using sub-buffers to illustrate and to fulfill assignment objectives,
	// in practice, we'd either use pointer arithmetic or precompute the sub-buffers since they are immutable.

	for (int tileRow = 0; tileRow < M; tileRow += TILE_M) {
		for (int tileColumn = 0; tileColumn < N; tileColumn += TILE_N) {
			cl_buffer_region regionA = {
				.origin = sizeof(float) * (tileRow * K),
				.size = sizeof(float) * (TILE_M * K)
			};

			cl_buffer_region regionB = {
				.origin = sizeof(float) * tileColumn ,
				.size = sizeof(float) * (K * TILE_N)
			};

			cl_buffer_region regionC = {
				.origin = sizeof(float) * (tileRow * N + tileColumn),
				.size = sizeof(float) * (TILE_M * TILE_N)
			};

			cl_mem subA = clCreateSubBuffer(
				bufferA,
				CL_MEM_READ_ONLY,
				CL_BUFFER_CREATE_TYPE_REGION,
				&regionA,
				&err);

			if (err != CL_SUCCESS) {
				std::cerr << "Failed to create sub-buffer for A." << std::endl;
				free(A);
				free(B);
				free(C);
				clReleaseContext(context);
				clReleaseCommandQueue(commandQueue);
				clReleaseProgram(program);
				return -1;
			}

			cl_mem subB = clCreateSubBuffer(
				bufferB,
				CL_MEM_READ_ONLY,
				CL_BUFFER_CREATE_TYPE_REGION,
				&regionB,
				&err);

			if (err != CL_SUCCESS) {
				std::cerr << "Failed to create sub-buffer for B." << std::endl;
				free(A);
				free(B);
				free(C);
				clReleaseContext(context);
				clReleaseCommandQueue(commandQueue);
				clReleaseProgram(program);
				clReleaseMemObject(subA);
				return -1;
			}

			cl_mem subC = clCreateSubBuffer(
				bufferC,
				CL_MEM_WRITE_ONLY,
				CL_BUFFER_CREATE_TYPE_REGION,
				&regionC,
				&err);

			if (err != CL_SUCCESS) {
				std::cerr << "Failed to create sub-buffer for C." << std::endl;
				free(A);
				free(B);
				free(C);
				clReleaseContext(context);
				clReleaseCommandQueue(commandQueue);
				clReleaseProgram(program);
				clReleaseMemObject(subA);
				clReleaseMemObject(subB);
				return -1;
			}

			int m = M;
			int n = N;
			int k = K;

			int vecWidth = 4; // Using float4

			size_t local[2] = {TILE_M, TILE_N / vecWidth};
			size_t global[2] = {TILE_M, TILE_N / vecWidth};

			int N_pitch = TILE_N;

			cl_event event;
			cl_ulong timeStart;
			cl_ulong timeEnd;


			clSetKernelArg(kernel, 0, sizeof(cl_mem), &subA);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &subB);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &subC);
			clSetKernelArg(kernel, 3, sizeof(int), &k);  // full inner dimension
			clSetKernelArg(kernel, 4, sizeof(float) * 4 * TILE_M * (TILE_K / vecWidth), NULL); // local A tile
			clSetKernelArg(kernel, 5, sizeof(float) * 4 * (TILE_N / vecWidth) * (TILE_K / vecWidth), NULL); // local B tile
			clSetKernelArg(kernel, 6, sizeof(int), &N_pitch);


			clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global, local, 0, NULL, &event);

			clWaitForEvents(1, &event);

			// Limits parallelism but simplifies synchronization for this example
			clFinish(commandQueue);

			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, NULL);
			clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, NULL);

			double kernelTimeMs = (timeEnd - timeStart) * 1e-6;  // Convert ns to ms
			totalKernelTimeMs += kernelTimeMs;
			std::cout << "Tile (" << tileRow << "," << tileColumn << ") kernel time: " << kernelTimeMs << " ms" << std::endl;

			clReleaseEvent(event);

			clReleaseMemObject(subA);
			clReleaseMemObject(subB);
			clReleaseMemObject(subC);
		}
	}
	
	err = clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, sizeof(float) * M * N, C, 0, NULL, NULL);
	
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to read data from buffer C." << std::endl;
		free(A);
		free(B);
		free(C);
		clReleaseContext(context);
		clReleaseCommandQueue(commandQueue);
		clReleaseProgram(program);
	}
	
	std::cout << "Total kernel execution time: " << totalKernelTimeMs << " ms" << std::endl;

	WriteMatrixToFile("inputA.txt", A, M, K);
	WriteMatrixToFile("inputB.txt", B, K, N);
	WriteMatrixToFile("result.txt", C, M, N);

	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);

	// cleanup
	Cleanup(context, commandQueue, program, A, B, C, kernel);
	return 0;
}

cl_context CreateContext(cl_device_id *deviceID) {
	cl_int err;
	cl_uint numPlatforms;
	cl_platform_id platform;
	cl_context context = NULL;

	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to find an OpenCL platform." << std::endl;
		return NULL;	
	}

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, deviceID, NULL);
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to find GPU device." << std::endl;
		return NULL;	
	}

	context = clCreateContext(
		NULL,
		1,
		deviceID,
		NULL,
		NULL,
		&err);
	
	if (err != CL_SUCCESS) {
		std::cerr << "Failed to create OpenCL context." << std::endl;
		return NULL;	
	}

	return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id deviceID) {
    cl_int err;
    cl_command_queue queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!queue || err != CL_SUCCESS) {
        std::cerr << "Failed to create command queue\n";
        return NULL;
    }
    return queue;
}


cl_program CreateProgram(cl_context context, cl_device_id deviceID, const char* fileName){
	cl_int err;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open()) {
		std::cerr << "Failed to open kernel file: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char * srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, &err);

	if (err != CL_SUCCESS || program == NULL) {
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		char buildLog[16384];
		clGetProgramBuildInfo(
			program,
			deviceID,
			CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog),
			buildLog,
			NULL);
		
		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog << std::endl;

		clReleaseProgram(program);
		return NULL;
	}
	return program;
}

void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, float* A, float* B, float* C, cl_kernel kernel) {
	if (program != NULL) {
		clReleaseProgram(program);
	}

	if (commandQueue != NULL) {
		clReleaseCommandQueue(commandQueue);
	}

	if (context != NULL) {
		clReleaseContext(context);
	}

	if (kernel != NULL) {
		clReleaseKernel(kernel);
	}

	free(A);
	free(B);
	free(C);
}

void RandomFill(float* matrix, int rows, int cols){
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i * cols + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}
	}
}

void WriteMatrixToFile(const char* filename, float* matrix, int rows, int cols) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file " << filename << " for writing.\n";
        return;
    }

    // Print numbers with fixed point, 6 decimals, aligned in 12-character width
    outFile << std::fixed << std::setprecision(6);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            outFile << std::setw(12) << matrix[i * cols + j];
        }
        outFile << "\n";
    }

    outFile.close();
}

void TransposeMatrix(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // input: rows x cols -> output: cols x rows
            output[j * rows + i] = input[i * cols + j];
        }
    }
}