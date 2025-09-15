all: assignment.c assignment.cu
	gcc assignment.c -o assignment
	nvcc assignment.cu -o cudaassignment

clean: 
	rm assignment
	rm cudaassignment