NVCC = /usr/local/cuda-7.5/bin/nvcc

all: cuda

cuda:
	$(NVCC) -O2 -std=c++11 -arch=sm_30 *.cu *.cpp -o cuda_exec

cuda_exec: *.o
	g++ -O2 -std=c++11 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 *.cpp $^ -lcuda -lcudart -o cuda_exec

*.o:
	$(NVCC) -std=c++11 -c -arch=sm_30 *.cu 

clean:
	rm -rf *o cuda_exec
