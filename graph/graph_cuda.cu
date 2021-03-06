#include "graph_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cudamem.h"
#include "util.h"

bool GraphCUDA::CudaInitialized = false;

int GraphCUDA::add_vertex()
{
	int value = Graph::add_vertex();
	this->dirty = true;

	return value;
}
void GraphCUDA::add_edge(int from, int to, unsigned int cost)
{
	Graph::add_edge(from, to, cost);
	this->dirty = true;
}

void GraphCUDA::initCuda()
{
	if (!GraphCUDA::CudaInitialized)
	{
		cudaSetDeviceFlags(cudaDeviceMapHost);
		GraphCUDA::CudaInitialized = true;
	}
}

__global__ void bfsKernel(LinearizedVertex* vertices, Edge* edges, int visitStart, int visitCounter, int target, int* stop, size_t size)
{
	int offset = (blockDim.x * blockDim.y) * blockIdx.x;	// how many blocks skipped
	int blockPos = blockDim.x * threadIdx.y + threadIdx.x;	// position in block
	int pos = offset + blockPos;

	if (pos >= size) return;

	if (vertices[pos].visitIndex == visitCounter)
	{
		int edgeCount = vertices[pos].edgeCount;
		int edgeIndex = vertices[pos].edgeIndex;

		stop[0] = 0;
		if (pos == target)
		{
			stop[1] = 1;
		}

		for (size_t i = 0; i < edgeCount; i++)
		{
			int edge = edges[edgeIndex + i].target;

			if (vertices[edge].visitIndex < visitStart)	// hasnt been visited in this BFS
			{
				vertices[edge].visitIndex = visitCounter + 1;
			}
		}
	}
}
bool GraphCUDA::is_connected(int from, int to)
{
	if (!this->has_vertex(from) || !this->has_vertex(to)) return false;

	this->relinearizeVertices();
	this->initCuda();

	if (this->edges.size() < 1) return false;

	int graphSize = (int) this->vertices.size();

	this->bfsVisitIndex++;
	int bfsVisitStart = this->bfsVisitIndex;
	this->linearizedVertices[from].visitIndex = bfsVisitStart;

	CudaMemory<LinearizedVertex> verticesCuda(graphSize, &(this->linearizedVertices[0]));
	CudaMemory<Edge> edgesCuda(this->edges.size(), &(this->edges[0]));
	CudaHostMemory<int> stopCuda(2);

	// computation
	dim3 blockDim(32, 32);
	int blockCount = (graphSize / (blockDim.x * blockDim.y)) + 1;
	dim3 gridDim(blockCount, 1);

	int* stopHost = stopCuda.host();
	stopHost[0] = 0;

	while (!stopHost[0])
	{
		stopHost[0] = 1;

		bfsKernel << <gridDim, blockDim >> >(*verticesCuda, *edgesCuda, bfsVisitStart, this->bfsVisitIndex++, to, stopCuda.device(), graphSize);
		cudaDeviceSynchronize();	// wait for kernel to end

		if (stopHost[1])
		{
			return true;
		}
	}

	return false;
}

__global__ void dijkstraKernel(LinearizedVertex* vertices, Edge* edges, unsigned int* __restrict__ costs, int visitCounter, int* __restrict__ stop, size_t size)
{
	int offset = (blockDim.x * blockDim.y) * blockIdx.x;	// how many blocks skipped
	int blockPos = blockDim.x * threadIdx.y + threadIdx.x;	// position in block
	int pos = offset + blockPos;

	if (pos >= size) return;

	if (vertices[pos].visitIndex == visitCounter)
	{
		unsigned int distance = costs[pos];

		int edgeCount = vertices[pos].edgeCount;
		int edgeIndex = vertices[pos].edgeIndex;

		for (size_t i = 0; i < edgeCount; i++)
		{
			Edge& edge = edges[edgeIndex + i];
			unsigned int newDistance = distance + edge.cost;
			if (atomicMin(&costs[edge.target], newDistance) > newDistance)
			{
				*stop = 0;
				vertices[edge.target].visitIndex = visitCounter + 1;
			}
		}
	}
}
unsigned int GraphCUDA::get_shortest_path(int from, int to)
{
	if (!this->has_vertex(from) || !this->has_vertex(to)) return UINT_MAX;

	this->relinearizeVertices();
	this->initCuda();

	if (this->edges.size() < 1) return UINT_MAX;

	int graphSize = (int) this->vertices.size();

	unsigned int visitCounter = 0;
	this->linearizedVertices[from].visitIndex = visitCounter;

	CudaMemory<LinearizedVertex> verticesCuda(graphSize, &(this->linearizedVertices[0]));
	CudaMemory<Edge> edgesCuda(this->edges.size(), &(this->edges[0]));

	CudaMemory<unsigned int> costsCuda(graphSize, 0xEE);
	//CudaHostMemory<int> stopCuda;
	CudaMemory<int> stopCuda(1, 0);

	// computation
	costsCuda.store(0, 1, from);

	dim3 blockDim(32, 32);
	int blockCount = (graphSize / (blockDim.x * blockDim.y)) + 1;
	dim3 gridDim(blockCount, 1);

	//int* stopHost = stopCuda.host();
	//*stopHost = 0;

	int stopHost = 0;

	while (!(stopHost))
	{
		//*stopHost = 1;
		stopCuda.store(1);

		dijkstraKernel << <gridDim, blockDim >> >(*verticesCuda, *edgesCuda, *costsCuda, visitCounter, stopCuda.device(), graphSize);
		cudaDeviceSynchronize();	// wait for kernel to end

		stopCuda.load(stopHost);

		visitCounter++;
	}

	std::vector<unsigned int> costs(graphSize);
	costsCuda.load(costs[0], graphSize);

	return costs[to];
}

void GraphCUDA::relinearizeVertices(bool force)
{
	if (this->dirty || force)
	{
		this->edges.clear();
		this->linearizedVertices.clear();

		for (const Vertex& vertex : this->vertices)
		{
			int edgeCount = (int)vertex.edges.size();
			int edgeIndex = (int)edges.size();

			this->edges.insert(this->edges.end(), vertex.edges.begin(), vertex.edges.end());
			this->linearizedVertices.emplace_back(edgeIndex, edgeCount);
		}

		this->bfsVisitIndex = 0;
	}

	this->dirty = false;
}
