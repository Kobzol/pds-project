#pragma once

#include <vector>
#include "graph.h"

/*
* Linearized vertex without edges that is used in CUDA kernels.
*/
struct LinearizedVertexHarnar
{
public:
	LinearizedVertexHarnar(int edgeIndex, int edgeCount) : edgeIndex(edgeIndex), edgeCount(edgeCount), visited(false), frontier(false), frontier_next(false)
	{

	}

	int edgeIndex;
	int edgeCount;
	bool frontier;
	bool visited;
	bool frontier_next;
};

class GraphHarnar : public Graph
{
public:
	GraphHarnar() {  }

	virtual int add_vertex() override;
	virtual void add_edge(int from, int to, unsigned int cost = 1.0) override;

	virtual bool is_connected(int from, int to) override;
	virtual unsigned int get_shortest_path(int from, int to) override;

private:
	void relinearizeVertices();
	void initCuda();

	static bool CudaInitialized;

	std::vector<Edge> edges;
	std::vector<LinearizedVertexHarnar> linearizedVertices;
	bool dirty = true;
};

void cudaInit();
