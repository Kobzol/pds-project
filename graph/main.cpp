#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include <cuda_profiler_api.h>

#include "cudamem.h"
#include "util.h"
#include "graph.h"
#include "graph_cuda.h"
#include "graph_cpu.h"
#include "graph_harnar.h"

void generate_graph(size_t vertexCount, size_t degree, std::vector<Vertex>& vertices)
{
	srand((unsigned int) time(nullptr));

	// generate graph
	for (size_t i = 0; i < vertexCount; i++)
	{
		vertices.push_back(Vertex());
		Vertex& vertex = vertices[vertices.size() - 1];

		size_t edgeCount = degree;

		for (size_t j = 0; j < edgeCount; j++)
		{
			int edge = rand() % vertexCount;
			if (edge == i)
			{
				j--;
				continue;
			}
			vertex.edges.push_back(Edge(edge, 1));
		}
	}
}
void load_sigmod_graph(std::string path, std::vector<Vertex>& vertices)
{
	std::fstream graphFile(path, std::ios::in);
	vertices.resize(1600000, Vertex());

	std::string line;
	while (std::getline(graphFile, line))
	{
		std::stringstream ss(line);
		int from, to;
		ss >> from >> to;

		vertices[from].edges.push_back(Edge(to, 1));
		vertices[to].edges.push_back(Edge(from, 1));
	}

	graphFile.close();
}
void load_dimacs_graph(std::string path, std::vector<Vertex>& vertices, int additionalEdges = 0)
{
	std::fstream graphFile(path, std::ios::in);
	std::uniform_int_distribution<int> randomGenerator;
	std::default_random_engine engine(1);

	std::string line;
	while (std::getline(graphFile, line))
	{
		if (line[0] == 'c')
		{
			continue;
		}
		else if (line[0] == 'p')
		{
			std::stringstream ss(line.substr(line.find("sp") + 2));
			int verticesCount, edgeCount;
			ss >> verticesCount >> edgeCount;
			vertices.resize(verticesCount + 1, Vertex());

			randomGenerator = std::uniform_int_distribution<int>(0, (int) vertices.size() - 1);
		}
		else if (line[0] == 'a')
		{
			std::stringstream ss(line.substr(line.find("a") + 1));
			int from, to, weight;
			ss >> from >> to >> weight;
			vertices[from].edges.push_back(Edge(to, weight));

			for (int i = 0; i < additionalEdges; i++)
			{
				vertices[from].edges.push_back(Edge(randomGenerator(engine), rand() % 10000));
			}
		}
	}

	graphFile.close();
}

void time_alg(Graph* graph, const std::vector<Vertex>& vertices, std::string alg, std::string type, int iterations)
{
	volatile int result;
	graph->vertices = vertices;
	std::uniform_int_distribution<int> randomGenerator(0, (int) graph->vertices.size() - 1);
	std::default_random_engine engine(1);
	int checkIterations = 3;
	long total_time = 0L;
	
	for (int i = 0; i < checkIterations; i++)
	{
		Timer timer;
		if (alg == "BFS")
		{
			for (int i = 0; i < iterations; i++)
			{	
				int from = randomGenerator(engine);
				int to = randomGenerator(engine);
				result = graph->is_connected(from, to);
			}
		}
		else if (alg == "Dijkstra")
		{
			for (int i = 0; i < iterations; i++)
			{	
				int from = randomGenerator(engine);
				int to = randomGenerator(engine);
				result = graph->get_shortest_path(from, to);
			}
		}
		total_time += timer.get_millis();
	}

	std::cout << "Time " + alg + " (" + type + ") " << (total_time / checkIterations) << " ms" << std::endl;

	delete graph;
}

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		std::cout << "graph_name additional_edges iteration_count" << std::endl;
		return 1;
	}

	std::vector<Vertex> vertices;
	load_dimacs_graph(argv[1], vertices, std::atoi(argv[2]));

	size_t count = 0;
	for (Vertex& vertex : vertices)
	{
		count += vertex.edges.size();
	}

	std::cout << "Average # of edges: " << count / (double) vertices.size() << std::endl;

	int iterationCount = std::atoi(argv[3]);

	time_alg(new GraphCPU(), vertices, "BFS", "cpu", iterationCount);
	time_alg(new GraphCUDA(), vertices, "BFS", "gpu", iterationCount);
	time_alg(new GraphHarnar(), vertices, "BFS", "harnar", iterationCount);

	time_alg(new GraphCPU(), vertices, "Dijkstra", "cpu", iterationCount);
	time_alg(new GraphCUDA(), vertices, "Dijkstra", "gpu", iterationCount);
	time_alg(new GraphHarnar(), vertices, "Dijkstra", "harnar", iterationCount);

	/*long times[3] = { 0 };
	const size_t ITERATION_COUNT = 30;
	std::default_random_engine engine(1);
	std::uniform_int_distribution<int> randomGenerator(0, (int) vertices.size() - 1);

	GraphCPU cpu;
	cpu.vertices = vertices;
	GraphCUDA g;
	g.vertices = vertices;
	GraphHarnar harnar;
	harnar.vertices = vertices;

	for (int i = 0; i < ITERATION_COUNT; i++)
	{
		int from = randomGenerator(engine);
		int to = randomGenerator(engine);
		
		Timer timer;
		unsigned int resultCPU = cpu.is_connected(from, to);
		times[0] += timer.get_millis();
		
		timer.start();
		unsigned int resultHarnar = harnar.is_connected(from, to);
		times[2] += timer.get_millis();

		timer.start();
		unsigned int resultGPU = g.is_connected(from, to);
		times[1] += timer.get_millis();

		if (resultGPU != resultCPU)
		{
			std::cout << "GPU error at query " << i << ": expected " << resultCPU << ", got " << resultGPU << std::endl;
		}

		if (resultHarnar != resultCPU)
		{
			std::cout << "Harnar error at query " << i << ": expected " << resultCPU << ", got " << resultHarnar << std::endl;
		}
	}

	std::cout << "CPU average: " << (times[0] / ITERATION_COUNT) << std::endl;
	std::cout << "GPU average: " << (times[1] / ITERATION_COUNT) << std::endl;
	std::cout << "Harnar average: " << (times[2] / ITERATION_COUNT) << std::endl;
	
	getchar();*/

   	return 0;
}
