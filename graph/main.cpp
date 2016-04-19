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

std::default_random_engine engine(1);

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

int main(int argc, char** argv)
{
	if (argc < 5) return 1;
	// graph_name additional_edges implementation_type iteration_count algorithm_type

	GraphCUDA g;
	load_dimacs_graph(argv[1], g.vertices, std::atoi(argv[2]));

	size_t count = 0;
	for (Vertex& vertex : g.vertices)
	{
		count += vertex.edges.size();
	}

	std::cout << "Average # of edges: " << count / (double) g.vertices.size() << std::endl;

	Graph* graph = nullptr;
	std::string graphType = argv[3];

	if (graphType == "cpu")
	{
		graph = new GraphCPU();
	}
	else if (graphType == "gpu")
	{
		graph = new GraphCUDA();
	}
	else graph = new GraphHarnar();

	std::cout << "Load finished" << std::endl;

	std::uniform_int_distribution<int> randomGenerator(0, (int)g.vertices.size() - 1);
	int iterationCount = std::atoi(argv[4]);
	std::string algorithmType = argv[5];

	for (int i = 0; i < iterationCount; i++)
	{
		int from = randomGenerator(engine);
		int to = randomGenerator(engine);

		if (algorithmType == "bfs")
		{
			graph->is_connected(from, to);
		}
		else graph->get_shortest_path(from, to);
	}

	/*long times[3] = { 0 };
	const size_t ITERATION_COUNT = 30;

	for (int i = 0; i < ITERATION_COUNT; i++)
	{
		int from = randomGenerator(engine);
		int to = randomGenerator(engine);
		
		Timer timer;
		unsigned int resultCPU = cpu.get_shortest_path(from, to);
		times[0] += timer.get_millis();
		
		timer.start();
		unsigned int resultHarnar = harnar.get_shortest_path(from, to);
		times[2] += timer.get_millis();

		timer.start();
		unsigned int resultGPU = g.get_shortest_path(from, to);
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
	
	getchar();
	*/

	delete graph;

    return 0;
}
