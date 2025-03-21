import os

from writers_readers import *


def generate_random_graphs(num_graphs: int, num_vertices: int, folder: str):
    for i in range(num_graphs):
        graph = PrecedenceGraph()
        graph.random_network(num_vertices)
        path_to_file = folder + "/graph_" + str(num_vertices) + "_" + str(i) + ".txt"
        write_graph(graph, path_to_file)

        print("GRAPH " + str(num_vertices) + ": " + str(i) + " Generated")


def generate_distribution(distribution_type: str, duration: int, tail: int):
    if distribution_type == "uniform":
        distribution = discrete_uniform_dist(duration, tail)
    elif distribution_type == "normal":
        std = np.random.uniform(0, 2)
        distribution = discrete_normal_dist(duration, std, tail)
    elif distribution_type == "exponential":
        mean = np.random.randint(1, 3)
        distribution = discrete_exponential_dist(mean, duration - mean, tail)
    else:
        raise ValueError("Distribution type must be either 'uniform', 'normal' or 'exponential'")
    return distribution


def generate_random_tasks(butch_size: int, num_tasks: int, distribution_type: str, folder: str):
    for i in range(butch_size):
        path_to_file = folder + "/tasks_" + distribution_type + "_" + str(num_tasks) + "_" + str(i) + ".txt"
        with open(path_to_file, "w") as file:
            first_dummy_task = Task(0, 1, {1: 1.})
            file.write(task_to_string(first_dummy_task) + '\n')
            for j in range(1, num_tasks-1):
                task_id = j
                duration = np.random.randint(5, 11)
                tail = np.random.randint(1, 4)
                distribution = generate_distribution(distribution_type, duration, tail)
                task = Task(task_id, duration, distribution)
                file.write(task_to_string(task) + '\n')
            last_dummy_task = Task(num_tasks-1, 1, {1: 1.})
            file.write(task_to_string(last_dummy_task) + '\n')

            print("TASKS " + distribution_type + " " + str(num_tasks) + ": " + str(i) + " Generated")


if __name__ == '__main__':
    # generate_random_graphs(num_graphs=100, num_vertices=52,
    #                        folder="Data/PrecedenceGraphs/FasterGeneratedGraphs/50_notDummyVertices")
    # generate_random_graphs(num_graphs=100, num_vertices=102,
    #                        folder="Data/PrecedenceGraphs/FasterGeneratedGraphs/100_notDummyVertices")
    # generate_random_graphs(num_graphs=100, num_vertices=202,
    #                        folder="Data/PrecedenceGraphs/FasterGeneratedGraphs/200_notDummyVertices")

    distribution_types = ['uniform', 'normal', 'exponential']
    for distribution_type in distribution_types:
        folder = "Data/Tasks/" + distribution_type
        generate_random_tasks(butch_size=100, num_tasks=62, distribution_type=distribution_type, folder=folder)
        # generate_random_tasks(butch_size=100, num_tasks=102, distribution_type=distribution_type, folder=folder)
        # generate_random_tasks(butch_size=100, num_tasks=202, distribution_type=distribution_type, folder=folder)
