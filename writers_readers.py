from class_graph import *
from class_task import *
import ast


def write_graph(graph: PrecedenceGraph, path_to_file: str):
    all_edges = graph.get_copy_of_all_edges()
    with open(path_to_file, 'w') as f:
        for fr_id in all_edges.keys():
            f.write(str(fr_id) + ':')
            for to_id in all_edges[fr_id]:
                f.write(str(to_id) + ',')
            f.write('\n')


def read_graph(path_to_file: str) -> PrecedenceGraph:
    graph = PrecedenceGraph()
    with open(path_to_file, 'r') as f:
        for line in f:
            fr_id, to_ids = line.split(':')
            for to_id in to_ids.split(',')[:-1]:
                graph.add_edge(int(fr_id), int(to_id))
    return graph


def task_to_string(task: Task) -> str:
    task_str = str(task.get_id()) + ':'
    task_str += str(task.get_duration()) + ':'
    task_str += str(task.get_distribution())
    return task_str


def string_to_task(string: str) -> Task:
    str_id, str_duration, str_distribution = string.split(':', 2)
    task = Task(int(str_id), int(str_duration), ast.literal_eval(str_distribution))
    return task


def read_tasks(path_to_file: str) -> list[Task]:
    tasks = []
    with open(path_to_file, 'r') as f:
        for line in f:
            task = string_to_task(line)
            tasks.append(task)
    return tasks


def read_tasks_to_dict(path_to_file: str) -> dict[int, Task]:
    tasks = dict()
    with open(path_to_file, 'r') as f:
        for line in f:
            task = string_to_task(line)
            tasks[task.get_id()] = task
    return tasks
