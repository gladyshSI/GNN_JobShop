import numpy as np
import torch
import tqdm
from torch_geometric.data import Data

from precedence_graph import PrecedenceGraph
from schedule import Schedule, SchAlgorithms
from utilities import get_avg_deltas, rand_f_geom


# Extract node and edge features from the schedule for dataset generating
def get_features_from_sch(sch: Schedule):
    node_features = []  # Node feature matrix with shape [num_nodes, num_node_features]
    edge_index = []  # Graph connectivity in COO format with shape [2, num_edges]
    edge_features = []  # Edge feature matrix with shape [num_edges, num_edge_features]

    # node_features = [id, rank, p, p_min, p_max]
    scha = SchAlgorithms(sch)
    ranks = scha.ranking()
    for t in sch._pg._vertices:
        id = t._id
        rank = ranks[id]
        p = t._duration
        p_min = t._d_min
        p_max = t._d_max
        node_features.append([id, rank, p, p_min, p_max])
    node_features = torch.tensor(node_features)

    # edge_index
    # edge_features = [0, 1, t_lag] (initial precedence constr) or [1, 0, t_lag] (schedule)
    from_ids = []
    to_ids = []
    for fr, to_ts in sch._edges.items():
        from_ids += [fr] * len(to_ts)
        for to, t_lag in to_ts.items():
            to_ids.append(to)
            # find edge_type
            is_initial = False
            if fr in sch._pg._edges.keys() and to in set(sch._pg._edges[fr]):
                is_initial = True
            edge_features.append([0, 1, t_lag] if is_initial else [1, 0, t_lag])

    edge_index = torch.tensor([from_ids, to_ids], dtype=torch.long)
    edge_features = torch.tensor(edge_features)

    return node_features, edge_index, edge_features


def get_answ_f_geom_mean(scha):
    return get_avg_deltas(scha, 100, rand_f=rand_f_geom, agg_f=lambda x: np.mean(x))


# Generate Data List with one initial graph
# X = [[id, rank, p, p_min, p_max] for each node]
# edge_index = [[from][to]] for each edge (from, to) in schedule._edges
# edge_attr = [[0, 1, t_lag] (for initial precedence relations),
#               [1, 0, t_lag] (for schedule edges)]
# Y = [get_answ_f for each node]
def gen_DList_1gr(pg, t_to_res, get_answ_f, _len, path_to_file):
    data_list = []
    for i in tqdm.tqdm(range(_len)):
        sch = Schedule(pg, t_to_res)
        sch.rand_sgs()
        node_fs, edge_idx, edge_fs = get_features_from_sch(sch)
        scha = SchAlgorithms(sch)
        y_dict = get_answ_f(scha)
        y_list = torch.tensor([[y_dict[t._id]] for t in sch._pg._vertices])

        d = Data(
            x=node_fs.type(torch.float32),
            edge_index=edge_idx,
            edge_attr=edge_fs.type(torch.float32),
            y=y_list.type(torch.float32)
        )
        data_list.append(d)

        # BACKUP:
        if len(data_list) % 10 == 0:
            torch.save(data_list, path_to_file)
    torch.save(data_list, path_to_file)

    return data_list


def generate_graph(v_num=50, ds_min=None, ds_max=None, seed=1, startstart_num_diap=(2, 4), end_num_diap=(2, 4)):
    pg = PrecedenceGraph()
    pg.random_v(v_num=v_num, ds_min=ds_min, ds_max=ds_max, seed=seed)
    pg.random_network(start_num_diap=startstart_num_diap, end_num_diap=end_num_diap, seed=seed)
    return pg


# Dictionary from task_id to list of available resources
def generate_complete_t_to_res(v_num, res_num):
    t_to_res = dict()
    for t in range(v_num):
        t_to_res[t] = [i for i in range(res_num)]
    return t_to_res


if __name__ == '__main__':
    GENERATE_GRAPH = True
    DATASET_SIZE = 2000
    DS_NAME = '2000'
    CASE = 1

    pg = PrecedenceGraph()
    if GENERATE_GRAPH:
        # Generate Precedence Graph:
        path_to_the_graph = 'Data/PrecedenceGraphs/pg.txt'
        pg = generate_graph()
        pg.write_to_file(path_to_the_graph)
    else:
        # or read graph from the file:
        path_to_the_graph = 'Data/PrecedenceGraphs/pg.txt'
        pg.read_from_file(path_to_the_graph)

    # Create map from task_id to list of available resources
    v_num = len(pg._vertices)
    t_to_res = generate_complete_t_to_res(v_num, 6)

    if CASE == 1:
        data_list = gen_DList_1gr(pg=pg,
                                  t_to_res=t_to_res,
                                  get_answ_f=get_answ_f_geom_mean,
                                  _len=DATASET_SIZE,
                                  path_to_file='Data/DataSets/dataList_1gr' + DS_NAME + '.pt')
