import numpy as np
import torch
from torch_geometric.data import Data

from dadaset_generation import generate_graph, generate_t_to_res, get_features_from_sch, get_answ_f_geom_mean
from models import ResTransformer
from schedule import Schedule, SchAlgorithms, print_schedule
from train import train
from utilities import get_colors_from_output, get_avg_deltas, rand_f_geom

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Net = ResTransformer
model = Net(n_hid=20, n_blocks=6, n_binary=1, node_dim=5, edge_dim=3).to(device)
model.to(device)

# Test graph and schedule creation
pg = generate_graph()
t_to_res = generate_t_to_res(pg)

# Before training:
sch_valid1 = Schedule(pg, t_to_res)
sch_valid1.rand_sgs()
node_fs, edge_idx, edge_fs = get_features_from_sch(sch_valid1)
scha = SchAlgorithms(sch_valid1)
y_dict = get_answ_f_geom_mean(scha)
y_list = torch.tensor([[y_dict[t._id]] for t in sch_valid1._pg._vertices])

d = Data(
    x=node_fs.type(torch.float32),
    edge_index=edge_idx,
    edge_attr=edge_fs.type(torch.float32),
    y=y_list.type(torch.float32)
)
d.to(device)
output = model(d)

colors = get_colors_from_output(output.detach().numpy().tolist())
print_schedule(sch_valid1, colors)

#################
# DATA LIST
path_to_datalist = 'Data/DataSets/dataList_1gr2000.pt'
dl = torch.load(path_to_datalist)

#################
# TRAINING
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train(
    data_list=dl,
    model=model,
    optimizer=optimizer,
    loss_function=torch.nn.MSELoss(),
    device=device
)
#################

output = model(d)

colors = get_colors_from_output(output.detach().numpy().tolist())
print("Predicted:")
print_schedule(sch_valid1, colors)

scha = SchAlgorithms(sch_valid1)
deltas = get_avg_deltas(scha, 1000, rand_f=rand_f_geom, agg_f=lambda x: np.mean(x))
print("Answer:")
print_schedule(sch_valid1, colors=deltas)
