import torch
from torch_geometric.nn import TransformerConv, Sequential
from math import sqrt


class ResTransformer(torch.nn.Module):
    N_HIDDEN = 20
    N_BLOCKS = 7
    N_BINARY = 1
    NODE_INPUT_DIM = 11
    EDGE_FEATURE_DIM = 2
    transformer_options = dict()

    def __init__(self, n_hid, n_blocks, n_binary, node_dim, edge_dim):
        super(ResTransformer, self).__init__()
        self.N_HIDDEN = n_hid
        self.N_BLOCKS = n_blocks
        self.N_BINARY = n_binary
        self.NODE_INPUT_DIM = node_dim
        self.EDGE_FEATURE_DIM = edge_dim
        self.conv1 = TransformerConv(in_channels=self.NODE_INPUT_DIM,
                                     out_channels=self.N_HIDDEN,
                                     edge_dim=self.EDGE_FEATURE_DIM,
                                     **self.transformer_options)

        residual_block_maker = lambda: (
            Sequential('x, edge_index, edge_attr', [
                (lambda x: torch.relu(x / x.std(-1, keepdim=True).detach()), 'x -> x2'),
                (TransformerConv(in_channels=self.N_HIDDEN,
                                 out_channels=self.N_HIDDEN,
                                 edge_dim=self.EDGE_FEATURE_DIM,
                                 **self.transformer_options), 'x2, edge_index, edge_attr -> y'),
                (lambda x, y: (x + y), 'x, y -> x'),
            ]))
        self.conv2 = Sequential('x, edge_index, edge_attr', [
            (residual_block_maker(), 'x, edge_index, edge_attr -> x')
            for _ in range(self.N_BLOCKS)
        ])

        # TODO: exponentiation? normalization by constant?
        self.linear = torch.nn.Linear(self.N_HIDDEN, self.N_BINARY)
        self.scale = torch.nn.Parameter(torch.tensor([2 ** i for i in range(self.N_BINARY)]), requires_grad=False)
        # self.scale = torch.nn.Parameter(torch.tensor([100]), requires_grad=False)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = (x + 1.).log()
        edge_attr = (edge_attr + 1.).log()

        x = self.conv1(x, edge_index, edge_attr)
        assert not x.isnan().any()
        x = self.conv2(x, edge_index, edge_attr)
        x = x / sqrt(self.N_BLOCKS)  # TODO: performance improves by removing this line ??
        x = self.linear(x)

        # Build output from binary representation: e.g. [1,0,0] -> [1], [0,1,0] -> [2], [1,0,1] -> [5], ...
        # This scale & sum does: x[..., 0] * 2^0 + x[..., 1] * 2^1 + x[..., 2] * 2^2 + ... + x[..., 9]  * 2^9
        # x = (x * self.scale).sum(-1, keepdim=True)

        return x
