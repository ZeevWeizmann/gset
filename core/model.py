import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GATEncoder(nn.Module):
    def __init__(self, n_genes: int, d_hid: int = 64, d_out: int = 16, n_heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(n_genes, d_hid // n_heads, heads=n_heads,
                             add_self_loops=False, dropout=0.0, edge_dim=1)
        self.conv2 = GATConv(d_hid, d_out, heads=1,
                             add_self_loops=False, dropout=0.0, edge_dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(d_out * 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        return self.conv2(x, data.edge_index, data.edge_attr)

    def decode(self, z, src, dst):
        return self.decoder(torch.cat([z[src], z[dst]], dim=1)).squeeze()


class GCNEncoder(nn.Module):
    def __init__(self, n_genes: int, d_hid: int = 64, d_out: int = 16):
        super().__init__()
        self.conv1 = GCNConv(n_genes, d_hid, add_self_loops=False)
        self.conv2 = GCNConv(d_hid, d_out, add_self_loops=False)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        return self.conv2(x, data.edge_index, data.edge_attr)

    def decode(self, z, src, dst):
        return (z[src] * z[dst]).sum(dim=1)
