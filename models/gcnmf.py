"""
version 1.0
date 2021/02/04
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import GCNConv
from . import GCNmfConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNmf(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.1, n_components=5):
        super(GCNmf, self).__init__()
        num_features = data.num_node_features
        self.gc1 = GCNmfConv(num_features, nhid, data, n_components, dropout)
        self.gc2 = GCNConv(nhid, 1)

        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(x.shape[0], x.shape[0])).to(device)
        adj = adj.to_dense()
        x = self.gc1(x, adj)
        x = self.gc2(x, edge_index)
        x = global_add_pool(x, batch)
        return F.sigmoid(x)
