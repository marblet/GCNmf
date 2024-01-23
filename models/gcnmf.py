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
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNmf(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.1, n_components=5):
        super(GCNmf, self).__init__()
        num_features = data.num_node_features
        self.gc1 = GCNmfConv(num_features, nhid, data, n_components, dropout)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout
        
        # Leaf network for observations
        obs_layers = [
            nn.Linear(2, nhid),
            nn.Dropout(dropout),
            nn.ReLU(),
        ]
    
        # for _ in range(1 - 1):
        #     obs_layers.append(nn.Linear(nhid, nhid))
        #     obs_layers.append(nn.Dropout(0.1))
        #     obs_layers.append(nn.ReLU())
            
        obs_layers.append(nn.Linear(nhid, nhid))
        
        self.leaf_network_obs = nn.Sequential(*obs_layers)
        
        backbone_layers = [
            nn.Linear(nhid, nhid),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(nhid, 1),
        ]
        self.backbone_fc = nn.Sequential(*backbone_layers)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        for layer in self.backbone_fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.414)
                layer.bias.data.fill_(0)
                
        for layer in self.leaf_network_obs:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.414)
                layer.bias.data.fill_(0)
        
    def test_reset_parameters(self):
        self.gc1.test_reset_parameters()
        self.gc2.reset_parameters()

    # def forward(self, data):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(x.shape[0], x.shape[0])).to(device)
        # adj = adj.to_dense()
        # x = self.gc1(x, adj)
    #     x = self.gc2(x, edge_index)
    #     x = global_add_pool(x, batch)
    #     return F.sigmoid(x)
    
    def forward(self, data):
        x, edge_index, batch, obs = data.x, data.edge_index, data.batch, torch.tensor(np.array(data.obs), dtype=torch.float32)
        obs = obs.cpu().to(device)
        
        #Create mask object for obs, since some values have been padded
        mask = obs[:, :, 0] >= 0 
        row_sum = torch.sum(mask, dim=1, keepdim=True) + 1E-9
        mask = (mask / row_sum).unsqueeze(-1).detach()
        
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(x.shape[0], x.shape[0])).to(device)
        adj = adj.to_dense()
        
        x = self.gc1(x, adj)
        x = self.gc2(x, edge_index)
        
        x = global_add_pool(x, batch)
        
        obs = self.leaf_network_obs(obs)
        obs_pooled = torch.sum(obs * mask, dim=1)
        
        node_features_w_obs = x + obs_pooled
        
        logits = self.backbone_fc(node_features_w_obs)

        return F.sigmoid(logits)
