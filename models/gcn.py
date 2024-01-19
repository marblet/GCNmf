"""
version 1.0
date 2021/02/04
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, Sequential
from torch_geometric.nn.conv import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN(nn.Module):
    def __init__(self, data, nhid=32, dropout=0.1, n_layers = 2):
        super(GCN, self).__init__()
        num_features = data.num_node_features
        
        gcn_layers = [
            (nn.Dropout(dropout), 'x -> x'),
            (GCNConv(num_features, nhid), 'x, edge_index -> x'),
            (nn.ReLU(inplace=True), 'x -> x')
        ]
        
        for _ in range(n_layers - 1):
            gcn_layers.append((nn.Dropout(dropout), 'x -> x'))
            gcn_layers.append((GCNConv(nhid, nhid), 'x, edge_index -> x'))
            gcn_layers.append((nn.ReLU(inplace=True), 'x -> x'))
        
        gcn_layers.append((GCNConv(nhid, nhid), 'x, edge_index -> x'))
        self.gcn = Sequential('x, edge_index', gcn_layers)
        
        # Leaf network for observations
        obs_layers = [
            nn.Linear(2, nhid),
            nn.Dropout(dropout),
            nn.ReLU(),
        ]
    
        for _ in range(n_layers - 1):
            obs_layers.append(nn.Linear(nhid, nhid))
            obs_layers.append(nn.Dropout(0.1))
            obs_layers.append(nn.ReLU())
            
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
        for layer in self.gcn:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for layer in self.backbone_fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.414)
                layer.bias.data.fill_(0)

    # def forward(self, data):
    #     x, edge_index, batch = data.x, data.edge_index, data.batch
        
    #     x = self.gcn(x, edge_index)
        
    #     x = global_add_pool(x, batch)
        
    #     logits = self.backbone_fc(x)

    #     return F.sigmoid(logits)
    
    def forward(self, data):
        x, edge_index, batch, obs = data.x, data.edge_index, data.batch, torch.tensor(np.array(data.obs), dtype=torch.float32)
        obs = obs.cpu().to(device)
        
        #Create mask object for obs, since some values have been padded
        mask = obs[:, :, 0] >= 0 
        row_sum = torch.sum(mask, dim=1, keepdim=True) + 1E-9
        mask = (mask / row_sum).unsqueeze(-1).detach()
        
        
        x = self.gcn(x, edge_index)
        
        x = global_add_pool(x, batch)
        
        obs = self.leaf_network_obs(obs)
        obs_pooled = torch.sum(obs * mask, dim=1)
        
        node_features_w_obs = x + obs_pooled
        
        logits = self.backbone_fc(node_features_w_obs)

        return F.sigmoid(logits)


class GCNConvo(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GCNConvo, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.dropout = dropout
        self.fc = nn.Linear(in_features, out_features)
        
        self.reset_parameters()
        

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        x = torch.spmm(adj, x)
        return x