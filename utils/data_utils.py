import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import torch


class Data:
    def __init__(self, dataset_str):
        if dataset_str in ['cora', 'citeseer']:
            data = load_planetoid_data(dataset_str)
        elif dataset_str in ['amaphoto', 'amacomp']:
            data = load_amazon_data(dataset_str)
        else:
            raise ValueError("Dataset {0} does not exist".format(dataset_str))
        self.adj = data['adj']
        self.edge_list = data['edge_list']
        self.features = data['features']
        self.labels = data['labels']
        self.num_features = self.features.size(1)

    def to(self, device):
        self.adj = self.adj.to(device)
        self.edge_list = self.edge_list.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)


class NodeClsData(Data):
    def __init__(self, dataset_str):
        super(NodeClsData, self).__init__(dataset_str)
        if dataset_str in ['cora', 'citeseer']:
            train_mask, val_mask, test_mask = split_planetoid_data(dataset_str, self.labels)
        else:  # in ['amaphoto', 'amacomp']
            train_mask, val_mask, test_mask = split_amazon_data(dataset_str, self.labels)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = int(torch.max(self.labels)) + 1

    def to(self, device):
        super().to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)


def load_planetoid_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("data/planetoid/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.tensor(out))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1), dtype=torch.int)
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    edge_list = adj_list_from_dict(graph)
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj(edge_list)

    data = {
        'adj': adj,
        'edge_list': edge_list,
        'features': features,
        'labels': labels
    }
    return data


def load_amazon_data(dataset_str):
    with np.load('data/amazon/' + dataset_str + '.npz', allow_pickle=True) as loader:
        loader = dict(loader)

    feature_mat = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                shape=loader['attr_shape']).todense()
    features = torch.tensor(feature_mat)

    adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                            shape=loader['adj_shape']).tocoo()
    edges = [(u, v) for u, v in zip(adj_mat.row.tolist(), adj_mat.col.tolist())]
    G = nx.Graph()
    G.add_nodes_from(list(range(features.size(0))))
    G.add_edges_from(edges)

    edges = torch.tensor([[u, v] for u, v in G.edges()]).t()
    edge_list = torch.cat([edges, torch.stack([edges[1], edges[0]])], dim=1)
    edge_list = add_self_loops(edge_list, loader['adj_shape'][0])
    adj = normalize_adj(edge_list)

    labels = loader['labels']
    labels = torch.tensor(labels).long()

    data = {
        'adj': adj,
        'edge_list': edge_list,
        'features': features,
        'labels': labels
    }
    return data


def split_planetoid_data(dataset_str, labels):
    with open("data/planetoid/ind.{}.y".format(dataset_str), 'rb') as f:
        y = torch.tensor(pkl.load(f, encoding='latin1'))
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    train_mask = index_to_mask(train_idx, labels.shape[0])
    val_mask = index_to_mask(val_idx, labels.shape[0])
    test_mask = index_to_mask(test_idx, labels.shape[0])
    return train_mask, val_mask, test_mask


def split_amazon_data(dataset_str, labels):
    with np.load('data/amazon/' + dataset_str + '_mask.npz', allow_pickle=True) as masks:
        train_idx, val_idx, test_idx = masks['train_idx'], masks['val_idx'], masks['test_idx']
    train_mask = index_to_mask(train_idx, labels.size(0))
    val_mask = index_to_mask(val_idx, labels.size(0))
    test_mask = index_to_mask(test_idx, labels.size(0))
    return train_mask, val_mask, test_mask


def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def get_degree(edge_list):
    row, col = edge_list
    deg = torch.bincount(row)
    return deg


def normalize_adj(edge_list):
    deg = get_degree(edge_list)
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj
