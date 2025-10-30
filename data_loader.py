import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import yaml
import os
from model import Simple_Model,Custom_Model
from utils import *


def load_data(args):
    if args.dataset in ['cora', 'citeseer']:
        adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
        features, _ = preprocess_features(features)

        dgl_graph = adj_to_dgl_graph(adj)
        adj = normalize_adj(adj)
        adj = (adj + sp.eye(adj.shape[0])).todense()
        adj = torch.FloatTensor(adj[np.newaxis])

    elif args.dataset in ['BlogCatalog','Flickr']:
        adj_or, features, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
        adj_path = os.path.join("dataset", f"{args.dataset}.csv")
        adj = np.loadtxt(adj_path, delimiter=',', dtype=float)
        features, _ = preprocess_features(features)

        # if isinstance(adj, np.ndarray):
        #     adj = sp.csr_matrix(adj)
        dgl_graph = adj_to_dgl_graph(adj_or)
        # adj = normalize_adj(adj)
        # adj = (adj + sp.eye(adj.shape[0])).todense()
        # features = torch.FloatTensor(features[np.newaxis])
        adj = torch.FloatTensor(adj[np.newaxis])
        adj = torch.as_tensor(adj, dtype=torch.float32)

    elif args.dataset == 'books':
        data = torch.load(f"./dataset/{args.dataset}.pt")
        features = data.x
        ano_label = (data.y).numpy()

        adj = np.loadtxt(f"./dataset/{args.dataset}adj.csv", delimiter=',', dtype='str')
        features = sp.lil_matrix(features)
        features, _ = preprocess_features(features)

        adj = adj.astype(float).astype(np.int32)
        dgl_graph = adj_to_dgl_graph(sp.csr_matrix(adj))
        adj = torch.FloatTensor((adj.astype(float))[np.newaxis])

    elif args.dataset == 'Weibo':
        data = torch.load(f"./dataset/{args.dataset}.pt")
        features = data.x
        features = prepro_features(features)
        ano_label = (data.y).numpy()

        adj = np.loadtxt(f"./dataset/{args.dataset}.csv", delimiter=',', dtype='str')
        features = sp.lil_matrix(features)
        features, _ = preprocess_features(features)

        adj = adj.astype(float).astype(np.int32)
        dgl_graph = adj_to_dgl_graph(sp.csr_matrix(adj))
        adj = torch.FloatTensor((adj.astype(float))[np.newaxis])

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return adj, features, ano_label, dgl_graph



