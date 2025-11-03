from utils import *
import argparse
import os
# # Adding Noise
# def perturb_adj_with_addition(adj, drop_rate=0.1, add_rate=0.1):
#     """
#     :param adj: scipy sparse adjacency matrix (should be symmetric for undirected graph)
#     :param drop_rate: proportion of edges to randomly drop
#     :param add_rate: proportion of edges to randomly add
#     :return: perturbed sparse adjacency matrix (symmetric)
#     """
#     adj = adj.tocoo()
#     N = adj.shape[0]
#
#     # === Step 1: drop existing edges ===
#     num_edges = len(adj.row)
#     perm = np.random.permutation(num_edges)
#     keep = perm[:int(num_edges * (1 - drop_rate))]
#
#     row_kept = adj.row[keep]
#     col_kept = adj.col[keep]
#     data_kept = adj.data[keep]
#
#     dropped_adj = sp.coo_matrix((data_kept, (row_kept, col_kept)), shape=adj.shape)
#
#     # === Step 2: add random edges ===
#     num_add = int(num_edges * add_rate)
#     added_edges = set()
#     exist_edges = set(zip(row_kept, col_kept))
#
#     while len(added_edges) < num_add:
#         i = np.random.randint(0, N)
#         j = np.random.randint(0, N)
#         if i != j and (i, j) not in exist_edges and (j, i) not in exist_edges:
#             added_edges.add((i, j))
#             added_edges.add((j, i))
#
#     row_add, col_add = zip(*added_edges)
#     data_add = np.ones(len(row_add))
#
#     add_adj = sp.coo_matrix((data_add, (row_add, col_add)), shape=adj.shape)
#
#     # === Combine ===
#     new_adj = dropped_adj + add_adj
#     new_adj.data = np.clip(new_adj.data, 0, 1)
#
#     return new_adj.tocsr()

def adjacency_list_tensor_to_adjacency_matrix(adjacency_tensor):
    num_nodes = max(torch.max(adjacency_tensor[0]), torch.max(adjacency_tensor[1])) + 1
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))

    for i in range(adjacency_tensor.size(1)):
        src_node = adjacency_tensor[0, i].item()
        dest_node = adjacency_tensor[1, i].item()
        adjacency_matrix[src_node, dest_node] = 1

    return adjacency_matrix

parser = argparse.ArgumentParser(description='Address Anomalies at Critical Crossroads for Graph Anomaly Detection')
parser.add_argument('--dataset', type=str, default='Flickr')  # 'cora' 'citeseer' 'books' 'BlogCatalog' 'Flickr' 'Weibo'
parser.add_argument('--k1', type=float)
parser.add_argument('--k2', type=float)
args = parser.parse_args()

if args.k1 is None:
    if args.dataset == 'BlogCatalog':
        args.k1 = 0.5
    elif args.dataset == 'Flickr':
        args.k1 = 0.45
    elif args.dataset == 'Weibo':
        args.k1 = 0.95

if args.k2 is None:
    if args.dataset == 'BlogCatalog':
        args.k2 = 0.25
    elif args.dataset == 'Flickr':
        args.k2 = 0.15
    elif args.dataset == 'Weibo':
        args.k2 = 0.60

if args.dataset in ['cora','citeseer','BlogCatalog','Flickr']:
    adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
elif args.dataset in ['books','Weibo']:
    data = torch.load("./dataset/{}.pt".format(args.dataset))
    adj = adjacency_list_tensor_to_adjacency_matrix(data.edge_index)
    # adj = perturb_adj_with_addition(adj, drop_rate=0.025, add_rate=0.025)


adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
adj_matrix = torch.FloatTensor(adj)


if args.dataset == 'books':
    np.savetxt(f"./dataset/{args.dataset}adj.csv", adj_matrix, fmt="%.4f", delimiter=",")

elif args.dataset in ['BlogCatalog','Flickr','Weibo']:
    adj_ = adj_matrix
    adj_[(adj_matrix != 0) & (adj_matrix != 1)] = 1
    node_weights = torch.sum(adj_matrix, axis=1)
    edge_nums = torch.sum(adj_)
    sorted_nodes = torch.argsort(node_weights, descending=True)
    top_percent_nodes = sorted_nodes[:int(args.k1 * len(sorted_nodes))]

    for node in top_percent_nodes:
        edges = np.where(adj_matrix[node] > 0)[0]
        edges = edges[edges != (node.item())]
        sorted_edges_by_count = sorted(edges, key=lambda edge: node_weights[edge], reverse=False)
        drop_num = int(edge_nums / 2 / adj_matrix.shape[0] * args.k2)
        edges_to_remove = sorted_edges_by_count[:drop_num]
        for edge in edges_to_remove:
            adj_matrix[node, edge] = 0
            adj_matrix[edge, node] = 0

    save_path = os.path.join("dataset", f"{args.dataset}.csv")
    np.savetxt(save_path, adj_matrix, fmt="%.4f", delimiter=",")


