import argparse
# Set argument
def get_parser():
    parser = argparse.ArgumentParser(description='Address Anomalies at Critical Crossroads for Graph Anomaly Detection')
    parser.add_argument('--dataset', type=str, default='Flickr')  # 'cora' 'citeseer' 'books' 'BlogCatalog' 'Flickr' 'Weibo'
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--n_hid', type=int, default=128)
    parser.add_argument('--fc_hidden', type=int)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--subgraph_size', type=int)
    parser.add_argument('--readout', type=str, default='avg')  # max min  weighted_sum
    parser.add_argument('--auc_test_rounds', type=int, default=256)
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--margin', type=float)

    args = parser.parse_args()

    if args.embedding_dim is None:
        if args.dataset in ['cora','citeseer','books','Weibo','Flickr']:
            args.embedding_dim = 64
        elif args.dataset == 'BlogCatalog':
            args.embedding_dim = 32

    if args.lr is None:
        if args.dataset =='cora':
            args.lr = 0.013
        elif args.dataset == 'citeseer':
            args.lr = 0.009
        elif args.dataset == 'books':
            args.lr = 0.0002
        elif args.dataset == 'BlogCatalog':
            args.lr = 0.003
        elif args.dataset == 'Flickr':
            args.lr = 0.0014
        elif args.dataset == 'Weibo':
            args.lr = 0.000085

    if args.num_epoch is None:
        if args.dataset in ['cora']:
            args.num_epoch = 120
        if args.dataset in ['citeseer', 'books']:
            args.num_epoch = 100
        elif args.dataset in ['BlogCatalog','Flickr']:
            args.num_epoch = 400
        elif args.dataset in ['Weibo']:
            args.num_epoch = 450


    if args.subgraph_size is None:
        if args.dataset in ['cora','citeseer']:
            args.subgraph_size = 4
        elif args.dataset in ['books','BlogCatalog','Flickr','Weibo']:
            args.subgraph_size = 5

    if args.gamma is None:
        if args.dataset in ['cora', 'books']:
            args.gamma = 0.8
        elif args.dataset in ['citeseer','BlogCatalog']:
            args.gamma = 0.95
        elif args.dataset in ['Flickr']:
            args.gamma = 0.85   
        elif args.dataset in ['Weibo']:
            args.gamma = 0.7   

    if args.margin is None:
        if args.dataset in ['cora','citeseer','BlogCatalog','Flickr']:
            args.margin = 0.75
        elif args.dataset == 'books':
            args.margin = 0.65
        elif args.dataset == 'Weibo':
            args.margin = 0.1

    if args.fc_hidden is None:
        if args.dataset in ['Flickr']:
            args.fc_hidden = 8000
        elif args.dataset == 'BlogCatalog':
            args.fc_hidden = 2600

    return args
