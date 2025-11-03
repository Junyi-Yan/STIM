import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
from torch_geometric.nn import GCNConv
import numpy as np
from parsers import get_parser

class Simple_GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(Simple_GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)  
        output = torch.matmul(adj, support)  

        return output


class Custom_GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, fc_hidden_dim,act):
        super(Custom_GCN, self).__init__()
        self.fc = nn.Linear(in_features, fc_hidden_dim)
        self.gc1 = GraphConvolution(fc_hidden_dim, hidden_features)
        self.gc2 = GraphConvolution(hidden_features, out_features)
        
    def forward(self, x, adj):
        x = torch.relu(self.fc(x))
        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = self.gc2(x, adj)
        x = torch.relu(x)

        return x

class CustomGCNDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(CustomGCNDecoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv1 = self.gcn_layer(in_channels, out_channels)

    def gcn_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)   

class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values

class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values

class WSReadout(nn.Module):   
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round,margin):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.margin_loss = torch.nn.MarginRankingLoss(margin=margin, reduce=False)
        self.margin = margin

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl):
        scs = []
        # positive
        scs.append(self.f_k(h_pl, c))

        pos_dis = F.pairwise_distance(h_pl, c)
        # negative
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-2:-1,:], c_mi[:-1,:]),0)
            scs.append(self.f_k(h_pl, c_mi))
        neg_dis = F.pairwise_distance(h_pl, c_mi)
        logits = torch.cat(tuple(scs))
        margin_label = -1 * torch.ones_like(neg_dis)
        loss2 = self.margin_loss(pos_dis, neg_dis, margin_label)
        return logits,loss2


class Custom_Model(nn.Module):
    def __init__(self,  n_in, n_hid,n_h, fc, activation, negsamp_round, readout,margin):
        super(Custom_Model, self).__init__()
        self.read_mode = readout
        self.gcn = Custom_GCN(n_in, n_hid, n_h, fc,activation)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round, margin)


    def forward(self, seq1, adj, sparse=False):
        h_1 = self.gcn(seq1, adj)
        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:,: -1,:])
            h_mv = h_1[:,-1,:]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:,: -1,:], h_1[:,-2: -1, :])


        ret, loss2 = self.disc(c, h_mv)

        return ret, loss2

class Simple_Model(nn.Module):
    def __init__(self, n_in, n_h, activation, negsamp_round, readout, margin):
        super(Simple_Model, self).__init__()
        self.read_mode = readout
        self.gcn = Simple_GCN(n_in, n_h, activation)
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h, negsamp_round, margin)

    def forward(self, seq1, adj, sparse=False):
        h_1 = self.gcn(seq1, adj, sparse)
        if self.read_mode != 'weighted_sum':
            c = self.read(h_1[:, : -1, :])
            h_mv = h_1[:, -1, :]
        else:
            h_mv = h_1[:, -1, :]
            c = self.read(h_1[:, : -1, :], h_1[:, -2: -1, :])

        ret, loss2 = self.disc(c, h_mv)

        return ret, loss2