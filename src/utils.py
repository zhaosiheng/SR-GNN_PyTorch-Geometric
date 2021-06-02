import netorkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeDistance:

    def __init__(self, adj, nclass=4):
        """
        :param graph: Networkx Graph.
        """
        self.adj = adj
        self.graph = nx.from_scipy_sparse_matrix(adj)
        self.nclass = nclass

    def get_label(self):
        path_length = dict(nx.all_pairs_shortest_path_length(self.graph, cutoff=self.nclass-1))
        distance = - np.ones((len(self.graph), len(self.graph))).astype(int)

        for u, p in path_length.items():
            for v, d in p.items():
                distance[u][v] = d

        distance[distance==-1] = distance.max() + 1
        distance = np.triu(distance)
        self.distance = distance
        return torch.LongTensor(distance) - 1

class Base:

    def __init__(self, adj, features, device):
        self.adj = adj
        self.features = features.to(device)
        self.device = device
        self.cached_adj_norm = None

    def get_adj_norm(self):
        if self.cached_adj_norm is None:
            adj_norm = preprocess_adj(self.adj, self.device)
            self.cached_adj_norm= adj_norm
        return self.cached_adj_norm

    def make_loss(self, embeddings):
        return 0

    def transform_data(self):
        return self.get_adj_norm(), self.features
     
class PairwiseDistance(Base):

    def __init__(self, adj, features, nhid, device, idx_train, regression=False):
        self.adj = adj
        self.features = features.to(device)
        self.nfeat = features.shape[1]
        self.cached_adj_norm = None
        self.device = device

        self.labeled = idx_train.cpu().numpy()
        self.all = np.arange(adj.shape[0])
        self.unlabeled = np.array([n for n in self.all if n not in idx_train])

        self.regression = regression
        self.nclass = 4
        if regression:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        else:
            self.linear = nn.Linear(nhid, self.nclass).to(device)

        self.pseudo_labels = None

    def transform_data(self):
        return self.get_adj_norm(), self.features

    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)

    def classification_loss(self, embeddings):
        if self.pseudo_labels is None:
            self.agent = NodeDistance(self.adj, nclass=self.nclass)
            self.pseudo_labels = self.agent.get_label().to(self.device)

        # embeddings = F.dropout(embeddings, 0, training=True)
        self.node_pairs = self.sample(self.agent.distance)
        node_pairs = self.node_pairs
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]

        embeddings = self.linear(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(embeddings, dim=1)
        loss = F.nll_loss(output, self.pseudo_labels[node_pairs])
        print(loss)
        # from metric import accuracy
        # acc = accuracy(output, self.pseudo_labels[node_pairs])
        # print(acc)
        return loss

    def sample(self, labels, ratio=0.1, k=4000):
        node_pairs = []
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]

    def _sample(self, labels, ratio=0.1, k=400):
        # first sample k nodes
        candidates = self.all
        # perm = np.random.choice(candidates, int(ratio*len(candidates)), replace=False)
        perm = np.random.choice(candidates, 300, replace=False)

        node_pairs = []
        # then sample k other nodes to make sure class balance
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
            tmp_0 = tmp[:, 0]
            targets = np.where(tmp_0.reshape(tmp_0.size, 1) == perm)[0]
            # targets = np.array([True if x in perm else False for x in tmp[:, 0]])
            # indices = np.random.choice(np.arange(len(tmp))[targets], k, replace=False)
            indices = np.random.choice(targets, k, replace=False)
            node_pairs.append(tmp[indices])
        node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()
        return node_pairs[0], node_pairs[1]

