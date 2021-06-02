import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeDistance:

    def __init__(self, n_node, edges, nclass=4):
        """
        :param graph: Networkx Graph.
        """
        
        G = nx.DiGraph()

        G.add_edges_from(np.transpose(edges))
        self.graph = G
        print(len(self.graph))
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

     
class PairwiseDistance():

    def __init__(self, n_node, edges, nhid, device, regression=False):
        self.n_node = n_node
        self.edges = edges
        self.device = device

        self.regression = regression
        self.nclass = 4
        if regression:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        else:
            self.linear = nn.Linear(nhid, self.nclass).to(device)

        self.pseudo_labels = None


    def make_loss(self, embeddings):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings)

    def classification_loss(self, embeddings):
        if self.pseudo_labels is None:
            self.agent = NodeDistance(self.n_node, self.edges, nclass=self.nclass)
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

