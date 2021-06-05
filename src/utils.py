import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

class NodeDistance:

    def __init__(self, data, nclass=4):
        """
        :param graph: Networkx Graph.
        """
        
        
        G = to_networkx(data)
        
        self.graph = G
        
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

    def __init__(self,  nhid, device, regression=False):
        
        
        self.device = device

        self.regression = regression
        self.nclass = 4
        if regression:
            self.linear = nn.Linear(nhid, self.nclass).to(device)
        else:
            self.linear = nn.Linear(nhid, self.nclass).to(device)

        self.pseudo_labels = None


    def make_loss(self, embeddings, data):
        if self.regression:
            return self.regression_loss(embeddings)
        else:
            return self.classification_loss(embeddings, data)

    def classification_loss(self, embeddings, data):
        
        agent = NodeDistance(data, nclass=self.nclass)
        self.pseudo_labels = agent.get_label().to(self.device)

        # embeddings = F.dropout(embeddings, 0, training=True)
        self.node_pairs = self.sample(agent.distance)
        node_pairs = self.node_pairs
        
        embeddings0 = embeddings[node_pairs[0]]
        embeddings1 = embeddings[node_pairs[1]]

        h = self.linear(torch.abs(embeddings0 - embeddings1))
        output = F.log_softmax(h, dim=1)
        loss = F.nll_loss(output, self.pseudo_labels[node_pairs])
        
        # from metric import accuracy
        # acc = accuracy(output, self.pseudo_labels[node_pairs])
        # print(acc)
        return loss

    def sample(self, labels, ratio=0.1, k=4000):
        node_pairs = []
        for i in range(1, labels.max()+1):
            tmp = np.array(np.where(labels==i)).transpose()
       #     indices = np.random.choice(np.arange(len(tmp)), k, replace=False)
            indices = np.random.choice(np.arange(len(tmp)), int(50), replace=True)
            node_pairs.append(tmp[indices])
        node_pairs = np.vstack(node_pairs).transpose()
       # node_pairs = np.array(node_pairs).reshape(-1, 2).transpose()

        return node_pairs[0], node_pairs[1]

