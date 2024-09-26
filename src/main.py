# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
"""

import os
import argparse
import logging
from utils import *
from tqdm import tqdm
from dataset import MultiSessionsGraph
from torch_geometric.data import DataLoader
from model import *
from train import forward
from tensorboardX import SummaryWriter


# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--top_k', type=int, default=20, help='top K indicator for evaluation')
parser.add_argument('--task_node', type=bool, default=False)
parser.add_argument('--task_graph', type=bool, default=False)
parser.add_argument('--ssl_task',type=str,default=None)
parser.add_argument('--lamda',type=float,default=1.0)
parser.add_argument('--device',type=str,default=None)
parser.add_argument('--gnn_model',type=str,default='ggnn')
opt = parser.parse_args()
logging.warning(opt)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.device is not None:
        device = torch.device(opt.device)

    cur_dir = os.getcwd()
    train_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = MultiSessionsGraph(cur_dir + '/../datasets/' + opt.dataset, phrase='test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    all_edges = []
    for i, batch in enumerate(train_loader):
        all_edges.append(batch.edge_index)
    full_graph_edges = torch.cat(all_edges, dim=1).to(device)
    
    log_dir = cur_dir + '/../log/' + str(opt.dataset) + '/' + str(opt)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)

    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37483
    else:
        n_node = 309

    model = GNNModel(full_graph_edges, hidden_size=opt.hidden_size, n_node=n_node, gnn_model=opt.gnn_model).to(device)
    
    if opt.ssl_task is not None:
        ssl = eval(opt.ssl_task)(nhid=opt.hidden_size, device=device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(ssl.linear.parameters()), lr=opt.lr, weight_decay=opt.l2)
    else:
        ssl = None
        optimizer = torch.optim.Adam(model.parameters() , lr=opt.lr, weight_decay=opt.l2)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    logging.warning(model)
    
    for epoch in tqdm(range(opt.epoch)):
        scheduler.step()
        forward(ssl, opt, model, train_loader, device, writer, epoch, top_k=opt.top_k, optimizer=optimizer, train_flag=True)
        with torch.no_grad():
            forward(ssl, opt, model, test_loader, device, writer, epoch, top_k=opt.top_k, train_flag=False)


if __name__ == '__main__':
    main()
