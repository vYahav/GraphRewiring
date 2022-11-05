
# following pytorch geometric example from https://dsgiitr.com/blogs/gat/

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import base64, io

# import numpy as np
# from collections import deque, namedtuple


torch.manual_seed(42)  # seed for reproducible numbers

# PYG imports

from torch_geometric.data import Data
from torch_geometric.nn import GATConv,TransformerConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
import torch_geometric.transforms as T

import matplotlib.pyplot as plt
# %matplotlib notebook

import warnings
warnings.filterwarnings("ignore")
name_data = 'Cora'
dataset = Planetoid(root= '/tmp/' + name_data, name = name_data)
dataset.transform = T.NormalizeFeatures()

print(f"Number of Classes in {name_data}:", dataset.num_classes)
print(f"Number of Node Features in {name_data}:", dataset.num_node_features)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Dropout before the GAT layer is used to avoid overfitting in small datasets like Cora.
        # One can skip them if the dataset is sufficiently large.

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def embed(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # shape = [2708,128]
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)  # shape = [2708,8]

        return torch.flatten(x).T

# Training Loop

embed_model = GAT().to(device)

data = dataset[0].to(device)
optimizer = torch.optim.Adam(embed_model.parameters(), lr=0.005, weight_decay=5e-4)


embed_model.train()
for epoch in range(1000):
    embed_model.train()
    optimizer.zero_grad()
    out = embed_model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    if epoch % 200 == 0:
        print(loss)

    loss.backward()
    optimizer.step()



embed_model.eval()
_, pred = embed_model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))

curr_gnn = GAT().to(device)


def train_gnn(state):

    state = torch.reshape(state, [2708, 2708])
    edge_index = dense_to_sparse(state)
    data = dataset[0].to(device)
    data.edge_index = edge_index[0]

    model = GAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    model.train()
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        if epoch % 200 == 0:
            print(loss)

        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()

    del data
    del model
    torch.cuda.empty_cache()
    return acc


def eval_gnn(state):
    with torch.no_grad():
        state = torch.reshape(state, [2708, 2708])
        edge_index = dense_to_sparse(state)
        # data = dataset[0].to(device)
        data.edge_index = edge_index[0]
        embed_model.eval()
        _, pred = embed_model(data).max(dim=1)
        correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        acc = correct / data.test_mask.sum().item()
    return acc


def embed_state(state):
    with torch.no_grad():
        state = torch.reshape(state, [2708, 2708])
        edge_index = dense_to_sparse(state)
        data = dataset[0].to(device)
        data.edge_index = edge_index[0]
        embedding = embed_model.embed(data)

    return embedding
