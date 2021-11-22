import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random, networkx as nx, pickle, numpy as np
from pathlib import Path
from pytorch_lightning import seed_everything

from graph_augmented_pt.constants import *
from graph_augmented_pt.utils.utils import *

seed_everything(0)

raw_data                = Path(RAW_DATASETS_DIR)
ogb                     = raw_data /'ogb'
cached_graph_path       = ogb / 'ogbn_mag_graph.pkl'

G                       = nx.read_gpickle(cached_graph_path)
G_copy                  = G.copy()
train_nodes             = []
val_nodes               = []
test_nodes              = []

TRAIN_FRAC              = 0.50
VAL_FRAC                = 0.25
TEST_FRAC               = 0.25

assert np.isclose(TRAIN_FRAC + VAL_FRAC + TEST_FRAC, 1.0)

edge_list = list(G.edges)
remaining_nodes = set(G.nodes)
random.shuffle(edge_list)

for idx, edge in enumerate(edge_list):    
    u, v = edge
    if u in remaining_nodes and v in remaining_nodes:
        G.remove_nodes_from([u, v])
        remaining_nodes.remove(u)
        remaining_nodes.remove(v)

        samp = random.random()
        if samp < TRAIN_FRAC:
            train_nodes.extend([u, v])
        elif samp < TRAIN_FRAC + VAL_FRAC:
            val_nodes.extend([u, v])
        else:
            test_nodes.extend([u, v])

train_nodes.sort()
val_nodes.sort()
test_nodes.sort()

assert set(train_nodes).intersection(set(val_nodes)) == set()
assert set(test_nodes).intersection(set(val_nodes)) == set()
assert set(train_nodes).intersection(set(test_nodes)) == set()

train_nodes_path        = ogb / 'train_node_ids.pkl'
val_nodes_path          = ogb / 'val_node_ids.pkl'
test_nodes_path         = ogb / 'test_node_ids.pkl'

enpickle(train_nodes, train_nodes_path)
enpickle(val_nodes, val_nodes_path)
enpickle(test_nodes, test_nodes_path)

train_graph_path        = ogb / 'train_graph.pkl'
val_graph_path          = ogb / 'val_graph.pkl'
test_graph_path         = ogb / 'test_graph.pkl'

train_graph             = nx.convert_node_labels_to_integers(G_copy.subgraph(train_nodes), ordering='sorted')
val_graph               = nx.convert_node_labels_to_integers(G_copy.subgraph(val_nodes), ordering='sorted')
test_graph              = nx.convert_node_labels_to_integers(G_copy.subgraph(test_nodes), ordering='sorted')

nx.write_gpickle(train_graph, train_graph_path)
nx.write_gpickle(val_graph, val_graph_path)
nx.write_gpickle(test_graph, test_graph_path)
