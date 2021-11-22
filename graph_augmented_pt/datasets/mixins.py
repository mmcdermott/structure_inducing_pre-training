import random, torch
from pytorch_lightning import seed_everything

import copy, faiss, itertools, pickle, random, torch, numpy as np, pandas as pd, networkx as nx
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, Tuple, Any
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
import networkx as nx, numpy as np, random, itertools, torch, matplotlib.pyplot as plt, math, tape
from typing import Optional, Callable, Any
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.generators.ego import ego_graph
from networkx.relabel import convert_node_labels_to_integers
from pytorch_lightning import LightningDataModule, seed_everything

class PretrainingDatasetMixins():
    @staticmethod
    def to_tensor(d):
        out = {}
        for k, v in d.items():
            if k in ('input_ids', 'labels'):
                out[k] = torch.LongTensor(v)
            elif k in ('attention_mask', 'head_mask'):
                out[k] = torch.FloatTensor(v)
            elif k in ('context_subgraph',):
                out[k] = v
            else:
                raise AssertError("Invalid key!")
        return out

    def _last_seed(self, name='Generate'):
        for idx, (s, n, time) in enumerate(self._past_seeds[::-1]):
            if n == name:
                idx = len(self._past_seeds) - 1 - idx
                return idx, s

        print(f"Failed to find seed with name {name}!")
        return -1, None

    def _seed(self, seed, key=None):
        if seed is None: seed = random.randint(0, int(1e8))
        if key is None: key = ''
        time = str(datetime.now())

        self.seed = seed
        if hasattr(self, '_past_seeds'): self._past_seeds.append((self.seed, key, time))
        else: self._past_seeds = [(self.seed, key, time)]

        seed_everything(seed)
        return seed
