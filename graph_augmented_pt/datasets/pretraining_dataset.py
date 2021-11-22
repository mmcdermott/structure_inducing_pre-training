import networkx as nx, numpy as np, random, itertools, torch, matplotlib.pyplot as plt, math, tape
from typing import Optional, Callable
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.generators.ego import ego_graph
from networkx.relabel import convert_node_labels_to_integers
from pytorch_lightning import LightningDataModule, seed_everything

import sys
sys.path.append("../PLUS")

from plus.preprocess import preprocess_seq_for_tfm


COLORS = [
    'r', 'c', 'g', 'y', 'purple', 'b', 'brown', 'orange', 'gray', 'pink',
    '#88A97B', '#819283', '#F47A2D', '#41768D', '#412332'
]
SHAPES = [
    "o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|",
    "_",".",",",
]

OPTIONS = list(itertools.product(SHAPES, COLORS))

@dataclass
class PretrainingDataset(LightningDataModule):
    """
    Base LightningDataModule containing graph, node features,
    and necessary information to run PT experiments.
    """
    G:                         nx.Graph
    node_features:             np.ndarray
    add_cls:                   bool                      = True
    mask_rate:                 float                     = 0.15
    batch_size:                int                       = None
    label_sets:                dict                      = None
    tokenizer:                 Optional[BertTokenizer]   = None
    do_subgraph:               bool                      = False
    ego_graph_radius:          int                       = 3
    use_negative_ego_graph_fn: bool                      = False
    num_dataloader_workers:    int                       = 16
    fit_on_init:               bool                      = True # This is used for debug use-cases.
    do_flat_batch:             bool                      = False
    split_by_word:             bool                      = False
    do_from_plus:              bool                      = False

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

    def __post_init__(self):
        # layout graphs with positions using graphviz neato
        if self.batch_size == 'ALL' or self.batch_size is None: self.batch_size = len(self.G)
        if self.label_sets is None: self.label_sets = {}

        self.K = len(self.G)

        self.initialized               = True
        self.graph_initialized         = True
        self.node_features_initialized = True
        self.label_sets_initialized    = True
        self.has_pos                   = False

        if self.do_flat_batch:
            assert not self.do_subgraph, "do_subgraph and do_flat_batch unsupported!"
            self.__collate_fn = self._flat_collate_fn
        else: self.__collate_fn = self._pairs_collate_fn

        if self.fit_on_init: self._fit()

        if self.do_subgraph:
            from torch_geometric.data import Data
            from torch_geometric.data.batch import Batch

        self.use_negative_sample_fn = hasattr(self, 'negative_sample_fn') and self.negative_sample_fn is not None

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

    def __len__(self):
        return len(self.G)

    def homophily_measure(self):
        agreement_fractions = {k: [] for k in self.label_sets}
        for n in self.G:

            neighbors = self.G[n]
            for k, v in self.label_sets.items():
                l = v[n]
                neighbor_labels = np.array([v[nn] for nn in neighbors])

                agreement_fractions[k].append(np.mean(neighbor_labels == l))

        return {k: (np.mean(v), np.std(v)) for k, v in agreement_fractions.items()}, agreement_fractions

    def _compute_pos(self):
        self._pos    = graphviz_layout(self.G, prog="neato")
        self.has_pos = True

    def display(
        self, figsize=5, do_print=True, ax=None, labels_to_display='ALL'
    ):
        assert self.graph_initialized
        assert self.node_features_initialized

        if not self.has_pos: self._compute_pos()

        if labels_to_display is None: labels_to_display = []
        elif labels_to_display == 'ALL':
            labels_to_display = list(self.label_sets.keys())

        assert set(labels_to_display).issubset(set(self.label_sets.keys()))

        if len(labels_to_display) > 1: assert ax is None, "Can't plot multiple labels on the same graph!"

        if do_print:
            print(f"graph has {nx.number_of_nodes(self.G)} nodes with {nx.number_of_edges(self.G)} edges")
            print("Node Features:")

            n_to_print = do_print if type(do_print) is int else len(self.node_features)
            print("\n".join([f"{i}: {x}" for i, x in enumerate(self.node_features[:n_to_print])]))

        n_plts = len(labels_to_display) + 1
        n_cols = int(math.ceil(math.sqrt(n_plts)))
        n_rows = int(math.ceil(n_plts / n_cols))

        if ax is None:
            fig, axes = plt.subplots(
                nrows=n_rows, ncols=n_cols, figsize=(figsize*n_cols, figsize*n_rows)
            )
            if n_cols == 1 and n_rows == 1: axes = [[axes]]
            elif n_rows == 1: axes = [axes]
        else: axes = [[ax]]

        flat_axes = list(itertools.chain.from_iterable(axes))
        for ax in flat_axes: ax.axis('off')

        ax = flat_axes[0]
        ax.set_title("Raw Graph")

        shared_kwargs = {'G': self.G, 'pos': self._pos}
        edge_draw_kwargs = {**shared_kwargs}
        node_draw_kwargs = {**shared_kwargs, 'node_size': 40, 'vmin': 0.0, 'vmax': 1.0}
        all_draw_kwargs = {**edge_draw_kwargs, **node_draw_kwargs}

        nx.draw(**all_draw_kwargs, with_labels=True, ax=ax)

        for i, label in enumerate(labels_to_display):
            ax = flat_axes[i + 1]
            ax.set_title(f"By {label}")

            labels = self.label_sets[label]

            shapes = [OPTIONS[l][0] for l in labels]
            colors = [OPTIONS[l][1] for l in labels]
            for n, (c, s) in enumerate(zip(colors, shapes)):
                nx.draw_networkx_nodes(**node_draw_kwargs, nodelist=[n], ax=ax, node_color=c, node_shape=s)
            nx.draw_networkx_edges(**edge_draw_kwargs, ax=ax)

    def _fit(self):
        assert self.node_features_initialized

        max_len = max(len(s) for s in self.node_features)
        if self.add_cls: max_len += 1

        if self.tokenizer is not None:
            # TODO: This doesn't seem right...
            self.tokenized_node_features = []
            for s in self.node_features:
                tokenized = self.encode(s, max_len)
                self.tokenized_node_features.append(tokenized)

        else:
            self.vocab = set()
            if self.split_by_word:
                self.vocab.update(itertools.chain.from_iterable([s.split() for s in self.node_features]))
            else:
                self.vocab.update(itertools.chain.from_iterable(self.node_features))
            self.vocab = (
                ['[PAD]', '[MASK]'] + (['[CLS]'] if self.add_cls else [])
                + list(self.vocab)
            )
            self.idxmap = {c: i for i, c in enumerate(self.vocab)}
            self.mask_id = self.idxmap['[MASK]']

            self.tokenized_node_features = []
            for s in self.node_features:
                tokenized = [self.idxmap['[CLS]']] if self.add_cls else []

                if self.split_by_word: tokenized += [self.idxmap[c] for c in s.split()]
                else: tokenized += [self.idxmap[c] for c in s]

                tokenized += [self.idxmap['[PAD]']] * (max_len - len(tokenized))
                self.tokenized_node_features.append(tokenized)

            self.pad_id = self.idxmap['[PAD]']

        self.is_fit = True

    def pad_batch(self, s):
        return tape.datasets.pad_sequences(s, constant_value = self.pad_id)

    def collate_fn(self, batch, seed=None):
        self._seed(seed, "collate_fn")

        # We need to validate that the batch looks like what we expect.
        batch_type = type(batch)
        try:
            if batch_type not in (list, tuple): batch = [batch]

            assert type(batch[0]) is dict
        except:
            print(batch_type, batch)
            raise

        keys = batch[0].keys()

        stacked_batch = {k: [batch[0][k]] for k in keys}
        for b in batch[1:]:
            assert b.keys() == keys
            for k in keys: stacked_batch[k].append(b[k])

        if self.do_from_plus:
            if self.do_flat_batch:
                stacked_batch = {
                    k: np.concatenate(v, axis=0) if 'index' in k else v for k, v in stacked_batch.items()
                }
            else: pass
        else:
            stacked_batch = {k: np.concatenate(v, axis=0) if 'subgraph' not in k else v \
                for k, v in stacked_batch.items()}

        # self.__collate_fn is defined in __post_init above, set to the appropriate collate_fn based on the
        # do_flat_collate member variable to avoid doing an unnecessary comparison on each collate call.
        return self.__collate_fn(stacked_batch)

    def __mask(self, np_array):
        mask_probs  = np.random.random(np_array.shape)
        mask_flags  = (mask_probs < self.mask_rate) & (np_array != self.pad_id)
        mask_encoded = self.mask_id
        mask_fills  = np.ones_like(np_array) * mask_encoded
        mask_points = np.where(mask_flags, mask_fills, np_array)
        mlm_labels  = np.where(mask_flags, np_array, np.ones_like(np_array) * -100)

        return mask_points, mlm_labels

    def __build_point_kwargs(self, input_ids):
        # TODO: Padding
        if self.do_from_plus:
            # print(input_ids[0].shape)
            sequences = [torch.from_numpy(sequence).long().squeeze(0) for sequence in input_ids]
            instances = []
            for seq in sequences:
                # tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights
                instance = preprocess_seq_for_tfm(seq, cfg=self.run_config, max_len=self.max_seq_len, augment=True)
                instances.append(instance)

            tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights =\
                tuple(torch.stack([a[i] for a in instances], 0) for i in range(6))

            return {
                'tokens': tokens,
                'segments': segments,
                'input_mask': input_mask,
                'masked_pos': masked_pos,
                'masked_tokens': masked_tokens,
                'masked_weights': masked_weights,
            }

        else:
            mask_points, mlm_labels = self.__mask(input_ids)
            attn_points = (input_ids != self.pad_id).astype(int)

            return self.to_tensor({
                'input_ids':      mask_points,
                'attention_mask': attn_points,
                'labels':         mlm_labels,
            })

    def __build_gml_kwargs(self, stacked_batch):
        N = len(stacked_batch['point_features'])
        if self.do_subgraph:
            raise NotImplementedError('Not supported yet.')

        else:
            labels = np.array([1]*N + [0]*N)               # 2*batch_size

            if self.do_from_plus:
                context_input_ids = stacked_batch['positive_features'] + stacked_batch['negative_features']
                sequences = [torch.from_numpy(sequence).long().squeeze(0) for sequence in context_input_ids]
                instances = []
                for seq in sequences:
                    # tokens, segments, input_mask, masked_pos, masked_tokens, masked_weights
                    instance = preprocess_seq_for_tfm(seq, cfg=self.run_config, max_len=self.max_seq_len, augment=False)
                    instances.append(instance)

                tokens, segments, input_mask =\
                    tuple(torch.stack([a[i] for a in instances], 0) for i in range(3))

                return {
                    'tokens': tokens,
                    'segments': segments,
                    'input_mask': input_mask,
                    'labels': torch.LongTensor(labels)
                }

            else:
                context_input_ids = np.concatenate(            # [2*batch_size x max_len]
                    [stacked_batch['positive_features'], stacked_batch['negative_features']],
                    axis=0
                )
                context_attn = (context_input_ids != self.pad_id)   # [2*batch_size x max_len]
                gml_kwargs = {
                    'input_ids': context_input_ids,
                    'attention_mask': context_attn,
                    'labels': labels
                }
                return self.to_tensor(gml_kwargs)

    def _flat_collate_fn(self, stacked_batch):
        input_ids  = stacked_batch['point_features']

        point_kwargs = self.__build_point_kwargs(input_ids)
        gml_kwargs = self.__build_gml_kwargs(stacked_batch)

        orig_labels = gml_kwargs.pop('labels')

        # TODO: Can we save mem here by passing only an upper triangular matrix? Can we save mem here by
        # making this a 1-bit type?
        N = len(input_ids)
        # TODO: Is this going to be slow?
        node_idx_map = defaultdict(list)
        for i, point_idx in enumerate(stacked_batch['point_index']):  node_idx_map[point_idx].append(i)
        for i, pos_idx in enumerate(stacked_batch['positive_index']): node_idx_map[pos_idx].append(i + N)
        for i, neg_idx in enumerate(stacked_batch['negative_index']): node_idx_map[neg_idx].append(i + 2*N)

        # TODO: This will work in the subgraph case too, save for the fact that it may be the case that
        # two distinct points could generate the same ego-graph, and we need to account for this somehow.
        # When run at a subgraph level of $k$, we need to do something like compute the correspondence between
        # the traph $G$ and its ego graph $E_r(G)$ (where $r$ is the radius), where nodes in $E_r(G)$
        # correspond to $r$-element subgraphs of $G$ that are a valid ego-graph for some node $n_i$ in $G$ (or
        # really a set of nodes $\{n_i \in G\}$ and two nodes of $E_r(G)$ (subgraphs of $G$) are linked in
        # $E_r(G)$ if their subgraphs are linked in $G$ (could also be if their nodesets $n_i$ are linked in
        # $G$ or if they intersect in $G$)?
        batch_subgraph = self.G.subgraph(node_idx_map.keys())
        adj_matrix = np.identity(N*3)
        for node_i, node_j in batch_subgraph.edges():
            for batch_i in node_idx_map[node_i]:
                for batch_j in node_idx_map[node_j]:
                    adj_matrix[batch_i][batch_j] = 1
                    adj_matrix[batch_j][batch_i] = 1

        gml_kwargs['adj_matrix'] = torch.Tensor(adj_matrix)

        return point_kwargs, gml_kwargs

    def _pairs_collate_fn(self, stacked_batch):
        if self.do_from_plus:
            input_ids = stacked_batch['point_features'] + stacked_batch['point_features']

        else:
            input_ids = np.concatenate(                           # [2*batch_size x max_len]
                [stacked_batch['point_features'], stacked_batch['point_features']],
                axis=0
            )

        point_kwargs = self.__build_point_kwargs(input_ids)
        gml_kwargs = self.__build_gml_kwargs(stacked_batch)

        return point_kwargs, gml_kwargs

    def __getitem__(self, i, seed=None):
        self._seed(seed, f"__getitem__({i})")

        assert self.is_fit

        point_features = np.array([self.tokenized_node_features[i]])

        if self.do_subgraph:
            positive_ego_graph = ego_graph(self.G, i, self.ego_graph_radius)
            positive_ego_graph.remove_node(i)

            positive_features = np.take(self.tokenized_node_features, list(positive_ego_graph.nodes), axis=0)
            positive_features = torch.tensor(positive_features)

            positive_ego_graph_relabeled = convert_node_labels_to_integers(positive_ego_graph)
            positive_edges = torch.tensor(list(positive_ego_graph_relabeled.edges)).long().T

            positive_subgraph = Data(
                x = positive_features,
                edge_index = positive_edges
            )

            # Sample negative ego graphs until we get something different than the positive graph.
            while True:
                if self.negative_ego_graph_fn is not None:
                    negative_sample_index = random.choice(list(set(range(self.K)) - set([i])))
                else:
                    negative_sample_index = random.choice(self.negative_ego_graph_fn(i))
                negative_ego_graph = ego_graph(self.G, negative_sample_index, self.ego_graph_radius)
                negative_ego_graph.remove_node(negative_sample_index)
                if positive_ego_graph.nodes != negative_ego_graph.nodes:
                    break

            negative_features = np.take(self.tokenized_node_features, list(negative_ego_graph.nodes), axis=0)
            negative_features = torch.tensor(negative_features)

            negative_ego_graph_relabeled = convert_node_labels_to_integers(negative_ego_graph)
            negative_edges = torch.tensor(list(negative_ego_graph_relabeled.edges)).long().T

            negative_subgraph = Data(
                x = negative_features,
                edge_index = negative_edges
            )

            return {
                'point_index':       [i],
                'point_features':    point_features,
                'positive_index':    [i],
                'positive_subgraph': positive_subgraph,
                'negative_index':    [negative_sample_index],
                'negative_subgraph': negative_subgraph,
            }
        else:
            positive_sample_index = random.choice(list(self.G.neighbors(i)))

            if self.use_negative_sample_fn:
                negative_sample_index = random.choice(self.negative_sample_fn(i))
            else:
                negative_sample_index = random.choice(list(set(range(self.K)) - set(self.G.neighbors(i)) - set([i])))

            positive_features = np.array([self.tokenized_node_features[positive_sample_index]])
            negative_features = np.array([self.tokenized_node_features[negative_sample_index]])

            return {
                'point_index':       [i],
                'point_features':    point_features,                        # [1 x max_len]
                'positive_index':    [positive_sample_index],
                'positive_features': positive_features,                     # [1 x max_len]
                'negative_index':    [negative_sample_index],
                'negative_features': negative_features,                     # [1 x max_len]
            }

    def min_degree(self): return min(d for n, d in self.G.degree)

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_dataloader_workers,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
