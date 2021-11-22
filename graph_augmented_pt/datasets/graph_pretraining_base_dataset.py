import copy, math, faiss, itertools, pickle, random, torch
import matplotlib.pyplot as plt, numpy as np, pandas as pd, networkx as nx
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from pytorch_lightning import LightningDataModule, seed_everything
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from typing import Any, Callable, Optional, Tuple

from .mixins import *
from ..constants import *
from ..utils.utils import depickle, enpickle

@dataclass
class GraphPretrainingDataset(LightningDataModule, PretrainingDatasetMixins):
    """
    Base LightningDataModule containing graph, node features,
    and necessary information to run PT experiments.
    """
    labelset_G:                nx.Graph
    dataset:                   Any
    batch_size:                int                       = None
    num_dataloader_workers:    int                       = 16
    do_flat_batch:             bool                      = False
    do_sppt:                   bool                      = True

    def __post_init__(self):
        # layout graphs with positions using graphviz neato
        if self.batch_size == 'ALL' or self.batch_size is None: self.batch_size = len(self.dataset)

        self.K = len(self.dataset)

        self.initialized               = True
        self.graph_initialized         = True
        self.label_sets_initialized    = True
        self.has_pos                   = False

        self._set_collate_fn()

    def _set_collate_fn(self):
        if self.do_flat_batch: self.__collate_fn = self._flat_collate_fn
        else: self.__collate_fn = self._pairs_collate_fn

    def __len__(self): return len(self.dataset)

    def homophily_measure(self):
        agreement_means, agreement_stds = [], []

        for n, labelset in enumerate(self.dataset.Y_labelset):
            neighbor_labelsets = self.labelset_G[labelset]
            neighboring_nodes = [
                nn for ls in neighbor_labelsets for nn in self.dataset.labelset_to_samp_idx[ls] if nn != n
            ]
            neighboring_Ys = self.dataset.Y_np[neighboring_nodes]
            neighboring_Ys_eq = (neighboring_Ys == self.dataset.Y_np[n])

            agreement_means.append(neighboring_Ys_eq.mean(axis=0))
            agreement_stds.append(neighboring_Ys_eq.std(axis=0))

        return (
            (np.mean(agreement_means, axis=0), np.std(agreement_means, axis=0)),
            (np.mean(agreement_stds, axis=0), np.std(agreement_stds, axis=0)),
        ), (agreement_means, agreement_stds)

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

        graph_feats = ('point_features', 'positive_features', 'negative_features')
        stacked_batch = {
            k: np.concatenate(v, axis=0) if k not in graph_feats else v for k, v in stacked_batch.items()
        }

        # self.__collate_fn is defined in __post_init above, set to the appropriate collate_fn based on the
        # do_flat_collate member variable to avoid doing an unnecessary comparison on each collate call.
        return self.__collate_fn(stacked_batch)

    @property
    def pt_batch_cls(self): return self.dataset.pt_batch_cls

    def __build_point_kwargs(self, input_ids):
        return {'input_ids': self.pt_batch_cls.from_data_list(input_ids), 'attention_mask': None}

    def __build_gml_kwargs(self, stacked_batch):
        if not self.do_sppt: return {}, [[], []]

        N = len(stacked_batch['point_features'])
        drop_indexes = [
            [i for i, v in enumerate(stacked_batch['positive_features']) if v is None],
            [i for i, v in enumerate(stacked_batch['negative_features']) if v is None],
        ]
        points = [
            [v for i, v in enumerate(stacked_batch['positive_features']) if v is not None],
            [v for i, v in enumerate(stacked_batch['negative_features']) if v is not None],
        ]
        labels = np.array([1]*len(points[0]) + [0]*len(points[1]))

        context_input_ids = Batch.from_data_list(points[0] + points[1])

        return {
            'input_ids': context_input_ids, 'labels': torch.LongTensor(labels),
            'attention_mask': None,
        }, drop_indexes

    def _flat_collate_fn(self, stacked_batch):
        gml_kwargs, drop_indexes = self.__build_gml_kwargs(stacked_batch)
        point_kwargs = self.__build_point_kwargs(stacked_batch['point_features'])

        if not self.do_sppt: return point_kwargs, gml_kwargs

        orig_labels = gml_kwargs.pop('labels')

        N = len(stacked_batch['point_features'])
        node_idx_map = defaultdict(list)

        for i, point_idx in enumerate(stacked_batch['point_index']):  node_idx_map[point_idx].append(i)

        total_pos_idx = 0
        for i, pos_idx in enumerate(stacked_batch['positive_index']):
            if pos_idx is None: continue
            total_pos_idx += 1
            node_idx_map[pos_idx].append(i + N)

        total_neg_idx = 0
        for i, neg_idx in enumerate(stacked_batch['negative_index']):
            if neg_idx is None: continue
            total_neg_idx += 1
            node_idx_map[neg_idx].append(i + N + total_pos_idx)

        batch_subgraph = nx.Graph()
        batch_subgraph.add_nodes_from(node_idx_map.keys())

        nodes_set = set(node_idx_map.keys())

        for n in node_idx_map.keys():
            batch_subgraph.add_edges_from((n, nn) for nn in nodes_set.intersection(self.neighbors_of_node(n)))

        adj_matrix = np.identity(N + total_pos_idx + total_neg_idx)
        for node_i, node_j in batch_subgraph.edges():
            for batch_i in node_idx_map[node_i]:
                for batch_j in node_idx_map[node_j]:
                    adj_matrix[batch_i][batch_j] = 1
                    adj_matrix[batch_j][batch_i] = 1

        gml_kwargs['adj_matrix'] = torch.Tensor(adj_matrix)

        return point_kwargs, gml_kwargs

    def _pairs_collate_fn(self, stacked_batch):
        gml_kwargs, drop_indexes = self.__build_gml_kwargs(stacked_batch)
        points = [
            [v for i, v in enumerate(stacked_batch['point_features']) if i not in drop_indexes[0]],
            [v for i, v in enumerate(stacked_batch['point_features']) if i not in drop_indexes[1]],
        ]
        point_kwargs = self.__build_point_kwargs(points[0] + points[1])

        return point_kwargs, gml_kwargs

    def neighbors_of_node(self, i):
        point_ls = self.dataset.Y_labelset[i]
        return list(set([
            n for ls in self.labelset_G.neighbors(point_ls) for n in self.dataset.labelset_to_samp_idx[ls]
        ] + [
            n for n in self.dataset.labelset_to_samp_idx[point_ls]
        ]) - {i})

    @property
    def all_nodes(self): return list(range(self.K))

    def __getitem__(self, i, seed=None):
        self._seed(seed, f"__getitem__({i})")

        point_features = self.dataset[i]

        if self.do_sppt:
            all_possible_positive_indexes = self.neighbors_of_node(i)
            all_possible_negative_indexes = list(set(self.all_nodes) - set(all_possible_positive_indexes) - {i})

            has_pos = len(all_possible_positive_indexes) > 0
            has_neg = len(all_possible_negative_indexes) > 0

            positive_sample_index = random.choice(all_possible_positive_indexes) if has_pos else None
            negative_sample_index = random.choice(all_possible_negative_indexes) if has_neg else None

            positive_features = self.dataset[positive_sample_index] if has_pos else None
            negative_features = self.dataset[negative_sample_index] if has_neg else None
        else:
            positive_sample_index = None
            negative_sample_index = None
            positive_features = None
            negative_features = None

        return {
            'point_index':       [i],
            'point_features':    point_features,                        # [1 x max_len]
            'positive_index':    [positive_sample_index],
            'positive_features': positive_features,                     # [1 x max_len]
            'negative_index':    [negative_sample_index],
            'negative_features': negative_features,                     # [1 x max_len]
        }

    @property
    def node_degrees(self):
        if not hasattr(self, '_node_degrees'):
            self._node_degrees = []
            for n, L in self.dataset.labelset_to_samp_idx.items():
                degree = L + sum(len(self.dataset.labelset_to_samp_idx[ls]) for ls in self.labelset_G[n])
                self._node_degrees.append((degree, L))
        return self._node_degrees

    @property
    def min_degree(self):
        if not hasattr(self, '_min_degree'):
            self._min_degree = min(degree for degree, num_nodes in self._node_degrees)
        return self._min_degree

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

class AugmentedDatasetMixin:
    _DISTS_FN_PFX   = 'dists'
    _SPLITS_FN      = 'splits.pkl'
    _DATA_SPLIT_DIR = 'SPPT_processed_{held_out_data_frac}_{held_out_task_frac}_{seed}'
    _Y_VAR          = 'TODO: overwrite'

    def _init_mixin(
        self, dataset_dir,
        n_probe:               Optional[int]          = None,
        max_nearest_neighbors: Optional[int]          = None,
        held_out_data_frac:    Optional[Tuple[float]] = None,
        held_out_task_frac:    Optional[Tuple[float]] = None,
        seed:                  Optional[int]          = None,
        num_layers:             Optional[int]          = None,
        context_size:          Optional[int]          = None,
        do_context_pred:       bool                   = False,
        do_masking:            bool                   = True,
        mask_rate:             Optional[float]        = None,
        mask_edge:             Optional[int]          = None,
        do_true_dists:         bool                   = True,
        do_compute_dists:      bool                   = True,
    ):
        self.do_compute_dists   = do_compute_dists

        if self.do_compute_dists:
            assert do_true_dists or ((n_probe is not None) and (max_nearest_neighbors is not None))

        self.do_true_dists      = do_true_dists

        self.raw_data_indices = np.array(list(self.indices()))

        self.total_n_samps = len(self.raw_data_indices)
        self.total_n_tasks = len(self.raw_task_indices)

        if (held_out_data_frac is None) or (held_out_task_frac is None):
            assert (held_out_data_frac is None) and (held_out_task_frac is None)
        else:
            assert seed is not None
            assert sum(held_out_data_frac) < 1 and sum(held_out_task_frac) < 1, \
                f"These should sum to less than 1; the leftover is assumed to be the 'train' set."
            assert len(held_out_data_frac) == len(held_out_task_frac)

        self.dataset_dir           = dataset_dir
        self.n_probe               = n_probe
        self.max_nearest_neighbors = max_nearest_neighbors
        self.held_out_data_frac    = held_out_data_frac
        self.held_out_task_frac    = held_out_task_frac
        self.seed                  = seed

        self.set_graph_PT_params(
            do_context_pred, do_masking, num_layers, context_size, mask_rate, mask_edge
        )

        if self.do_true_dists: self._dist_kwargs = {}
        else: self._dist_kwargs = {'n_probe': n_probe, 'max_nearest_neighbors': max_nearest_neighbors}

        self._cls_kwargs = {
            'held_out_data_frac':    self.held_out_data_frac,
            'held_out_task_frac':    self.held_out_task_frac,
            'seed':                  self.seed,
        }

        self._process_and_save()

    @property
    def pt_batch_cls(self):
        raise NotImplementedError("Must be overwritten")
        #if self.do_context_pred: return BatchSubstructContext
        #elif self.do_masking: return BatchMasking
        #else: return Batch

    @classmethod
    def _labelset_key(self, y_np_arr):
        raise NotImplementedError("Should overwrite in base!")
        # If missingness
        # return frozenset(np.where(y_np_arr == 1)[0]), frozenset(np.where(y_np_arr == 0)[0])
        # If no missingness and y = 0 is more common
        # return frozenset(np.where(y_np_arr == 0)[0])
        # If no missingness and y = 1 is more common
        # return frozenset(np.where(y_np_arr == 1)[0])

    @staticmethod
    def _get_dist(ls_1, ls_2, n_tasks):
        raise NotImplementedError("Should overwrite in derived class!")

        # If missingness
        #(ls_true, ls_false) = ls_1
        #(ls_true_2, ls_false_2) = ls_2

        #i_true_j_false   = len(ls_true.intersection(ls_false_2))
        #i_false_j_true   = len(ls_true_2.intersection(ls_false))
        #i_true_j_true    = len(ls_true.intersection(ls_true_2))
        #i_false_j_false  = len(ls_false.intersection(ls_false_2))

        #return (1/2) * (n_tasks + i_true_j_false + i_false_j_true - i_true_j_true - i_false_j_false)

        # If no missingness
        #(ls_true, ls_false) = ls_1
        #(ls_true_2, ls_false_2) = ls_2

        #return len(ls_true.symmetric_difference(ls_true_2))

    @property
    def do_hold_out(self): return not (self.held_out_data_frac is None or self.held_out_task_frac is None)
    @property
    def num_tasks(self): return len(self[0][self._Y_VAR])
    @property
    def num_samples(self): return len(self)

    def _init_split_indexes(self):
        if not self.do_hold_out: return

        if self.split_filepath.is_file(): self._data_index, self._task_index = depickle(self.split_filepath)
        else:
            seed_everything(self.seed)

            self._data_index = np.random.permutation(self.raw_data_indices)
            self._task_index = np.random.permutation(self.raw_task_indices)

    @property
    def data_splits(self):
        if not hasattr(self, '_data_splits'): self._init_split()
        return self._data_splits
    @property
    def task_splits(self):
        if not hasattr(self, '_task_splits'): self._init_split()
        return self._task_splits

    def _init_split(self):
        if not self.do_hold_out: return
        self._init_split_indexes()

        held_out_data_samples = [int(round(f * self.total_n_samps)) for f in self.held_out_data_frac]
        held_out_task_samples = [int(round(f * self.total_n_tasks)) for f in self.held_out_task_frac]

        st_data = self.total_n_samps - sum(held_out_data_samples)
        st_task = self.total_n_tasks - sum(held_out_task_samples)

        data_splits, task_splits = [list(self._data_index[:st_data])], [list(self._task_index[:st_task])]

        for size in held_out_data_samples:
            data_splits.append(list(self._data_index[st_data:st_data+size]))
            st_data += size
        for size in held_out_task_samples:
            task_splits.append(list(self._task_index[st_task:st_task+size]))
            st_task += size

        self._data_splits, self._task_splits = data_splits, task_splits
        self.set_split(0, 0)

    def set_graph_PT_params(
        self, do_context_pred: bool, do_masking: bool,
        num_layers:    Optional[int]   = None,
        context_size: Optional[int]   = None,
        mask_rate:    Optional[float] = None,
        mask_edge:    Optional[int]   = None,
    ):
        assert not (do_masking and do_context_pred)
        if do_masking: assert (mask_rate is not None) and (mask_edge is not None)
        if do_context_pred: assert (num_layers is not None) and (context_size is not None)

        self.do_graph_PT     = do_context_pred or do_masking

        self.do_masking      = do_masking
        self.mask_rate       = mask_rate if do_masking else None
        self.mask_edge       = mask_edge if do_masking else None

        self.do_context_pred = do_context_pred
        self.num_layers       = num_layers if do_context_pred else None
        self.context_size    = context_size if do_context_pred else None
        self.context_l1      = (num_layers - 1) if do_context_pred else None
        self.context_l2      = (self.context_l1 + context_size) if do_context_pred else None

        if hasattr(self, '_graph_PT_transform'): delattr(self, '_graph_PT_transform')
        self.set_transform()

    def _set_graph_PT_transform(self):
        raise NotImplementedError("Overwrite in derived class!")
        #if not self.do_graph_PT: self._graph_PT_transform = torch.nn.Identity()
        #elif self.do_context_pred:
        #    self._graph_PT_transform = ExtractSubstructureContextPair(
        #        self.num_layers, self.context_l1, self.context_l2
        #    )
        #elif self.do_masking:
        #    self._graph_PT_transform = MaskAtom(
        #        num_atom_type = 119, num_edge_type = 5,
        #        mask_rate = self.mask_rate, mask_edge=self.mask_edge
        #    )

    @property
    def graph_PT_transform(self):
        if not self.do_graph_PT: return torch.nn.Identity()

        if not hasattr(self, '_graph_PT_transform'): self._set_graph_PT_transform()
        return self._graph_PT_transform

    def lim_tasks(self, data: Data):
        out = copy.copy(data)
        out[self._Y_VAR] = out[self._Y_VAR][self.raw_task_indices[self.task_splits[self._task_split]]]

        if self.do_graph_PT: out = self.graph_PT_transform(out)
        return out

    def set_transform(self):
        if self.do_hold_out: self.transform = self.lim_tasks
        elif self.do_graph_PT: self.transform = self.graph_PT_transform
        else: self.transform = None

    def set_data_split(self, data_split: int):
        assert self.do_hold_out
        if not hasattr(self, 'data_splits'): self._init_split()
        assert data_split < len(self.data_splits)

        self._data_split = data_split
        self.__indices__ = self.raw_data_indices[self.data_splits[self._data_split]]

    def set_task_split(self, task_split: int):
        assert self.do_hold_out
        if not hasattr(self, 'task_splits'): self._init_split()
        assert task_split < len(self.task_splits)

        self._task_split = task_split
        self.set_transform()

    def set_split(self, data_split: int, task_split: int):
        self.set_data_split(data_split)
        self.set_task_split(task_split)

    @property
    def Y(self):
        if self.do_hold_out: assert self._data_split == 0
        if not hasattr(self, '_Y'):
            self._Y = torch.vstack([self[i][self._Y_VAR] for i in range(len(self))])
        return self._Y
    @property
    def Y_means(self):
        if not hasattr(self, '_Y_means'): self._Y_means = np.nanmean(self.Y_np, axis=0)
        return self._Y_means
    @property
    def Y_np(self):
        if self.do_hold_out: assert self._data_split == 0
        if not hasattr(self, '_Y_np'):
            Y_np = self.Y.float().detach().cpu().numpy()

            is_missing = (Y_np == 0)
            Y_np[is_missing] = np.NaN
            Y_np = (Y_np + 1) / 2

            self._Y_np = Y_np
        return self._Y_np
    @property
    def labelset_to_samp_idx(self):
        if self.do_hold_out: assert self._data_split == 0
        if not hasattr(self, '_labelset_to_samp_idx'):
            idxmap = defaultdict(list)
            for y_idx, y_labelset_idx in enumerate(self.Y_labelset):
                idxmap[y_labelset_idx].append(y_idx)
            self._labelset_to_samp_idx = idxmap
        return self._labelset_to_samp_idx
    @property
    def Y_labelset(self):
        if not hasattr(self, '_Y_labelset'): self._init_labelsets()
        return self._Y_labelset
    @property
    def unique_labels(self):
        if not hasattr(self, '_unique_labels'): self._init_labelsets()
        return self._unique_labels
    @property
    def labelset_vocab(self):
        if not hasattr(self, '_labelset_vocab'): self._init_labelsets()
        return self._labelset_vocab
    @property
    def labelset_idxmap(self):
        if not hasattr(self, '_labelset_idxmap'): self._init_labelsets()
        return self._labelset_idxmap
    @property
    def n_labelsets(self):
        if not hasattr(self, '_n_labelsets'): self._init_labelsets()
        return self._n_labelsets

    def _init_labelsets(self):
        if not self.do_compute_dists: return
        if self.do_hold_out: assert self._data_split == 0

        self._unique_labels = defaultdict(set)
        _Y_labelset   = []
        for i, y in tqdm(enumerate(self.Y_np), total=self.num_samples, leave=False):
            ls = self._labelset_key(y)
            self._unique_labels[ls].add(i)
            _Y_labelset.append(ls)

        self._labelset_vocab = sorted(self._unique_labels.keys(), key=lambda y: -len(self._unique_labels[y]))
        self._labelset_idxmap = {y: i for i, y in enumerate(self._labelset_vocab)}

        self._Y_labelset = [self._labelset_idxmap[ls] for ls in _Y_labelset]
        self._n_labelsets = len(self._labelset_vocab)

    def _init_Y_dist(self):
        if not self.do_compute_dists: return

        if self.dists_filepath.is_file(): self._load_dists()
        elif self.do_true_dists:
            print(f"{self.dists_filepath} does not exist. Computing true dists...")
            self._init_true_Y_dists()
        else:
            print(f"{self.dists_filepath} does not exist. Computing faiss dists...")
            self._init_faiss_Y_dist()

    @property
    def recoverable_attrs(self):
        attrs = (
            '_Y', '_Y_np', '_Y_means', '_Y_labelset', '_labelset_idxmap', '_labelset_vocab', '_n_labelsets',
            '_unique_labels', '_data_splits', '_task_splits', '_labelset_dists', '_labelset_indexes',
        )
        if hasattr(self, '_save_dist_attrs'): attrs += self._save_dist_attrs

        return [a for a in attrs if hasattr(self, a)]

    def _drop_recoverable_attrs(self, save={}):
        out = {attr: getattr(self, attr) for attr in self.recoverable_attrs if attr not in save}
        for attr in out.keys(): delattr(self, attr)
        return out

    def _init_true_Y_dists(self):
        n_labelsets = self.n_labelsets # to initialize if needed
        n_tasks     = self.num_tasks

        self._drop_recoverable_attrs(save={'_labelset_vocab',})

        print("computing directly")
        self._true_dists = np.zeros((n_labelsets, n_labelsets))

        for i, ls_1 in tqdm(enumerate(self.labelset_vocab), total=self.n_labelsets, leave=False):
            for j, ls_2 in enumerate(self.labelset_vocab):
                self._true_dists[i, j] = self._get_dist(ls_1, ls_2, n_tasks)

        self._save_dist_attrs = tuple(sorted(('_true_dists',)))
        self._postprocess_dists()

        self._init_labelsets()

    @property
    def true_dists(self):
        assert self.do_compute_dists and self.do_true_dists
        if not hasattr(self, '_true_dists'): self._init_Y_dist()
        return self._true_dists
    @property
    def labelset_indexes(self):
        if not hasattr(self, '_labelset_indexes'): self._postprocess_dists()
        return self._labelset_indexes
    @property
    def labelset_dists(self):
        if not hasattr(self, '_labelset_dists'): self._postprocess_dists()
        return self._labelset_dists

    def _postprocess_dists(self):
        if not self.do_true_dists: return
        self._labelset_indexes = np.argsort(self.true_dists, axis=1)
        self._labelset_dists   = np.take_along_axis(self.true_dists, self._labelset_indexes, axis=1)

    def _load_dists(self):
        dist_attrs = depickle(self.dists_filepath)
        for attr, val in dist_attrs.items(): setattr(self, attr, val)

        self._save_dist_attrs = tuple(sorted(dist_attrs.keys()))

        self._postprocess_dists()

    def _init_faiss_Y_dist(self):
        raise NotImplementedError(
            "Currently doesn't support data with non-imputable missingness. "
            "See prior commits for partial code."
        )

    def set_dists_params(self, **dist_kwargs):
        if not self.do_compute_dists: return

        self._dist_kwargs = dist_kwargs.keys()
        for attr, val in dist_kwargs.items(): setattr(self, attr, val)

        else: self._init_Y_dist()

    @classmethod
    def _load(
        cls, dataset_dir: Path, context_pred_kwargs: dict, cls_kwargs: dict,
        dist_kwargs: Optional[dict] = None, do_compute_dists: bool = False,
    ):
        if dist_kwargs is None: dist_kwargs = {}

        out = cls(
            dataset_dir, do_compute_dists=do_compute_dists,
            **cls_kwargs, **context_pred_kwargs, **dist_kwargs
        )

        return out

    @property
    def dir(self): return self.dataset_dir / self._DATA_SPLIT_DIR.format(**self._cls_kwargs)
    @property
    def _dists_filename(self):
        if not(self._dist_kwargs): return f"{self._DISTS_FN_PFX}.pkl"
        dists_sfx_keys = sorted(list(self._dist_kwargs.keys()))
        dists_sfx = '|'.join("{k}:{self._dist_kwargs[k]}" for k in dists_sfx_keys)
        return f"{self._DISTS_FN_PFX}-{dists_sfx}.pkl"
    @property
    def dists_filepath(self): return self.dir / self._dists_filename
    @property
    def split_filepath(self): return self.dir / self._SPLITS_FN

    def _process_and_save(self):
        self._init_split()
        self._init_Y_dist()
        self._save()

    def _save(self):
        self.dir.mkdir(parents=True, exist_ok=True)

        if self.do_compute_dists:
            to_save = {attr: getattr(self, attr) for attr in self._save_dist_attrs}
            if not self.dists_filepath.is_file(): enpickle(to_save, self.dists_filepath)
        if self.do_hold_out:
            to_save = (self._data_index, self._task_index)
            if not self.split_filepath.is_file(): enpickle(to_save, self.split_filepath)

    @property
    def queried_dists(self):
        return enumerate(zip(self.dists, self.indexes))

class GraphDerivedPretrainingDataset(GraphPretrainingDataset):
    _DATA_SUBPATH = '' #TODO: overwrite
    _DATASET_CLS = None # TODO: overwrite

    def __init__(
        self,
        seed:               int                    = 1,
        data_dir:           Path                   = RAW_DATASETS_DIR,
        n_neighbors:        Optional[int]          = 25,
        radius:             Optional[float]        = None,
        held_out_data_frac: Optional[Tuple[float]] = (0.1, 0.1),
        held_out_task_frac: Optional[Tuple[float]] = (0.1, 0.1),
        do_masking:         bool                   = True,
        do_context_pred:    bool                   = False,
        num_layers:         Optional[int]          = 5,
        context_size:       Optional[int]          = 3,
        mask_rate:          Optional[float]        = None,
        mask_edge:          Optional[int]          = None,
        do_compute_dists:   bool                   = True,
        n_probe:               Optional[int]       = None,
        max_nearest_neighbors: Optional[int]       = None,
        **dataset_kwargs
    ):
        self._seed(seed, key="Construction")

        if do_compute_dists: assert (radius is not None) or (n_neighbors is not None)

        self.data_dir        = data_dir
        self.n_neighbors     = n_neighbors
        self.radius          = radius if radius is not None else float('inf')

        self.do_compute_dists = do_compute_dists
        self.do_knn_graph     = do_compute_dists and (n_neighbors is not None)
        self.do_true_dists    = ((n_probe is None) and (max_nearest_neighbors is None))

        self.raw_data_dir    = Path(self.data_dir) / 'graph_pretraining' / self._DATA_SUBPATH
        self.dataset_cls_kwargs = {
            'held_out_data_frac': held_out_data_frac,
            'held_out_task_frac': held_out_task_frac,
            'seed':               seed,
        }
        self.dataset_graph_PT_kwargs = {
            'do_masking':      do_masking,
            'do_context_pred': do_context_pred,
            'num_layers':      num_layers,
            'context_size':    context_size,
            'mask_rate':       mask_rate,
            'mask_edge':       mask_edge,
        }
        self.dataset_dist_kwargs = {
            'do_true_dists': self.do_true_dists,
            'n_probe': n_probe,
            'max_nearest_neighbors': max_nearest_neighbors,
        }
        self.dataset = self._DATASET_CLS._load(
            self.raw_data_dir,
            self.dataset_graph_PT_kwargs,
            self.dataset_cls_kwargs,
            dist_kwargs = self.dataset_dist_kwargs,
            do_compute_dists = do_compute_dists,
        )

        self.generate()

        super().__init__(
            labelset_G = self.labelset_G,
            dataset    = self.dataset,
            do_sppt    = do_compute_dists,
            **dataset_kwargs
        )

    def generate(
        self,
        seed = None,
    ):
        self._seed(seed, "Generate")

        # Building labelset graph

        if self.do_compute_dists:
            if self.do_knn_graph:
                labelset_G = nx.Graph()
                labelset_G.add_nodes_from(np.arange(len(self.dataset.labelset_vocab)))

                print("Building a kNN graph")
                labelset_G.add_edges_from((
                    (r, c) for r, cs in enumerate(
                        self.dataset.labelset_indexes[:, :self.n_neighbors]
                    ) for c in cs if c != r
                ))
            elif self.do_true_dists:
                print("Constructing the graph directly from the true_dists matrix")
                labelset_G = nx.from_numpy_matrix(self.dataset.true_dists < self.radius)
            else:
                labelset_G = nx.Graph()
                labelset_G.add_nodes_from(np.arange(len(self.dataset.labelset_vocab)))

                print("Building the rNN graph from kNN output")
                labelset_G.add_edges_from((
                    (r, c) for r, (cs, ds) in enumerate(zip(
                        self.dataset.labelset_indexes, self.dataset.labelset_dists
                    )) for c, d in zip(cs, ds) if (c != r) and (d < self.radius)
                ))
        else: labelset_G = None

        self.labelset_G = labelset_G

        print('Finished setup!')
        self.graph_initialized = True
        self.node_features_initialized = True
