import numpy as np, pandas as pd, networkx as nx, pickle, random
from Bio import SeqIO
from pathlib import Path
from tape.tokenizers import TAPETokenizer

from .pretraining_dataset import PretrainingDataset
from ..constants import *

import sys
sys.path.append("../PLUS")

from plus.data.alphabets import Protein
from plus.preprocess import preprocess_seq_for_tfm
from plus.config import RunConfig


class TreeoflifeDataset(PretrainingDataset):
    def __init__(
        self,
        seed                = None,
        data_dir            = RAW_DATASETS_DIR,
        species             = [],
        species_filename    = '',
        max_seq_len         = None,
        min_sample_nodes    = 50,
        do_from_tape        = False,
        do_from_plus        = False,
        do_use_sample_cache = True,
        **dataset_kwargs
    ):
        self._seed(seed, key="Construction")

        self.data_dir           = data_dir
        self.species            = species
        self.species_filename   = species_filename
        self.max_seq_len        = max_seq_len
        self.min_sample_nodes   = min_sample_nodes
        self.do_from_tape       = do_from_tape
        self.do_from_plus       = do_from_plus
        self.do_use_sample_cache= do_use_sample_cache

        self.raw_data            = Path(self.data_dir)
        self.interaction_path    = self.raw_data / 'treeoflife/treeoflife.interactomes'
        self.species_path        = self.raw_data / 'string/species'
        self.cached_graphs_path  = self.raw_data / 'treeoflife/species_graphs'
        self.run_cache           = self.cached_graphs_path / self.species_filename
        self.cached_graph_file   = self.raw_data / 'treeoflife/species_files' / (self.species_filename+'.gpickle')

        for path in (self.interaction_path, self.species_path):
            assert path.exists()
        if not self.run_cache.exists(): os.mkdir(self.run_cache)

        self.generate()

        if self.do_from_tape:
            tokenizer = TAPETokenizer()
            dataset_kwargs['tokenizer'] = tokenizer

            def encode(s, max_len):
                tokenized = [tokenizer.convert_token_to_id('<cls>')] if self.add_cls else []
                for c in s:
                    if c == 'J': c = 'L'
                    tokenized += [tokenizer.convert_token_to_id(c)]
                tokenized += [tokenizer.convert_token_to_id('<pad>')] * (max_len - len(tokenized))
                return tokenized

            self.encode = encode
            self.mask_id = tokenizer.convert_token_to_id('<mask>')
            self.pad_id = tokenizer.convert_token_to_id('<pad>')

        elif self.do_from_plus:
            tokenizer = Protein()
            dataset_kwargs['tokenizer'] = tokenizer

            def encode(s, max_len):
                s_bytes = str.encode(s)
                tokenized = tokenizer.encode(s_bytes)
                return tokenized

            self.encode = encode

            self.run_config_file = '../PLUS/config/run/plus-tfm_pfam.json'
            self.run_config = RunConfig(
                file = self.run_config_file,
            )

        super().__init__(
            G = self.G,
            node_features = self.node_features,
            do_from_plus = self.do_from_plus,
            **dataset_kwargs
        )

        # Build the sample cache, if it exists on file already.
        self.sample_species_subgraph_cache = {}

        for i, species_id in enumerate(self.species):
            species_cached_sample_path = self.run_cache / (species_id + '_sample.gpickle')
            if species_cached_sample_path.exists() and self.do_use_sample_cache:
                print(f'Reading {species_id} SAMPLE from cache')
                G_sample = nx.read_gpickle(species_cached_sample_path)
                self.sample_species_subgraph_cache[i] = G_sample

    def generate(
        self,
        seed = None,
    ):
        self._seed(seed, "Generate")

        self.initialized = False
        self.__gen_graph()

    def find(self, i):
        # Find the species graph associated with node index i.
        assert self.graph_initialized
        for idx, (start_idx, end_idx) in enumerate(self.index_ranges):
            if start_idx <= i and i <= end_idx:
                return self.Gs_by_species[idx]

    def negative_sample_fn(self, i):
        G = self.find(i)
        return list(set(G.nodes) - set(G.neighbors(i)) - set([i]))

    def negative_ego_graph_fn(self, i):
        G = self.find(i)
        return list(set(G.nodes) - set([i]))

    def sample_species_subgraph(self, species_idx):
        if species_idx in self.sample_species_subgraph_cache:
            return self.sample_species_subgraph_cache[species_idx]

        G = self.Gs_by_species[species_idx]
        nodes = list(G.nodes)

        # Sample min_sample_nodes and add in all their neighbors.
        if self.min_sample_nodes != 0:
            assert self.min_sample_nodes > 0
            sample = random.sample(nodes, self.min_sample_nodes)
            sample_including_neighbors = []
            for node in sample:
                sample_including_neighbors.extend(list(G.neighbors(node)))
                sample_including_neighbors.append(node)
            sample_including_neighbors = set(sample_including_neighbors)
            G_sample = G.subgraph(sample_including_neighbors)

        else:
            G_sample = G

        # Save the sample in the cache for this run.
        self.sample_species_subgraph_cache[species_idx] = G_sample

        # Also write it out to file for use in future runs.
        if self.do_use_sample_cache:
            species_id = self.species[species_idx]                  # id = identifier, idx = 0....1840.
            print(f'Writing {species_id} SAMPLE to cache')
            species_cached_sample_path = self.run_cache / (species_id + '_sample.gpickle')
            nx.write_gpickle(G_sample, species_cached_sample_path)
        return G_sample

    def __gen_graph(self, seed=None):
        self._seed(seed, key="Generate Graph")

        self.G                        = nx.Graph()
        self.Gs_by_species            = []
        self.node_features            = []
        self.node_features_by_species = []
        self.index_ranges             = []   # (start_idx, end_idx), same len as self.Gs_by_species

        for species_id in self.species:
            species_cached_graph_path = self.run_cache / (species_id + '.gpickle')
            species_cached_feats_path = self.run_cache / (species_id + '.pkl')

            if species_cached_graph_path.exists():
                print(f'Reading {species_id} from cache')
                try:
                  G = nx.read_gpickle(species_cached_graph_path)
                except ValueError as e:
                  print(f"Failed to read cached graph {species_cached_graph_path}")
                  raise

                with open(species_cached_feats_path, 'rb') as fh:
                    node_features = pickle.load(fh)

            else:
                interaction_file = self.interaction_path / f'{species_id}.txt'
                with open(interaction_file, 'r') as f:

                    all_ids, id_to_index, edges = [], {}, []
                    for ln in f:
                        idx_pair = []
                        for i in ln.strip().split(' '):
                            i = f"{species_id}.{i}"
                            if i in id_to_index:
                                idx_pair.append(id_to_index[i])
                                continue

                            idx = len(all_ids)
                            id_to_index[i] = idx
                            all_ids.append(i)
                            idx_pair.append(idx)

                        edges.append(tuple(idx_pair))

                sequences = {}
                species_file = self.species_path / f'{species_id}.fa'

                with open(species_file, 'r') as f:
                    for record in SeqIO.parse(str(species_file), 'fasta'):
                        if record.id in all_ids:
                            sequences[record.id] = str(record.seq)

                # Truncate from front or back.
                if self.max_seq_len is not None:
                    for k, v in sequences.items():
                        if random.random() > 0.5:
                            sequences[k] = v[:self.max_seq_len]
                        else:
                            if len(v) > self.max_seq_len: sequences[k] = v[len(v)-self.max_seq_len:]
                            else: sequences[k] = v
                        assert len(sequences[k]) <= self.max_seq_len

                G = nx.Graph()
                nodes = list(range(len(all_ids)))
                node_features = [sequences[id] for id in all_ids]

                # Keep indices disjoint.
                N = len(self.node_features)
                nodes = [v+N for v in nodes]
                edges = [(u+N, v+N) for u, v in edges]

                G.add_nodes_from(nodes)
                G.add_edges_from(edges)

                nx.write_gpickle(G, species_cached_graph_path)
                with open(species_cached_feats_path, 'wb') as fh:
                    pickle.dump(node_features, fh)
                print(f'Writing {species_id} to cache')

            start_idx = len(self.node_features)
            self.Gs_by_species.append(G)
            self.node_features_by_species.append(node_features)
            self.node_features.extend(node_features)
            end_idx = len(self.node_features) - 1

            assert min(list(G.nodes)) == start_idx
            assert max(list(G.nodes)) == end_idx
            assert start_idx <= list(G.edges())[0][0] and list(G.edges())[0][0] <= end_idx

            self.index_ranges.append((start_idx, end_idx))

            if not self.cached_graph_file.exists():
                self.G.add_nodes_from(list(G.nodes))
                self.G.add_edges_from(list(G.edges))

        if self.cached_graph_file.exists():
            print(f'Reading entire graph from cache... {self.cached_graph_file}')
            try:
              self.G = nx.read_gpickle(self.cached_graph_file)
            except ValueError as e:
              print(f"Failed to read {self.cached_graph_file}: {e}.\nContinuing.")
              pass
        else:
            print('Writing entire graph to cache...')
            nx.write_gpickle(self.G, self.cached_graph_file)

        print('Finished setup!')
        self.graph_initialized = True
        self.node_features_initialized = True

    def _fit(self):
        super()._fit()

        # For fast species-wise access.
        self.tokenized_node_features_by_species = []
        for species_idx, nf in enumerate(self.node_features_by_species):
            species_tokenized = []
            species_start_idx = self.index_ranges[species_idx][0]
            for i in range(len(nf)):
                species_tokenized.append(self.tokenized_node_features[species_start_idx + i])
            self.tokenized_node_features_by_species.append(species_tokenized)

        # Check that this looks correct.
        if self.tokenized_node_features_by_species:
            print('Num species:', len(self.tokenized_node_features_by_species))
            print('Num sequences in first species:', len(self.tokenized_node_features_by_species[0]))
            print('Length of first sequence in first species:', len(self.tokenized_node_features_by_species[0][0]))
