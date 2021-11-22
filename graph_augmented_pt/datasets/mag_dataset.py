import pickle, random, traceback, numpy as np, pandas as pd, networkx as nx, random
from pathlib import Path
from ogb.nodeproppred import NodePropPredDataset
from transformers import AutoTokenizer

from .pretraining_dataset import PretrainingDataset
from ..constants import *
from ..utils.utils import *


# TODO: There's an opportunity here to leverage auxiliary nodes via the authors or to infer edges between
# papers based on shared authorship.
class MAGDataset(PretrainingDataset):

    D_NAME = 'ogbn-mag'
    START_WORDS_TO_DROP = ['abstract']
    MAPPING_READERS = {
        'paper':  lambda p: pd.read_csv(p, sep=',', index_col='ent idx'),
        'author': lambda p: pd.read_csv(p).set_index('ent idx'),
        'field':  lambda p: pd.read_csv(p).set_index('ent idx'),
        'inst':   lambda p: pd.read_csv(p).set_index('ent idx'),
    }

    def __init__(
        self,
        seed             = None,
        data_dir         = RAW_DATASETS_DIR,
        max_len          = 512,
        run_name         = '',
        min_sample_nodes = 50,
        node_types       = ('paper',),
        edge_types       = ('cites',),
        tqdm             = None,
        overwrite        = False,
        remove_isolates  = True,
        do_debug_graph   = False,
        hf_model_name    = SCIBERT_SCIVOCAB_UNCASED,
        **dataset_kwargs
    ):
        assert set(node_types) == set(('paper',)), f"We only support 'paper' nodes now. Got {node_types}"
        assert set(edge_types) == set(('cites',)), f"We only support 'cites' edges now. Got {edge_types}"

        self._seed(seed, key="Construction")

        self.data_dir           = data_dir
        self.root               = os.path.join(data_dir, 'ogb')
        self.max_len            = max_len
        self.run_name           = run_name
        self.min_sample_nodes   = min_sample_nodes

        self.node_types         = set(node_types)
        self.edge_types         = set(edge_types)

        self.tqdm               = tqdm
        self.overwrite          = overwrite

        self.remove_isolates    = remove_isolates
        self.do_debug_graph     = do_debug_graph
        self.hf_model_name      = hf_model_name

        # TODO: For now, we're not passing this along b/c it wouldn't be used correctly.
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        # This isn't needed yet, given the assertion above.
        #self.node_types_to_tokenize = set(('papers',)).intersection(self.node_types)

        self.generate()

        # TODO: We probably need a negative sample fn

        super().__init__(
            G = self.G,
            node_features      = self.node_features,
            tokenizer          = self.tokenizer,
            **dataset_kwargs
        )

    def _fit(self):
        assert self.node_features_initialized

        raw_data        = Path(self.data_dir)
        ogb             = raw_data / 'ogb'

        postfix = self.hf_model_name.split('/')[-1]
        cached_tokenized_path = ogb / f'ogbn_mag_tokenized_{postfix}.pkl'

        if self.do_debug_graph:
            cached_tokenized_path = ogb / f'debug_tokenized_{postfix}.pkl'

        if not self.overwrite and cached_tokenized_path.exists():
            print("Reading tokenized features from cache.")
            self.tokenized_node_features = depickle(cached_tokenized_path)

        else:
            print("Writing tokenized features to cache.")
            self.raw_tokenized_node_features = self.tokenizer(
                self.node_features, add_special_tokens=True, padding='longest'
            )
            try:
                self.tokenized_node_features = self.raw_tokenized_node_features.data['input_ids']
                enpickle(self.tokenized_node_features, cached_tokenized_path)
            except Exception as e:
                print("Failed to fit!", e)
                traceback.print_exc()
                self.is_fit = False
                pass

        self.mask_id = self.tokenizer.mask_token_id
        self.pad_id = self.tokenizer.pad_token_id
        max_len = max(len(l) for l in self.tokenized_node_features)
        if max_len > self.max_len:
            self.tokenized_node_features = [t[:self.max_len] for t in self.tokenized_node_features]
        self.is_fit = True


    def generate(
        self,
        seed = None,
    ):
        self._seed(seed, "Generate")

        self.initialized = False
        self.__gen_graph()

    def __is_valid_edge(self, edge):
        head, rel, tail = edge
        return (head in self.node_types and tail in self.node_types and rel in self.edge_types)

    def _preprocess_abstract(self, abstract):
        for w in self.START_WORDS_TO_DROP:
            if abstract.lower().startswith(f"{w} "):
                abstract = abstract[len(w)+1:]
                break
        return abstract

    def __gen_graph(self, seed=None):
        self._seed(seed, key="Generate Graph")
        # TODO: Move a bunch of this up to the init where it belongs.

        raw_data        = Path(self.data_dir)
        ogb             = raw_data / 'ogb'
        abstracts_path  = ogb / 'abstracts.txt'
        papers_path     = ogb / 'ogbn_mag_papers.txt'

        self.node_mapping_paths = {
            'paper':  ogb / 'ogbn_mag' / 'mapping' / 'paper_entidx2name.csv',
            'author': ogb / 'ogbn_mag' / 'mapping' / 'author_entidx2name.csv',
            'field':  ogb / 'ogbn_mag' / 'mapping' / 'field_of_study_entidx2name.csv',
            'inst':   ogb / 'ogbn_mag' / 'mapping' / 'institution_entidx2name.csv',
        }

        cached_graph_path = ogb / 'ogbn_mag_graph.pkl'
        cached_nodes_path = ogb / 'ogbn_mag_nodes.pkl'

        if self.do_debug_graph:
            cached_graph_path = ogb / 'debug_graph.pkl'
            cached_nodes_path = ogb / 'debug_nodes.pkl'

        if not self.overwrite and cached_graph_path.exists():
            assert cached_nodes_path.exists()
            print('Reading entire graph from cache...')
            self.G = nx.read_gpickle(cached_graph_path)
            self.node_features = depickle(cached_nodes_path)

            self.graph_initialized = True
            self.node_features_initialized = True
            print("Finished setup!")
            return

        node_mappings = {k: self.MAPPING_READERS[k](self.node_mapping_paths[k]) for k in self.node_types}

        print(f"Loading abstracts from {abstracts_path}")
        abstracts = pd.read_csv(abstracts_path, sep=',', index_col='ent idx')

        print(f"Loading dataset @ name = {self.D_NAME}, root = {self.root}")
        dataset   = NodePropPredDataset(name = self.D_NAME, root = self.root)

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0] # graph: library-agnostic graph object

        self.ogb_graph, self.ogb_label = graph, label

        edge_types = set(e for e in graph['edge_index_dict'].keys() if self.__is_valid_edge(e))
        num_nodes = {k: graph['num_nodes_dict'][k] for k in self.node_types}

        self.invalid_nodes = {k: set([]) for k in self.node_types}
        node_type_rng = zip_dicts(num_nodes, node_mappings)
        if self.tqdm is not None and len(num_nodes) > 2:
            node_type_rng = self.tqdm(node_type_rng, desc="Getting valid nodes")

        for node_type, num, mapping in node_type_rng:
            node_rng = range(num)
            if self.tqdm is not None and num > 100:
                node_rng = self.tqdm(node_rng, leave=False, desc=f"Pre-filtering {node_type} nodes")

            for node_idx in node_rng:
                if node_type == 'paper' and node_idx not in abstracts.index:
                    self.invalid_nodes[node_type].update([node_idx])

        self.valid_node_degrees = {k: [0] * n for k, n in num_nodes.items()}

        edge_types_rng = edge_types
        if self.tqdm is not None and len(edge_types) > 2:
            edge_type_rng = self.tqdm(edge_type_rng, desc="Collecting Node Degrees")

        for head_type, edge_type, tail_type in edge_types_rng:
            edge_rng = graph['edge_index_dict'][(head_type, edge_type, tail_type)].T
            if self.tqdm is not None and len(edge_rng) > 100:
                edge_rng = self.tqdm(edge_rng, desc=f"{edge_type} Edges")

            head_invalid_nodes = self.invalid_nodes[head_type]
            tail_invalid_nodes = self.invalid_nodes[tail_type]
            for head_node_idx, tail_node_idx in edge_rng:
                if head_node_idx in head_invalid_nodes or tail_node_idx in tail_invalid_nodes: continue

                self.valid_node_degrees[head_type][head_node_idx] += 1
                if (head_type, head_node_idx) != (tail_type, tail_node_idx):
                    self.valid_node_degrees[tail_type][tail_node_idx] += 1

        nodes                 = []
        node_metadata_by_type = {k: [] for k in self.node_types}
        node_metadata         = []
        node_idx_to_graph_idx = {k: {} for k in self.node_types}

        node_type_rng = zip_dicts(num_nodes, node_mappings)
        if self.tqdm is not None and len(num_nodes) > 2: node_type_rng = self.tqdm(node_type_rng)

        for node_type, num, mapping in node_type_rng:
            node_rng = range(num)
            if self.tqdm is not None and num > 100:
                node_rng = self.tqdm(node_rng, leave=False, desc="Building Nodes")

            for node_idx in node_rng:
                if node_idx in self.invalid_nodes[node_type]: continue
                elif self.remove_isolates and self.valid_node_degrees[node_type][node_idx] == 0: continue
                elif node_type == 'paper':
                    abstract = abstracts.loc[node_idx, 'abstract']
                    str_features = self._preprocess_abstract(abstract)
                else:
                    abstract = None
                    str_features = f"[{node_type}-{node_idx}]"

                graph_node_idx = len(nodes)
                nodes.append(str_features)
                node_idx_to_graph_idx[node_type][node_idx] = graph_node_idx

                metadata_keys = ['Graph Index', 'Node Type', 'Node Type Index', 'Original ID']
                metadata = [graph_node_idx, node_idx, mapping.loc[node_idx]]

                node_metadata_by_type[node_type].append(metadata)
                node_metadata.append({k: v for k, v in zip(metadata_keys, metadata)})

        assert len(nodes) == len(node_metadata)
        self.pre_G_nodes = nodes
        self.pre_G_node_metadata = node_metadata

        self.G = nx.Graph()
        self.G.add_nodes_from(zip(np.arange(len(nodes)), node_metadata))

        assert len(self.G) == len(nodes)

        edge_types_rng = edge_types
        if self.tqdm is not None and len(edge_types) > 2:
            edge_type_rng = self.tqdm(edge_type_rng, desc="Building Edges")

        for head_type, edge_type, tail_type in edge_types_rng:
            head_idxmap = node_idx_to_graph_idx[head_type]
            tail_idxmap = node_idx_to_graph_idx[tail_type]

            edge_rng = graph['edge_index_dict'][(head_type, edge_type, tail_type)].T
            if self.tqdm is not None and len(edge_rng) > 100: edge_rng = self.tqdm(edge_rng)
            for head_node_idx, tail_node_idx in edge_rng:
                if (head_node_idx not in head_idxmap) or (tail_node_idx not in tail_idxmap): continue

                head_graph_idx = head_idxmap[head_node_idx]
                tail_graph_idx = tail_idxmap[tail_node_idx]

                assert head_graph_idx < len(nodes), f"{head_graph_idx},{head_node_idx} Invalid!"
                assert tail_graph_idx < len(nodes), f"{tail_graph_idx},{tail_node_idx} Invalid!"
                self.G.add_edge(head_graph_idx, tail_graph_idx)

        assert len(self.G) == len(nodes)

        print("Built graph, writing to file.")
        self.node_features = nodes
        nx.write_gpickle(self.G, cached_graph_path)
        enpickle(nodes, cached_nodes_path)

        print('Finished setup!')
        self.graph_initialized = True
        self.node_features_initialized = True
