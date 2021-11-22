"""These classes generate various synthetic dataset, with graphs and node features suitable for pre-training.


"""

# Need to modify the path slightly to import the graph_augmented_pt module.
import sys
sys.path.append('..')

from abc import abstractmethod
import copy, itertools, random, traceback, string, numpy as np
from sklearn.cluster import KMeans

from graph_augmented_pt.datasets.pretraining_dataset import *

from simplicial_manfiolds import *

class SyntheticGraphGeneratorDataset(PretrainingDataset):
    """This class constructs the graph underlying the two subsequent Synthetic Datasets.

    TODO:
      0. Full docstring
      1. types
      2. Noise
      3. Redundant Edges
      4. Multiple Cycles
      5. Better homophily signature labels
    """
    def __init__(
        self,
        graphlet_atlas,
        seed                       = None,
        base_cycle_size            = 9,
        num_graphlets              = 3,
        adjoining_graphlets        = ["random"], # w/in graphlet_atlas
        structural_atlas           = 5,
        n_structural_labels        = 3,
        neighborhood_labels_radius = 3,
        n_neighborhood_labels      = 3,
        redundancy_factor          = 0,
        max_len                    = 350,
        **dataset_kwargs,
    ):
        self._seed(seed, key="Construction")

        self.graphlet_atlas     = graphlet_atlas
        self.negative_sample_fn = None # TODO: This probably belongs somewhere else.

        self.generate(
            base_cycle_size            = base_cycle_size,
            num_graphlets              = num_graphlets,
            adjoining_graphlets        = adjoining_graphlets,
            structural_atlas           = structural_atlas,
            n_structural_labels        = n_structural_labels,
            neighborhood_labels_radius = neighborhood_labels_radius,
            n_neighborhood_labels      = n_neighborhood_labels,
            redundancy_factor          = redundancy_factor,
            re_post_init               = False,
        )

        super().__init__(
            G = self.G,
            node_features = self.node_features,
            label_sets = self.label_sets,
            **dataset_kwargs
        )

    def generate(
        self,
        re_post_init = True,
        seed         = None,
        **kwargs,
    ):
        self._seed(seed, "Generate")

        self.initialized = False

        for kwarg, kwval in kwargs.items():
            if re_post_init:
                assert hasattr(self, kwarg), f"Cannot generate with {kwarg}, as this isn't a member variable!"
            if kwval is None: kwval = getattr(self, kwarg)
            else: setattr(self, kwarg, kwval)

        self.__gen_graph()
        self.gen_label_sets()
        self.gen_node_features()

        if re_post_init: self.__post_init__()

    def __gen_graph(self, seed=None):
        assert type(self.base_cycle_size) is int and self.base_cycle_size > 0
        assert type(self.num_graphlets) is int
        assert self.num_graphlets > 0 and self.num_graphlets <= self.base_cycle_size
        assert type(self.adjoining_graphlets) is list and len(self.adjoining_graphlets) > 0

        self._seed(seed, key="Generate Graph")

        # If you reset the graph, you need to reset the rest...
        self.initialized = False

        cycle = nx.cycle_graph(self.base_cycle_size)

        graphlet_period = self.base_cycle_size // self.num_graphlets
        N_nodes = self.base_cycle_size
        idxs, graphlets, new_edges = [], [], []

        for cycle_pos in range(0, self.base_cycle_size, graphlet_period):
            idx, graphlet = self.graphlet_atlas.resolve_graphlet(
                random.choice(self.adjoining_graphlets)
            )
            idxs.append(idx)
            graphlets.append(graphlet)
            connector_node = N_nodes + random.randint(0, len(graphlet) - 1)

            new_edges.append((cycle_pos, connector_node))
            N_nodes += len(graphlet)

        G = nx.algorithms.disjoint_union_all([cycle] + graphlets)
        G.add_edges_from(new_edges)

        delta_cutoffs = [self.base_cycle_size] + [len(g) for g in graphlets]
        motifs = ['CYCLE'] + idxs

        motif_ids = []
        for motif_idx, delta_cutoff in enumerate(delta_cutoffs):
            for i in range(delta_cutoff): motif_ids.append(motif_idx)

        # Adding redundancy
        copies = [[list(G.nodes()), list(G.edges())]]
        new_nodes, new_edges = [], []
        new_cycles, new_graphlets, new_idxs = [], [], []
        for copy_num in range(self.redundancy_factor):
            new_copy = [
                [n + N_nodes for n in copies[-1][0]], [(l+N_nodes, r+N_nodes) for l, r in copies[-1][1]]
            ]
            new_nodes.extend(new_copy[0])
            new_edges.extend(new_copy[1])

            new_idxs.extend(idxs)
            new_graphlets.extend(graphlets)
            new_cycles.extend(cycle)

        for n in G.nodes():
            new_edges.extend([
                (n+i*N_nodes, n+j*N_nodes) for i, j in \
                    itertools.combinations(range(self.redundancy_factor), 2)
            ])

        G.add_nodes_from(new_nodes)
        G.add_edges_from(new_edges)

        self.G = G
        self.node_structural_features = self.structural_atlas.graphlet_degree_vectors(self.G)
        self.motif_ids = motif_ids * (self.redundancy_factor + 1)

        self.motif_ids_initialized = True
        self.graph_initialized = True

    @abstractmethod
    def gen_node_features(self, seed=None):
        raise NotImplementedError("This method needs to be derived.")

    def gen_label_sets(self, seed=None):
        self._seed(seed, "gen_label_sets")

        self.label_sets_initialized = False

        self.structural_labels_kmeans = KMeans(n_clusters=self.n_structural_labels, n_init=50)
        structural_labels = self.structural_labels_kmeans.fit_predict(self.node_structural_features)

        ego_graph_memberships = np.zeros((len(self.G), len(self.G)))
        for i in self.G:
            for node in nx.ego_graph(self.G, n=i, radius=self.neighborhood_labels_radius):
                ego_graph_memberships[i, node] = 1

        self.neighborhood_labels_kmeans = KMeans(n_clusters=self.n_neighborhood_labels, n_init=50)
        neighborhood_labels = self.neighborhood_labels_kmeans.fit_predict(ego_graph_memberships)

        self.label_sets = {
            'Motif': self.motif_ids,
            'Structural': list(structural_labels),
            'Ego-graph Similarity': list(neighborhood_labels),
        }
        self.label_sets_initialized = True

class ClusteredSentencesSyntheticDataset(SyntheticGraphGeneratorDataset):
    """This class assigns each node a real-world sentence in such a way that a particular label-set
       (defined over the graph) reflects a topical division among the sentences. In this way, the class
       reflects only a limited notion of graph structure, much as we might expect in the real world.

       For example, if our graph is as below, with the cycle in the middle, single-node motif A on the left,
       and 4-node motif B on the right

            *--*      *
           /    \    / \
       *--*      *--*---*
           \    /    \
            *--*      *

       And we use this class with `node_features_correspond_to_label` = 'Motif', then we might assign the node
       in Motif A a sentence to do with sports, the nodes in the cycle sentences having to do with scientific
       discoveries, and those in motif B on the right sentences having to do with politics. The internal edges
       within these motifs would *not* be reflected with this assignment.
    """

    def __init__(
        self,
        *args,
        sentences,
        topic_probabilities,
        node_features_correspond_to_label = 'Motif',
        FT_dataset_size                   = 0,
        max_len                           = 350,
        **kwargs,
    ):
        self.sentences                         = sentences
        self.topic_probabilities               = topic_probabilities
        self.node_features_correspond_to_label = node_features_correspond_to_label
        self.n_topics                          = self.topic_probabilities.shape[1]
        self.FT_dataset_size                   = FT_dataset_size
        self.max_len                           = max_len

        first_topics = self.topic_probabilities.argmax(axis=1)
        self.sents_by_topic = {t: [] for t in range(self.n_topics)}
        for i, (t, all_p, sent) in enumerate(zip(first_topics, self.topic_probabilities, self.sentences)):
            self.sents_by_topic[t].append((sent, all_p[t], i))

        for t in range(self.n_topics):
            self.sents_by_topic[t] = sorted(self.sents_by_topic[t], key=lambda x: x[1], reverse=True)

        super().__init__(*args, **kwargs)

    def gen_node_features(self, seed=None):
        assert self.graph_initialized
        assert self.label_sets_initialized
        assert self.node_features_correspond_to_label in self.label_sets

        # To support multiple label sets (e.g., self.node_features_correspond_to_label is a list),
        # general procedure idea is as follows:
        # V0
        # 1. Let N be the total # of classes across all labels in self.node_features_correspond_to_label
        # 2. Choose N random topics T_1, ..., T_N
        # 3. For each node, with label set L = {l_1, ..., l_m}, 1 \le l_i \le N, choose a sentence such that it
        #    maximizes \prod_i T_{l_i} - \sum_j T_{j} | j \notin L - \prod_j T_{j} | j \notin L
        # V1
        # 1. N times, do V0. Choose the topic assignment that yields the maximal probability
        #    \prod_i T_{l_i} across all nodes.
        # V2. Do V1, but instead of N random choices, do a markov chain like deal, where in each case a single
        #     topic is swapped for another with transition probability proportional to the gain in overall
        #     score?

        self._seed(seed, key="Generate Node Features")

        N = len(self.G)

        data_indices = np.arange(N)
        random.shuffle(data_indices)

        self.PT_indices = data_indices[self.FT_dataset_size:]
        self.FT_indices = data_indices[:self.FT_dataset_size]


        node_labels = self.label_sets[self.node_features_correspond_to_label]
        topics_by_label = {l: random.choice(list(range(self.n_topics))) for l in set(node_labels)}

        self.node_features = []
        idx_by_topic = {t: 0 for t in range(self.n_topics)}
        for node_label in node_labels:
            t = topics_by_label[node_label]
            idx = idx_by_topic[t]
            idx_by_topic[t] += 1

            sents = self.sents_by_topic[t]
            self.node_features.append(sents[idx][0].strip()[:self.max_len])

        self.full_G             = copy.deepcopy(self.G)
        self.full_label_sets    = copy.deepcopy(self.label_sets)
        self.full_node_features = copy.deepcopy(self.node_features)

        self.FT_label_sets = {k: np.array(ls)[self.FT_indices] for k, ls in self.full_label_sets.items()}
        self.label_sets = {k: np.array(ls)[self.PT_indices] for k, ls in self.full_label_sets.items()}

        self.FT_G = nx.relabel.relabel_nodes(
            nx.subgraph(self.full_G, self.FT_indices),
            {overall_idx: i for i, overall_idx in enumerate(self.FT_indices)}
        )
        self.G = nx.relabel.relabel_nodes(
            nx.subgraph(self.full_G, self.PT_indices),
            {overall_idx: i for i, overall_idx in enumerate(self.PT_indices)}
        )

        self.FT_node_features = copy.deepcopy(np.array(self.node_features)[self.FT_indices])
        self.node_features = copy.deepcopy(np.array(self.node_features)[self.PT_indices])

        self.node_features_initialized = True

    def gen_FT_dataset(self, seed=None):
        # We also need to handle the tokenized_FT features here, as they won't be handled via fit.
        max_len = max(len(s) for s in self.FT_node_features) + 1

        self.FT_node_features = [[c for c in s if c in self.idxmap] for s in self.FT_node_features]

        self.FT_tokenized_node_features = []
        for s in self.FT_node_features:
            tokenized = [self.idxmap['[CLS]']] if self.add_cls else []
            tokenized += [self.idxmap[c] for c in s]
            tokenized += [self.idxmap['[PAD]']] * (max_len - len(tokenized))
            self.FT_tokenized_node_features.append(tokenized)

        return self.FT_node_features, self.FT_tokenized_node_features, FT_topics
