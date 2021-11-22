"""These classes generate various synthetic dataset, with graphs and node features suitable for pre-training.
"""

# Need to modify the path slightly to import the graph_augmented_pt module.
import sys
sys.path.append('..')

from abc import abstractmethod
import copy, pickle, inspect, math, itertools, random, traceback, string, numpy as np
from sklearn.cluster import KMeans

from graph_augmented_pt.datasets.pretraining_dataset import *

from simplicial_manfiolds import *

class BasicTopicSentenceDataset(PretrainingDataset):
    DROP_BEFORE_SAVE_ATTRS = [
        'source_sentences', 'simplex_probabilities', 'simplex_entropies', 'simplexes', 'source_original_sentence_idxs',
        'topic_clique', 'manifold', 'source_topic_probabilities',
    ]
    REQUIRED_KWARGS = []
    DEFAULT_KWARGS = {}

    def __init__(
        self,
        dataset_size      = 20,
        FT_dataset_size   = 20,
        max_len           = 350,
        noise_rate        = 0,
        noise_style       = 'random_add', # or 'random_flip',
        seed              = None,
        **kwargs,
    ):
        for kwarg in self.REQUIRED_KWARGS: assert kwarg in kwargs, f"Missing {kwarg}!"
        for kwarg, val in self.DEFAULT_KWARGS.items():
            if kwarg not in kwargs: kwargs[kwarg] = val

        self.generate(
            dataset_size    = dataset_size,
            FT_dataset_size = FT_dataset_size,
            seed            = seed,
            max_len         = max_len,
            noise_rate      = noise_rate,
            noise_style     = noise_style,
            re_post_init    = False,
            **kwargs,
        )

        self.negative_sample_fn = None

        arg_names = inspect.getfullargspec(super().__init__).args
        dataset_kwargs = {k: v for k, v in kwargs.items() if k in arg_names}

        super().__init__(
            G = self.G,
            node_features = self.node_features,
            label_sets = self.label_sets,
            **dataset_kwargs,
        )

    def _save(self, fp):
        attrs = {}
        for a in self.DROP_BEFORE_SAVE_ATTRS:
            if hasattr(self, a):
                attrs[a] = getattr(self, a)
                setattr(self, a, None)

        with open(fp, mode='wb') as f: pickle.dump(self, f)

        for a, v in attrs.items(): setattr(self, a, v)

    def _custom_load(self, kwargs):
        """If you want this to do something, overwrite it in the derived class."""
        return

    @staticmethod
    def _load(fp, **kwargs):
        with open(fp, mode='rb') as f: dataset = pickle.load(f)

        for a, v in kwargs.items(): setattr(dataset, a, v)

        dataset._custom_load(kwargs)

        return dataset

    def generate(
        self,
        seed          = None,
        re_post_init  = True,
        re_set_topics = True,
        **kwargs,
    ):
        self._seed(seed, "Generate")

        # assert isinstance(noise_rate, (int, float)) and noise_rate >= 0 and noise_rate <= 1
        # assert isinstance(noise_style, str) and noise_style in ('random_add', 'random_flip')
        # assert isinstance(max_len, int) and max_len > 0
        # assert isinstance(dataset_size, int) and dataset_size > 0

        self.initialized = False

        changing_kwargs = set()
        for kwarg, kwval in kwargs.items():
            if re_post_init:
                assert hasattr(self, kwarg), \
                    f"Cannot generate with {kwarg}, as this isn't a member variable!"
            if kwval is None and hasattr(self, kwarg): kwval = getattr(self, kwarg)
            elif hasattr(self, kwarg) and kwval != getattr(self, kwarg):
                changing_kwargs.add(kwarg)
                setattr(self, kwarg, kwval)
            else: setattr(self, kwarg, kwval)

        if 'FT_dataset_size' in kwargs or hasattr(self, 'FT_dataset_size'):
            self.total_dataset_size = self.FT_dataset_size + self.dataset_size
        else: self.total_dataset_size = self.dataset_size

        self.node_features_initialized = False
        self.label_sets_initialized    = False

        if re_set_topics:
            err = None
            if hasattr(self, 'defined_topics'): delattr(self, 'defined_topics')

            try: topic_assignment_exists = self.set_topics(changing_kwargs=changing_kwargs)
            except Exception as e:
                print(f"Failed to find topic assignment!\n{e}")
                traceback.print_exc()

                err = e
                topic_assignment_exists = False

            if not topic_assignment_exists:
                print("Failed to find topic assignment!")
                self.err = err
                self.G = nx.Graph()
                self.label_sets = {'Topics': [None]}
                self.node_features = ['']

                if re_post_init: self.__post_init__()
                return False

        try:
            self._set_PT_data(changing_kwargs=changing_kwargs, re_set_topics=re_set_topics)
            self.node_features_initialized = True
            self.label_sets_initialized    = True
        except Exception as e:
            print(f"Failed during PT graph/label generation!\n{e}")
            traceback.print_exc()

            return False

        try:
            self._fit()
            self._set_FT_data()
        except Exception as e:
            print(f"Failed during FT generation!\n{e}")
            traceback.print_exc()

            return False

        if re_post_init: self.__post_init__()

        return True

    @abstractmethod
    def _set_topics_internal(self, changing_kwargs=None, seed=None):
        # This method should define:
        # self.defined_sentences
        # self.defined_topic_probabilities
        # self.defined_source_sentence_idxs
        raise NotImplementedError("Should be defined in derived class!")

    def set_topics(self, changing_kwargs=None, seed=None):
        self._seed(seed, key="Choose Topics/Sentences")

        self._set_topics_internal(changing_kwargs=changing_kwargs)

        self.defined_N = len(self.defined_sentences)
        assert self.defined_N > self.total_dataset_size, f"Not enough sentences to support PT & FT!"

        self.vocab = set(' '.join(self.defined_sentences).split())

        self.defined_sentences            = np.array(self.defined_sentences)
        self.defined_topic_probabilities  = np.array(self.defined_topic_probabilities)
        self.defined_source_sentence_idxs = np.array(self.defined_source_sentence_idxs)

        defined_topics = self.defined_topic_probabilities.argmax(axis=1)
        if hasattr(self, 'defined_topics'):
            assert (np.array(self.defined_topics) == defined_topics).all()
        else: self.defined_topics = defined_topics

        return True

    @abstractmethod
    def _gen_graph(self, changing_kwargs=None, re_set_topics=False):
        raise NotImplementedError("Should be defined in derived class!")

    def _noise_graph(self, seed=None):
        self._seed(seed, key="Noise Graph")
        if not hasattr(self, 'noise_rate') or self.noise_rate == 0: return

        assert hasattr(self, 'noise_style')

        old_edges = set(frozenset(e) for e in self.G.edges())
        edges_to_add = []
        edges_to_remove = []

        ps = np.random.random(size=(len(self.G)*(len(self.G)-1),))
        for p, (u, v) in zip(ps, itertools.combinations(self.G.nodes(), 2)):
            if p >= self.noise_rate: continue

            is_edge = frozenset((u, v)) in old_edges

            if not is_edge: edges_to_add.append((u, v))
            elif self.noise_style == 'random_flip': edges_to_remove.append((u, v))

        self.G.remove_edges_from(edges_to_remove)
        self.G.add_edges_from(edges_to_add)

    def _get_node_sents(self, idxs):
        node_features    = [self.defined_sentences[i] for i in idxs]
        topics           = [self.defined_topics[i] for i in idxs]

        if hasattr(self, 'max_len') and self.max_len is not None and self.max_len > 0:
            node_features = [n[:self.max_len-1] for n in node_features]

        return node_features, topics

    def _get_indices(self, N, disjoint_from=None, seed=None):
        self._seed(seed, key="Get Indices")

        idxs = set(np.arange(len(self.defined_sentences)))
        if disjoint_from is not None: idxs -= set(disjoint_from)

        assert len(idxs) >= N, f"Can't retrieve {N} indices--not enough possible!"

        idxs = list(idxs)
        random.shuffle(idxs)
        return idxs[:N]

    def _set_PT_data(self, seed=None, changing_kwargs=None, re_set_topics=True):
        self._seed(seed, key="Set PT Dataset")

        self.PT_idxs = self._get_indices(self.dataset_size, seed=seed)
        self.node_features, self.topics = self._get_node_sents(self.PT_idxs)

        self._gen_graph(changing_kwargs=changing_kwargs, re_set_topics=re_set_topics)
        self._noise_graph()
        self.label_sets = {'Topics': list(self.topics)}

    def _set_FT_data(self, seed=None):
        self._seed(seed, key="Set FT Dataset")

        self.FT_idxs = self._get_indices(self.FT_dataset_size, disjoint_from=self.PT_idxs, seed=seed)
        self.FT_node_features, self.FT_topics = self._get_node_sents(self.FT_idxs)
        self.FT_label_sets = {'Topics': list(self.FT_topics)}

        FT_tokenized_node_features = []
        for s in self.FT_node_features:
            tokenized = [self.idxmap['[CLS]']] if self.add_cls else []
            tokenized += [self.idxmap[c] for c in s if c in self.idxmap]
            tokenized += [self.idxmap['[PAD]']] * (self.max_len - len(tokenized))
            FT_tokenized_node_features.append(tokenized)

        self.FT_tokenized_node_features = FT_tokenized_node_features

class DisconnectedCliquesDataset(BasicTopicSentenceDataset):
    REQUIRED_KWARGS = ['source_topic_probabilities', 'source_sents_by_topic']
    DEFAULT_KWARGS = {'topic_thresh': 0.9, 'total_topics': 15, 'equalize_topics': True}

    def _gen_graph(self, changing_kwargs=None, re_set_topics=False):
        nodes = list(range(len(self.node_features)))
        edges = []
        for node, topic in zip(nodes, self.topics):
            for other_node, other_topic in zip(nodes[node+1:], self.topics[node+1:]):
                if topic == other_topic: edges.extend([(node, other_node), (other_node, node)])

        self.G = nx.Graph()
        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)

    def _set_topics_internal(self, seed=None, changing_kwargs=None):
        self._seed(seed, key="Choose Topics/Sentences")

        n_topics           = self.source_topic_probabilities.shape[1]
        n_attempts         = 0
        valid_topics_found = False

        min_viable = math.ceil(self.total_dataset_size / self.total_topics) if self.equalize_topics else 5

        valid_sents_by_topic = {t: [] for t in range(n_topics)}
        for t in range(n_topics):
            for sent, p, idx in self.source_sents_by_topic[t]:
                #if p < self.topic_thresh: break
                if p >= self.topic_thresh: valid_sents_by_topic[t].append((sent, p, idx))

        valid_sents_by_topic = {t: vs for t, vs in  valid_sents_by_topic.items() if len(vs) >= min_viable}
        all_topics           = list(valid_sents_by_topic.keys())

        while n_attempts < 10 and not valid_topics_found:
            n_attempts += 1

            random.shuffle(all_topics)
            selected_topics = all_topics[:self.total_topics]

            cnt_per_topic = {t: len(valid_sents_by_topic[t]) for t in selected_topics}

            if self.equalize_topics:
                min_cnt = min(cnt_per_topic.values())
                cnt_per_topic = {t: min_cnt for t in cnt_per_topic}

            if sum(cnt_per_topic.values()) > self.total_dataset_size:
                valid_topics_found = True

        assert sum(cnt_per_topic.values()) > self.total_dataset_size, (
            f"Not enough Topics/Sents! Want {self.dataset_size} got {sum(cnt_per_topic.values())}, "
            f"{cnt_per_topic}\n{str({t: len(v) for t, v in valid_sents_by_topic.items()})}"
        )

        self.defined_sentences            = []
        self.defined_topic_probabilities  = []
        self.defined_source_sentence_idxs = []
        for t in selected_topics:
            sents = copy.deepcopy(valid_sents_by_topic[t])
            random.shuffle(sents)
            for i, (sent, p, idx) in enumerate(sents):
                if i > cnt_per_topic[t]: break
                elif p < self.topic_thresh: continue

                self.defined_sentences.append(
                    sent.translate(str.maketrans('', '', string.punctuation)).lower()
                )
                self.defined_topic_probabilities.append(self.source_topic_probabilities[idx])
                self.defined_source_sentence_idxs.append(idx)

        return True

class ManifoldNearestNeighborDataset(BasicTopicSentenceDataset):
    REQUIRED_KWARGS = [
        'manifold_kwargs', 'source_sentences', 'source_topic_probabilities', 'simplex_probabilities',
        'simplex_entropies', 'topic_simplices', 'source_original_sentence_idxs', 'topic_clique',
        'agg_entropy_per_simplex',
    ]
    DEFAULT_KWARGS = {
        'subsample_data': -1,
        'num_samples_per_simplex': 25,
        'manifold_radius_graph_r': 0.3,
    }

    def _custom_load(self, kwargs):
        if 'manifold' not in kwargs: self.manifold = LabeledSimplicialManifold(**self.manifold_kwargs)

    def _set_topics_internal(self, changing_kwargs=None, seed=None):
        self._seed(seed, "Assign topics to simplex vertices")

        if changing_kwargs is None: changing_kwargs = {}

        if not hasattr(self, 'manifold') or 'manifold_kwargs' in changing_kwargs:
            self.manifold = LabeledSimplicialManifold(**self.manifold_kwargs)

        all_pts_size = len(self.manifold.simplices)*self.num_samples_per_simplex

        assert self.dataset_size <= all_pts_size
        assert self.subsample_data < self.dataset_size
        assert len(self.manifold.vocab) < len(self.topic_clique)

        topic_clique = list(self.topic_clique)
        random.shuffle(topic_clique)
        self.simplex_vertices_to_topics = {mv: t for mv, t in zip(self.manifold.vocab, topic_clique)}
        self.topics_to_simplex_vertices = {t: mv for mv, t in self.simplex_vertices_to_topics.items()}
        self.valid_topics = set(self.simplex_vertices_to_topics.values())

        return self._assign_sentences_to_simplices()

    def _assign_sentences_to_simplices(self, seed=None):
        self._seed(seed, "Generate topic assignment")

        if self.subsample_data > 0:
            source_sentences              = self.source_sentences[:sefl.subsample_data]
            simplex_probabilities         = self.simplex_probabilities[:self.subsample_data]
            simplex_entropies             = self.simplex_entropies[:self.subsample_data]
            source_original_sentence_idxs = self.source_original_sentence_idxs[:self.subsample_data]
            topic_simplices               = self.topic_simplices[:self.subsample_data]
        else:
            source_sentences              = self.source_sentences
            simplex_probabilities         = self.simplex_probabilities
            simplex_entropies             = self.simplex_entropies
            source_original_sentence_idxs = self.source_original_sentence_idxs
            topic_simplices               = self.topic_simplices

        self.defined_original_sentence_idxs, self.defined_topics, self.defined_local_coordinates = [], [], []
        self.defined_sentences_per_simplex = {frozenset(s): [] for s in self.manifold.simplices}

        self.defined_source_sentence_idxs = []

        source_sentence_idxs = np.arange(len(source_sentences))

        simplices_rng = self.manifold.simplices
        if hasattr(self, 'tqdm') and self.tqdm is not None:
            simplices_rng = self.tqdm(simplices_rng, desc="Mapping Simplices", leave=False)

        for simplex in simplices_rng:
            simplex_key = frozenset(simplex)

            topic_simplex = frozenset([self.simplex_vertices_to_topics[v] for v in simplex])

            mask = [frozenset(t_row) == topic_simplex for t_row in self.topic_simplices]

            local_source_sentence_idxs      = source_sentence_idxs[mask]
            local_topic_simplices           = topic_simplices[mask]
            local_simplex_probabilities     = simplex_probabilities[mask]
            local_simplex_entropies         = simplex_entropies[mask]
            local_original_sentence_idxs    = source_original_sentence_idxs[mask]

            local_topic_labels              = local_topic_simplices[
                np.arange(len(local_topic_simplices)), np.argmax(local_simplex_probabilities, axis=1)
            ]
            check_topic_labels              = self.source_topic_probabilities[
                local_source_sentence_idxs
            ].argmax(axis=1)
            assert (local_topic_labels == check_topic_labels).all(), (
                f"{str(local_topic_labels)} \n {str(check_topic_labels)} \n {topic_simplex}"
            )

            N_sents = len(local_original_sentence_idxs)

            local_coordinates           = [
                {self.topics_to_simplex_vertices[t]: p for t, p in zip(t_row, p_row)}
                    for t_row, p_row in zip(local_topic_simplices, local_simplex_probabilities)
            ]

            _, _, entropy_hist = self.agg_entropy_per_simplex[topic_simplex]
            hist_counts, hist_bucket_endpoints = entropy_hist
            hist_probs = hist_counts / hist_counts.sum()
            n_buckets = len(hist_bucket_endpoints) - 1

            P = []
            for e in local_simplex_entropies:
                try:
                    bucket_idx = next(i for i in range(n_buckets) if hist_bucket_endpoints[i] > e)
                except StopIteration:
                    bucket_idx = n_buckets - 1

                sample_p = 1/max(1, hist_probs[bucket_idx])
                assert not np.isnan(sample_p) and sample_p > 0
                P.append(sample_p)

            P = np.array(P)
            P /= P.sum()

            num_samples = min(N_sents, self.num_samples_per_simplex)
            sample_idxs = np.random.choice(np.arange(N_sents), num_samples, replace=False, p=P)

            self.defined_source_sentence_idxs.extend(local_source_sentence_idxs[sample_idxs])
            self.defined_original_sentence_idxs.extend(local_original_sentence_idxs[sample_idxs])
            self.defined_topics.extend(local_topic_labels[sample_idxs])
            self.defined_local_coordinates.extend([local_coordinates[i] for i in sample_idxs])

            self.defined_sentences_per_simplex[simplex_key].extend(local_original_sentence_idxs[sample_idxs])

        self.defined_simplex_per_sentence = {
            i: k for k, v in self.defined_sentences_per_simplex.items() for i in v
        }

        self.defined_sentences, self.defined_topic_probabilities = [], []
        for idx in zip(self.defined_source_sentence_idxs):
            # Note this requires we pass in unaltered sentences.
            self.defined_sentences.append(
                self.source_sentences[idx].translate(str.maketrans('', '', string.punctuation)).lower()
            )
            self.defined_topic_probabilities.append(self.source_topic_probabilities[idx])

        return True

    def _gen_graph(self, changing_kwargs=None, re_set_topics=False):
        needs_new_distances = (
            (not hasattr(self, 'pairwise_geodesic_distances')) or
            re_set_topics or
            'dataset_size' in changing_kwargs
        )
        if needs_new_distances:
            print("Generating Distances & Graph")
            precomputed_distance_matrix = None
        else:
            print("Generating Graph using Pre-computed Distances")
            precomputed_distance_matrix = self.pairwise_geodesic_distances

        self.G, self.pairwise_geodesic_distances = self.manifold.radius_nearest_neighbor_graph(
            self.manifold_radius_graph_r, *[self.defined_local_coordinates[i] for i in self.PT_idxs],
            precomputed_distance_matrix = precomputed_distance_matrix
        )
