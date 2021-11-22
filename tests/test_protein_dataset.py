import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph_augmented_pt.datasets.protein_dataset import *

species = SPECIES_1840[:-16] + '2_species.txt'
species_filename = str(species).split('/')[-1].strip('.txt')
with open(species, 'r') as f:
    species = [ln.strip() for ln in f]

train_dataset = TreeoflifeDataset(
        species                 = species,
        species_filename        = species_filename,
        max_seq_len             = 512,   # Force error if not using cached node features.
        min_sample_nodes        = 0,    # Use entire graphs,
        batch_size              = 4,
        do_from_tape            = True,
        do_use_sample_cache     = False, # Don't use cached samples.
    )

assert len(train_dataset.G)                                         == 344
assert len(train_dataset.G.edges)                                   == 554

assert len(train_dataset.Gs_by_species)                             == 2
assert len(train_dataset.Gs_by_species[0])                          == 192
assert len(train_dataset.Gs_by_species[1])                          == 152

assert len(train_dataset.node_features)                             == 344
assert len(train_dataset.node_features_by_species[0])               == 192
assert len(train_dataset.node_features_by_species[1])               == 152

assert train_dataset.index_ranges                                   == [(0, 191), (192, 343)]
assert len(train_dataset.tokenized_node_features_by_species[0])     == 192
assert len(train_dataset.tokenized_node_features_by_species[1])     == 152

with open('data/node_features.txt', 'r') as f:
    node_features = f.readline().strip()
with open('data/tokenized_node_features.txt', 'r') as f:
    tokenized_node_features = eval(f.readline().strip())

assert train_dataset.node_features[0]                               == node_features
assert train_dataset.node_features_by_species[0][0]                 == node_features
assert train_dataset.tokenized_node_features[0]                     == tokenized_node_features
assert train_dataset.tokenized_node_features_by_species[0][0]       == tokenized_node_features

print('Dataset tests passed!')