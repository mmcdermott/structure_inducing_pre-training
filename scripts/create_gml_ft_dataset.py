import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, torch, time, networkx as nx, numpy as np
from pytorch_lightning import seed_everything
from graph_augmented_pt.datasets.protein_dataset import *

species_file_train = os.path.join(RAW_DATASETS_DIR, 'treeoflife/species_files/train_species.txt')
species_file_val   = os.path.join(RAW_DATASETS_DIR, 'treeoflife/species_files/val_species.txt')
species_file_test  = os.path.join(RAW_DATASETS_DIR, 'treeoflife/species_files/test_species.txt')

def main(
    output_dir:                     str,
):
    # Setup dataset.
    output_dir                  = Path(output_dir)
    embeddings_dir              = output_dir / 'embeddings'
    ft_datasets_dir             = output_dir / 'finetune' / 'data'
    train_dir                   = ft_datasets_dir / 'train'
    val_dir                     = ft_datasets_dir / 'val'
    test_dir                    = ft_datasets_dir / 'test'

    if not ft_datasets_dir.exists(): 
        os.mkdir(ft_datasets_dir)

    assert not train_dir.exists(), 'Dataset already exists!'
    os.mkdir(train_dir)
    os.mkdir(val_dir)
    os.mkdir(test_dir)

    def get_species_list(filename):
        with open(filename, 'r') as f:
            return [ln.strip() for ln in f]
    
    all_species                 = get_species_list(SPECIES_1840)
    train_species               = get_species_list(species_file_train)
    val_species                 = get_species_list(species_file_val)
    test_species                = get_species_list(species_file_test)

    assert set(all_species) == set(train_species).union(set(val_species), set(test_species))
    assert len(all_species) == len(train_species) + len(val_species) + len(test_species)

    species_filename = str(SPECIES_1840).split('/')[-1].strip('.txt')
    batch_size = 4
    train_dataset = TreeoflifeDataset(
        species                 = all_species,
        species_filename        = species_filename,
        max_seq_len             = -1,   # Force error if not using cached node features.
        min_sample_nodes        = 0,    # Use entire graphs, 
        batch_size              = batch_size,     
        do_from_tape            = True,
        do_use_sample_cache     = False, # Don't use cached samples.
    )

    point_embeddings            = {'train': [], 'val': [], 'test': []}
    context_embeddings          = {'train': [], 'val': [], 'test': []}
    labels                      = {'train': [], 'val': [], 'test': []}

    start = time.time()

    count = {k: 0 for k in point_embeddings}

    seed = 0
    seed_everything(seed)
    
    for species_idx, species_name in enumerate(train_dataset.species):
        assert (species_name in train_species) + (species_name in val_species) + (species_name in test_species) == 1

        if species_name in train_species:   split = 'train'
        if species_name in val_species:     split = 'val'
        if species_name in test_species:    split = 'test'

        embedding_file = embeddings_dir / (str(species_idx) + '.pt')
        all_embeddings = torch.load(embedding_file)

        start_idx, end_idx = train_dataset.index_ranges[species_idx]
        for i in range(start_idx, end_idx+1):
            count[split] += 1
            positive_sample_index = random.choice(list(train_dataset.G.neighbors(i)))
            negative_sample_index = random.choice(train_dataset.negative_sample_fn(i))
            
            assert i in range(start_idx, end_idx+1)
            assert positive_sample_index in range(start_idx, end_idx+1)
            assert negative_sample_index in range(start_idx, end_idx+1), f'{i} {positive_sample_index} {negative_sample_index} {start_idx} {end_idx}'

            point_embed = all_embeddings[i - start_idx]
            pos_embed   = all_embeddings[positive_sample_index - start_idx]
            neg_embed   = all_embeddings[negative_sample_index - start_idx]

            point_embed = point_embed.unsqueeze(0)
            pos_embed   = pos_embed.unsqueeze(0)
            neg_embed   = neg_embed.unsqueeze(0)

            point_embeddings[split].extend([point_embed, point_embed])
            context_embeddings[split].extend([pos_embed, neg_embed])
            labels[split].extend([1,0]) 

        print(f'{species_idx} of {len(train_dataset.species)}. Elapsed: {round(time.time() - start, 2)}s')
        
    for split in ('train', 'val', 'test'):
        point_embeddings_tensor = torch.cat(point_embeddings[split])
        context_embeddings_tensor = torch.cat(context_embeddings[split])
        labels_tensor = torch.tensor(labels[split])

        torch.save(point_embeddings_tensor, ft_datasets_dir / split / 'point_embeddings.pt')
        torch.save(context_embeddings_tensor, ft_datasets_dir / split / 'context_embeddings.pt')
        torch.save(labels_tensor, ft_datasets_dir / split / 'labels.pt')

    print(count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
            type=str)
    args = parser.parse_args()
    arg_dict = vars(args)
    main(**arg_dict)
