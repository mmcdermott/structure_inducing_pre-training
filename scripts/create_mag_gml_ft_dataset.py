import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, torch, networkx as nx
from pytorch_lightning import seed_everything

from graph_augmented_pt.datasets.mag_dataset import *
from graph_augmented_pt.utils.utils import *

MAX_SEQ_LEN = 512
seed_everything(0)

def main(
    output_dir:         str,
):
    # Setup dataset.
    output_dir          = Path(output_dir)
    embeddings_dir      = output_dir / 'embeddings'
    ft_datasets_dir     = output_dir / 'finetune' / 'data'
    train_dir           = ft_datasets_dir / 'train'
    val_dir             = ft_datasets_dir / 'val'
    test_dir            = ft_datasets_dir / 'test'

    raw_data            = Path(RAW_DATASETS_DIR)
    ogb                 = raw_data /'ogb'
    train_nodes_path    = ogb / 'train_node_ids.pkl'
    val_nodes_path      = ogb / 'val_node_ids.pkl'
    test_nodes_path     = ogb / 'test_node_ids.pkl'

    if not ft_datasets_dir.exists():
        os.mkdir(ft_datasets_dir)

    assert not train_dir.exists(), 'Dataset already exists!'
    os.mkdir(train_dir)
    os.mkdir(val_dir)
    os.mkdir(test_dir)

    train_nodes         = set(depickle(train_nodes_path))
    val_nodes           = set(depickle(val_nodes_path))
    test_nodes          = set(depickle(test_nodes_path))

    dataset = MAGDataset(
        max_len         = MAX_SEQ_LEN,
    )

    G                   = dataset.G
    train_graph         = G.subgraph(train_nodes)
    val_graph           = G.subgraph(val_nodes)
    test_graph          = G.subgraph(test_nodes)
    split_graphs                = {'train': train_graph, 'val': val_graph, 'test': test_graph}
    point_embeddings            = {'train': [], 'val': [], 'test': []}
    context_embeddings          = {'train': [], 'val': [], 'test': []}
    labels                      = {'train': [], 'val': [], 'test': []}
    count = {k: 0 for k in point_embeddings}
    
    embedding_file              = embeddings_dir / 'mag_embed.pt'
    all_embeddings              = torch.load(embedding_file)
    assert all_embeddings.shape[0] == len(G)

    for i in range(len(dataset.G)):
        if i % 1000 == 0: print(i)
        assert (i in train_nodes) + (i in val_nodes) + (i in test_nodes) <= 1

        if i in train_nodes:   split = 'train'; split_nodes = train_nodes
        elif i in val_nodes:     split = 'val'; split_nodes = val_nodes
        elif i in test_nodes:    split = 'test'; split_nodes = test_nodes
        else: continue 

        count[split] += 1
        i_graph = split_graphs[split]

        positive_sample_index = random.choice(list(i_graph.neighbors(i)))
        negative_sample_index = random.choice(list(split_nodes - set(i_graph.neighbors(i)) - set([i])))
        
        assert positive_sample_index in split_nodes
        assert negative_sample_index in split_nodes

        point_embed = all_embeddings[i]
        pos_embed   = all_embeddings[positive_sample_index]
        neg_embed   = all_embeddings[negative_sample_index]

        point_embed = point_embed.unsqueeze(0)
        pos_embed   = pos_embed.unsqueeze(0)
        neg_embed   = neg_embed.unsqueeze(0)   

        point_embeddings[split].extend([point_embed, point_embed])
        context_embeddings[split].extend([pos_embed, neg_embed])
        labels[split].extend([1,0])
        
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
