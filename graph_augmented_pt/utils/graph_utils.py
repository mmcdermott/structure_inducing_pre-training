import torch, random

def select(total_encoded, indices):
    """
    Selects encodings for a batch of point or context indices.
    """
    indices = indices.squeeze()
    summary = torch.index_select(total_encoded['summary'], dim=0, index=indices)
    granular = torch.index_select(total_encoded['granular'], dim=0, index=indices)

    return {
        'summary': summary,
        'granular': granular,
    }

def get_sample_from_graph(G, min_sample_nodes):
    num_nodes_to_sample = min(min_sample_nodes, len(G))
    sample = random.sample(list(G.nodes), num_nodes_to_sample)
    sample_including_neighbors = []
    for n in sample:
        sample_including_neighbors.extend(list(G.neighbors(n)))
        sample_including_neighbors.append(n)
    sample_including_neighbors = set(sample_including_neighbors)
    G_sample = G.subgraph(sample_including_neighbors)
    return sorted(list(sample_including_neighbors)), G_sample
