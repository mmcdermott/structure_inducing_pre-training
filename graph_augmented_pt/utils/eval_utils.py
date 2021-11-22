import torch, numpy as np, networkx as nx, collections, torch.optim as optim, time, collections
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import label_ranking_average_precision_score, ndcg_score, average_precision_score
from scipy.stats import rankdata

from ..torch_modules import *
from ..constants import *
from ..datasets.pretraining_dataset import *


class RunAccumulator:
    def __init__(self):
        self.run_stats = collections.defaultdict(list)

    def mean_std(self, key):
        assert key in self.run_stats, f'{key} not present.'
        stats = self.run_stats[key]
        return np.mean(stats), np.std(stats)

    def update(self, stats):
        for k, v in stats.items():
            self.run_stats[k].append(v)


def get_baseline_preds(A):
    mask = torch.zeros_like(torch.Tensor(A))
    for i in range(mask.shape[0]):
        mask[i][i] = 0.5

    # Generate random symmetric matrix.
    preds = torch.rand_like(torch.Tensor(A))
    preds *= mask                   # Multiply diagonal values by 0.5.
    preds = torch.triu(preds)       # Zero the below-diagonal elements.
    preds += preds.T                # Add the transpose.
    assert torch.all(torch.isclose(preds, preds.T)), f'{preds}'
    preds = preds.numpy()
    return preds


def get_embeddings(
    encoder_module,
    V,
    tokenizer = None,
    device = 'cuda',
    pad_id = 0,
):
    nodes = tokenizer.extract(V) if tokenizer is not None else V
    nodes = torch.tensor(nodes)
    K = len(nodes)

    input_ids = nodes.to(device)
    attention_mask = (nodes != pad_id).to(device)

    out = encoder_module(
        input_ids,
        attention_mask
    )

    embeddings = out['summary']
    embeddings = embeddings.detach().cpu()

    return embeddings


def get_embeddings_from_model(
    train_dataset,
    species_idx,
    G_sample,
    eval_batch_size,
    encoder_module,
    embedding_file = None,
    device = 'cuda',
):
    start_idx, end_idx = train_dataset.index_ranges[species_idx]
    take_indices = [x-start_idx for x in G_sample.nodes]
    features = np.take(
        train_dataset.tokenized_node_features_by_species[species_idx],
        take_indices,
        axis=0
    )

    features_split = np.array_split(
        features, 
        max(len(features) // eval_batch_size, 1)
    )

    embeddings_split = []
    for subarray in features_split:
        embeddings = get_embeddings(
            encoder_module,
            subarray,
            tokenizer = None,
            device = device,
        )
        embeddings_split.append(embeddings)
    all_embeddings = torch.cat(embeddings_split)

    if embedding_file is not None:
        print(f'Saving {species_idx} embedding.')
        torch.save(all_embeddings, embedding_file)
    return all_embeddings


def neighbor_prediction_eval(
    G,
    embeddings,
    metrics_dict = None,
):
    if metrics_dict is None: metrics_dict = collections.defaultdict(list)
    A = nx.adjacency_matrix(G, nodelist=sorted(list(G.nodes))).todense()
    A = np.array(A)

    preds = -1. * torch.cdist(embeddings, embeddings)
    preds = preds.numpy()
    n = len(preds)

    # Remove diagonals, since self-loops cannot be present. 
    def remove_diagonal(X):
        return X[~np.eye(X.shape[0],dtype=bool)].reshape(X.shape[0],-1)
    A = remove_diagonal(A)
    preds = remove_diagonal(preds)
    # NDCG requires nonnegative inputs, so shift everything by the min.
    preds = preds - preds.min()         
    # New shapes should be n x (n-1).
    assert A.shape == preds.shape and A.shape[0] == A.shape[1] + 1 and A.shape[0] == n

    # Compute all ranking-related metrics.
    lrap = label_ranking_average_precision_score(A, preds)
    ndcg = ndcg_score(A, preds, ignore_ties=True)
    ap = average_precision_score(A.reshape(-1), preds.reshape(-1))

    # Rankdata actually ranks smallest as best, so flip distances back.
    # ranks = rankdata(-1*preds, axis=1)        # doesn't work in scipy-1.3.1 (Power9)
    ranks = np.vstack([rankdata(row) for row in (-1 * preds)]) 
    rank_first_match = []
    for i, row in enumerate(A):
        matches = np.where(row == 1)
        ranks_matches = ranks[i, matches]
        rank_first_match.append(np.min(ranks_matches))
    rank_first_match = np.array(rank_first_match).squeeze()
    mrr = np.mean(1 / rank_first_match)

    # Compute hits @k
    k_list = [1, 5, 10, 25, 100]
    hits_at_k = {k: [] for k in k_list}
    for i, row in enumerate(A):
        matches = np.where(row == 1)
        ranks_matches = ranks[i, matches]
        ranks_matches = ranks_matches.squeeze(0) 
        for k in k_list:
            hits_at_k[k].append((ranks_matches <= k).sum())
    hits_at_k = {k: np.array(v) for k, v in hits_at_k.items()}

    precision_at_1 = np.mean(hits_at_k[1] / 1)
    precision_at_5 = np.mean(hits_at_k[5]/ 5)
    precision_at_10 = np.mean(hits_at_k[10] / 10)
    precision_at_25 = np.mean(hits_at_k[25]/ 25)
    precision_at_100 = np.mean(hits_at_k[100] / 100)

    recall_at_1 = hits_at_k[1].sum() / A.sum()
    recall_at_5 = hits_at_k[5].sum() / A.sum()
    recall_at_10 = hits_at_k[10].sum() / A.sum()
    recall_at_25 = hits_at_k[25].sum() / A.sum()
    recall_at_100 = hits_at_k[100].sum() / A.sum()

    # Update metrics dict.
    metrics_dict['num_nodes'].append(len(G))
    metrics_dict['num_directed_edges'].append(A.sum())

    metrics_dict['lrap'].append(lrap)
    metrics_dict['ndcg'].append(ndcg)
    metrics_dict['ap'].append(ap)
    metrics_dict['mrr'].append(mrr)

    metrics_dict['p_at_1'].append(precision_at_1)
    metrics_dict['p_at_5'].append(precision_at_5)
    metrics_dict['p_at_10'].append(precision_at_10)
    metrics_dict['p_at_25'].append(precision_at_25)
    metrics_dict['p_at_100'].append(precision_at_100)

    metrics_dict['r_at_1'].append(recall_at_1)
    metrics_dict['r_at_5'].append(recall_at_5)
    metrics_dict['r_at_10'].append(recall_at_10)
    metrics_dict['r_at_25'].append(recall_at_25)
    metrics_dict['r_at_100'].append(recall_at_100)

    return metrics_dict


def calc_total_metrics(metrics_dict):
    def calc_weighted(val_list, weight_list):
        return sum(v * w for v, w in zip(val_list, weight_list)) / sum(weight_list)

    def f1(pr, re):
        if pr == 0 and re == 0: return 0
        return 2*pr*re /(pr+re)

    total_metrics = {}
    
    total_metrics['lrap']      = calc_weighted(metrics_dict['lrap'],       metrics_dict['num_nodes'])
    total_metrics['ndcg']      = calc_weighted(metrics_dict['ndcg'],       metrics_dict['num_nodes'])
    total_metrics['ap']        = calc_weighted(metrics_dict['ap'],         metrics_dict['num_nodes'])
    total_metrics['mrr']       = calc_weighted(metrics_dict['mrr'],        metrics_dict['num_nodes'])

    total_metrics['p_at_1']    = calc_weighted(metrics_dict['p_at_1'],     metrics_dict['num_nodes'])
    total_metrics['p_at_5']    = calc_weighted(metrics_dict['p_at_5'],     metrics_dict['num_nodes'])
    total_metrics['p_at_10']   = calc_weighted(metrics_dict['p_at_10'],    metrics_dict['num_nodes'])
    total_metrics['p_at_25']   = calc_weighted(metrics_dict['p_at_25'],    metrics_dict['num_nodes'])
    total_metrics['p_at_100']  = calc_weighted(metrics_dict['p_at_100'],   metrics_dict['num_nodes'])

    total_metrics['r_at_1']    = calc_weighted(metrics_dict['r_at_1'],     metrics_dict['num_directed_edges'])
    total_metrics['r_at_5']    = calc_weighted(metrics_dict['r_at_5'],     metrics_dict['num_directed_edges'])
    total_metrics['r_at_10']   = calc_weighted(metrics_dict['r_at_10'],    metrics_dict['num_directed_edges'])
    total_metrics['r_at_25']   = calc_weighted(metrics_dict['r_at_25'],    metrics_dict['num_directed_edges'])
    total_metrics['r_at_100']  = calc_weighted(metrics_dict['r_at_100'],   metrics_dict['num_directed_edges'])

    total_metrics['f1_at_1']   = f1(total_metrics['p_at_1'],                total_metrics['r_at_1'])
    total_metrics['f1_at_5']   = f1(total_metrics['p_at_5'],                total_metrics['r_at_5'])
    total_metrics['f1_at_10']  = f1(total_metrics['p_at_10'],               total_metrics['r_at_10'])
    total_metrics['f1_at_25']  = f1(total_metrics['p_at_25'],               total_metrics['r_at_25'])
    total_metrics['f1_at_100'] = f1(total_metrics['p_at_100'],              total_metrics['r_at_100'])

    return total_metrics    

      
class ProteinNeighborPredictionEvalCallback(Callback):
    def __init__(
        self,
        train_dataset,
        eval_batch_size,
        neighbor_prediction_eval_freq,
    ):
        super().__init__()
        self.train_dataset                  = train_dataset
        self.eval_batch_size                = eval_batch_size
        self.neighbor_prediction_eval_freq  = neighbor_prediction_eval_freq

        self.prev_step                      = -1    # Used to make grad_accum > 1 work.

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        global_step = trainer.global_step
        if pl_module.local_rank != 0: return
        if global_step % self.neighbor_prediction_eval_freq != 0: return
        if global_step == self.prev_step: return

        start = time.time()
        pl_module.eval()
        keys = [
            'num_nodes', 'num_directed_edges',                      # Helps us cumulate metrics.
            'lrap', 'ndcg', 'ap', 'mrr',                            # Cumulate via num_nodes
            'p_at_1', 'p_at_5', 'p_at_10', 'p_at_25', 'p_at_100',   # Cumulate via num_nodes
            'r_at_1', 'r_at_5', 'r_at_10', 'r_at_25', 'r_at_100',   # Cumulate via num_directed_edges
        ]
        metrics_dict = {k: [] for k in keys}

        for species_idx, species_name in enumerate(self.train_dataset.species):
            G_sample = self.train_dataset.sample_species_subgraph(species_idx)

            embeddings = get_embeddings_from_model(
                self.train_dataset,
                species_idx,
                G_sample,
                self.eval_batch_size,
                pl_module.encoder_module,
                device = pl_module.device,
            )

            neighbor_prediction_eval(
                G            = G_sample,
                embeddings   = embeddings,
                metrics_dict = metrics_dict,
            )

            with open('progress.txt', 'a') as f:
                f.write(f'{species_idx}: {round(time.time()-start, 2)} elapsed\n')

            print(f'{species_idx}: {round(time.time()-start, 2)} elapsed')
            print(f'Total sampled: {sum(metrics_dict["num_nodes"])}')

        total_metrics = calc_total_metrics(metrics_dict)
        for name, val in total_metrics.items():
            print(f'{name} = {val}')
            pl_module.logger.experiment.add_scalar(f'Eval/{name}', val, global_step)

        self.prev_step = global_step
        pl_module.train()
