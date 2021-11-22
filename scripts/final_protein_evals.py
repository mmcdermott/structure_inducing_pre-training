import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, torch, time, torch.nn as nn, networkx as nx, numpy as np, scipy

from sklearn.metrics import label_ranking_average_precision_score, ndcg_score, average_precision_score
from scipy.stats import rankdata
from scipy.spatial.distance import cdist
from transformers import BertConfig, BertModel
from transformers.modeling_bert import BertOnlyMLMHead
# from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from graph_augmented_pt.torch_modules import *
from graph_augmented_pt.datasets.protein_dataset import *
from graph_augmented_pt.utils.eval_utils import *
from graph_augmented_pt.utils.utils import *


# Load resaved models onto Power9.
def load_model_txt(model, path):
    data_dict = {}
    fin = open(path, 'r')
    i = 0
    odd = 1
    prev_key = None
    while True:
        s = fin.readline().strip()
        if not s:
            break
        if odd:
            prev_key = s
        else:
            print('Iter', i)
            val = eval(s)
            if type(val) != type([]):
                data_dict[prev_key] = torch.FloatTensor([eval(s)])[0]
            else:
                data_dict[prev_key] = torch.FloatTensor(eval(s))
            i += 1
        odd = (odd + 1) % 2

    # Replace existing values with loaded
    print('Loading...')
    own_state = model.state_dict()
    print('Items:', len(own_state.items()))
    for k, v in data_dict.items():
        if not k in own_state:
            print('Parameter', k, 'not found in own_state!!!')
        else:
            try:
                own_state[k].copy_(v)
            except:
                print('Key:', k)
                print('Old:', own_state[k])
                print('New:', v)
                sys.exit(0)
    print('Model loaded')


# Setup models if not using cached embeddings.
def get_modules(
    output_dir,
    data_module, 
    tokenizer,
    eval_type,
    no_do_power9, 
    no_do_half,
    do_ft_eval,
    ft_task,
):
    config = BertConfig(
        vocab_size = len(tokenizer.vocab),
        hidden_size = 768,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 10,
        max_position_embeddings=4096,
        output_hidden_states = True,
        return_dict = True,
        hidden_dropout_prob=0
    )

    modules = ()

    def load_and_prepare_model(model, path, resaved_path, device='cuda'):
        if not no_do_power9:
            load_model_txt(model, resaved_path)
        else:
            model.load_state_dict(torch.load(path))
        model.to(device)
        if not no_do_half: model.half()
        model.eval()

    if do_ft_eval:
        encoder_module = None
        modules += (encoder_module,)
    else:
        encoder_module = EncoderModule(
            config,
            BertModel,
            cls_pooler
        )
        encoder_module.sequence_model = ProteinBertModel.from_pretrained('bert-base')
        load_and_prepare_model(
            model = encoder_module, 
            path = output_dir / 'encoder.pt', 
            resaved_path = output_dir / 'encoder.pt_new'
        )
        modules += (encoder_module,)

    if eval_type == 'mlm':
        point_pretraining_head = MLMPretrainingHead(
            config,
            BertOnlyMLMHead,
            mask_id = data_module.mask_id
        )
        load_and_prepare_model(
            model = point_pretraining_head,
            path = output_dir / 'point_pretraining_head.pt',
            resaved_path = output_dir / 'point_pretraining_head.pt_new'
        )
        modules += (point_pretraining_head,)

    if do_ft_eval:
        assert eval_type == ALL_RANK
        if ft_task == EUCLIDEAN_DISTANCE:
            encoder_hidden_size, contrastive_margin = 768, 1e1
            linker = MarginEuclideanLinker(
                encoder_hidden_size, 
                encoder_hidden_size,
                negative_margin=contrastive_margin,
            )
            head_cls = GmlFinetuningHead
            head_kwargs = {
                'linker': linker,
                'do_initial_fc': True,
                'fc_input_dim': encoder_hidden_size,
                'fc_output_dim': encoder_hidden_size,
            }

        elif ft_task == COSINE_DISTANCE:
            encoder_hidden_size = 768
            linker = BilinearCosineLinker(
                encoder_hidden_size,
                encoder_hidden_size,
            )
            head_cls = GmlFinetuningHead
            head_kwargs = {
                'linker': linker,
                'do_initial_fc': False
            }

        else:
            raise NotImplementedError

        finetuning_head = head_cls(**head_kwargs)
        load_and_prepare_model(
            model = finetuning_head,
            path = output_dir / 'finetuning_head.pt',
            resaved_path = output_dir / 'finetuning_head.pt_new',
            device = 'cpu'
        )
        modules += (finetuning_head,)

    return modules


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


def get_embeddings_from_model(
    train_dataset,
    species_idx,
    G_sample,
    eval_batch_size,
    encoder_module,
    embedding_file,
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
            device = 'cuda',
        )
        embeddings_split.append(embeddings)
    all_embeddings = torch.cat(embeddings_split)

    print(f'Saving {species_idx} embedding.')
    torch.save(all_embeddings, embedding_file)
    return all_embeddings


def calc_total_metrics(metrics_dict):
    def calc_weighted(val_list, weight_list):
        return sum(v * w for v, w in zip(val_list, weight_list)) / sum(weight_list)

    def f1(pr, re):
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


def eval_all_rank(
    output_dir,
    train_dataset, 
    use_cached_embeddings,
    embeddings_dir, 
    do_baseline, 
    do_ft_eval,
    ft_task,
    encoder_module=None,
    finetuning_head=None,
):
    keys = [
        'num_nodes', 'num_directed_edges',                      # Helps us cumulate metrics.
        'lrap', 'ndcg', 'ap', 'mrr',                            # Cumulate via num_nodes
        'p_at_1', 'p_at_5', 'p_at_10', 'p_at_25', 'p_at_100',   # Cumulate via num_nodes
        'r_at_1', 'r_at_5', 'r_at_10', 'r_at_25', 'r_at_100',   # Cumulate via num_directed_edges
    ]
    metrics_dict = {k: [] for k in keys}
    start = time.time()
    eval_batch_size = 4

    if do_ft_eval:
        species_file_train = os.path.join(RAW_DATASETS_DIR, 'treeoflife/species_files/train_species.txt')
        species_file_val   = os.path.join(RAW_DATASETS_DIR, 'treeoflife/species_files/val_species.txt')
        species_file_test  = os.path.join(RAW_DATASETS_DIR, 'treeoflife/species_files/test_species.txt')

        def get_species_list(filename):
            with open(filename, 'r') as f:
                return [ln.strip() for ln in f]
        
        train_species               = get_species_list(species_file_train)
        val_species                 = get_species_list(species_file_val)
        test_species                = get_species_list(species_file_test)

    count_species = 0

    for species_idx, species_name in enumerate(train_dataset.species):
        if do_ft_eval:
            if species_name not in test_species:
                assert species_name in train_species or species_name in val_species
                continue

        count_species += 1

        G_sample = train_dataset.sample_species_subgraph(species_idx)
        embedding_file = embeddings_dir / (str(species_idx) + '.pt')

        # Generate adj matrix.
        A = nx.adjacency_matrix(G_sample).todense()
        A = np.array(A)

        # Generate random pairwise distances.         no_do_power9   
        if do_baseline:
            preds = get_baseline_preds(A)

        # Get embeddings.
        else:
            if not use_cached_embeddings:
                all_embeddings = get_embeddings_from_model(
                    train_dataset,
                    species_idx,
                    G_sample,
                    eval_batch_size,
                    encoder_module,
                    embedding_file,
                )
            else:
                print(f'Reading {species_idx} embedding.')
                all_embeddings = torch.load(embedding_file)

            if do_ft_eval and ft_task == EUCLIDEAN_DISTANCE:
                all_embeddings = finetuning_head(all_embeddings)
                all_embeddings = all_embeddings.detach()

            if do_ft_eval and ft_task == COSINE_DISTANCE:
                n, _ = A.shape
                p_list = []

                for i in range(n):
                    input1 = all_embeddings
                    input2 = all_embeddings[i].repeat(n, 1)
                    dist = finetuning_head(
                        input1, input2
                    ).unsqueeze(0)
                    p_list.append(dist)   

                preds = -1. * torch.cat(p_list).squeeze()
                preds = preds.detach().numpy()
            else:
                # Want highest-scoring pairs to be smallest distance.
                preds = -1. * torch.cdist(all_embeddings, all_embeddings)
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
        # ranks = rankdata(-1*preds, axis=1)        # doesn't work in scipy-1.3.1 (power9)
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
        metrics_dict['num_nodes'].append(len(G_sample))
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

        end = time.time()
        print(f'{species_idx}: {round(end-start, 2)} elapsed')
        print(f'Total sampled: {sum(metrics_dict["num_nodes"])}')

        if species_idx > 0 and species_idx % 1 == 0:
            total_metrics = calc_total_metrics(metrics_dict)
            for name, val in total_metrics.items():
                print(f'{name} = {val}')
            with open(output_dir / 'progress.txt', 'w') as f:
                f.write(str(species_idx) + ' ' + str(count_species) + '\n')

    total_metrics = calc_total_metrics(metrics_dict)

    with open(output_dir / f'all_eval_results.txt' , 'w') as f:
        for name, val in total_metrics.items():
            print(f'{name} = {val}')
            f.write(f'{name} = {val}\n')


def eval_mlm(
    output_dir,
    encoder_module,
    point_pretraining_head,
    sample_rate,
    num_samples,
    batch_size,
    train_dataset,
):
    accum = RunAccumulator()
    PRINT_FREQ = 100
    assert (sample_rate > 0 and sample_rate <= 1)
    
    for sample in range(num_samples):
        start = time.time()
        loss_list, acc_overall_list, acc_masked_list, num_overall_list, num_masked_list = [], [], [], [], []
        train_dataloader = train_dataset.train_dataloader(shuffle=True)
        total_batches = int(sample_rate * len(train_dataloader))

        dataloader_iterator = iter(train_dataloader)
        for idx in range(total_batches):
            try: batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_dataloader)
                batch = next(dataloader_iterator)

            if idx > 0 and idx % PRINT_FREQ == 0:
                avg_acc_masked = sum(a * n for a, n in zip(acc_masked_list, num_masked_list)) / sum(num_masked_list)
                num_sampled = len(num_masked_list) * batch_size
                print(f'{idx} of {total_batches}/{len(train_dataloader)}: Avg acc: {avg_acc_masked}, Num sampled: {num_sampled}. {round(time.time() - start, 2)}s elapsed.')
                
            point_kwargs, gml_kwargs = batch
            for k, v in point_kwargs.items(): point_kwargs[k] = v.to('cuda')

            points_encoded = encoder_module(
                point_kwargs['input_ids'],
                point_kwargs['attention_mask']
            )
            point_pretraining_out = point_pretraining_head(
                points_encoded, point_kwargs
            )

            is_masked = point_kwargs['input_ids'] == train_dataset.mask_id
            is_present = point_kwargs['attention_mask'] == 1
            present_and_masked = is_present & is_masked
            num_present_overall = is_present.sum()
            num_present_masked = present_and_masked.sum()

            loss_list.append(
                point_pretraining_out['loss'].item()
            )
            acc_overall_list.append(
                point_pretraining_out['metrics']['accuracy_overall'].item()
            )
            acc_masked_list.append(
                point_pretraining_out['metrics']['accuracy_masked'].item()
            )
            num_overall_list.append(
                num_present_overall.item()
            )
            num_masked_list.append(
                num_present_masked.item()
            )

        avg_loss = sum(loss_list) / len(loss_list)
        avg_acc_overall = sum(a * n for a, n in zip(acc_overall_list, num_overall_list)) / sum(num_overall_list)
        avg_acc_masked = sum(a * n for a, n in zip(acc_masked_list, num_masked_list)) / sum(num_masked_list)                
        num_sampled = len(num_masked_list) * batch_size

        print(f'Loss = {avg_loss}')
        print(f'Overall acc = {avg_acc_overall}')
        print(f'Masked acc = {avg_acc_masked}')
        print(f'Num sampled = {num_sampled}')
        print()

        accum.update({
            'avg_loss': avg_loss,
            'avg_acc_overall': avg_acc_overall,
            'avg_acc_masked': avg_acc_masked,
            'num_sampled': num_sampled,
        })

        with open(output_dir / f'mlm_eval_results_{str(sample)}.txt' , 'w') as f:
            f.write(f'Loss = {avg_loss}\n')
            f.write(f'Overall acc = {avg_acc_overall}\n')
            f.write(f'Masked acc = {avg_acc_masked}\n')
            f.write(f'Num sampled = {num_sampled}\n')

    with open(output_dir / f'mlm_eval_results.txt' , 'w') as f:
        lm, ls = accum.mean_std('avg_loss')
        aom, aos = accum.mean_std('avg_acc_overall')
        amm, ams = accum.mean_std('avg_acc_masked')
        nsm, nss = accum.mean_std('num_sampled')
        f.write(f'Loss = {lm}±{ls}\n')
        f.write(f'Overall acc = {aom}±{aos}\n')
        f.write(f'Masked acc = {amm}±{ams}\n')
        f.write(f'Num sampled = {nsm}\n') 


def main(
    output_dir:                     str,
    species:                        str     = SPECIES_1840,
    eval_type:                      str     = ALL_RANK,
    no_do_power9:                   bool    = False,
    no_do_half:                     bool    = False,
    sample_rate:                    float   = 1,
    num_samples:                    int     = 1,
    do_baseline:                    bool    = False,
    do_ft_eval:                     bool    = False,
    ft_task:                        str     = EUCLIDEAN_DISTANCE,
):
    # Setup dataset.
    output_dir = Path(output_dir)
    species_filename = str(species).split('/')[-1].strip('.txt')
    with open(species, 'r') as f:
        species = [ln.strip() for ln in f]
    batch_size = 4

    train_dataset = TreeoflifeDataset(
        species                 = species,
        species_filename        = species_filename,
        max_seq_len             = -1,   # Force error if not using cached node features.
        min_sample_nodes        = 0,    # Use entire graphs, 
        batch_size              = batch_size,     
        do_from_tape            = True,
        do_use_sample_cache     = False, # Don't use cached samples.
    )
    data_module = train_dataset
    tokenizer = data_module.tokenizer if data_module.tokenizer else data_module

    try: embeddings_dir = find_in_parent_dirs(output_dir, 'embeddings')
    except: embeddings_dir = output_dir / 'embeddings'

    use_cached_embeddings = (eval_type == ALL_RANK and embeddings_dir.exists())
    if do_ft_eval: assert use_cached_embeddings, 'Must have cached embeddings for FT eval!'

    if eval_type == ALL_RANK: no_do_half = True
    if eval_type == 'mlm' and (not no_do_power9): no_do_half = True             # Old pytorch version doesn't allow half() with some of the np operations in mlm_accuracy().
    eval_all_kwargs = {}

    need_modules = not (do_baseline or use_cached_embeddings) or do_ft_eval
    if need_modules:
        if eval_type != 'mlm' and not do_ft_eval: os.mkdir(embeddings_dir)
        modules = get_modules(
            output_dir,
            data_module, 
            tokenizer, 
            eval_type, 
            no_do_power9, 
            no_do_half,
            do_ft_eval,
            ft_task,
        )
        encoder_module = modules[0]
        if len(modules) > 1: 
            if do_ft_eval: 
                finetuning_head = modules[1]
                if ft_task == EUCLIDEAN_DISTANCE:   finetuning_head = finetuning_head.fc
                elif ft_task == COSINE_DISTANCE:    finetuning_head = finetuning_head.linker.combiner
                else:                               raise ValueError('No FT head found!')
                eval_all_kwargs['finetuning_head'] = finetuning_head
            else: 
                assert eval_type == 'mlm', '2 modules returned, but not needed!'
                point_pretraining_head = modules[1]
        eval_all_kwargs['encoder_module'] = encoder_module
        
    # Do all evaluations related to ranking.
    if eval_type == ALL_RANK:
        eval_all_rank(
            output_dir,
            train_dataset, 
            use_cached_embeddings, 
            embeddings_dir, 
            do_baseline,
            do_ft_eval,
            ft_task,
            **eval_all_kwargs,
        )

    # Run MLM eval.
    elif eval_type == 'mlm':
        eval_mlm(
            output_dir,
            encoder_module,
            point_pretraining_head,
            sample_rate,
            num_samples,
            batch_size,
            train_dataset,
        )
    
    else: raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
            type=str)
    parser.add_argument('--species', 
            type=str, default=SPECIES_1840)
    parser.add_argument('--eval_type', 
            type=str, default=ALL_RANK)
    parser.add_argument('--no_do_power9',
            action='store_true')
    parser.add_argument('--no_do_half',
            action='store_true')
    parser.add_argument('--sample_rate',
            type=float, default=1)
    parser.add_argument('--num_samples',
            type=int, default=1)
    parser.add_argument('--do_baseline',
            action='store_true')
    parser.add_argument('--do_ft_eval',
            action='store_true')
    parser.add_argument('--ft_task',
            type=str, default=EUCLIDEAN_DISTANCE)
    args = parser.parse_args()
    arg_dict = vars(args)

    main(**arg_dict)
