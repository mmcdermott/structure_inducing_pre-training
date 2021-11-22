import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, torch, time

from transformers import BertConfig, BertModel, AutoModel
from pytorch_lightning import seed_everything

from graph_augmented_pt.torch_modules import *
from graph_augmented_pt.datasets.mag_dataset import *
from graph_augmented_pt.utils.eval_utils import *
from graph_augmented_pt.utils.utils import *
from graph_augmented_pt.utils.graph_utils import *

seed_everything(0)

MAX_SEQ_LEN = 512

# Setup models if not using cached embeddings.
def get_modules(
    output_dir,
    dataset, 
    do_ft_eval,
    hf_model_name,
):
    tokenizer = dataset.tokenizer
    if hasattr(tokenizer, 'vocab_size'): vocab_size = tokenizer.vocab_size
    else: vocab_size = len(tokenizer.vocab)

    config = BertConfig(
        vocab_size = vocab_size,
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

    def load_and_prepare_model(model, path, device='cuda', half=True):
        model.load_state_dict(torch.load(path))
        model.to(device)
        if half: model.half()
        model.eval()

    if do_ft_eval:
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

        finetuning_head = head_cls(**head_kwargs)
        load_and_prepare_model(
            model = finetuning_head,
            path = output_dir / 'finetuning_head.pt',
            device = 'cpu',
            half = False,
        )
        modules += (finetuning_head.fc,)

    else:
        encoder_module = EncoderModule(
            config,
            BertModel,
            cls_pooler,
        )
        print('Loading model...')
        encoder_module.sequence_model = AutoModel.from_pretrained(hf_model_name, cache_dir='./pretrained_model')
        load_and_prepare_model(
            model = encoder_module,
            path = output_dir / 'encoder.pt'
        )
        modules += (encoder_module,)
        
    return modules


def get_mag_embeddings_from_model(
    output_dir,
    dataset,
    eval_batch_size,
    encoder_module,
    embedding_file = None,
    device = 'cuda',
):
    features = dataset.tokenized_node_features
    features_split = np.array_split(
        features,
        len(features) // eval_batch_size,
    )

    start = time.time()

    embeddings_split = []
    for batch_idx, subarray in enumerate(features_split):
        end = time.time()
        with open(output_dir / 'progress.txt', 'w') as f:
            f.write(f'{batch_idx} of {len(features_split)} batches embedded. {round(end - start, 2)}s elapsed.')
        embeddings = get_embeddings(
            encoder_module,
            subarray,
            device = device,
            pad_id = dataset.pad_id,
        )
        embeddings_split.append(embeddings)
    all_embeddings = torch.cat(embeddings_split)
    all_embeddings = all_embeddings.float()

    if embedding_file is not None:
        print(f'Saving embedding to {str(embedding_file)}.')
        torch.save(all_embeddings, embedding_file)
    return all_embeddings


def eval_all_rank(
    output_dir,
    dataset,
    do_debug_graph,
    eval_batch_size,
    use_cached_embeddings,
    embeddings_dir,
    do_baseline,
    do_ft_eval,
    num_samples,
    min_sample_nodes,
    encoder_module = None,
    finetuning_head = None,
):
    start = time.time()
    graph_name = 'debug' if do_debug_graph else 'mag'
    embedding_file = embeddings_dir / f'{graph_name}_embed.pt'

    if do_baseline:
        all_embeddings = torch.rand(649880, 768)

    elif use_cached_embeddings:
        print(f'Reading MAG embedding.')
        all_embeddings = torch.load(embedding_file)
        if do_ft_eval:
            all_embeddings = finetuning_head(all_embeddings)
            all_embeddings = all_embeddings.detach()

    else:
        print('Writing MAG embedding.')
        os.mkdir(embeddings_dir)
        all_embeddings = get_mag_embeddings_from_model(
            output_dir      = output_dir,
            dataset         = dataset,
            eval_batch_size = eval_batch_size,
            encoder_module  = encoder_module,
            embedding_file  = embedding_file,
        )

    raw_data            = Path(RAW_DATASETS_DIR)
    ogb                 = raw_data /'ogb'
    test_node_ids_path  = ogb / 'test_node_ids.pkl'
    test_graph_path     = ogb / 'test_graph.pkl'        

    print('Reading test graph.')
    test_node_ids       = depickle(test_node_ids_path)
    test_graph          = nx.read_gpickle(test_graph_path)

    test_embeddings = np.take(
        all_embeddings,
        test_node_ids,
        axis=0
    )

    TRIALS = 5
    trial_accum = RunAccumulator()
    print('Starting trials.')

    for trial_idx in range(TRIALS):
        print(f'Running trial {trial_idx}')
        sample_accum = RunAccumulator()
        for sample_idx in range(num_samples):
            print(sample_idx)
            sample_node_idxs, sample_graph = get_sample_from_graph(test_graph, min_sample_nodes)
            sample_graph = nx.convert_node_labels_to_integers(sample_graph, ordering='sorted')
            sample_embeddings = np.take(
                test_embeddings,
                sample_node_idxs,
                axis=0
            )

            metrics_dict = neighbor_prediction_eval(
                    G               = sample_graph,
                    embeddings      = sample_embeddings,
            )

            total_metrics = calc_total_metrics(metrics_dict)
            sample_accum.update(total_metrics)
        trial_accum.update({name: sample_accum.mean_std(name)[0] for name in sample_accum.run_stats})

    with open(output_dir / 'all_eval_results.txt', 'w') as f:
        for name in trial_accum.run_stats:
            mean, std = trial_accum.mean_std(name)
            print(f'{name} = {mean} ± {std}')
            f.write(f'{name} = {mean} ± {std}\n')


def main(
    output_dir:                     str,
    eval_batch_size:                int     = 8,
    do_debug_graph:                 bool    = False,
    do_ft_eval:                     bool    = False,
    do_baseline:                    bool    = False,
    num_samples:                    int     = 100,
    min_sample_nodes:               int     = 200,
    hf_model_name:                  str     = SCIBERT_SCIVOCAB_UNCASED,
):
    # Setup dataset.
    output_dir = Path(output_dir)
    
    try: embeddings_dir = find_in_parent_dirs(output_dir, 'embeddings')
    except: embeddings_dir = output_dir / 'embeddings'

    use_cached_embeddings = embeddings_dir.exists()
    if do_ft_eval: assert use_cached_embeddings, 'Must have cached embeddings for FT eval!' # Else we couldn't have done FT.
    
    if not use_cached_embeddings:
        dataset = MAGDataset(
            max_len             = MAX_SEQ_LEN,
            do_debug_graph      = do_debug_graph,
            hf_model_name       = hf_model_name,
        )
        print('Done reading dataset.')
    else: dataset = None

    need_modules = (not do_baseline and not use_cached_embeddings) or do_ft_eval
    all_rank_kwargs = {}
    if need_modules:
        modules = get_modules(
            output_dir          = output_dir,
            dataset             = dataset,
            do_ft_eval          = do_ft_eval,
            hf_model_name       = hf_model_name,
        )
        if do_ft_eval:
            all_rank_kwargs['finetuning_head'] = modules[0]
        else:
            all_rank_kwargs['encoder_module'] = modules[0]
    
    eval_all_rank(
        output_dir              = output_dir,
        dataset                 = dataset,
        do_debug_graph          = do_debug_graph,
        eval_batch_size         = eval_batch_size,
        use_cached_embeddings   = use_cached_embeddings,
        embeddings_dir          = embeddings_dir,
        do_baseline             = do_baseline,
        do_ft_eval              = do_ft_eval,
        num_samples             = num_samples,
        min_sample_nodes        = min_sample_nodes,
        **all_rank_kwargs
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
            type=str)
    parser.add_argument('--eval_batch_size',
            type=int, default=8)
    parser.add_argument('--do_debug_graph',
            action='store_true')
    parser.add_argument('--do_ft_eval',
            action='store_true')
    parser.add_argument('--do_baseline',
            action='store_true')
    parser.add_argument('--hf_model_name',
            type=str, default=SCIBERT_SCIVOCAB_UNCASED)
    args = parser.parse_args()
    arg_dict = vars(args)

    main(**arg_dict)
