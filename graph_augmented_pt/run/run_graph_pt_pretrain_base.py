import os, sys, time
from pathlib import Path
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
from typing import Any, Callable

from ..constants import *
from ..utils.eval_utils import *
from ..utils.train_utils import *

from inspect import signature

# create a decorator to copy signatures
def copy_signature(source_fct):
    def copy(target_fct):
        target_fct.__signature__ = signature(source_fct)
        return target_fct
    return copy

def run_graph_pt_pretrain_base(
    output_dir:                     str,
    data_cls:                       Any,
    singleton_fn:                   Callable,
    seed:                           int     = 42,
    gml_weight:                     float   = 1.0,
    point_weight:                   float   = 1.0,
    gml_encoder:                    str     = POINT_ENCODER,
    gml_head:                       str     = EUCLIDEAN_DISTANCE,
    n_neighbors:                    int     = None,
    max_n_neighbors:                int     = 50,
    radius:                         float   = 1.5,
    do_masking:                     bool    = True,
    do_multitask_PT:                bool    = True,
    mask_rate:                      float   = 0.15,
    do_mask_edge:                   bool    = False,
    contrastive_margin:             float   = 1e1,
    pos_weight:                     float   = 1,
    neg_weight:                     float   = 1,
    thresh:                         float   = 0.1,
    encoder_emb_dim:                int     = 300,
    encoder_hidden_size:            int     = 300,
    encoder_num_layer:              int     = 5,
    encoder_JK:                     str     = 'last',
    encoder_dropout_ratio:          float   = 0.1,
    encoder_graph_pooling:          str     = 'mean',
    encoder_do_masking:             bool    = True,
    encoder_do_context_pred:        bool    = False,
    encoder_do_mask_edge:           bool    = False,
    grad_accum:                     int     = 1,
    batch_size:                     str     = 128,
    lr:                             float   = 3e-4,
    num_epochs:                     int     = 125,
    save_freq:                      int     = 100,
    num_samples:                    int     = 1,
    do_log_on_epoch:                bool    = True,
    do_half:                        bool    = False,
    do_ddp:                         bool    = False,
    gpus:                           int     = 1,
    weights_summary:                str     = 'top',
    do_simple_profiler:             bool    = False,
    do_advanced_profiler:           bool    = False,
    min_sample_nodes:               int     = 50,
    eval_batch_size:                int     = 32,
    do_normalize_embeds:            bool    = False,
):
    output_dir = Path(output_dir)
    seed_everything(seed)

    do_subgraph = False         # Not supported for now.
    if batch_size != 'ALL': batch_size = int(batch_size)
    do_flat_batch = gml_head in (MULTISIMILARITY,)

    train_dataset = data_cls(
        n_neighbors             = n_neighbors,
        radius                  = radius,
        do_masking              = do_masking,
        mask_rate               = mask_rate,
        mask_edge               = 1 if do_mask_edge else 0, # they expect an int for who nows what reason
        batch_size              = batch_size,
        do_flat_batch           = do_flat_batch,
        do_compute_dists        = gml_weight > 0,
        seed                    = seed,
    )

    if train_dataset.do_compute_dists:
        train_dataset[0] # This initializes some dataset internals prior to getting farmed out for learning

    start = time.time()

    def do_sample(sample_num):
        sample_dir = output_dir / str(sample_num)
        if not sample_dir.exists(): os.mkdir(sample_dir)

        callbacks = []

        if save_freq > 0: callbacks.append(
            SaveCallback(
                save_freq                       = save_freq,
                sample_dir                      = sample_dir,
                start                           = start,
            )
        )

        gml_head_kwargs = {
            'margin': contrastive_margin,
            'negative_margin': contrastive_margin,
            'pos_weight': pos_weight,
            'neg_weight': neg_weight,
            'thresh': thresh,
            'do_normalize_embeds': do_normalize_embeds,
        }

        run_singleton_inputs = {
            'sample_dir': sample_dir,
            'data_module': train_dataset,
            'gml_weight': gml_weight,
            'point_weight': point_weight,
            'gml_encoder': gml_encoder,
            'gml_head': gml_head,
            'gml_head_kwargs': gml_head_kwargs,
            'encoder_emb_dim': encoder_emb_dim,
            'encoder_hidden_size': encoder_hidden_size,
            'encoder_num_layer': encoder_num_layer,
            'encoder_JK': encoder_JK,
            'encoder_dropout_ratio': encoder_dropout_ratio,
            'encoder_graph_pooling': encoder_graph_pooling,
            'encoder_do_masking': encoder_do_masking,
            'encoder_do_context_pred': encoder_do_context_pred,
            'encoder_do_mask_edge': encoder_do_mask_edge,
            'do_multitask_PT': do_multitask_PT,
            'grad_accum': grad_accum,
            'lr': lr,
            'do_log_on_epoch': do_log_on_epoch,
            'do_half': do_half,
            'do_ddp': do_ddp,
            'gpus': gpus,
            'max_epochs': num_epochs,
            'weights_summary': weights_summary,
            'do_simple_profiler': do_simple_profiler,
            'do_advanced_profiler': do_advanced_profiler,
            'callbacks': callbacks,
            'do_normalize_embeds': do_normalize_embeds,
        }

        model, trainer = singleton_fn(**run_singleton_inputs)

        if model.local_rank == 0:
            final_dir = sample_dir / 'final_save'
            save_pl_module(trainer, model, final_dir)
            end = time.time()
            elapsed = round((end - start) / 3600, 2)
            with open(final_dir / 'elapsed.txt', 'w') as f:
                f.write(str(elapsed) + ' hours elapsed.')

    samples_rng = range(num_samples)
    samples_rng = tqdm(samples_rng, total=num_samples, leave=False)

    for sample_num in samples_rng:
        do_sample(sample_num)
