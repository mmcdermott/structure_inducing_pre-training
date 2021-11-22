import os, sys, time
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback

from ..constants import *
from ..datasets.mag_dataset import *
from ..utils.eval_utils import *
from ..utils.train_utils import *
from .run_singleton import *

def run_mag_pretrain(
    output_dir:                     str,
    seed:                           int     = 42,
    gml_weight:                     float   = 1.0,
    gml_encoder:                    str     = POINT_ENCODER,
    gml_head:                       str     = COSINE_DISTANCE,
    contrastive_margin:             float   = 1e1,
    pos_weight:                     float   = 1,
    neg_weight:                     float   = 1,
    thresh:                         float   = 0.1,
    ego_graph_radius:               int     = 3,
    encoder_hidden_size:            int     = 10,
    encoder_num_hidden_layers:      int     = 2,
    encoder_num_attention_heads:    int     = 2,
    encoder_intermediate_size:      int     = 8,
    context_encoder_hidden_size:    int     = 10,
    context_encoder_hidden_layers:  int     = 2,
    max_seq_len:                    int     = 4096,
    grad_accum:                     int     = 1,
    batch_size:                     str     = 'ALL',
    lr:                             float   = 3e-4,
    num_epochs:                     int     = 2500,
    save_freq:                      int     = 100,
    num_samples:                    int     = 25,
    do_log_on_epoch:                bool    = True,
    do_half:                        bool    = False,
    do_ddp:                         bool    = False,
    gpus:                           int     = 1,
    weights_summary:                str     = 'top',
    do_simple_profiler:             bool    = False,
    do_advanced_profiler:           bool    = False,
    neighbor_prediction_eval_freq:  int     = 100,
    do_neighbor_prediction_eval:    bool    = True,
    min_sample_nodes:               int     = 50,
    eval_batch_size:                int     = 32,
    do_from_scibert:                bool    = False,
    hf_model_name:                  str     = SCIBERT_SCIVOCAB_UNCASED,
    do_debug_graph:                 bool    = False,
    do_normalize_embeds:            bool    = False,
):
    output_dir = Path(output_dir)
    seed_everything(seed)

    if batch_size != 'ALL': batch_size = int(batch_size)
    do_flat_batch = gml_head in (MULTISIMILARITY,)

    train_dataset = MAGDataset(
        seed                    = seed,
        max_len                 = max_seq_len,
        run_name                = str(output_dir),
        min_sample_nodes        = min_sample_nodes,
        hf_model_name           = hf_model_name,
        batch_size              = batch_size,
        do_debug_graph          = do_debug_graph,
        do_flat_batch           = do_flat_batch
    )

    start = time.time()

    def do_sample(sample_num):
        sample_dir = output_dir / str(sample_num)
        if not sample_dir.exists(): os.mkdir(sample_dir)

        callbacks = []

        if do_neighbor_prediction_eval: 
            pass # TODO: Add callback.

        if save_freq > 0: callbacks.append(
            SaveCallback(
                save_freq   = save_freq,
                sample_dir  = sample_dir,
                start       = start,
            )
        )

        gml_head_kwargs ={
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
            'gml_encoder': gml_encoder,
            'gml_head': gml_head,
            'gml_head_kwargs': gml_head_kwargs,
            'encoder_hidden_size': encoder_hidden_size,
            'encoder_num_hidden_layers': encoder_num_hidden_layers,
            'encoder_num_attention_heads': encoder_num_attention_heads,
            'encoder_intermediate_size': encoder_intermediate_size,
            'context_encoder_hidden_size': context_encoder_hidden_size,
            'context_encoder_hidden_layers': context_encoder_hidden_layers,
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
            'do_from_scibert': do_from_scibert,
            'hf_model_name': hf_model_name,
            'callbacks': callbacks,
            'do_normalize_embeds': do_normalize_embeds,
        }

        model, trainer = run_singleton(**run_singleton_inputs)

        if model.local_rank == 0:
            save_pl_module(trainer, model, sample_dir / 'final_save')
            end = time.time()
            elapsed = round((end - start) / 3600, 2)
            with open(sample_dir / 'elapsed.txt', 'w') as f:
                f.write(str(elapsed) + ' hours elapsed.')

    samples_rng = range(num_samples)
    samples_rng = tqdm(samples_rng)

    for sample_num in samples_rng:
        do_sample(sample_num)
