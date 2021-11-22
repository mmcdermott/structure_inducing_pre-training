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

def run_graph_pt_finetune_base(
    output_dir:                     str,
    data_cls:                       Any,
    singleton_fn:                   Callable,
    seed:                           int     = 1,
    encoder_emb_dim:                int     = 300,
    encoder_hidden_size:            int     = 300,
    encoder_num_layer:              int     = 5,
    encoder_JK:                     str     = 'last',
    encoder_dropout_ratio:          float   = 0.1,
    encoder_graph_pooling:          str     = 'mean',
    grad_accum:                     int     = 1,
    train_batch_size:               str     = 128,
    lr:                             float   = 3e-4,
    num_epochs:                     int     = 125,
    save_freq:                      int     = 100,
    num_samples:                    int     = 1,
    do_half:                        bool    = False,
    do_ddp:                         bool    = False,
    gpus:                           int     = 1,
    weights_summary:                str     = 'top',
    do_simple_profiler:             bool    = False,
    do_advanced_profiler:           bool    = False,
    eval_batch_size:                int     = 32,
    patience:                       int     = 10,
    do_freeze_encoder:              bool    = False,
    do_schedule:                    bool    = False,
    task:                           str     = '',
    do_run_without_pretraining:     bool    = False,
):
    output_dir = Path(output_dir)
    seed_everything(seed)

    train_dataset = data_cls(seed=seed, batch_size=train_batch_size, split='train')
    val_dataset = data_cls(seed=seed, batch_size=eval_batch_size,  split='valid')

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

        run_singleton_inputs = {
            'sample_dir': sample_dir,
            'train_dataset': train_dataset,
            'do_freeze_encoder': do_freeze_encoder,
            'val_dataset': val_dataset,
            'encoder_emb_dim': encoder_emb_dim,
            'encoder_hidden_size': encoder_hidden_size,
            'encoder_num_layer': encoder_num_layer,
            'encoder_JK': encoder_JK,
            'encoder_dropout_ratio': encoder_dropout_ratio,
            'encoder_graph_pooling': encoder_graph_pooling,
            'grad_accum': grad_accum,
            'lr': lr,
            'do_half': do_half,
            'do_ddp': do_ddp,
            'gpus': gpus,
            'max_epochs': num_epochs,
            'weights_summary': weights_summary,
            'do_simple_profiler': do_simple_profiler,
            'do_advanced_profiler': do_advanced_profiler,
            'callbacks': callbacks,
            'patience': patience,
            'do_schedule': do_schedule,
            'head_kwargs': {'y_label': task},
            'do_run_without_pretraining': do_run_without_pretraining,
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
