import os, time
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback

from ..constants import *
from ..datasets.ranking_dataset import *
from ..utils.train_utils import *
from .run_finetune_singleton import *

def run_ranking_finetune(
    output_dir:                     str,
    task:                           str     = EUCLIDEAN_DISTANCE,
    seed:                           int     = 42,
    encoder_hidden_size:            int     = 10,
    contrastive_margin:             float   = 1e1,
    grad_accum:                     int     = 1,
    train_set_frac:                 float   = 1.0,
    do_upsample:                    bool    = False,
    train_batch_size:               int     = 32,
    eval_batch_size:                int     = 32,
    lr:                             float   = 1e-4,
    num_epochs:                     int     = 10,
    patience:                       int     = 10,
    num_samples:                    int     = 3,
    do_half:                        bool    = False,
    do_ddp:                         bool    = False,
    gpus:                           int     = 1,
    weights_summary:                str     = 'top',
    do_simple_profiler:             bool    = False,
    do_advanced_profiler:           bool    = False,
):
    output_dir = Path(output_dir)
    data_dir = find_in_parent_dirs(
        current_dir = output_dir,
        target = 'data'
    )
    seed_everything(seed)

    head_kwargs_map = {
        EUCLIDEAN_DISTANCE: {
            'linker': MarginEuclideanLinker(encoder_hidden_size, encoder_hidden_size, negative_margin=contrastive_margin),
            'do_initial_fc': True,
            'fc_input_dim': encoder_hidden_size,
            'fc_output_dim': encoder_hidden_size,
        },
        COSINE_DISTANCE: {
            'linker': BilinearCosineLinker(encoder_hidden_size, encoder_hidden_size),
            'do_initial_fc': False,
        }
    }

    dataset_cls = RankingDataset
    head_cls = GmlFinetuningHead
    head_kwargs = head_kwargs_map[task]
    metric_name = None

    def do_sample(sample_num):
        train_dataset = dataset_cls(
            split             = 'train',
            data_dir          = data_dir,
            batch_size        = train_batch_size,
        )
        val_dataset = dataset_cls(
            split             = 'val',
            data_dir          = data_dir,
            batch_size        = eval_batch_size,
        )
        
        start = time.time()
        sample_dir = output_dir / str(sample_num)
        final_save_dir = sample_dir / 'final_save'
        if final_save_dir.exists():
            return                      # Finished already.
        if not sample_dir.exists(): 
            os.mkdir(sample_dir)        # Only need one process to mkdir.

        callbacks = []

        save_freq = int(1 / train_set_frac) if do_upsample else 1
        callbacks.append(
            FinetuneSaveCallback(
               sample_dir       = sample_dir,
               start            = start,
               save_freq        = save_freq,
           )
        )

        run_finetune_singleton_inputs = {
            'sample_dir': sample_dir,
            'task': task,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'head_cls': head_cls,
            'head_kwargs': head_kwargs,
            'do_head_only': True,
            'grad_accum': grad_accum,
            'train_set_frac': train_set_frac,
            'do_upsample': do_upsample,
            'lr': lr,
            'do_half': do_half,
            'do_ddp': do_ddp,
            'gpus': gpus,
            'max_epochs': num_epochs,
            'patience': patience,
            'metric_name': metric_name,
            'weights_summary': weights_summary,
            'do_simple_profiler': do_simple_profiler,
            'do_advanced_profiler': do_advanced_profiler,
            'callbacks': callbacks,
        }

        model, trainer = run_finetune_singleton(**run_finetune_singleton_inputs)

        if model.local_rank == 0:
            os.mkdir(final_save_dir)
            save_ft_module(model, final_save_dir)
            end = time.time()
            elapsed = round((end - start) / 3600, 2)
            with open(final_save_dir / 'elapsed.txt', 'w') as f:
                f.write(str(elapsed) + ' hours elapsed.')

    samples_rng = range(num_samples)
    samples_rng = tqdm(samples_rng)

    for sample_num in samples_rng:
        do_sample(sample_num)
