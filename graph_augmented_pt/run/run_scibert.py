import os, time
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from ..constants import *
from ..datasets.scibert_dataset import *
from ..utils.train_utils import *
from .run_finetune_singleton import *

def run_scibert_finetune(
    output_dir:                     str,
    task:                           str     = MAG,
    seed:                           int     = 42,
    encoder_hidden_size:            int     = 10,
    encoder_num_hidden_layers:      int     = 2,
    encoder_num_attention_heads:    int     = 2,
    encoder_intermediate_size:      int     = 8,
    max_seq_len:                    int     = 4096,
    grad_accum:                     int     = 1,
    train_set_frac:                 float   = 1.0,
    do_upsample:                    bool    = False,
    train_batch_size:               int     = 32,
    eval_batch_size:                int     = 32,
    lr:                             float   = 1e-4,
    warmup_frac:                    float   = 0.2,
    num_epochs:                     int     = 10,
    patience:                       int     = 10,
    num_samples:                    int     = 3,
    do_half:                        bool    = False,
    do_ddp:                         bool    = False,
    gpus:                           int     = 1,
    hf_model_name:                  str     = SCIBERT_SCIVOCAB_UNCASED,
    weights_summary:                str     = 'top',
    do_simple_profiler:             bool    = False,
    do_advanced_profiler:           bool    = False,
    do_freeze_encoder:              bool    = False,
):
    output_dir = Path(output_dir)
    seed_everything(seed)

    dataset_cls = ScibertDataset
    head_cls = BertSequenceClassificationFinetuningHead
    metric_name = 'accuracy'

    def do_sample(sample_num):
        train_dataset = dataset_cls(
            task              = task,
            split             = 'train',
            max_seq_len       = max_seq_len,
            batch_size        = train_batch_size,
            hf_model_name     = hf_model_name,
        )
        val_dataset = dataset_cls(
            task              = task,
            split             = 'dev',
            max_seq_len       = max_seq_len,
            batch_size        = eval_batch_size,
            hf_model_name     = hf_model_name,
        )
        head_kwargs = {
            'num_labels': train_dataset.num_labels,  
            'hidden_dropout_prob': 0.1,
            'hidden_size': encoder_hidden_size,
        }

        start = time.time()
        sample_dir = output_dir / str(sample_num)
        final_save_dir = sample_dir / 'final_save'
        test_results = sample_dir / 'test_results.txt'
        if final_save_dir.exists() or test_results.exists():
            return                      # Finished already.
        if sample_dir.exists(): return
        if not sample_dir.exists(): 
            os.mkdir(sample_dir)        # Only need one process to mkdir.

        callbacks = []

        if train_set_frac < ONE_MINUS_EPSILON:
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
            'encoder_hidden_size': encoder_hidden_size,
            'encoder_num_hidden_layers': encoder_num_hidden_layers,
            'encoder_num_attention_heads': encoder_num_attention_heads,
            'encoder_intermediate_size': encoder_intermediate_size,
            'grad_accum': grad_accum,
            'train_set_frac': train_set_frac,
            'do_upsample': do_upsample,
            'lr': lr,
            'warmup_frac': warmup_frac,
            'do_half': do_half,
            'do_ddp': do_ddp,
            'gpus': gpus,
            'hf_model_name': hf_model_name,
            'max_epochs': num_epochs,
            'patience': patience,
            'metric_name': metric_name,
            'weights_summary': weights_summary,
            'do_simple_profiler': do_simple_profiler,
            'do_advanced_profiler': do_advanced_profiler,
            'do_freeze_encoder': do_freeze_encoder,
            'callbacks': callbacks,
        }

        model, trainer = run_finetune_singleton(**run_finetune_singleton_inputs)

        if model.local_rank == 0 and train_set_frac > ONE_MINUS_EPSILON:
            os.mkdir(final_save_dir)
            save_ft_module(model, final_save_dir)
            end = time.time()
            elapsed = round((end - start) / 3600, 2)
            with open(final_save_dir / 'elapsed.txt', 'w') as f:
                f.write(str(elapsed) + ' hours elapsed.')

        # Run evals.
        for split in ('dev', 'test'):
            eval_dataset = dataset_cls(
                task              = task,
                split             = split,
                max_seq_len       = max_seq_len,
                batch_size        = eval_batch_size,
                hf_model_name     = hf_model_name,
            )
            results_file = sample_dir / f'{split}_results.txt'

            run_finetune_eval_inputs = {
                'sample_dir': sample_dir,
                'task': task,
                'test_dataset': eval_dataset,
                'head_cls': head_cls,
                'head_kwargs': head_kwargs,
                'encoder_hidden_size': encoder_hidden_size,
                'encoder_num_hidden_layers': encoder_num_hidden_layers,
                'encoder_num_attention_heads': encoder_num_attention_heads,
                'encoder_intermediate_size': encoder_intermediate_size,
                'train_set_frac': train_set_frac,
                'do_upsample': do_upsample,
                'do_half': do_half,
                'metric_name': metric_name,
                'hf_model_name': hf_model_name,
            }

            model, trainer, _ = run_finetune_eval(**run_finetune_eval_inputs)
            targets = model.test_results['targets']
            predictions = model.test_results['predictions']

            accuracy = accuracy_score(targets, predictions)
            f1 = f1_score(targets, predictions, average='macro', labels=list(eval_dataset.data.label_map.values()), zero_division=0)

            with open(sample_dir / f'{split}_results.txt', 'w') as f:
                f.write(f'Accuracy: {accuracy}\n')
                f.write(f'F1: {f1}\n')

    samples_rng = range(num_samples)
    samples_rng = tqdm(samples_rng)

    for sample_num in samples_rng:
        do_sample(sample_num)
