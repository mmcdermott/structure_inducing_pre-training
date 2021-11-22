import os, time
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import Callback
from tape.metrics import spearmanr, accuracy

try: from tape.metrics import sequence_accuracy   # Present in earlier versions of TAPE.
except: sequence_accuracy = accuracy              # In recent versions, handled as one function.

from ..constants import *
from ..datasets.tape_dataset import *
from ..utils.train_utils import *
from .run_finetune_singleton import *

def run_tape_finetune(
    output_dir:                     str,
    task:                           str     = TAPE_RH,
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
    num_epochs:                     int     = 10,
    patience:                       int     = 10,
    num_samples:                    int     = 3,
    do_half:                        bool    = False,
    do_ddp:                         bool    = False,
    gpus:                           int     = 1,
    weights_summary:                str     = 'top',
    do_simple_profiler:             bool    = False,
    do_advanced_profiler:           bool    = False,
    do_freeze_encoder:              bool    = False,
    do_from_plus:                   bool    = False,
):
    output_dir = Path(output_dir)
    seed_everything(seed)

    dataset_map = {
        TAPE_RH: RemoteHomologyDataset,
        TAPE_FL: FluorescenceDataset,
        TAPE_ST: StabilityDataset,
        TAPE_SS: SecondaryStructureDataset,
    }
    head_cls_map = {
        TAPE_RH: SequenceClassificationFinetuningHead,   
        TAPE_FL: ValuePredictionFinetuningHead,
        TAPE_ST: ValuePredictionFinetuningHead,
        TAPE_SS: SequenceToSequenceClassificationFinetuningHead,
    }
    head_kwargs_map = {
        TAPE_RH: {
           'hidden_size': encoder_hidden_size,
           'num_labels': 1195,
        },
        TAPE_FL: {
           'hidden_size': encoder_hidden_size,
        },
        TAPE_ST: {
           'hidden_size': encoder_hidden_size,
        },
        TAPE_SS: {
           'hidden_size': encoder_hidden_size,
           'num_labels': 3,
           'ignore_index': -1,
        },
    }
    metric_name_map = {
        TAPE_RH: 'accuracy',
        TAPE_FL: 'spearmanr',
        TAPE_ST: 'spearmanr',
        TAPE_SS: 'accuracy'
    }

    dataset_cls, head_cls, head_kwargs, metric_name = dataset_map[task], head_cls_map[task], head_kwargs_map[task], metric_name_map[task]

    def do_sample(sample_num):
        train_dataset = dataset_cls(
            split             = 'train',
            max_seq_len       = max_seq_len,
            batch_size        = train_batch_size,
            do_from_plus      = do_from_plus,
        )
        val_dataset = dataset_cls(
            split             = 'valid',
            max_seq_len       = max_seq_len,
            batch_size        = eval_batch_size,
            do_from_plus      = do_from_plus,
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
            'encoder_hidden_size': encoder_hidden_size,
            'encoder_num_hidden_layers': encoder_num_hidden_layers,
            'encoder_num_attention_heads': encoder_num_attention_heads,
            'encoder_intermediate_size': encoder_intermediate_size,
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
            'do_freeze_encoder': do_freeze_encoder,
            'callbacks': callbacks,
            'do_from_plus': do_from_plus,
        }

        model, trainer = run_finetune_singleton(**run_finetune_singleton_inputs)

        if model.local_rank == 0:
            os.mkdir(final_save_dir)
            # save_ft_module(model, final_save_dir)
            end = time.time()
            elapsed = round((end - start) / 3600, 2)
            with open(final_save_dir / 'elapsed.txt', 'w') as f:
                f.write(str(elapsed) + ' hours elapsed.')

    samples_rng = range(num_samples)
    samples_rng = tqdm(samples_rng)

    for sample_num in samples_rng:
        do_sample(sample_num)


def run_tape_eval(
    output_dir:                     str,
    task:                           str     = TAPE_RH,
    eval_split:                     str     = 'test',
    seed:                           int     = 42,
    encoder_hidden_size:            int     = 10,
    encoder_num_hidden_layers:      int     = 2,
    encoder_num_attention_heads:    int     = 2,
    encoder_intermediate_size:      int     = 8,
    max_seq_len:                    int     = 4096,
    train_set_frac:                 float   = 1.0,
    do_upsample:                    bool    = False,
    eval_batch_size:                int     = 32,
    num_samples:                    int     = 3,
    do_half:                        bool    = False,
    do_from_plus:                   bool    = False,
):
    output_dir = Path(output_dir)
    seed_everything(seed)

    dataset_map = {
        TAPE_RH: RemoteHomologyDataset,
        TAPE_FL: FluorescenceDataset,
        TAPE_ST: StabilityDataset,
        TAPE_SS: SecondaryStructureDataset,
    }
    head_cls_map = {
        TAPE_RH: SequenceClassificationFinetuningHead,   
        TAPE_FL: ValuePredictionFinetuningHead,
        TAPE_ST: ValuePredictionFinetuningHead,
        TAPE_SS: SequenceToSequenceClassificationFinetuningHead,
    }
    head_kwargs_map = {
        TAPE_RH: {
           'hidden_size': encoder_hidden_size,
           'num_labels': 1195,
        },
        TAPE_FL: {
           'hidden_size': encoder_hidden_size,
        },
        TAPE_ST: {
           'hidden_size': encoder_hidden_size,
        },
        TAPE_SS: {
           'hidden_size': encoder_hidden_size,
           'num_labels': 3,
           'ignore_index': -1,
        },
    }
    metric_name_map = {
        TAPE_RH: 'accuracy',
        TAPE_FL: 'spearmanr',
        TAPE_ST: 'spearmanr',
        TAPE_SS: 'sequence_accuracy'
    }

    dataset_cls, head_cls, head_kwargs, metric_name = dataset_map[task], head_cls_map[task], head_kwargs_map[task], metric_name_map[task]
    split = 'valid' if eval_split == 'valid' else dataset_cls.test_split_name
    test_dataset = dataset_cls(
        split             = split,
        max_seq_len       = max_seq_len,
        batch_size        = eval_batch_size,
        do_from_plus      = do_from_plus,
    )

    def do_sample(sample_num):
        sample_dir = output_dir / str(sample_num)
        final_save_dir = sample_dir / 'final_save'
        results_file = sample_dir / f'{split}_results.txt'
        if not final_save_dir.exists():
            return                      # Not finished yet.
        if results_file.exists():       # Already evaluated.
            return

        run_finetune_eval_inputs = {
            'sample_dir': sample_dir,
            'task': task,
            'test_dataset': test_dataset,
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
            'do_from_plus': do_from_plus,
        }

        model, trainer, outputs = run_finetune_eval(**run_finetune_eval_inputs)
        targets = model.test_results['targets']
        predictions = model.test_results['predictions']
        
        # if metric_name == 'accuracy': metric_fn = accuracy
        # elif metric_name == 'sequence_accuracy': metric_fn = sequence_accuracy
        # elif metric_name == 'spearmanr': metric_fn = spearmanr
        # else: raise ValueError('Metric not recognized.')
        
        # metric = metric_fn(targets, predictions)
        # best_epoch = outputs[0]
        enpickle(targets, sample_dir / f'{split}_targets.pkl')
        enpickle(predictions, sample_dir / f'{split}_preds.pkl')

        # with open(results_file, 'w') as f:
        #     f.write(f'{metric_name} = {metric}')
        #     f.write(f'\nBest epoch: {best_epoch}')

    samples_rng = range(num_samples)
    samples_rng = tqdm(samples_rng)

    for sample_num in samples_rng:
        do_sample(sample_num)

    run_str = f'python calc_tape_metrics.py --output_dir={output_dir} --split={split} --metric_name={metric_name}'
    os.system(run_str)
