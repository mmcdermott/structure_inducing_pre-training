from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from typing import List, Optional, Any
from pathlib import Path

from ..constants import *
from ..lightning_modules import *
from ..metric_learning_heads import *
from ..utils.train_utils import *
from ..utils.utils import *

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from tqdm.auto import tqdm

import torch, traceback
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

# Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/765#issuecomment-593703168
class GlobalProgressBar(Callback):
    """Global progress bar.
    TODO: add progress bar for training, validation and testing loop.
    """

    def __init__(self, global_progress: bool = True, leave_global_progress: bool = True):
        super().__init__()

        self.global_progress = global_progress
        self.global_desc = "Epoch: {epoch}/{max_epoch}"
        self.leave_global_progress = leave_global_progress
        self.global_pb = None

    def on_train_start(self, trainer, pl_module):
        desc = self.global_desc.format(epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs)

        self.global_pb = tqdm(
            desc=desc,
            total=trainer.max_epochs,
            initial=trainer.current_epoch,
            leave=self.leave_global_progress,
            disable=not self.global_progress,
        )

    def on_train_end(self, trainer, pl_module):
        self.global_pb.close()
        self.global_pb = None

    def on_epoch_end(self, trainer, pl_module):

        # Set description
        desc = self.global_desc.format(epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs)
        self.global_pb.set_description(desc)

        # Set logs and metrics
        try:
            logs = pl_module.logs
            for k, v in logs.items():
                if isinstance(v, torch.Tensor):
                    logs[k] = v.squeeze().item()
            self.global_pb.set_postfix(logs)
        except: pass

        # Update progress
        self.global_pb.update(1)

def run_singleton_finetune_base(
    encoder_cls,
    sample_dir:                     str,
    train_dataset:                  LightningDataModule,
    val_dataset:                    LightningDataModule,
    head_cls:                       nn.Module,
    head_kwargs:                    dict,
    do_head_only:                   bool = False,

    encoder_emb_dim:                int = 10,
    encoder_hidden_size:            int = 10,
    encoder_num_layer:              int = 2,
    encoder_JK:                     str = 'last',
    encoder_dropout_ratio:          float = 0.1,
    encoder_graph_pooling:          str = 'mean',

    grad_accum:                     int = 1,
    lr:                             float = 3e-4,
    do_half:                        bool = False,
    do_ddp:                         bool = False,
    gpus:                           int = None,
    max_epochs:                     int = 1,
    weights_summary:                str = 'top',
    do_simple_profiler:             bool = False,
    do_advanced_profiler:           bool = False,
    callbacks:                      List[Callback] = [],
    do_checkpointing:               bool = True,
    progress_bar_refresh_rate:      int  = 1,
    do_progress_bar:                bool = True,
    just_get_model:                 bool = False,
    do_anomaly_detection:           bool = False, # Useful for debugging, but slow.
    manual_checkpoint_dir:          str  = None,
    train_set_frac:                 float = 1.0,
    do_upsample:                    bool = False,
    warmup_frac:                    float = 0.2,
    patience:                       int = 10,
    metric_name:                    str = 'accuracy',

    do_freeze_encoder:              bool = False,
    do_schedule:                    bool = False,
    do_use_auroc:                   bool = False,
    do_run_without_pretraining:     bool = False,
):
    if gpus is None:
        gpus = 1 if torch.cuda.is_available() else None

    config = {
        'emb_dim': encoder_emb_dim,
        'hidden_dim': encoder_hidden_size,
        'JK': encoder_JK,
        'dropout_ratio': encoder_dropout_ratio,
        'graph_pooling': encoder_graph_pooling,
        'num_layer': encoder_num_layer,
        'do_masking': False,
        'do_context_pred': False,
        'do_mask_edge': 0,
    }

    encoder_module = encoder_cls(config)
    if not do_run_without_pretraining:
        print("LOADING PT ENCODER")
        encoder_path = find_in_parent_dirs(
            current_dir = sample_dir,
            target = 'encoder.pt'
        )
        if torch.cuda.is_available():
            encoder_module.load_state_dict(torch.load(encoder_path))
        else:
            encoder_module.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
    else:
        print("NOT LOADING PT ENCODER")

    finetuning_head = head_cls(config, **head_kwargs)

    train_loader = train_dataset.train_dataloader(train_set_frac)
    val_loader = val_dataset.val_dataloader()

    num_training_steps = len(train_loader) * max_epochs
    model = FinetuningModule(
        encoder_module,
        finetuning_head,
        lr = lr,
        do_head_only = False,
        do_freeze_encoder = do_freeze_encoder,
        do_schedule = do_schedule,
        num_training_steps = num_training_steps,
        warmup_frac = warmup_frac,
        do_from_plus = False,
        do_use_auroc = do_use_auroc,
    )

    if do_progress_bar:
        callbacks.extend([
            GlobalProgressBar(leave_global_progress=False),
            #LocalProgressBar(leave_local_progress=False),
        ])

    trainer_kwargs = {}

    profile_filepath = sample_dir / 'profile.txt'

    if do_simple_profiler: trainer_kwargs['profiler'] = SimpleProfiler(output_filename = profile_filepath)
    if do_advanced_profiler: trainer_kwargs['profiler'] = AdvancedProfiler(output_filename = profile_filepath)
    if do_ddp: trainer_kwargs['accelerator'] = 'ddp'
    if not do_checkpointing: trainer_kwargs['checkpoint_callback'] = False
    trainer_kwargs['progress_bar_refresh_rate'] = progress_bar_refresh_rate if do_progress_bar else 0
    if do_half: trainer_kwargs['precision'] = 16
    if patience >= 0:
        callbacks.append(
            EarlyStopping(
                monitor=f'Val/loss',
                patience=patience,
                mode='min',
            )
        )
    if grad_accum > 1: trainer_kwargs['accumulate_grad_batches'] = grad_accum

    if do_upsample:
        max_epochs = int(max_epochs / train_set_frac)
        trainer_kwargs['check_val_every_n_epoch'] = int(1 / train_set_frac)

    if do_schedule:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

    trainer = Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        deterministic=True,
        default_root_dir=sample_dir,
        weights_summary=weights_summary,
        callbacks=callbacks,
        **trainer_kwargs
    )

    if len(train_loader) > 0: trainer.fit(model, train_loader, val_loader)

    # if not do_checkpointing: trainer.save_checkpoint(sample_dir / 'final_model.ckpt')

    return model, trainer

def run_finetune_eval_base(
    encoder_cls,
    sample_dir:                     str,
    test_dataset:                   LightningDataModule,
    head_cls:                       nn.Module,
    head_kwargs:                    dict,

    encoder_emb_dim:                int = 10,
    encoder_hidden_size:            int = 10,
    encoder_num_layer:              int = 2,
    encoder_JK:                     str = 'last',
    encoder_dropout_ratio:          float = 0.1,
    encoder_graph_pooling:          str = 'mean',

    train_set_frac:                 float = 1.0,
    do_upsample:                    bool = False,
    do_half:                        bool = False,
    metric_name:                    str = 'accuracy',
):
    gpus = 1 if torch.cuda.is_available() else None

    config = {
        'emb_dim': encoder_emb_dim,
        'hidden_dim': encoder_hidden_size,
        'JK': encoder_JK,
        'dropout_ratio': encoder_dropout_ratio,
        'graph_pooling': encoder_graph_pooling,
        'num_layer': encoder_num_layer,
        'do_masking': False,
        'do_context_pred': False,
        'do_mask_edge': 0,
    }

    encoder_module = encoder_cls(config)
    outputs = ()

    best_epoch = get_best_epoch(sample_dir, remove_others=False)

    encoder_path = sample_dir / 'epochs'  / str(best_epoch) / 'encoder.pt'
    finetuning_head_path = sample_dir / 'epochs' / str(best_epoch) / 'finetuning_head.pt'
    outputs += (best_epoch,)

    if torch.cuda.is_available():
        encoder_module.load_state_dict(torch.load(encoder_path))
    else:
        encoder_module.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))

    finetuning_head = head_cls(config, **head_kwargs)
    if torch.cuda.is_available():
        finetuning_head.load_state_dict(torch.load(finetuning_head_path))
    else:
        finetuning_head.load_state_dict(torch.load(finetuning_head_path, map_location=torch.device('cpu')))

    model = FinetuningModule(
        encoder_module,
        finetuning_head,
        metric_name = metric_name,
        do_from_plus = False,
    )

    trainer_kwargs = {}
    if do_half: trainer_kwargs['precision'] = 16

    trainer = Trainer(
        gpus=gpus,
        deterministic=True,
        default_root_dir=sample_dir,
        **trainer_kwargs
    )

    test_loader = test_dataset.test_dataloader()
    trainer.test(
        model,
        test_dataloaders = test_loader
    )

    return model, trainer, outputs
