from transformers import BertConfig, BertModel, AutoModel
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pathlib import Path

from ..constants import *
from ..lightning_modules import *
from ..torch_modules import *
from ..utils.eval_utils import *
from ..utils.train_utils import *
from ..utils.utils import *
from ..utils.tensorboard_utils import *

import sys
sys.path.append("../PLUS")
from plus.config import ModelConfig
from plus.model.plus_tfm import PLUS_TFM

from tqdm.auto import tqdm
from typing import List, Dict

import torch
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

def run_finetune_singleton(
    sample_dir:                     str,
    task:                           str,
    train_dataset:                  LightningDataModule,
    val_dataset:                    LightningDataModule,
    head_cls:                       nn.Module,
    head_kwargs:                    Dict,
    do_head_only:                   bool = False,
    encoder_hidden_size:            int = 10,
    encoder_num_hidden_layers:      int = 2,
    encoder_num_attention_heads:    int = 2,
    encoder_intermediate_size:      int = 8,
    grad_accum:                     int = 1,
    train_set_frac:                 float = 1.0,
    do_upsample:                    bool = False,
    lr:                             float = 1e-4,
    warmup_frac:                    float = 0.2,
    do_half:                        bool = False,
    do_ddp:                         bool = False,
    hf_model_name:                  str = SCIBERT_SCIVOCAB_UNCASED,
    gpus:                           int = 1,
    max_epochs:                     int = 10,
    patience:                       int = 10,
    metric_name:                    str = 'accuracy',
    weights_summary:                str = 'top',
    do_simple_profiler:             bool = False,
    do_advanced_profiler:           bool = False,
    do_freeze_encoder:              bool = False,
    callbacks:                      List[Callback] = [],
    do_checkpointing:               bool = False,
    progress_bar_refresh_rate:      int  = 1,
    do_progress_bar:                bool = True,
    do_from_plus:                   bool = True,
):
    if gpus is None:
        gpus = 1 if torch.cuda.is_available() else None

    tokenizer = train_dataset.tokenizer if train_dataset.tokenizer else train_dataset

    if do_head_only:
        encoder_module = None
    
    elif do_from_plus:
        if 'num_labels' in head_kwargs: num_classes = head_kwargs['num_labels']
        else: num_classes = 1   # Regression task

        config = ModelConfig(
            file="../PLUS/config/model/plus-tfm.json",
            model_type="TFM",
            input_dim = len(tokenizer), num_classes=num_classes
        )

        # Needed for positional encoding to work for TAPE_SS, which shouldn't have truncated sequences.
        if task == TAPE_SS: config.max_len = None

        flag_map = {
            TAPE_FL: {'reg_protein': True},
            TAPE_RH: {'cls_protein': True},
            TAPE_SS: {'cls_amino': True},
            TAPE_ST: {'reg_protein': True}
        }
        flags = flag_map[task]

        encoder_module = PLUSEncoderModule(
            config = config,
            sequence_model_cnstr = PLUS_TFM,
            pooling_model = cls_pooler,
            **flags,
        )

        encoder_path = find_in_parent_dirs(
            current_dir = sample_dir,
            target = 'encoder.pt'
        )
        encoder_module.load_state_dict(
            filter_keys_for_plus(torch.load(encoder_path), encoder_module.state_dict(), task)
        )

    else:
        if hasattr(tokenizer, 'vocab_size'): vocab_size = tokenizer.vocab_size
        else: vocab_size = len(tokenizer.vocab)

        config = BertConfig(
            vocab_size = vocab_size,
            hidden_size = encoder_hidden_size,
            num_hidden_layers = encoder_num_hidden_layers,
            num_attention_heads = encoder_num_attention_heads,
            intermediate_size = encoder_intermediate_size,
            max_position_embeddings=4096,
            output_hidden_states = True,
            return_dict = True,
            hidden_dropout_prob=0
        )

        encoder_module = EncoderModule(
            config,
            BertModel,
            cls_pooler
        )

        if task in TAPE_TASKS:
            encoder_module.sequence_model = ProteinBertModel.from_pretrained('bert-base')
        elif task in SCIBERT_TASKS:
            encoder_module.sequence_model = AutoModel.from_pretrained(hf_model_name)
        else:
            raise NotImplementedError
        
        encoder_path = find_in_parent_dirs(
            current_dir = sample_dir,
            target = 'encoder.pt'
        )
        encoder_module.load_state_dict(torch.load(encoder_path))

    train_loader = train_dataset.train_dataloader(train_set_frac)
    val_loader = val_dataset.val_dataloader()

    finetuning_head = head_cls(**head_kwargs)

    do_schedule = task in SCIBERT_TASKS
    num_training_steps = len(train_loader) * max_epochs
    model = FinetuningModule(
        encoder_module,
        finetuning_head,
        lr = lr,
        do_head_only = do_head_only,
        do_freeze_encoder = do_freeze_encoder,
        do_schedule = do_schedule,
        num_training_steps = num_training_steps,
        warmup_frac = warmup_frac,
        do_from_plus = do_from_plus,
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
    if patience >= 0 and (task in (TAPE_TASKS + RANKING_TASKS) or train_set_frac < ONE_MINUS_EPSILON):    # Only do this for TAPE tasks and Scibert with downsampling.
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
    

def run_finetune_eval(
    sample_dir:                     str,
    task:                           str,
    test_dataset:                   LightningDataModule,
    head_cls:                       nn.Module,
    head_kwargs:                    Dict,
    encoder_hidden_size:            int = 10,
    encoder_num_hidden_layers:      int = 2,
    encoder_num_attention_heads:    int = 2,
    encoder_intermediate_size:      int = 8,
    train_set_frac:                 float = 1.0,
    do_upsample:                    bool = False,
    do_half:                        bool = False,
    metric_name:                    str = 'accuracy',
    hf_model_name:                  str = SCIBERT_SCIVOCAB_UNCASED,
    do_from_plus:                   bool = False,
):
    gpus = 1 if torch.cuda.is_available() else None

    tokenizer = test_dataset.tokenizer if test_dataset.tokenizer else test_dataset

    if do_from_plus:
        if 'num_labels' in head_kwargs: num_classes = head_kwargs['num_labels']
        else: num_classes = 1   # Regression task

        config = ModelConfig(
            file="../PLUS/config/model/plus-tfm.json",
            model_type="TFM",
            input_dim = len(tokenizer), num_classes=num_classes
        )

        if task == TAPE_SS: config.max_len = None

        flag_map = {
            TAPE_FL: {'reg_protein': True},
            TAPE_RH: {'cls_protein': True},
            TAPE_SS: {'cls_amino': True},
            TAPE_ST: {'reg_protein': True}
        }
        flags = flag_map[task]

        encoder_module = PLUSEncoderModule(
            config = config,
            sequence_model_cnstr = PLUS_TFM,
            pooling_model = cls_pooler,
            **flags,
        )

    else:
        if hasattr(tokenizer, 'vocab_size'): vocab_size = tokenizer.vocab_size
        else: vocab_size = len(tokenizer.vocab)

        config = BertConfig(
            vocab_size = vocab_size,
            hidden_size = encoder_hidden_size,
            num_hidden_layers = encoder_num_hidden_layers,
            num_attention_heads = encoder_num_attention_heads,
            intermediate_size = encoder_intermediate_size,
            max_position_embeddings=4096,
            output_hidden_states = True,
            return_dict = True,
            hidden_dropout_prob=0
        )

        encoder_module = EncoderModule(
            config,
            BertModel,
            cls_pooler
        )

    outputs = ()

    if (task in TAPE_TASKS or (task in SCIBERT_TASKS and train_set_frac < ONE_MINUS_EPSILON)):
        best_epoch = get_best_epoch(sample_dir)
        if do_upsample: best_epoch *= int(1 / train_set_frac)
        print(f'Best epoch: {best_epoch}')

    if task in TAPE_TASKS:
        if do_from_plus:
            pass
        else:
            encoder_module.sequence_model = ProteinBertModel.from_pretrained('bert-base')

        encoder_path = sample_dir / 'epochs'  / str(best_epoch) / 'encoder.pt'
        finetuning_head_path = sample_dir / 'epochs' / str(best_epoch) / 'finetuning_head.pt'
        outputs += (best_epoch,)

    elif task in SCIBERT_TASKS:
        encoder_module.sequence_model = AutoModel.from_pretrained(hf_model_name)
        if train_set_frac < ONE_MINUS_EPSILON:
            encoder_path = sample_dir / 'epochs'  / str(best_epoch) / 'encoder.pt'
            finetuning_head_path = sample_dir / 'epochs' / str(best_epoch) / 'finetuning_head.pt'
        else:
            encoder_path = sample_dir / 'final_save' / 'encoder.pt'
            finetuning_head_path = sample_dir / 'final_save' / 'finetuning_head.pt'

    else:
        raise NotImplementedError

    encoder_module.load_state_dict(torch.load(encoder_path))

    finetuning_head = head_cls(**head_kwargs)
    finetuning_head.load_state_dict(torch.load(finetuning_head_path))

    model = FinetuningModule(
        encoder_module,
        finetuning_head,
        metric_name = metric_name,
        do_from_plus = do_from_plus,
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
