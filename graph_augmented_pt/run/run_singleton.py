from transformers import BertConfig, BertModel, AutoModel
from transformers.modeling_bert import BertOnlyMLMHead
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import Callback
from typing import List, Optional, Any
from pathlib import Path

from ..constants import *
from ..lightning_modules import *
from ..torch_modules import *
from ..metric_learning_heads import *
from ..utils.train_utils import *

import sys
sys.path.append("../PLUS")
from plus.config import ModelConfig
from plus.model.plus_tfm import PLUS_TFM

from tqdm.auto import tqdm

import torch, traceback
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler

# TODO: Does this really belong here???
BASIC_GML_HEADS = {
    COSINE_DISTANCE: BilinearCosineLinker,
    EUCLIDEAN_DISTANCE: MarginEuclideanLinker,
    MULTISIMILARITY: lambda *args, **kwargs: RDMLHead(
        *args, MultisimilarityCriterion(
            kwargs['pos_weight'], kwargs['neg_weight'], kwargs['margin'], kwargs['thresh']
        )
    ),
    DISTANCE_WEIGHTED_TRIPLET: lambda *args, **kwargs: RDMLHead(
        *args, TripletCriterion(
            kwargs['margin'], DistanceWeightedBatchMiner(kwargs['lower_cutoff'], kwargs['upper_cutoff'])
        )
    ),
}

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

def run_singleton(
    sample_dir:                     str,
    data_module:                    LightningDataModule,
    gml_weight:                     float = 1.0,
    point_weight:                   float = 1.0,
    gml_encoder:                    str = POINT_ENCODER,
    gml_head:                       str = COSINE_DISTANCE,
    do_gru:                         bool = False,
    gml_head_kwargs:                dict = dict(negative_margin=1e1),
    encoder_hidden_size:            int = 10,
    encoder_num_hidden_layers:      int = 2,
    encoder_num_attention_heads:    int = 2,
    encoder_intermediate_size:      int = 8,
    context_encoder_hidden_size:    int = 10,
    context_encoder_hidden_layers:  int = 2,
    grad_accum:                     int = 1,
    lr:                             float = 3e-4,
    do_log_on_epoch:                bool = True,
    do_half:                        bool = False,
    do_ddp:                         bool = False,
    gpus:                           int = None,
    max_epochs:                     int = 1,
    weights_summary:                str = 'top',
    do_simple_profiler:             bool = False,
    do_advanced_profiler:           bool = False,
    callbacks:                      List[Callback] = [],
    do_from_tape:                   bool = False,
    do_from_plus:                   bool = False,
    do_from_scibert:                bool = False,
    hf_model_name:                  str = SCIBERT_SCIVOCAB_UNCASED,
    do_checkpointing:               bool = True,
    progress_bar_refresh_rate:      int  = 1,
    do_progress_bar:                bool = True,
    just_get_model:                 bool = False,
    do_anomaly_detection:           bool = False, # Useful for debugging, but slow.
    do_normalize_embeds:            bool = False,
    manual_checkpoint_dir:          str  = None,
    gru_pooling_strategy:           str  = None,
    gru_do_bidirectional:           bool = None,
    do_bow_ae:                      bool = False,
):
    if gpus is None:
        gpus = 1 if torch.cuda.is_available() else None

    tokenizer = data_module.tokenizer if data_module.tokenizer else data_module

    if do_from_plus:
        config = ModelConfig(
            file="../PLUS/config/model/plus-tfm.json",
            model_type="TFM",
            input_dim = len(tokenizer), num_classes=2
        )

        encoder_module = PLUSEncoderModule(
            config = config,
            sequence_model_cnstr = PLUS_TFM,
            pooling_model = cls_pooler,
            lm = True,
        )
        PLUS_DIR = Path(RAW_MODELS_DIR) / 'PLUS'
        PLUS_TFM_FILEPATH = PLUS_DIR / 'PLUS-TFM.pt'
        encoder_module.sequence_model.load_weights(PLUS_TFM_FILEPATH)

        point_pretraining_head = PointContextEncoder()  # Just need this to be an nn.Module(), it won't be used.

    else:
        if hasattr(tokenizer, 'vocab_size'): vocab_size = tokenizer.vocab_size
        else: vocab_size = len(tokenizer.vocab)

        if do_gru and gru_do_bidirectional: encoder_hidden_size *= 2

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


        point_pretraining_head = MLMPretrainingHead(
            config,
            BertOnlyMLMHead,
            mask_id = data_module.mask_id
        )

        if do_gru:
            gru_config = GRUConfig(
                len(tokenizer.vocab), encoder_intermediate_size, encoder_hidden_size,
                do_max_pool      = gru_pooling_strategy == 'max',
                do_avg_pool      = gru_pooling_strategy == 'avg',
                do_attn_pool     = gru_pooling_strategy == 'attn',
                do_last_pool     = gru_pooling_strategy == 'last',
                do_bidirectional = gru_do_bidirectional,
            )
            encoder_module = EncoderModule(
                gru_config,
                GRUEncoder,
                GRUPooler(gru_config),
            )
        elif do_bow_ae:
            bow_ae_config = BOWAEConfig(
                len(tokenizer.vocab), encoder_intermediate_size, encoder_hidden_size,
            )
            encoder_module = EncoderModule(
                bow_ae_config,
                BOWEncoder,
                bow_pooler,
            )
            point_pretraining_head = BOWAEPretrainingHead(bow_ae_config, mask_id = data_module.mask_id)
        else:
            encoder_module = EncoderModule(
                config,
                BertModel,
                cls_pooler
            )

        if do_from_tape:
            encoder_module.sequence_model = ProteinBertModel.from_pretrained('bert-base')
        if do_from_scibert:
            encoder_module.sequence_model = AutoModel.from_pretrained(hf_model_name)

    ctx_encoder_kwargs, ctx_encoder_out_size = {}, encoder_hidden_size
    if gml_encoder == POINT_ENCODER:
        ctx_encoder_cnstr = PointContextEncoder
    else:
        raise NotImplementedError

    context_encoder_module = ctx_encoder_cnstr(**ctx_encoder_kwargs)

    gml_cnstr = BASIC_GML_HEADS[gml_head]
    gml_head = gml_cnstr(
        encoder_hidden_size,
        ctx_encoder_out_size,
        **gml_head_kwargs,
    )

    hparams = {
        'lr': lr, 'gml_weight': gml_weight, 'point_weight': point_weight, 'do_log_on_epoch': do_log_on_epoch,
    }

    final_save_dir = sample_dir / 'final_save'
    batches_dir = sample_dir / 'batches'

    if manual_checkpoint_dir is not None: latest_checkpoint_dir = manual_checkpoint_dir
    elif final_save_dir.exists(): latest_checkpoint_dir = final_save_dir
    elif batches_dir.exists():
        batches = [int(b) for b in os.listdir(batches_dir)]
        max_batch = max(batches)
        # TODO: this is bad form

        for c in callbacks:
            if isinstance(c, SaveCallback): c.offset = max_batch + 1

        print(torch.cuda.is_available())

        latest_checkpoint_dir = batches_dir / f"{max_batch}"
    else: latest_checkpoint_dir, final_checkpoint_exists = None, False

    model = None
    if latest_checkpoint_dir and latest_checkpoint_dir.exists():
        print(f"Attempting to reload model from {latest_checkpoint_dir}")

        # TODO: Check that hparams agree with stored args.

        try:
            model = PretrainingModule.load_from(
                latest_checkpoint_dir,
                #directory=latest_checkpoint_dir,
                encoder_module, point_pretraining_head, context_encoder_module, gml_head,
                **hparams,
            )
            print(type(model), type(model.encoder_module), type(model.point_pretraining_head),
                type(model.context_encoder_module), type(model.gml_head))
        except Exception as e:
            print("Failed to reload!")
            print("Error:", e)
            print("Traceback:")
            traceback.print_exc()
            raise
    else:
        model = PretrainingModule(
            encoder_module,
            point_pretraining_head,
            context_encoder_module,
            gml_head,
            **hparams,
        )

    assert model is not None

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_of_model = num_parameters * 16 if do_half else num_parameters * 32

    if just_get_model: return model

    if do_progress_bar:
        callbacks.extend([
            GlobalProgressBar(leave_global_progress=False),
        ])

    trainer_kwargs = {}

    profile_filepath = sample_dir / 'profile.txt'

    if do_simple_profiler: trainer_kwargs['profiler'] = SimpleProfiler(output_filename = profile_filepath)
    if do_advanced_profiler: trainer_kwargs['profiler'] = AdvancedProfiler(output_filename = profile_filepath)
    if do_ddp: trainer_kwargs['accelerator'] = 'ddp'
    if not do_checkpointing: trainer_kwargs['checkpoint_callback'] = False
    trainer_kwargs['progress_bar_refresh_rate'] = progress_bar_refresh_rate if do_progress_bar else 0
    if do_half: trainer_kwargs['precision'] = 16
    if grad_accum > 1: trainer_kwargs['accumulate_grad_batches'] = grad_accum

    trainer = Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        deterministic=True,
        default_root_dir=sample_dir,
        weights_summary=weights_summary,
        callbacks=callbacks,
        **trainer_kwargs
    )

    train_loader = data_module.train_dataloader()

    if len(train_loader) > 0:
        if do_anomaly_detection:
            with torch.autograd.set_detect_anomaly(True): trainer.fit(model, train_loader)
        else: trainer.fit(model, train_loader)

    trainer.save_checkpoint(sample_dir / 'final_model.ckpt')

    return model, trainer
