from .run_graph_pt_singleton_finetune_base import run_singleton_finetune_base, run_finetune_eval_base
from ..graph_pt_bio_torch_modules import *

def run_singleton_finetune(*args, **kwargs):
    return run_singleton_finetune_base(
        *args,
        encoder_cls=GCNNEncoderModule,
        head_cls=FTClassificationHead,
        do_use_auroc=True,
        **kwargs
    )

def run_finetune_eval(*args, **kwargs):
    return run_finetune_eval_base(
        *args,
        encoder_cls=GCNNEncoderModule,
        head_cls=FTClassificationHead,
        do_use_auroc=True,
        **kwargs
    )
