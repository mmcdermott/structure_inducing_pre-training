from ..datasets.graph_pretraining_bio_ft_dataset import GraphBioFinetuningDataset
from .run_graph_pt_finetune_base import run_graph_pt_finetune_base, copy_signature
from .run_graph_pt_bio_singleton_finetune import run_singleton_finetune

@copy_signature(run_graph_pt_finetune_base)
def run_graph_pt_finetune(*args, **kwargs):
    run_graph_pt_finetune_base(
        *args, **kwargs, data_cls=GraphBioFinetuningDataset, singleton_fn=run_singleton_finetune,
    )
