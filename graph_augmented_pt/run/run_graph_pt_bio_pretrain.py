from ..datasets.graph_pretraining_bio_dataset import GraphBioPretrainingDataset
from .run_graph_pt_pretrain_base import run_graph_pt_pretrain_base, copy_signature
from .run_graph_pt_bio_singleton import run_singleton

@copy_signature(run_graph_pt_pretrain_base)
def run_graph_pt_pretrain(*args, **kwargs):
    run_graph_pt_pretrain_base(
        *args, **kwargs, data_cls=GraphBioPretrainingDataset, singleton_fn=run_singleton,
    )
