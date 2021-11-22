from .run_graph_pt_singleton_base import run_singleton_base
from ..graph_pt_bio_torch_modules import *

def run_singleton(*args, **kwargs):
    if kwargs.pop('do_multitask_PT'): ppt_cls = FakeGGMLClassificationHead
    else: ppt_cls = GCNNMaskingPTHead
    return run_singleton_base(
        *args,
        encoder_cls=GCNNEncoderModule,
        ppt_cls=ppt_cls,
        pce_cls=PointContextEncoder,
        **kwargs
    )
