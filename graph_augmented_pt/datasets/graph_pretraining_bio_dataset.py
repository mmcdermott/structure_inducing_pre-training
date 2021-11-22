from .graph_pretraining_base_dataset import *

import sys
sys.path.append(str(COMPANION_CODE_DIR / 'pretrain-gnns/bio'))

from loader import BioDataset
from batch import BatchSubstructContext, BatchMasking, BatchFinetune
from util import ExtractSubstructureContextPair, MaskEdge

class AugmentedBioDataset(BioDataset, AugmentedDatasetMixin):
    _Y_VAR = 'go_target_pretrain'

    def __init__(
        self, dataset_dir,
        **kwargs
    ):
        super().__init__(dataset_dir, data_type='supervised')
        self.raw_task_indices = np.arange(len(super().get(0)[self._Y_VAR]))
        self._init_mixin(dataset_dir, **kwargs)

    @property
    def pt_batch_cls(self):
        if self.do_context_pred: return BatchSubstructContext
        elif self.do_masking: return BatchMasking
        else: return BatchFinetune

    @classmethod
    def _labelset_key(self, y_np_arr): return frozenset(np.where(y_np_arr == 1)[0])

    @staticmethod
    def _get_dist(ls_true, ls_true_2, n_tasks): return len(ls_true.symmetric_difference(ls_true_2))

    def _set_graph_PT_transform(self):
        if not self.do_graph_PT: self._graph_PT_transform = torch.nn.Identity()
        elif self.do_context_pred:
            self._graph_PT_transform = ExtractSubstructureContextPair(
                self.num_layer, self.context_l1, self.context_l2
            )
        elif self.do_masking: self._graph_PT_transform = MaskEdge(mask_rate = self.mask_rate)

class GraphBioPretrainingDataset(GraphDerivedPretrainingDataset):
    _DATA_SUBPATH = 'bio/supervised'
    _DATASET_CLS = AugmentedBioDataset
