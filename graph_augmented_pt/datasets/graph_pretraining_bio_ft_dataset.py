import torch, numpy as np

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from .finetuning_dataset import *
from ..utils.utils import *
from ..constants import *

from .graph_pretraining_bio_dataset import GraphBioPretrainingDataset

import sys
sys.path.append(str(COMPANION_CODE_DIR / 'pretrain-gnns/bio'))

from batch import BatchFinetune

class GraphBioFinetuningDataset(FinetuningDataset):
    test_split_name = 'test'

    def __init__(
        self,
        split:      str  = 'train',
        data_dir:   Path = RAW_DATASETS_DIR,
        seed:       int  = 1,
        **dataset_kwargs
    ):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test']")

        self.seed         = seed
        self._data_kwargs = {
            'seed':               self.seed,
            'do_compute_dists':   False,
            'held_out_data_frac': (0.1, 0.1),
            'held_out_task_frac': (0.1, 0.1),
            'do_masking':         False,
            'do_context_pred':    False,
        }

        splits = {'train': (0, 1), 'valid': (1, 1), 'test': (2, 1)}

        self.data = GraphBioPretrainingDataset(**self._data_kwargs)
        self.data.dataset.set_split(*splits[split])

        super().__init__(**dataset_kwargs)

    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, index: int): return self.data[index]['point_features']
    def collate_fn(self, *args, **kwargs): return {
        'input_ids': BatchFinetune.from_data_list(*args, **kwargs),
        'input_mask': None
    }
