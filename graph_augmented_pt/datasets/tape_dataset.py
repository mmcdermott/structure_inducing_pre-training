import torch, numpy as np

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from torch.utils.data import DataLoader, Dataset
from tape.tokenizers import TAPETokenizer
from tape.datasets import dataset_factory, pad_sequences

from .finetuning_dataset import *

import sys
sys.path.append("../PLUS")

from plus.data.alphabets import Protein
from plus.preprocess import preprocess_seq_for_tfm


class RemoteHomologyDataset(FinetuningDataset):
    test_split_name       = 'test_fold_holdout'

    def __init__(
        self,
        split,
        data_dir          = RAW_DATASETS_DIR,
        in_memory         = True,
        max_seq_len       = None,
        do_from_plus      = False,
        **dataset_kwargs
    ):

        if split not in (
            'train', 'valid', 
            'test_fold_holdout', 'test_family_holdout', 'test_superfamily_holdout'
        ):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'test_fold_holdout', "
                             f"'test_family_holdout', 'test_superfamily_holdout']")

        if do_from_plus: tokenizer = Protein()
        else: tokenizer = TAPETokenizer(vocab='iupac')
        self.tokenizer = tokenizer

        data_dir  = Path(data_dir)
        data_file = f'TAPE/remote_homology/remote_homology_{split}.lmdb'
        self.data = dataset_factory(data_dir / data_file, in_memory)
        self.max_seq_len = max_seq_len
        self.do_from_plus = do_from_plus

        super().__init__(
            **dataset_kwargs
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        if self.do_from_plus:
            s_bytes = str.encode(item['primary'])
            input_ids = self.tokenizer.encode(s_bytes)
            return input_ids, item['fold_label']

        else:
            token_ids = self.tokenizer.encode(item['primary'])
            input_mask = np.ones_like(token_ids)
            if self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
                input_mask = input_mask[:self.max_seq_len]
            return token_ids, input_mask, item['fold_label']

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        if self.do_from_plus:
            input_ids, fold_label = tuple(zip(*batch))
            sequences = [torch.from_numpy(sequence).long().squeeze(0) for sequence in input_ids]
            instances = []
            for seq in sequences:
                # tokens, segments, input_mask
                instance = preprocess_seq_for_tfm(seq, max_len=self.max_seq_len, augment=False)
                instances.append(instance)
        
            tokens, segments, input_mask =\
                tuple(torch.stack([a[i] for a in instances], 0) for i in range(3))

            fold_label = torch.LongTensor(fold_label)  # type: ignore

            return {'tokens': tokens,
                    'segments': segments,
                    'input_mask': input_mask,
                    'targets': fold_label}

        else:
            input_ids, input_mask, fold_label = tuple(zip(*batch))
            input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
            input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
            fold_label = torch.LongTensor(fold_label)  # type: ignore

            return {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'targets': fold_label}


class StabilityDataset(FinetuningDataset):
    test_split_name       = 'test'

    def __init__(
        self,
        split,
        data_dir          = RAW_DATASETS_DIR,
        in_memory         = True,
        max_seq_len       = None,
        do_from_plus      = False,
        **dataset_kwargs
    ):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")
                             
        tokenizer = TAPETokenizer(vocab='iupac')
        if do_from_plus: tokenizer = Protein()
        else: tokenizer = TAPETokenizer(vocab='iupac')
        self.tokenizer = tokenizer

        data_dir  = Path(data_dir)
        data_file = f'TAPE/stability/stability_{split}.lmdb'
        self.data = dataset_factory(data_dir / data_file, in_memory)
        self.max_seq_len = max_seq_len
        self.do_from_plus = do_from_plus

        super().__init__(
            **dataset_kwargs
        )


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        if self.do_from_plus:
            s_bytes = str.encode(item['primary'])
            input_ids = self.tokenizer.encode(s_bytes)
            return input_ids, float(item['stability_score'][0])

        else:
            token_ids = self.tokenizer.encode(item['primary'])
            input_mask = np.ones_like(token_ids)
            if self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
                input_mask = input_mask[:self.max_seq_len]
            return token_ids, input_mask, float(item['stability_score'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        if self.do_from_plus:
            input_ids, stability_true_value = tuple(zip(*batch))
            sequences = [torch.from_numpy(sequence).long().squeeze(0) for sequence in input_ids]
            instances = []
            for seq in sequences:
                # tokens, segments, input_mask
                instance = preprocess_seq_for_tfm(seq, max_len=self.max_seq_len, augment=False)
                instances.append(instance)
        
            tokens, segments, input_mask =\
                tuple(torch.stack([a[i] for a in instances], 0) for i in range(3))

            stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore
            stability_true_value = stability_true_value.unsqueeze(1)

            return {'tokens': tokens,
                    'segments': segments,
                    'input_mask': input_mask,
                    'targets': stability_true_value}

        else:
            input_ids, input_mask, stability_true_value = tuple(zip(*batch))
            input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
            input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
            stability_true_value = torch.FloatTensor(stability_true_value)  # type: ignore
            stability_true_value = stability_true_value.unsqueeze(1)

            return {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'targets': stability_true_value}


class FluorescenceDataset(FinetuningDataset):
    test_split_name       = 'test'

    def __init__(
        self,
        split,
        data_dir          = RAW_DATASETS_DIR,
        in_memory         = True,
        max_seq_len       = None,
        do_from_plus      = False,
        **dataset_kwargs
    ):

        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        tokenizer = TAPETokenizer(vocab='iupac')
        if do_from_plus: tokenizer = Protein()
        else: tokenizer = TAPETokenizer(vocab='iupac')
        self.tokenizer = tokenizer

        data_dir  = Path(data_dir)
        data_file = f'TAPE/fluorescence/fluorescence_{split}.lmdb'
        self.data = dataset_factory(data_dir / data_file, in_memory)
        self.max_seq_len = max_seq_len
        self.do_from_plus = do_from_plus

        super().__init__(
            **dataset_kwargs
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        if self.do_from_plus:
            s_bytes = str.encode(item['primary'])
            input_ids = self.tokenizer.encode(s_bytes)
            return input_ids, float(item['log_fluorescence'][0])

        else:
            token_ids = self.tokenizer.encode(item['primary'])
            input_mask = np.ones_like(token_ids)
            if self.max_seq_len:
                token_ids = token_ids[:self.max_seq_len]
                input_mask = input_mask[:self.max_seq_len]
            return token_ids, input_mask, float(item['log_fluorescence'][0])

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        if self.do_from_plus:
            input_ids, fluorescence_true_value = tuple(zip(*batch))
            sequences = [torch.from_numpy(sequence).long().squeeze(0) for sequence in input_ids]
            instances = []
            for seq in sequences:
                # tokens, segments, input_mask
                instance = preprocess_seq_for_tfm(seq, max_len=self.max_seq_len, augment=False)
                instances.append(instance)
        
            tokens, segments, input_mask =\
                tuple(torch.stack([a[i] for a in instances], 0) for i in range(3))

            fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore
            fluorescence_true_value = fluorescence_true_value.unsqueeze(1)

            return {'tokens': tokens,
                    'segments': segments,
                    'input_mask': input_mask,
                    'targets': fluorescence_true_value}
        else:            
            input_ids, input_mask, fluorescence_true_value = tuple(zip(*batch))
            input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
            input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
            fluorescence_true_value = torch.FloatTensor(fluorescence_true_value)  # type: ignore
            fluorescence_true_value = fluorescence_true_value.unsqueeze(1)

            return {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'targets': fluorescence_true_value}


class SecondaryStructureDataset(FinetuningDataset):
    test_split_name       = 'cb513'

    def __init__(
        self,
        split,
        data_dir          = RAW_DATASETS_DIR,
        in_memory         = True,
        max_seq_len       = None,
        do_from_plus      = False,
        **dataset_kwargs
    ):

        if split not in ('train', 'valid', 'casp12', 'ts115', 'cb513'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'valid', 'casp12', "
                             f"'ts115', 'cb513']")

        tokenizer = TAPETokenizer(vocab='iupac')
        if do_from_plus: tokenizer = Protein()
        else: tokenizer = TAPETokenizer(vocab='iupac')
        self.tokenizer = tokenizer

        data_dir  = Path(data_dir)
        data_file = f'TAPE/secondary_structure/secondary_structure_{split}.lmdb'
        self.data = dataset_factory(data_dir / data_file, in_memory)
        self.max_seq_len = max_seq_len
        assert max_seq_len is None or max_seq_len == 0, 'Truncating not supported for sequence to sequence tasks!'
        self.do_from_plus = do_from_plus
        
        super().__init__(
            **dataset_kwargs
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        if self.do_from_plus:
            s_bytes = str.encode(item['primary'])
            input_ids = self.tokenizer.encode(s_bytes)

            # pad with -1s because of cls/sep tokens
            labels = np.asarray(item['ss3'], np.int64)
            labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)
            return input_ids, labels

        else:
            token_ids = self.tokenizer.encode(item['primary'])
            input_mask = np.ones_like(token_ids)

            # pad with -1s because of cls/sep tokens
            labels = np.asarray(item['ss3'], np.int64)
            labels = np.pad(labels, (1, 1), 'constant', constant_values=-1)

            return token_ids, input_mask, labels

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        if self.do_from_plus:
            input_ids, ss_label = tuple(zip(*batch))
            sequences = [torch.from_numpy(sequence).long().squeeze(0) for sequence in input_ids]
            instances = []
            max_len = max([len(s) for s in sequences]) + 2    # for [CLS], [SEP]

            for seq in sequences:
                # tokens, segments, input_mask
                instance = preprocess_seq_for_tfm(seq, max_len=max_len, augment=False)
                instances.append(instance)
        
            tokens, segments, input_mask =\
                tuple(torch.stack([a[i] for a in instances], 0) for i in range(3))

            ss_label = torch.from_numpy(pad_sequences(ss_label, -1))
            weights = torch.zeros_like(ss_label, dtype=torch.bool)
            for idx, seq in enumerate(sequences):
                weights[idx, 1:len(seq)+1] = True

            assert torch.all(weights == (ss_label != -1)), "Weights should be 1 if label != -1! (ignore_index)"

            return {'tokens': tokens,
                    'segments': segments,
                    'input_mask': input_mask,
                    'targets': ss_label,
                    'weights': weights}

        else:
            input_ids, input_mask, ss_label = tuple(zip(*batch))
            input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
            input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
            ss_label = torch.from_numpy(pad_sequences(ss_label, -1))

            output = {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'targets': ss_label}

            return output
