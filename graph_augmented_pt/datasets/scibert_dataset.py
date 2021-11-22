import torch, numpy as np

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import AutoTokenizer
from tape.datasets import pad_sequences

from .finetuning_dataset import *
from ..utils.utils import *


class JsonlinesDataset():
    def __init__(
        self,
        data_file: Path,
        tokenizer: AutoTokenizer,
        task: str,
        hf_model_name: str,
    ):
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.task = task

        self.raw_texts = []
        self.raw_labels = []
        
        if self.task == SCIBERT_MAG:
            self.label_map = {
                'politics': 0,
                'geography': 1,
                'category': 2,
                'medicine': 3,
                'economics': 4,
                'psychology': 5,
                'business': 6,
                'sociology': 7
            }
        elif self.task == SCIBERT_SCI_CITE:
            self.label_map = {'result': 0, 'background': 1, 'method': 2}
        elif self.task == SCIBERT_CITATION_INTENT:
            self.label_map = {'Extends': 0, 'Background': 1, 'Motivation': 2, 'Future': 3, 'Uses': 4, 'CompareOrContrast': 5}
        elif self.task == SCIBERT_OGBN_MAG:
            self.label_map = {k: k for k in range(349)}
        else: raise NotImplementedError

        self._read()

        hf_model_suffix = hf_model_name.split('/')[-1]
        self.cached_tokenized_path = str(self.data_file).strip('.txt') + '_' + hf_model_suffix + '.pkl'
        self.cached_tokenized_path = Path(self.cached_tokenized_path)

        if self.cached_tokenized_path.exists():
            print(f"Reading tokenized texts from {self.cached_tokenized_path}")
            self.input_ids = depickle(self.cached_tokenized_path)
        else:
            print(f"Writing tokenized texts to {self.cached_tokenized_path}")
            self.tokenized_texts = self.tokenizer(
                self.raw_texts, add_special_tokens=True, padding=False
            )
            self.input_ids = self.tokenized_texts.data['input_ids']
            enpickle(self.input_ids, self.cached_tokenized_path)

        self.num_labels = len(self.label_map)
        self.labels = [self.label_map[l] for l in self.raw_labels]

        assert len(self.input_ids) == len(self.labels)

    def _read(self):
        import jsonlines
        with jsonlines.open(self.data_file) as f_in:
            for json_object in f_in:
                raw_text = json_object.get('text')
                self.raw_texts.append(raw_text)
                raw_label = json_object.get('label')
                self.raw_labels.append(raw_label)

    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, index: int):
        if not 0 <= index < len(self.input_ids):
            raise IndexError(index)
        
        input_ids, label = self.input_ids[index], self.labels[index]
        return input_ids, label


class ScibertDataset(FinetuningDataset):
    def __init__(
        self,
        task,
        split,
        data_dir          = RAW_DATASETS_DIR,
        max_seq_len       = None,
        hf_model_name    = SCIBERT_SCIVOCAB_UNCASED,
        **dataset_kwargs
    ):
        if task not in ('mag', 'sci-cite', 'citation_intent', 'ogbn-mag'):
            raise ValueError(f"Unrecognized task: {task}."
                             f" Must be one of ['mag', 'sci-cite', 'citation_intent', 'ogbn-mag']")
        if split not in ('train', 'dev', 'test'):
            raise ValueError(f"Unrecognized split: {split}."
                             f" Must be one of ['train', 'dev', 'test']")
        
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        data_dir = Path(data_dir)
        data_file = f'scibert/data/text_classification/{task}/{split}.txt'
        self.data = JsonlinesDataset(data_dir / data_file, self.tokenizer, task, hf_model_name)
        self.num_labels = self.data.num_labels
        self.max_seq_len = max_seq_len

        super().__init__(
            **dataset_kwargs
        )
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        input_ids, label = self.data[index]
        input_ids = np.array(input_ids)
        input_mask = np.ones_like(input_ids)
        if self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            input_mask = input_mask[:self.max_seq_len]
        return input_ids, input_mask, label
    
    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, label = tuple(zip(*batch))
        input_ids = torch.from_numpy(pad_sequences(input_ids, self.tokenizer.pad_token_id))
        input_mask = torch.from_numpy(pad_sequences(input_mask, self.tokenizer.pad_token_id))
        label = torch.LongTensor(label)

        return {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'targets': label
        }
