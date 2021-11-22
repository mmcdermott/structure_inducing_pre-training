import torch
from pathlib import Path

from .finetuning_dataset import *


class RankingDataset(FinetuningDataset):
    def __init__(
        self,
        split,
        data_dir,
        **dataset_kwargs
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"Unrecognized split: {split}. Must be one of "
                             f"['train', 'val', 'test']")
        
        data_dir                    = Path(data_dir)
        split_dir                   = data_dir / split
        self.split_dir              = split_dir
        
        self._read_embeddings()

        super().__init__(
            **dataset_kwargs
        )

    def _read_embeddings(self):
        self.point_embeddings       = torch.load(self.split_dir / 'point_embeddings.pt')
        self.context_embeddings     = torch.load(self.split_dir / 'context_embeddings.pt')
        self.labels                 = torch.load(self.split_dir / 'labels.pt')

        assert len(self.point_embeddings) == len(self.context_embeddings), "Mismatched dims."
        assert len(self.point_embeddings) == len(self.labels), "Mismatched dims."

    def __len__(self):
        return len(self.point_embeddings)

    def __getitem__(self, index):
        point_embed                 = self.point_embeddings[index].unsqueeze(0)
        context_embed               = self.context_embeddings[index].unsqueeze(0)
        label                       = self.labels[index].unsqueeze(0)
        return point_embed, context_embed, label

    def collate_fn(self, batch):
        point_embeds, context_embeds, labels = tuple(zip(*batch))
        point_embeds                = torch.cat(point_embeds)
        context_embeds              = torch.cat(context_embeds)
        labels                      = torch.cat(labels)
        return {
            'point_embeds': point_embeds,
            'context_embeds': context_embeds,
            'labels': labels,
        }
