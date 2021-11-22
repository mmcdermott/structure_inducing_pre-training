import copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
# from torch_geometric.nn import global_mean_pool, GCNConv
from tape.models.modeling_bert import (
    ProteinBertAbstractModel, ProteinBertEmbeddings, ProteinBertEncoder, ProteinBertPooler
)

from .RDML_ports import *
from .constants import *

# RDML-based Heads
class RDMLHead(nn.Module):
    def __init__(self, dim1, dim2, criteria, **linker_kwargs):
        super().__init__()

        self.criteria = criteria

    def forward(self, points_encoded, context_encoded, gml_kwargs):
        args, kwargs = produce_rdml_batchminer_input(points_encoded, context_encoded, gml_kwargs)

        outputs = self.criteria(*args, **kwargs)
        if isinstance(outputs, tuple):
            loss, metrics = outputs
        else:
            loss = outputs 
            metrics = {}
        return {'loss': loss, 'predictions': {}, 'metrics': metrics}

# Simple, pair-based heads
class BilinearCosineLinker(nn.Module):
    def __init__(self, dim1, dim2, **linker_kwargs):
        super().__init__()

        self.combiner = nn.Bilinear(in1_features=dim1, in2_features=dim2, out_features=1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, points_encoded, context_encoded, gml_kwargs):
        logits = self.combiner(points_encoded['summary'], context_encoded).squeeze()
        labels = gml_kwargs['labels'].squeeze().float()
        loss = self.loss(logits, labels)
        return {
            'loss': loss,
            'predictions': {
                'logits': logits,
            },
            'metrics': {},
        }

class MarginEuclideanLinker(nn.Module):
    """
    This is a very simple contrastive euclidean linker, based on the paper below
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, dim1, dim2, negative_margin, positive_margin=0, do_normalize_embeds=False, **linker_kwargs):
        super().__init__()

        assert negative_margin > 0, f"Must pass a negative_margin > 0! Got {negative_margin}"
        self.negative_margin = negative_margin

        assert positive_margin >= 0, f"Must pass a non-negative positive_margin! Got {positive_margin}"
        self.positive_margin = positive_margin

        self.dist = nn.PairwiseDistance()
        self.do_normalize_embeds = do_normalize_embeds

    def forward(self, points_encoded, context_encoded, gml_kwargs):
        points_embeds = points_encoded['summary']
        context_embeds = context_encoded

        if self.do_normalize_embeds:
            points_embeds = F.normalize(points_embeds, p=2, dim=1)
            context_embeds = F.normalize(context_embeds, p=2, dim=1)

        dist = self.dist(points_embeds, context_embeds)

        labels = gml_kwargs['labels'].squeeze().float()

        linked_pairs   = (labels == 1)
        unlinked_pairs = (labels == 0)

        margin = torch.ones_like(labels)
        margin = torch.where(unlinked_pairs, margin * self.negative_margin, margin * self.positive_margin)
        diff = dist - margin

        mult = torch.ones_like(labels)
        mult = torch.where(unlinked_pairs, -1 * mult, mult)

        loss = F.relu(mult * diff).mean()

        return {
            'loss': loss,
            'predictions': {
                'dist': dist,
            },
            'metrics': {},
            'targets': None,
        }


## Old Heads
class gml_cosine_distance(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()

        self.combiner = nn.Linear(dim1 + dim2, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, points_encoded, context_encoded, gml_kwargs):
        logits = self.combiner(
            torch.cat((points_encoded['summary'], context_encoded),
            dim=1)
        )
        labels = gml_kwargs['labels'].squeeze()
        loss = self.loss(logits, labels)
        return {
            'loss': loss,
            'predictions': {
                'logits': logits,
            },
            'metrics': {},
        }


class gml_euclidean_distance(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()

        self.dist = nn.PairwiseDistance()

    def forward(self, points_encoded, context_encoded, gml_kwargs):
        dist = self.dist(points_encoded['summary'], context_encoded)

        labels = gml_kwargs['labels'].squeeze()
        mult = torch.ones_like(labels)
        mult = torch.where(labels == 0, -1 * mult, mult)

        loss = (mult * dist).mean()

        return {
            'loss': loss,
            'predictions': {
                'dist': dist,
            },
            'metrics': {},
        }
