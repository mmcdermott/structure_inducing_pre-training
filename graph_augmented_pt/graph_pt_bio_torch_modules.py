import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

import os, sys
from .constants import *
sys.path.append(os.path.join(COMPANION_CODE_DIR, "pretrain-gnns/bio"))

from model import GNN_graphpred
from pretrain_masking import compute_accuracy

class FTClassificationHead(nn.Module):
    TASK_DIMS = {
        'go_target_pretrain': 500,
        'go_target_downstream': 40,
    }

    def __init__(self, config: dict, y_label: str = 'go_target_pretrain'):
        super().__init__()

        self.y_label = y_label
        self.logits_layer = nn.Linear(config['hidden_dim'], self.TASK_DIMS[y_label])
        self.criterion = nn.BCEWithLogitsLoss()

    def _get_y(self, batch): return getattr(batch['input_ids'], self.y_label)

    def forward(self, points_encoded, batch):
        logits = self.logits_layer(points_encoded['summary'])
        y      = self._get_y(batch).view_as(logits).float()
        loss   = self.criterion(logits, y)

        acc    = ((logits > 0.5) == y).float().mean(dim=1).mean(dim=0).detach()
        return {
            'loss': loss,
            'metrics': {
                'accuracy': acc,
            },
            'predictions': logits.detach(),
            'targets': y.detach(),
        }

class FakeGGMLClassificationHead(FTClassificationHead):
    TASK_DIMS = {
        'go_target_pretrain': 4000,
        'go_target_downstream': 40,
    }
    def forward(self, points_encoded, point_kwargs):
        return super().forward(points_encoded, point_kwargs)

class GNNBoth(GNN_graphpred):
    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        pooled = self.pool(node_representation, batch)
        center_node_rep = node_representation[data.center_node_idx]

        graph_rep = torch.cat([pooled, center_node_rep], dim=1)

        return node_representation, self.graph_pred_linear(graph_rep)

class GCNNEncoderModule(nn.Module):
    def __init__(
        self, config,
        sequence_model_cnstr=None, pooling_model=None # Both are disregarded...
    ):
        super().__init__()
        self.config = config

        gnn_kwargs = {
            'emb_dim':       config['emb_dim'],
            'num_tasks':     config['hidden_dim'],
            'JK':            config['JK'],
            'drop_ratio':    config['dropout_ratio'],
            'graph_pooling': config['graph_pooling'],
            'gnn_type':      'gin',
        }

        num_layer = config['num_layer']
        self.encoder = GNNBoth(num_layer = num_layer, **gnn_kwargs)

        if self.config['do_context_pred']:
            raise NotImplementedError

    def forward(
        self,
        input_ids=None,
        attention_mask=None
    ):
        node_representation, graph_pooled = self.encoder(input_ids)
        return {
            'summary': graph_pooled,
            'granular': node_representation,
            'batch': input_ids,
        }

class GCNNMaskingPTHead(nn.Module):
    # Wraps a huggingface-style MLM PT head.
    def __init__(
        self, config
    ):
        super().__init__()
        self.config = config

        self.do_mask_edge = config['do_mask_edge']

        self.criterion         = torch.nn.CrossEntropyLoss()
        self.linear_pred_edges = torch.nn.Linear(config['emb_dim'], 7)

    def forward(
        self,
        points_encoded,
        point_kwargs
    ):
        batch = point_kwargs['input_ids']

        sequence_out = points_encoded['granular']

        masked_edge_idx = batch.edge_index[:, batch.masked_edge_idx]
        edge_rep = sequence_out[masked_edge_idx[0]] + sequence_out[masked_edge_idx[1]]
        pred_edge = self.linear_pred_edges(edge_rep)

        labels = torch.argmax(batch.mask_edge_label[:, :7], dim=1)

        acc_edge = compute_accuracy(pred_edge, labels)
        loss = self.criterion(pred_edge, labels)

        metrics = {'accuracy_masked_edge': acc_edge}
        predictions = {
            #'prediction_edges_scores': pred_edge.detach(),
            #'prediction_edges': pred_edge.argmax(dim=1).detach(),
        }

        return {'loss': loss, 'predictions': predictions, 'metrics': metrics}

class PointContextEncoder(nn.Module):
    # For context equal to a single point.
    # Returns the summary for that point.
    def __init__(
        self
    ):
        super().__init__()

    def forward(
        self,
        context_points_encoded,
        gml_kwargs
    ):
        out = context_points_encoded['summary']
        return out
