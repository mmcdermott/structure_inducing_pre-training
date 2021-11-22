import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tape.models.modeling_bert import (
    ProteinBertAbstractModel, ProteinBertEmbeddings, ProteinBertEncoder, ProteinBertPooler
)
from tape.models.modeling_utils import SequenceClassificationHead, ValuePredictionHead, SequenceToSequenceClassificationHead
from tape.metrics import spearmanr
from transformers import BertModel, RobertaModel

import sys
sys.path.append("../PLUS")
from plus.model.plus_tfm import evaluate_lm, evaluate_cls_protein, evaluate_cls_amino


class EncoderModule(nn.Module):
    # Wraps a huggingface-style sequence model.
    def __init__(
        self, config, sequence_model_cnstr, pooling_model
    ):
        super().__init__()
        self.config = config
        self.sequence_model = sequence_model_cnstr(self.config)
        self.pooling_model = pooling_model

    def forward(
        self,
        input_ids=None,
        attention_mask=None
    ):
        input_ids = input_ids
        attention_mask = attention_mask

        sequence_model_kwargs = {}
        if isinstance(self.sequence_model, (BertModel, RobertaModel)):
            sequence_model_kwargs = {
                'return_dict': True,
                'output_hidden_states': True
            }

        sequence_out = self.sequence_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **sequence_model_kwargs,
        )
        last_hidden_state = sequence_out.last_hidden_state
        pooled_out = self.pooling_model(last_hidden_state)

        return {
            'summary': pooled_out,
            'granular': last_hidden_state,
        }

def bow_pooler(tensor): return tensor[0]

class PLUSEncoderModule(EncoderModule):
    # Wraps a PLUS-style sequence model.
    def __init__(
        self,
        lm=False,
        reg_protein=False,
        cls_protein=False,
        cls_amino=False,
        **encoder_kwargs
    ):
        super().__init__(
            **encoder_kwargs
        )

        self.lm = lm
        self.reg_protein = reg_protein
        self.cls_protein = cls_protein
        self.cls_amino = cls_amino

    def forward(
        self,
        tokens=None,
        segments=None,
        input_mask=None,
        masked_pos=None,
        masked_tokens=None,
        masked_weights=None,
        embedding=False,
        targets=None,
        weights=None,
        **kwargs,
    ):
        per_seq = not self.cls_amino
        sequence_out = self.sequence_model(
            tokens=tokens,
            segments=segments,
            input_mask=input_mask,
            masked_pos=masked_pos,
            embedding=embedding,
            per_seq=per_seq,
        )
        if embedding: h = sequence_out
        else: logits_lm, logits_cls, h = sequence_out

        last_hidden_state = h
        pooled_out = self.pooling_model(last_hidden_state)

        encoded = {
            'summary': pooled_out,
            'granular': last_hidden_state,
        }

        if embedding: return encoded

        if self.lm:
            result = evaluate_lm(logits_lm, masked_tokens, masked_weights, flag={'acc': True})
            accuracy_masked = result['correct'] / result['n']
            accuracy_masked = torch.Tensor([accuracy_masked]).to(result['avg_loss'].device)

            point_pretraining_out = {
                'loss': result['avg_loss'],
                'predictions': {
                    'prediction_score': logits_lm,
                    'predictions': logits_lm.argmax(dim=2),
                },
                'metrics': {
                    'accuracy_masked': accuracy_masked,
                }
            }

            return encoded, point_pretraining_out

        elif self.reg_protein:
            result = evaluate_cls_protein(logits_cls, targets.unsqueeze(1), {'acc': False, 'pred': True}, {'regression': True})

            loss = result['avg_loss']
            logits = result['logits'][0]

            value_prediction = logits.detach().numpy()
            targets = targets.detach().cpu().numpy()

            metric_val = spearmanr(targets, value_prediction)
            metrics = {'spearmanr': torch.Tensor([metric_val]).to(loss.device)}

            return {
                'loss': loss,
                'predictions': value_prediction,
                'metrics': metrics,
                'targets': targets,
            }

        elif self.cls_protein:
            result = evaluate_cls_protein(logits_cls, targets.unsqueeze(1), {'acc': True, 'pred': True}, {'regression': False})

            loss = result['avg_loss']
            logits = result['logits'][0]
            accuracy = torch.Tensor([result['correct'] / result['n']]).to(loss.device)

            prediction_scores = logits.detach().numpy()
            targets = targets.detach().cpu().numpy()
            metrics = {'accuracy': accuracy}

            return {
                'loss': loss,
                'predictions': prediction_scores,
                'metrics': metrics,
                'targets': targets,
            }


        elif self.cls_amino:
            result = evaluate_cls_amino(logits_cls, targets, weights, {'acc': True, 'pred': True})

            loss = result['avg_loss']
            logits = result['logits'][0]
            accuracy = torch.Tensor([result['correct'] / result['n']]).to(loss.device)

            sequence_logits = logits.detach().numpy()
            targets = targets.detach().cpu().numpy()
            metrics = {'accuracy': accuracy}

            return {
                'loss': loss,
                'predictions': sequence_logits,
                'metrics': metrics,
                'targets': targets,
            }

def cls_pooler(tensor):
    return tensor[:, 0, :]

class GRUPooler(nn.Module):
    # TODO: This should take into account the attention mask!
    def __init__(self, config):
        super().__init__()

        self.do_max_pool, self.do_avg_pool, self.do_attn_pool, self.do_last_pool = False, False, False, False

        self.config = config
        if hasattr(config, 'do_max_pool') and config.do_max_pool:
            self.do_max_pool = True
        elif hasattr(config, 'do_avg_pool') and config.do_avg_pool:
            self.do_avg_pool = True
        elif hasattr(config, 'do_attn_pool') and config.do_attn_pool:
            self.do_attn_pool = True

            gru_out_dim = config.encoder_hidden_size

            self.attn_vector = nn.Parameter(torch.FloatTensor(gru_out_dim, 1))
            self.attn_vector = nn.init.xavier_normal(self.attn_vector)
        else:
            self.do_last_pool = True

    def forward(self, gru_out):
        if self.do_max_pool:    return gru_out.max(dim=1)[0]
        elif self.do_avg_pool:  return gru_out.mean(dim=1)
        elif self.do_last_pool:                   return gru_out[:, -1, :]
        elif self.do_attn_pool:
            attention_score = torch.matmul(gru_out, self.attn_vector)
            normalized_attention_score = F.softmax(attention_score, dim=1)
            weighted_average = (gru_out * normalized_attention_score).sum(dim=1)
            return weighted_average

        raise NotImplementedError(
            "Only supports do_avg_pool, do_max_pool, do_attn_pool, and do_last_pool for now."
        )

class BOWAEConfig:
    def __init__(
        self, vocab_size, embed_dim, encoder_hidden_size,
    ):
        self.vocab_size                      = vocab_size
        self.embed_dim                       = embed_dim
        self.encoder_hidden_size             = encoder_hidden_size

class GRUConfig:
    def __init__(
        self, vocab_size, embed_dim, encoder_hidden_size,
        do_max_pool = True, do_avg_pool = False, do_attn_pool = False, do_last_pool = False,
        do_bidirectional = False,
    ):
        self.vocab_size                      = vocab_size
        self.embed_dim                       = embed_dim
        self.encoder_hidden_size             = encoder_hidden_size
        self.do_max_pool, self.do_avg_pool   = do_max_pool, do_avg_pool
        self.do_attn_pool, self.do_last_pool = do_attn_pool, do_last_pool
        self.do_bidirectional                = do_bidirectional

class GRUEncoder(nn.Module):
    class GRUOutput:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    def __init__(
        self,
        config
    ):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.encoder_hidden_size = config.encoder_hidden_size
        self.do_bidirectional = hasattr(config, 'do_bidirectional') and config.do_bidirectional

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embed_dim,
            padding_idx = 0
        )

        # TODO: Why aren't you leveraging the attention mask at all here?

        if self.do_bidirectional:
            assert self.encoder_hidden_size % 2 == 0
            local_encoder_hidden_size = self.encoder_hidden_size // 2
        else: local_encoder_hidden_size = self.encoder_hidden_size

        self.gru = nn.GRU(
            input_size    = self.embed_dim,
            hidden_size   = local_encoder_hidden_size,
            bidirectional = self.do_bidirectional,
            batch_first   = True
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None
    ):
        embedded_seq = self.embedding(input_ids)
        last_hidden_state, _ = self.gru(embedded_seq) # sequence_out shape: (seq_len, batch, num_directions * hidden_size)
        return GRUEncoder.GRUOutput(last_hidden_state)

class BOWEncoder(nn.Module):
    class BOWOutput:
        def __init__(self, last_hidden_state, extra_state):
            self.last_hidden_state = (last_hidden_state, extra_state)

    def __init__(self, config):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.encoder_hidden_size = config.encoder_hidden_size

        self.embedding = nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embed_dim,
            padding_idx = 0
        )

        self.enc_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.encoder_hidden_size),
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None
    ):
        try:
            one_hot_repr = F.one_hot((input_ids * attention_mask).long(), self.vocab_size)
            bow_labels = one_hot_repr.sum(dim=1)
        except RuntimeError as e:
            print(input_ids.shape, attention_mask.shape, self.vocab_size)
            print(input_ids)
            raise

        embedded_seq = self.embedding(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(2)

        embedded_seq = embedded_seq * extended_attention_mask
        bow_sum = embedded_seq.sum(dim=1)
        return BOWEncoder.BOWOutput(self.enc_model(bow_sum), bow_labels)


def mlm_accuracy(
    point_kwargs,
    logits,
    mask_id
 ):
    for k, v in point_kwargs.items():
        point_kwargs[k] = v.detach().cpu()
    logits = logits.detach().cpu()

    match = (
        point_kwargs['labels'].squeeze() == torch.argmax(logits, dim=-1)
    )

    is_masked = point_kwargs['input_ids'] == mask_id
    is_present = point_kwargs['attention_mask'] == 1
    present_and_masked = is_present & is_masked

    num_correct_overall = np.where(
        is_present, match, np.zeros_like(match)
    ).sum(axis=-1)
    num_present_overall = is_present.sum(axis=-1)
    acc_overall = (num_correct_overall / num_present_overall.float().detach().numpy())

    num_correct_masked = np.where(
        present_and_masked, match, np.zeros_like(match)
    ).sum(axis=-1)
    num_present_masked = present_and_masked.sum(axis=-1)
    any_masked = num_present_masked > 0
    num_present_masked = np.where(
        any_masked, num_present_masked, np.ones_like(num_present_masked)
    )

    acc_masked = np.where(
        any_masked,
        num_correct_masked / num_present_masked,
        np.zeros_like(num_correct_masked),
    ).astype(float)

    return acc_overall.mean(), acc_masked[~np.isnan(acc_masked)].mean()

class BOWAEPretrainingHead(nn.Module):
    # Wraps a huggingface-style MLM PT head.
    def __init__(
        self, config, mask_id
    ):
        super().__init__()
        self.config = config
        self.dec = nn.Sequential(
            nn.Linear(config.encoder_hidden_size, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, config.vocab_size),
        )
        self.loss = nn.PoissonNLLLoss()
        self.mask_id = mask_id

    def forward(
        self,
        points_encoded,
        point_kwargs
    ):
        bow_enc, bow_targets = points_encoded['granular']

        prediction_scores = self.dec(bow_enc)

        try: point_loss = self.loss(prediction_scores, bow_targets)
        except RuntimeError as e:
            print(prediction_scores.shape)
            print(bow_targets.shape)
            raise

        return {
            'loss': point_loss,
            'predictions': { },
            'metrics': { }
        }

class MLMPretrainingHead(nn.Module):
    # Wraps a huggingface-style MLM PT head.
    def __init__(
        self, config, head_cnstr, mask_id
    ):
        super().__init__()
        self.config = config
        self.head = head_cnstr(self.config)
        self.mask_id = mask_id

    def forward(
        self,
        points_encoded,
        point_kwargs
    ):
        sequence_out = points_encoded['granular']
        labels = point_kwargs['labels']

        prediction_scores = self.head(sequence_out)
        loss_fct = nn.CrossEntropyLoss()
        point_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )

        accuracy_overall, accuracy_masked = mlm_accuracy(
            point_kwargs,
            prediction_scores,
            self.mask_id
        )

        accuracy_overall = torch.Tensor([float(accuracy_overall)]).type_as(point_loss)
        accuracy_masked = torch.Tensor([float(accuracy_masked)]).type_as(point_loss)

        return {
            'loss': point_loss,
            'predictions': {
                'prediction_scores': prediction_scores,
                'predictions': prediction_scores.argmax(dim=2),
            },
            'metrics': {
                'accuracy_overall': accuracy_overall,
                'accuracy_masked': accuracy_masked,
            }
        }


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


# Metric Learning Heads
## New Heads
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
            'targets': None,
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


class BilinearPredictionHead(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()

        self.bil = nn.Bilinear(dim1, dim2, 1)
        self.loss_fct = nn.BCELoss()

    def forward(self, input1, input2, labels):
        preds = torch.sigmoid(self.bil(input1, input2)).squeeze()
        loss = self.loss_fct(
            preds,
            labels
        )
        return preds, loss


class ProteinBertModel(ProteinBertAbstractModel):

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = ProteinBertEmbeddings(config)
        self.encoder = ProteinBertEncoder(config)
        self.pooler = ProteinBertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class ProteinModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    class ProteinBertOutput():
        def __init__(self, hidden_states, last_hidden_state):
            self.hidden_states = hidden_states
            self.last_hidden_state = last_hidden_state

    def forward(
        self,
        input_ids=None,
        attention_mask=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since input_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       chunks=None)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        return ProteinBertModel.ProteinBertOutput(
            hidden_states = [outputs[0]],
            last_hidden_state = outputs[0]
        )


def _accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


class SequenceClassificationFinetuningHead(nn.Module):
    def __init__(self, **head_kwargs):
        super().__init__()
        self.head = SequenceClassificationHead(**head_kwargs)

    def forward(
        self,
        points_encoded,
        batch,
    ):
        pooled_out = points_encoded['summary']
        outputs = self.head(pooled_out, batch['targets'])

        if isinstance(outputs[0], tuple):
            loss = outputs[0][0]
        else:
            loss = outputs[0]
        prediction_scores = outputs[1]
        metrics = {'accuracy': _accuracy(prediction_scores, batch['targets'])}
        prediction_scores = prediction_scores.detach().cpu().numpy()
        targets = batch['targets'].detach().cpu().numpy()

        return {
            'loss': loss,
            'predictions': prediction_scores,
            'metrics': metrics,
            'targets': targets,
        }


class ValuePredictionFinetuningHead(nn.Module):
    def __init__(self, **head_kwargs):
        super().__init__()
        self.head = ValuePredictionHead(**head_kwargs)

    def forward(
        self,
        points_encoded,
        batch,
    ):
        pooled_out = points_encoded['summary']
        outputs = self.head(pooled_out, batch['targets'])

        loss = outputs[0]
        value_prediction = outputs[1].detach().cpu().numpy()
        targets = batch['targets'].detach().cpu().numpy()

        metric_val = spearmanr(targets, value_prediction)
        metrics = {'spearmanr': torch.Tensor([metric_val]).to(loss.device)}

        return {
            'loss': loss,
            'predictions': value_prediction,
            'metrics': metrics,
            'targets': targets,
        }


class SequenceToSequenceClassificationFinetuningHead(nn.Module):
    def __init__(self, **head_kwargs):
        super().__init__()
        self.head = SequenceToSequenceClassificationHead(**head_kwargs)

    def forward(
        self,
        points_encoded,
        batch,
    ):
        sequence_out = points_encoded['granular']
        outputs = self.head(sequence_out, batch['targets'])


        loss = outputs[0]
        sequence_logits = outputs[1]
        metrics = {
            'sequence_accuracy': _accuracy(
                sequence_logits.view(-1, self.head.num_labels),
                batch['targets'].view(-1),
                self.head._ignore_index
            )
        }

        sequence_logits = sequence_logits.detach().cpu().numpy()
        targets = batch['targets'].detach().cpu().numpy()

        return {
            'loss': loss,
            'predictions': sequence_logits,
            'metrics': metrics,
            'targets': targets,
        }


class BertSequenceClassificationFinetuningHead(nn.Module):
    def __init__(self, num_labels, hidden_dropout_prob, hidden_size):
        super().__init__()
        self.num_labels = num_labels

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        points_encoded,
        batch,
    ):
        pooled_out = points_encoded['summary']

        pooled_out = self.dropout(pooled_out)
        logits = self.classifier(pooled_out)

        labels = batch['targets']

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        metrics = {'accuracy': _accuracy(logits, labels)}
        targets = labels.detach().cpu().numpy()
        predictions = logits.float().argmax(-1)
        predictions = predictions.detach().cpu().numpy()

        return {
            'loss': loss,
            'predictions': predictions,
            'metrics': metrics,
            'targets': targets,
        }


class GmlFinetuningHead(nn.Module):
    def __init__(self, linker = None, do_initial_fc = True, fc_input_dim = None, fc_output_dim = None):
        super().__init__()

        self.linker = linker
        self.do_initial_fc = do_initial_fc
        self.fc_input_dim = fc_input_dim
        self.fc_output_dim = fc_output_dim

        if self.do_initial_fc:
            self.fc = nn.Linear(
                in_features = fc_input_dim,
                out_features = fc_output_dim,
            )

    def forward(
        self,
        points_encoded,
        batch,
    ):
        if self.do_initial_fc:
            batch['point_embeds'] =self.fc(batch['point_embeds'])

        points_encoded = {'summary': batch['point_embeds']}
        context_encoded = batch['context_embeds']
        gml_kwargs = {'labels': batch['labels']}

        return self.linker(points_encoded, context_encoded, gml_kwargs)
