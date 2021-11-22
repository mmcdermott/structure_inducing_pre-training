import torch, torch.nn as nn, torch.optim as optim, numpy as np
from transformers import get_linear_schedule_with_warmup
from pytorch_lightning import (
    LightningModule, LightningDataModule, Trainer, metrics
)


class PretrainingModule(LightningModule):
    """
    This Lightning Module summarizes the system in _pre-training_ mode.
    It takes as input:

      `encoder_module`:
        A PyTorch module that (on forward) ingests a base object and yields a
        dictionary {
            'summary': <an overall summary of the object>,
            'granular': <a localized representation of the object (e.g., per-token)>,
        }

      `point_pretraining_head`:
        A PyTorch module that ingests point object encodings and point pretraining task args
        (eg. input_ids, attn mask, labels). It returns a dictionary {
            'loss': <mlm loss>
            'predictions': <predictions dictionary>
            'metrics': <metrics dictionary>
        }

      `context_encoder_module`:
         A PyTorch or PyTorch Geometric module that ingests a subgraph's point encodings and its structure.
         Returns <an overall summary of the context>.


      `graph_metric_learning_head`:
         A PyTorch module that ingests the encoding of a point and its context, as well as
         GML task args. It yields a dictionary {
            'loss': <gml loss>
            'predictions': <predictions dictionary>
            'metrics': <metrics dictionary>
         }

    """
    def __init__(
        self,
        encoder_module = None,
        point_pretraining_head = None,
        context_encoder_module = None,
        gml_head = None,
        lr = 1e-4,
        gml_weight = 1,
        point_weight = 1,
        do_log_on_epoch = True,
        do_from_plus = False,
    ):
        super().__init__()
        self.encoder_module = encoder_module
        self.point_pretraining_head = point_pretraining_head
        self.context_encoder_module = context_encoder_module
        self.gml_head = gml_head
        self.lr = lr

        weight_sum = gml_weight + point_weight
        assert weight_sum > 0 and gml_weight >= 0 and point_weight >= 0

        self.gml_weight = gml_weight / weight_sum
        self.point_weight = point_weight / weight_sum

        self.do_log_on_epoch = do_log_on_epoch
        self.do_from_plus = do_from_plus

    def forward(
        self,
        batch
    ):
        if self.do_from_plus:
            point_kwargs, gml_kwargs = batch
            points_encoded, point_pretraining_out = self.encoder_module(embedding=False, **point_kwargs)
            context_encoded = self.encoder_module(embedding=True, **gml_kwargs)
            context_encoded = context_encoded['summary']
            return points_encoded, context_encoded, point_pretraining_out

        else:
            point_kwargs, gml_kwargs = batch
            points_encoded = self.encoder_module(
                point_kwargs['input_ids'],
                point_kwargs['attention_mask']
            )

            if 'context_subgraph' in gml_kwargs:
                ctx_input_ids = gml_kwargs['context_subgraph'].x
                ctx_attention_mask = gml_kwargs['context_subgraph'].attention_mask
            else:
                ctx_input_ids = gml_kwargs['input_ids']
                ctx_attention_mask = gml_kwargs['attention_mask']

            context_points_encoded = self.encoder_module(
                ctx_input_ids,
                ctx_attention_mask
            )
            context_encoded = self.context_encoder_module(context_points_encoded, gml_kwargs)

            return points_encoded, context_encoded

    @classmethod
    def load_from(
        cls, load_dir,
        encoder_module,  point_module, context_module, gml_module,
        **kwargs
    ):
        # TODO: This should really use lightning utilities more.
        print(f'Attempting to load from {load_dir}')
        assert load_dir.exists()

        # TODO: These are state-dicts, not modules
        encoder                = torch.load(load_dir / 'encoder.pt')
        point_pretraining_head = torch.load(load_dir / 'point_pretraining_head.pt')
        context_encoder        = torch.load(load_dir / 'context_encoder_module.pt')
        gml_head               = torch.load(load_dir / 'gml_head.pt')

        encoder_module.load_state_dict(encoder)
        point_module.load_state_dict(point_pretraining_head)
        context_module.load_state_dict(context_encoder)
        gml_module.load_state_dict(gml_head)

        return cls(
            encoder_module         = encoder_module,
            point_pretraining_head = point_module,
            context_encoder_module = context_module,
            gml_head               = gml_module,
            **kwargs,
        )

    def loss(
        self,
        points_encoded, point_kwargs,
        context_encoded, gml_kwargs,
        point_pretraining_out = None,
    ):
        # Compute the point-level pre-training objective
        if point_pretraining_out is None:
            point_pretraining_out = self.point_pretraining_head(
                points_encoded, point_kwargs
            )

        # Compute the graph metric-learning pre-training objective
        gml_out = self.gml_head(
            points_encoded, context_encoded, gml_kwargs
        )

        point_pretraining_loss = point_pretraining_out['loss']
        gml_loss = gml_out['loss']

        return (
            point_pretraining_loss * self.point_weight + gml_loss * self.gml_weight,
            point_pretraining_out,
            gml_out,
        )


    def step(self, batch):
        point_kwargs, gml_kwargs = batch

        forward_out = self.forward(batch)
        if len(forward_out) == 2:
            points_encoded, context_encoded = forward_out
            point_pretraining_out = None
        elif len(forward_out) == 3:
            points_encoded, context_encoded, point_pretraining_out = forward_out

        loss, point_pretraining_out, gml_out = self.loss(
            points_encoded, point_kwargs,
            context_encoded, gml_kwargs,
            point_pretraining_out,
        )

        logs = {
            "loss": loss,
            "point_pretraining_loss": point_pretraining_out['loss'],
            "graph_metric_learning_loss": gml_out['loss'],
        }

        logs.update(point_pretraining_out['metrics'])
        logs.update(gml_out['metrics'])

        return logs


    def training_step(self, batch, _):
        logs = self.step(batch)
        for k, v in logs.items():
            self.log(
                f'Train/{k}', v, on_step=True, on_epoch=self.do_log_on_epoch, prog_bar=True, logger=True,
                sync_dist=True
            )
        return logs['loss']


    # This is commented out as don't currently use a pre-training validation step.
    #def validation_step(self, batch, _):
    #    logs = self.step(batch)
    #    for k, v in logs.items():
    #        self.log(
    #            f'Val/{k}', v, on_step=True, on_epoch=self.do_log_on_epoch, prog_bar=True, logger=True,
    #            sync_dist=True
    #        )
    #    return logs['loss']


    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)


    def epoch_end(self, outputs, stage_str):
        # self.custom_histogram_adder()
        pass


    def training_epoch_end(self, outputs):
        return self.epoch_end(outputs, 'Train')


    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, 'Valid')


    def configure_optimizers(self):
        #TODO: LR finder, scheduler?
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class FinetuningModule(LightningModule):
    def __init__(
        self,
        encoder_module = None,
        finetuning_head = None,
        lr = 1e-4,
        do_head_only = False,
        do_freeze_encoder = False,
        metric_name = '',
        do_schedule = False,
        num_training_steps = 0,
        warmup_frac = 0.2,
        do_from_plus = False,
    ):
        super().__init__()
        self.encoder_module = encoder_module
        self.finetuning_head = finetuning_head
        self.lr = lr
        self.do_head_only = do_head_only
        self.do_freeze_encoder = do_freeze_encoder
        self.metric_name = metric_name
        self.do_schedule = do_schedule
        self.num_training_steps = num_training_steps
        self.warmup_frac = warmup_frac
        self.do_from_plus = do_from_plus

    def forward(
        self,
        batch
    ):
        points_encoded = self.encoder_module(
            batch['input_ids'],
            batch['input_mask']
        )
        return points_encoded

    def loss(
        self,
        points_encoded,
        batch,
    ):
        finetune_out = self.finetuning_head(
            points_encoded, batch
        )

        loss = finetune_out['loss']

        return (
            loss,
            finetune_out,
        )

    def step(self, batch):
        if self.do_from_plus:
            finetune_out = self.encoder_module(**batch)
            loss = finetune_out['loss']

        else:
            if self.do_head_only:
                points_encoded = None
            else:
                points_encoded = self.forward(batch)

            loss, finetune_out = self.loss(
                points_encoded,
                batch,
            )

        logs = {
            "loss": loss,
            "predictions": finetune_out["predictions"],
            "targets": finetune_out["targets"]
        }

        logs.update(finetune_out['metrics'])

        return logs

    def training_step(self, batch, _):
        logs = self.step(batch)
        for k, v in logs.items():
            if k == 'predictions' or k == 'targets': continue
            self.log(
                f'Train/{k}', v, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                sync_dist=True
            )
        return logs

    def validation_step(self, batch, _):
       logs = self.step(batch)
       for k, v in logs.items():
           if k == 'predictions' or k == 'targets': continue
           self.log(
               f'Val/{k}', v, prog_bar=True, logger=True, sync_dist=True
           )
       return logs

    def test_step(self, batch, _):
        logs = self.step(batch)
        return logs

    def custom_histogram_adder(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def aggregate_outputs(self, outputs):
        agg_metrics = ('spearmanr', 'accuracy',)
        targets = [el['targets'] for el in outputs]
        if self.metric_name in agg_metrics: targets = np.concatenate(targets)

        predictions = [el['predictions'] for el in outputs]
        if self.metric_name in agg_metrics:  predictions = np.concatenate(predictions)
        return targets, predictions

    def test_epoch_end(self, outputs):
        targets, predictions = self.aggregate_outputs(outputs)
        self.test_results = {
            'targets': targets,
            'predictions': predictions,
        }

    def configure_optimizers(self):
        if self.do_freeze_encoder:
            for param in self.encoder_module.parameters(): param.requires_grad = False
            self.encoder_module.eval()
            opt_params = self.finetuning_head.parameters()
        else:
            opt_params = self.parameters()

        optimizer = optim.Adam(opt_params, lr=self.lr)

        if self.do_schedule:
            num_warmup_steps = int(self.num_training_steps * self.warmup_frac)
            scheduler = {
                'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps, self.num_training_steps),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1
            }

            return [optimizer], [scheduler]
        else:
            return optimizer
