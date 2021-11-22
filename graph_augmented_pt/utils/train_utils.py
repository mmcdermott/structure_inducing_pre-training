import os, time, torch
from pytorch_lightning.callbacks import Callback
from ..constants import *

def save_pl_module(trainer, pl_module, save_dir):
    os.mkdir(save_dir)

    # trainer.save_checkpoint(save_dir / 'model.ckpt')

    torch.save(
        pl_module.encoder_module.state_dict(), 
        save_dir / 'encoder.pt'
    )

    torch.save(
        pl_module.point_pretraining_head.state_dict(), 
        save_dir / 'point_pretraining_head.pt'
    )

    torch.save(
        pl_module.context_encoder_module.state_dict(), 
        save_dir / 'context_encoder_module.pt'
    )

    torch.save(
        pl_module.gml_head.state_dict(), 
        save_dir / 'gml_head.pt'
    )

class SaveCallback(Callback):
    def __init__(
        self,
        save_freq,
        sample_dir,
        start,
        offset=0,
    ):
        super().__init__()
        self.save_freq = save_freq
        self.sample_dir = sample_dir
        self.start = start
        self.offset = offset

        self.prev_step = -1   # Used to make grad_accum > 1 work.

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        global_step = trainer.global_step
        if pl_module.local_rank != 0: return
        if global_step % self.save_freq != 0: return
        if global_step == self.prev_step: return

        batches_dir = self.sample_dir / 'batches'
        if not batches_dir.exists(): os.mkdir(batches_dir)
        save_dir = batches_dir / str(global_step + self.offset)
        save_pl_module(trainer, pl_module, save_dir)
        end = time.time()
        elapsed = round((end - self.start) / 3600, 2)
        with open(save_dir / 'elapsed.txt', 'w') as f:
            f.write(str(elapsed) + ' hours elapsed.')
        
        self.prev_step = global_step

def save_ft_module(pl_module, save_dir):
    if not(save_dir).exists(): os.mkdir(save_dir)
    if pl_module.encoder_module is not None:
        torch.save(
            pl_module.encoder_module.state_dict(), 
            save_dir / 'encoder.pt'
        )

    torch.save(
        pl_module.finetuning_head.state_dict(), 
        save_dir / 'finetuning_head.pt'
    )

class FinetuneSaveCallback(Callback):
    def __init__(
        self,
        sample_dir,
        start,
        save_freq
    ):
        super().__init__()
        self.sample_dir = sample_dir
        self.start = start
        self.save_freq = save_freq

    def on_epoch_start(self, trainer, pl_module):
        if pl_module.local_rank != 0: return
        epoch = trainer.current_epoch
        if epoch % self.save_freq != 0: return

        epochs_dir = self.sample_dir / 'epochs'
        if not epochs_dir.exists(): os.mkdir(epochs_dir)
        save_dir = epochs_dir / str(epoch)
        save_ft_module(pl_module, save_dir)

        end = time.time()
        elapsed = round((end - self.start) / 3600, 2)
        with open(save_dir / 'elapsed.txt', 'w') as f:
            f.write(str(elapsed) + ' hours elapsed.')

# Keep classification logits from the fresholy initialized model.
# The cls logits from the saved model won't have the right shape.
def filter_keys_for_plus(saved_dict, fresh_dict, task):
    patterns_to_remove = ['sequence_model.cls']
    if task == TAPE_SS: patterns_to_remove.append('pos_embed')

    del_keys = set()
    for k in saved_dict:
        for p in patterns_to_remove:
            if p in k:
                del_keys.add(k)
                continue

    for k in del_keys: del saved_dict[k]
    saved_dict.update({k: fresh_dict[k] for k in del_keys})
    return saved_dict
    