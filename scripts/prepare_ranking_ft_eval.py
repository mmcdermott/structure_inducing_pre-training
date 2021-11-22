import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, torch
from pathlib import Path

from graph_augmented_pt.utils.utils import *
from graph_augmented_pt.utils.tensorboard_utils import *

"""
In an FT directory, creates a best_epoch/ subdirectory and writes the following files:
    finetuning_head.pt:     Best-performing head.
    epoch_num.txt:          Contains epoch number of best val performance.
"""


def main(
    output_dir:                 str,
    patience:                   int  = 25,
    num_epochs:                 int  = 200,
    do_upsample:                bool = False,
    train_set_frac:             float = 1.0,
):
    output_dir = Path(output_dir)
    best_epoch = get_best_epoch(output_dir)
    if do_upsample: best_epoch *= int(1 / train_set_frac)
    print(f'Best Epoch: {best_epoch}')

    best_dir = output_dir / 'best_epoch'
    assert not best_dir.exists(), 'Already computed best!'
    os.mkdir(best_dir)
    
    with open(best_dir / 'epoch_num.txt', 'w') as f:
        f.write(str(best_epoch) + '\n')

    epochs_dir = output_dir / 'epochs'
    copy_from = epochs_dir / str(best_epoch) / 'finetuning_head.pt'
    copy_to = best_dir
    COPY_CMD = f'cp {copy_from} {copy_to}'
    os.system(COPY_CMD)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', 
            type=str)
    parser.add_argument('--patience', 
            type=int, default=25)
    parser.add_argument('--num_epochs',
            type=int, default=200)
    parser.add_argument('--do_upsample',
            action='store_true')
    parser.add_argument('--train_set_frac',
                type=float, default=1.0)
    args = parser.parse_args()
    arg_dict = vars(args)

    main(**arg_dict)