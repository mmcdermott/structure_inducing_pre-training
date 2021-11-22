import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging, inspect

from graph_augmented_pt.args import *
from graph_augmented_pt.run.run_tape import *
from graph_augmented_pt.run.run_ranking import *
from graph_augmented_pt.run.run_scibert import *

logger = logging.getLogger(__name__)

def run_finetune():
    args = FinetuneArgs.from_commandline()
    arg_dict = vars(args)

    if args.dataset_type == TAPE:
        if args.do_eval_only:           finetune_fn = run_tape_eval
        else:                           finetune_fn = run_tape_finetune
    elif args.dataset_type == RANK:     finetune_fn = run_ranking_finetune
    elif args.dataset_type == SCIBERT:  finetune_fn = run_scibert_finetune
    else: raise NotImplementedError

    # TODO: Need to change to allow reload, possibly overwrite.
    if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)

    args.to_json_file(os.path.join(args.output_dir, FINETUNE_ARGS_FILENAME))

    arg_names = inspect.getfullargspec(finetune_fn).args

    missing = set(arg_names) - set(arg_dict.keys())
    extra = set(arg_dict.keys()) - set(arg_names)
    if missing:
        raise RuntimeError(f"Missing arguments: {missing}")
    if extra:
        logger.warning(f"Extra arguments that would be ignored: {extra}")
        for key in extra:
            del arg_dict[key]

    finetune_fn(**arg_dict)

if __name__ == '__main__':
    run_finetune()
