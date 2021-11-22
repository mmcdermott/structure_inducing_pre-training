import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging, inspect

from graph_augmented_pt.args import *
from graph_augmented_pt.run.run_protein_pretrain import *
from graph_augmented_pt.run.run_mag_pretrain import *

logger = logging.getLogger(__name__)

def run_pretrain():
    args = PretrainArgs.from_commandline()
    arg_dict = vars(args)

    if args.dataset_type == PROTEINS:     pretrain_fn = run_protein_pretrain
    elif args.dataset_type == MAG:          pretrain_fn = run_mag_pretrain
    else: raise NotImplementedError

    # TODO: Need to change to allow reload, possibly overwrite.
    if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)

    args.to_json_file(os.path.join(args.output_dir, PRETRAIN_ARGS_FILENAME))

    arg_names = inspect.getfullargspec(pretrain_fn).args

    missing = set(arg_names) - set(arg_dict.keys())
    extra = set(arg_dict.keys()) - set(arg_names)
    if missing:
        raise RuntimeError(f"Missing arguments: {missing}")
    if extra:
        logger.warning(f"Extra arguments that would be ignored: {extra}")
        for key in extra:
            del arg_dict[key]

    pretrain_fn(**arg_dict)

if __name__ == '__main__':
    run_pretrain()
