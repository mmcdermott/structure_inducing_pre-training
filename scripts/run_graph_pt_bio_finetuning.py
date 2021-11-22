import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging, inspect

from graph_augmented_pt.args import GraphPTFinetuneArgs
from graph_augmented_pt.constants import *
from graph_augmented_pt.run.run_graph_pt_bio_finetune import run_graph_pt_finetune

logger = logging.getLogger(__name__)

def run_finetune():
    args = GraphPTFinetuneArgs.from_commandline()
    arg_dict = vars(args)

    if not os.path.isdir(args.output_dir): os.mkdir(args.output_dir)
    args.to_json_file(os.path.join(args.output_dir, FINETUNE_ARGS_FILENAME))

    arg_names = inspect.getfullargspec(run_graph_pt_finetune).args

    missing = set(arg_names) - set(arg_dict.keys()) - {'data_cls', 'singleton_fn'}
    extra = set(arg_dict.keys()) - set(arg_names)
    if missing:
        raise RuntimeError(f"Missing arguments: {missing}")
    if extra:
        logger.warning(f"Extra arguments that would be ignored: {extra}")
        for key in extra:
            del arg_dict[key]

    run_graph_pt_finetune(**arg_dict)

if __name__ == '__main__':
    run_finetune()
