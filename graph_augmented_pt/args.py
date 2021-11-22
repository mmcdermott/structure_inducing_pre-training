import argparse, json, os, pickle
from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass, asdict

from .constants import *

class BaseArgs(ABC):
    @classmethod
    def from_json_file(cls, filepath):
        with open(filepath, mode='r') as f: return cls(**json.loads(f.read()))
    @staticmethod
    def from_pickle_file(filepath):
        with open(filepath, mode='rb') as f: return pickle.load(f)

    def to_dict(self): return asdict(self)
    def to_json_file(self, filepath):
        with open(filepath, mode='w') as f: f.write(json.dumps(asdict(self), indent=4))
    def to_pickle_file(self, filepath):
        with open(filepath, mode='wb') as f: pickle.dump(self, f)

    @classmethod
    @abstractmethod
    def _build_argparse_spec(cls, parser):
        raise NotImplementedError("Must overwrite in base class!")

    @staticmethod
    def add_bool_arg(parser, arg, default, help_msg='', required=False):
        """
        Adds a copy of `arg` and `no_{arg}` to the parser.
        """
        assert arg.startswith("do_"), "Arg should be of the form do_*! Got %s" % arg
        do_arg, no_do_arg = "--%s" % arg, "--no_%s" % arg
        parser.add_argument(
            do_arg, action='store_true', dest=arg, help=help_msg, default=default, required=required
        )
        parser.add_argument(no_do_arg, action='store_false', dest=arg)

    @classmethod
    def from_commandline(cls):
        parser = argparse.ArgumentParser()

        # To load from a output_directory (not synced to overall structure above):
        parser.add_argument(
            "--do_load_from_dir", action='store_true',
            help="Should the system reload from the sentinel args.json file in the specified run directory "
                 "(--output_dir) and use those args rather than consider those set here? If so, no other args "
                 "need be set (they will all be ignored).",
            default=False
        )

        main_dir_arg, args_filename = cls._build_argparse_spec(parser)

        args = parser.parse_args()

        if args.do_load_from_dir:
            load_dir = vars(args)[main_dir_arg]
            assert os.path.exists(load_dir), "Dir (%s) must exist!" % load_dir
            args_path = os.path.join(load_dir, args_filename)
            assert os.path.exists(args_path), "Args file (%s) must exist!" % args_path

            return cls.from_json_file(args_path)

        args_dict = vars(args)
        if 'do_load_from_dir' in args_dict: args_dict.pop('do_load_from_dir')

        return cls(**args_dict)

def intlt(bounds):
    start, end = bounds if type(bounds) is tuple else (0, bounds)
    def fntr(x):
        x = int(x)
        if x < start or x >= end: raise ValueError("%d must be in [%d, %d)" % (x, start, end))
        return x
    return fntr

def remap(options):
    def fntr(x):
        if x in options.values(): return x
        if x in options.keys(): return options[x]
        raise ValueError("%s is not a valid option: %s" % (str(x), str(options)))
    return fntr

@dataclass
class PretrainArgs(BaseArgs):
    output_dir:                     str 
    dataset_type:                   str = SYNTHETIC
    seed:                           int = 42
    do_noised:                      bool = False
    noise_rate:                     float = 0.05
    species:                        str = SPECIES_1840
    gml_weight:                     float = 1.0
    gml_encoder:                    str = POINT_ENCODER
    gml_head:                       str = COSINE_DISTANCE
    contrastive_margin:             float = 1e1
    pos_weight:                     float = 1
    neg_weight:                     float = 1
    thresh:                         float = 0.1
    ego_graph_radius:               int = 3
    encoder_hidden_size:            int = 10
    encoder_num_hidden_layers:      int = 2
    encoder_num_attention_heads:    int = 2
    encoder_intermediate_size:      int = 8
    context_encoder_hidden_size:    int = 10
    context_encoder_hidden_layers:  int = 2
    max_seq_len:                    int = 4096
    grad_accum:                     int = 1
    batch_size:                     str = 'ALL'
    lr:                             float = 3e-4
    num_epochs:                     int = 2500
    save_freq:                      int = 100
    num_samples:                    int = 25
    do_log_on_epoch:                bool = True
    do_half:                        bool = False
    do_ddp:                         bool = False
    gpus:                           int = 1
    print_first:                    int = 0
    show_first:                     int = 0
    weights_summary:                str = 'top'
    do_simple_profiler:             bool = False
    do_advanced_profiler:           bool = False
    neighbor_prediction_eval_freq:  int  = 100
    do_neighbor_prediction_eval:    bool = True
    min_sample_nodes:               int = 50
    eval_batch_size:                int = 32
    do_from_tape:                   bool = False
    do_from_scibert:                bool = False
    do_from_plus:                   bool = False
    hf_model_name:                  str = SCIBERT_SCIVOCAB_UNCASED
    do_debug_graph:                 bool = False
    do_normalize_embeds:            bool = False

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument('--output_dir', 
            type=str)
        parser.add_argument('--dataset_type', 
            type=str, default=SYNTHETIC)
        parser.add_argument('--seed', 
            type=int, default=42)
        cls.add_bool_arg(parser, 'do_noised', default=False)
        parser.add_argument('--noise_rate', 
            type=float, default=0.05)
        parser.add_argument('--species', 
            type=str, default=SPECIES_1840)
        parser.add_argument('--gml_weight', 
            type=float, default=1.0)
        parser.add_argument('--gml_encoder', 
            type=str, default=POINT_ENCODER)
        parser.add_argument('--gml_head', 
            type=str, default=COSINE_DISTANCE)
        parser.add_argument('--contrastive_margin', 
            type=float, default=1e1)
        parser.add_argument('--pos_weight', 
            type=float, default=1)
        parser.add_argument('--neg_weight', 
            type=float, default=1)
        parser.add_argument('--thresh', 
            type=float, default=0.1)
        parser.add_argument('--ego_graph_radius', 
            type=int, default=3)
        parser.add_argument('--encoder_hidden_size', 
            type=int, default=10)
        parser.add_argument('--encoder_num_hidden_layers', 
            type=int, default=2)
        parser.add_argument('--encoder_num_attention_heads', 
            type=int, default=2)
        parser.add_argument('--encoder_intermediate_size', 
            type=int, default=8)
        parser.add_argument('--context_encoder_hidden_size', 
            type=int, default=10)
        parser.add_argument('--context_encoder_hidden_layers', 
            type=int, default=2)
        parser.add_argument('--max_seq_len', 
            type=int, default=4096)
        parser.add_argument('--grad_accum', 
            type=int, default=1)
        parser.add_argument('--batch_size', 
            type=str, default='ALL')
        parser.add_argument('--lr', 
            type=float, default=3e-4)
        parser.add_argument('--num_epochs', 
            type=int, default=2500)
        parser.add_argument('--save_freq', 
            type=int, default=100)
        parser.add_argument('--num_samples', 
            type=int, default=25)
        cls.add_bool_arg(parser, 'do_log_on_epoch', default=True)
        cls.add_bool_arg(parser, 'do_half', default=False)
        cls.add_bool_arg(parser, 'do_ddp', default=False)
        parser.add_argument('--gpus', 
            type=int, default=1)
        parser.add_argument('--print_first', 
            type=int, default=0)
        parser.add_argument('--show_first', 
            type=int, default=0)
        parser.add_argument('--weights_summary', 
            type=str, default='top')
        cls.add_bool_arg(parser, 'do_simple_profiler', default=False)
        cls.add_bool_arg(parser, 'do_advanced_profiler', default=False)
        parser.add_argument('--neighbor_prediction_eval_freq', 
            type=int, default=100)
        cls.add_bool_arg(parser, 'do_neighbor_prediction_eval', default=True)
        parser.add_argument('--min_sample_nodes', 
            type=int, default=50)
        parser.add_argument('--eval_batch_size', 
            type=int, default=32)
        cls.add_bool_arg(parser, 'do_from_tape', default=False)
        cls.add_bool_arg(parser, 'do_from_scibert', default=False)
        cls.add_bool_arg(parser, 'do_from_plus', default=False)
        parser.add_argument('--hf_model_name', 
            type=str, default=SCIBERT_SCIVOCAB_UNCASED)
        cls.add_bool_arg(parser, 'do_debug_graph', default=False)
        cls.add_bool_arg(parser, 'do_normalize_embeds', default=False)

        return 'output_dir', PRETRAIN_ARGS_FILENAME


@dataclass
class FinetuneArgs(BaseArgs):
    output_dir:                     str 
    dataset_type:                   str = TAPE
    task:                           str = TAPE_RH
    eval_split:                     str = 'test'
    seed:                           int = 42
    encoder_hidden_size:            int = 10
    encoder_num_hidden_layers:      int = 2
    encoder_num_attention_heads:    int = 2
    encoder_intermediate_size:      int = 8
    contrastive_margin:             float = 1e1
    max_seq_len:                    int = 4096
    grad_accum:                     int = 1
    train_set_frac:                 float = 1.0
    do_upsample:                    bool = False
    train_batch_size:               int = 32
    eval_batch_size:                int = 32
    lr:                             float = 1e-4
    warmup_frac:                    float = 0.2
    num_epochs:                     int = 10
    patience:                       int = 10
    num_samples:                    int = 3
    do_half:                        bool = False
    do_ddp:                         bool = False
    gpus:                           int = 1
    hf_model_name:                  str = SCIBERT_SCIVOCAB_UNCASED
    weights_summary:                str = 'top'
    do_simple_profiler:             bool = False
    do_advanced_profiler:           bool = False
    do_freeze_encoder:              bool = False
    do_eval_only:                   bool = False
    do_from_plus:                   bool = False

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument('--output_dir', 
            type=str)
        parser.add_argument('--dataset_type', 
            type=str, default=TAPE)
        parser.add_argument('--task', 
            type=str, default=TAPE_RH)
        parser.add_argument('--eval_split', 
            type=str, default='test')
        parser.add_argument('--seed', 
            type=int, default=42)
        parser.add_argument('--encoder_hidden_size', 
            type=int, default=10)
        parser.add_argument('--encoder_num_hidden_layers', 
            type=int, default=2)
        parser.add_argument('--encoder_num_attention_heads', 
            type=int, default=2)
        parser.add_argument('--encoder_intermediate_size', 
            type=int, default=8)
        parser.add_argument('--contrastive_margin', 
            type=float, default=1e1)
        parser.add_argument('--max_seq_len', 
            type=int, default=4096)
        parser.add_argument('--grad_accum', 
            type=int, default=1)
        parser.add_argument('--train_set_frac', 
            type=float, default=1.0)
        cls.add_bool_arg(parser, 'do_upsample', default=False)
        parser.add_argument('--train_batch_size', 
            type=int, default=32)
        parser.add_argument('--eval_batch_size', 
            type=int, default=32)
        parser.add_argument('--lr', 
            type=float, default=1e-4)
        parser.add_argument('--warmup_frac', 
            type=float, default=0.2)
        parser.add_argument('--num_epochs', 
            type=int, default=10)
        parser.add_argument('--patience', 
            type=int, default=10)
        parser.add_argument('--num_samples', 
            type=int, default=3)
        cls.add_bool_arg(parser, 'do_half', default=False)
        cls.add_bool_arg(parser, 'do_ddp', default=False)
        parser.add_argument('--gpus', 
            type=int, default=1)
        parser.add_argument('--hf_model_name', 
            type=str, default=SCIBERT_SCIVOCAB_UNCASED)
        parser.add_argument('--weights_summary', 
            type=str, default='top')
        cls.add_bool_arg(parser, 'do_simple_profiler', default=False)
        cls.add_bool_arg(parser, 'do_advanced_profiler', default=False)
        cls.add_bool_arg(parser, 'do_freeze_encoder', default=False)
        cls.add_bool_arg(parser, 'do_eval_only', default=False)
        cls.add_bool_arg(parser, 'do_from_plus', default=False)

        return 'output_dir', FINETUNE_ARGS_FILENAME

# Default hyperparameters are determined from https://arxiv.org/pdf/1905.12265.pdf, Section 5.2 &
@dataclass
class GraphPTPretrainArgs(BaseArgs):
    output_dir:                     str
    seed:                           int = 42
    gml_weight:                     float = 1.0
    point_weight:                   float = 1.0
    gml_encoder:                    str = POINT_ENCODER
    gml_head:                       str = COSINE_DISTANCE
    contrastive_margin:             float = 1e1
    pos_weight:                     float = 1
    neg_weight:                     float = 1
    thresh:                         float = 0.1

    encoder_emb_dim:                int   = 300,    # Set
    encoder_hidden_size:            int   = 300,    # TODO -- Appendix A and main body disagree!
    encoder_num_layer:              int   = 5,
    encoder_graph_pooling:          str   = 'mean',
    encoder_JK:                     str   = 'last', # TODO --- this seems like it is right based on the code?
    encoder_dropout_ratio:          float = 0.1,    # Set --- original paper used 20\% for supervised PT.
    encoder_do_masking:             bool  = True,   # Set
    encoder_do_context_pred:        bool  = False,  # Set
    encoder_do_mask_edge:           bool  = False,  # Set

    n_neighbors:                    Optional[int] = None,
    max_n_neighbors:                int = 50,
    radius:                         float = 1.5,
    mask_rate:                      float = 0.15,
    do_mask_edge:                   bool = False,
    do_masking:                     bool = True,
    do_multitask_PT:                bool = False,

    grad_accum:                     int = 1
    batch_size:                     str = 'ALL'
    lr:                             float = 3e-4
    num_epochs:                     int = 2500
    save_freq:                      int = 100
    num_samples:                    int = 25
    do_log_on_epoch:                bool = True
    do_half:                        bool = False
    do_ddp:                         bool = False
    gpus:                           int = 1
    print_first:                    int = 0
    show_first:                     int = 0
    weights_summary:                str = 'top'
    do_simple_profiler:             bool = False
    do_advanced_profiler:           bool = False
    min_sample_nodes:               int = 50
    eval_batch_size:                int = 32
    do_debug_graph:                 bool = False
    do_normalize_embeds:            bool = False

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument('--output_dir',
            type=str)
        parser.add_argument('--seed',
            type=int, default=42)
        parser.add_argument('--gml_weight',
            type=float, default=1.0)
        parser.add_argument('--point_weight',
            type=float, default=1.0)
        parser.add_argument('--gml_encoder',
            type=str, default=POINT_ENCODER)
        parser.add_argument('--gml_head',
            type=str, default=COSINE_DISTANCE)
        parser.add_argument('--contrastive_margin',
            type=float, default=1e1)
        parser.add_argument('--pos_weight',
            type=float, default=1)
        parser.add_argument('--neg_weight',
            type=float, default=1)
        parser.add_argument('--thresh',
            type=float, default=0.1)

        parser.add_argument('--encoder_emb_dim',
            type=int, default = 300,    # Set
        )
        parser.add_argument('--encoder_hidden_size',
            type=int, default = 300,    # TODO -- Appendix A and main body disagree!
        )
        parser.add_argument('--encoder_num_layer',
            type=int, default = 5,
        )
        parser.add_argument('--encoder_graph_pooling', type=str, default = 'mean',)
        parser.add_argument('--encoder_JK',
            type=str, default = 'last', # TODO --- this seems like it is right based on the code?
        )
        parser.add_argument('--encoder_dropout_ratio',
            type=float, default = 0.1,    # Set --- original paper used 20\% for supervised PT.
        )
        parser.add_argument('--encoder_do_masking',
            type=bool, default = True,   # Set
        )
        parser.add_argument('--encoder_do_context_pred',
            type=bool, default = False,  # Set
        )
        parser.add_argument('--encoder_do_mask_edge',
            type=bool, default = False,  # Set
        )

        parser.add_argument('--n_neighbors',
            type=int, default = None,
        )
        parser.add_argument('--max_n_neighbors',
            type=int, default = 50,
        )
        parser.add_argument('--radius',
            type=float, default=1.5,
        )
        parser.add_argument('--mask_rate',
            type=float, default=0.15,
        )
        cls.add_bool_arg(parser, 'do_mask_edge', default=False)
        cls.add_bool_arg(parser, 'do_masking', default=True)
        cls.add_bool_arg(parser, 'do_multitask_PT', default=False)

        parser.add_argument('--grad_accum',
            type=int, default=1)
        parser.add_argument('--batch_size',
            type=str, default='ALL')
        parser.add_argument('--lr',
            type=float, default=3e-4)
        parser.add_argument('--num_epochs',
            type=int, default=2500)
        parser.add_argument('--save_freq',
            type=int, default=100)
        parser.add_argument('--num_samples',
            type=int, default=25)
        cls.add_bool_arg(parser, 'do_log_on_epoch', default=True)
        cls.add_bool_arg(parser, 'do_half', default=False)
        cls.add_bool_arg(parser, 'do_ddp', default=False)
        parser.add_argument('--gpus',
            type=int, default=1)
        parser.add_argument('--print_first',
            type=int, default=0)
        parser.add_argument('--show_first',
            type=int, default=0)
        parser.add_argument('--weights_summary',
            type=str, default='top')
        cls.add_bool_arg(parser, 'do_simple_profiler', default=False)
        cls.add_bool_arg(parser, 'do_advanced_profiler', default=False)
        parser.add_argument('--eval_batch_size',
            type=int, default=32)
        cls.add_bool_arg(parser, 'do_debug_graph', default=False)
        cls.add_bool_arg(parser, 'do_normalize_embeds', default=False)

        return 'output_dir', PRETRAIN_ARGS_FILENAME


@dataclass
class GraphPTFinetuneArgs(BaseArgs):
    output_dir:                     str
    eval_split:                     str   = 'test'
    seed:                           int   = 1
    task:                           str   = ''

    encoder_emb_dim:                int   = 300,    # Set
    encoder_hidden_size:            int   = 300,    # TODO -- Appendix A and main body disagree!
    encoder_num_layer:              int   = 5,
    encoder_graph_pooling:          str   = 'mean',
    encoder_JK:                     str   = 'last', # TODO --- this seems like it is right based on the code?
    encoder_dropout_ratio:          float = 0.1,    # Set --- original paper used 20\% for supervised PT.

    grad_accum:                     int = 1
    train_batch_size:               int = 32
    eval_batch_size:                int = 32
    lr:                             float = 1e-4
    warmup_frac:                    float = 0.2
    num_epochs:                     int = 10
    patience:                       int = 10
    num_samples:                    int = 3
    do_half:                        bool = False
    do_ddp:                         bool = False
    gpus:                           int = 1
    weights_summary:                str = 'top'
    do_simple_profiler:             bool = False
    do_advanced_profiler:           bool = False
    do_freeze_encoder:              bool = False
    do_eval_only:                   bool = False
    save_freq:                      int  = 100
    do_schedule:                    bool = False
    do_run_without_pretraining:     bool = False

    @classmethod
    def _build_argparse_spec(cls, parser):
        parser.add_argument('--output_dir',
            type=str)
        parser.add_argument('--eval_split',
            type=str, default='test')
        parser.add_argument('--seed',
            type=int, default=1)

        parser.add_argument('--encoder_emb_dim',
            type=int, default = 300,    # Set
        )
        parser.add_argument('--encoder_hidden_size',
            type=int, default = 300,    # TODO -- Appendix A and main body disagree!
        )
        parser.add_argument('--encoder_num_layer',
            type=int, default = 5,
        )
        parser.add_argument('--encoder_graph_pooling', type=str, default = 'mean',)
        parser.add_argument('--encoder_JK',
            type=str, default = 'last', # TODO --- this seems like it is right based on the code?
        )
        parser.add_argument('--encoder_dropout_ratio',
            type=float, default = 0.1,    # Set --- original paper used 20\% for supervised PT.
        )

        parser.add_argument('--grad_accum',
            type=int, default=1)
        parser.add_argument('--train_batch_size',
            type=int, default=32)
        parser.add_argument('--eval_batch_size',
            type=int, default=32)
        parser.add_argument('--lr',
            type=float, default=1e-4)
        parser.add_argument('--warmup_frac',
            type=float, default=0.2)
        parser.add_argument('--num_epochs',
            type=int, default=10)
        parser.add_argument('--patience',
            type=int, default=10)
        parser.add_argument('--num_samples',
            type=int, default=3)
        cls.add_bool_arg(parser, 'do_half', default=False)
        cls.add_bool_arg(parser, 'do_ddp', default=False)
        parser.add_argument('--gpus',
            type=int, default=1)
        parser.add_argument('--weights_summary',
            type=str, default='top')
        cls.add_bool_arg(parser, 'do_simple_profiler', default=False)
        cls.add_bool_arg(parser, 'do_advanced_profiler', default=False)
        cls.add_bool_arg(parser, 'do_freeze_encoder', default=False)
        cls.add_bool_arg(parser, 'do_eval_only', default=False)
        cls.add_bool_arg(parser, 'do_schedule', default=False)
        cls.add_bool_arg(parser, 'do_run_without_pretraining', default=False)
        parser.add_argument('--save_freq',
            type=int, default=100)

        return 'output_dir', FINETUNE_ARGS_FILENAME
