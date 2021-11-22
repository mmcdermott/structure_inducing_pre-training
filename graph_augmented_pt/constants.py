import logging, os

PROJECTS_BASE = os.environ['PROJECTS_BASE'] if 'PROJECTS_BASE' in os.environ else '/crimea'
PROJECT_NAME = 'graph_augmented_pt'
PROJECT_ROOT = os.path.join(PROJECTS_BASE, PROJECT_NAME)
assert os.path.isdir(PROJECT_ROOT), "Project root (%s) doesn't exist!" % PROJECT_ROOT

FINETUNE_ARGS_FILENAME = 'finetune_args.json'
PRETRAIN_ARGS_FILENAME = 'pretrain_args.json'

RAW_DATASETS_DIR = os.path.join(PROJECT_ROOT, 'raw_datasets')
RUNS_DIR = os.path.join(PROJECT_ROOT, 'runs')
RAW_MODELS_DIR = os.path.join(PROJECT_ROOT, 'raw_models')

SPECIES_1840 = RAW_DATASETS_DIR + '/treeoflife/species_files/1840_species.txt'

POINT_ENCODER = 'point'

ALL_RANK = 'all_rank_eval'

SYNTHETIC = 'synthetic'
PROTEINS = 'proteins'
MAG = 'mag'
TAPE = 'tape'
RANK = 'rank'
SCIBERT = 'scibert'

TAPE_RH = 'remote_homology'
TAPE_FL = 'fluorescence'
TAPE_ST = 'stability'
TAPE_SS = 'secondary_structure'

SCIBERT_MAG = 'mag'
SCIBERT_SCI_CITE = 'sci-cite'
SCIBERT_CITATION_INTENT = 'citation_intent'
SCIBERT_OGBN_MAG = 'ogbn-mag'

SCIBERT_SCIVOCAB_UNCASED = 'allenai/scibert_scivocab_uncased'
CS_ROBERTA_BASE = 'allenai/cs_roberta_base'
BM_ROBERTA_BASE = 'allenai/biomed_roberta_base'

MULTISIMILARITY = 'multisimilarity'
DISTANCE_WEIGHTED_TRIPLET = 'distance_triplet'

# TODO: Consider renaming these to reflect their usage?
COSINE_DISTANCE = 'cosine'
EUCLIDEAN_DISTANCE = 'euclidean'

TAPE_TASKS = (TAPE_RH, TAPE_FL, TAPE_ST, TAPE_SS)
SCIBERT_TASKS = (SCIBERT_MAG, SCIBERT_SCI_CITE, SCIBERT_CITATION_INTENT, SCIBERT_OGBN_MAG)
RANKING_TASKS = (COSINE_DISTANCE, EUCLIDEAN_DISTANCE)

ONE_MINUS_EPSILON = 0.99999999
