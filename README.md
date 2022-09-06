# Structure Inducing Pre-Training

## Directory Structure:
Our code expects that there exists a root directory, stored in an environment variable named `PROJECTS_BASE`
which will in turn contain a directory named `graph_augmented_pt` which stores the raw & processed data for
these experiments, as well as the eventual directories in which run checkpoints and result files will be
stored.

Within `PROJECTS_BASE`, you must have a directories `raw_datasets/treeoflife`, `raw_datasets/strings` (for our
protein PT experiments), `raw_datasets/TAPE` (for protein FT), `raw_datasets/ogbn_mag` (for scientific
articles PT), and `raw_datasets/scibert` (for scientific articles FT).

## Datasets & Pre-trained Models:
### Pre-training Datasets:
The `treeoflife` dataset can be obtained from the [Stanford SNAP
page](https://snap.stanford.edu/tree-of-life/data.html). The OGBN-MAG graph can be obtained from the OGB,
[here](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag), and the associated abstracts can be downloaded from
the [MAG](https://www.microsoft.com/en-us/research/project/open-academic-graph/).

Networks PT Datasets can be accessed via [the original github](https://github.com/snap-stanford/pretrain-gnns)

### Fine-tuning Datasets:
TAPE datasets can be obtained according to the directions in the
[TAPE Paper](https://proceedings.neurips.cc/paper/2019/file/37f65c068b7723cd7809ee2d31d7861c-Paper.pdf). Each
TAPE FT dataset should be stored within a separate directory in `PROJECTS_BASE/raw_datasets/TAPE`.
SciBERT Datasets can be obtained according to the directions in the
[SciBERT Paper](https://www.aclweb.org/anthology/D19-1371.pdf) (simply clone the entire SciBERT github into the `raw_datasets/scibert` folder).
The Networks dataset access instructions above include both PT and FT data.

### Initializing Models:
Follow the instructions in the [TAPE Repository](https://github.com/songlab-cal/tape) to obtain the initial
TAPE pre-trained model. SciBERT can be obtained directly via
[huggingface](https://huggingface.co/allenai/scibert_scivocab_uncased). The PLUS model can be obtained via the
instructions [here](https://github.com/mswzeus/PLUS/). Note that the PLUS model's base files must be
downloaded and stored in a `raw_models` subdirectory of `PROJECTS_BASE`.

### Synthetic Data:
The dump of sentences from wikipedia used as node features in our synthetic experiments can be downloaded
[here](https://www.kaggle.com/mikeortman/wikipedia-sentences?select=wikisent2.txt). It should be placed in the
directory `PROJECTS_BASE/synthetic_datasets/`. Additional Synthetic dataset processing should be run via the
notebook `synthetic_experiments/'Generate Synthetic Data Node Features & Topics.ipynb'`. 
For the manifolds experiments, you must additionally run `Preprocessing Topics for Simplicial
Alignment.ipynb`.

## Companion Code
### Networks
You need to download [this forked repo](https://anonymous.4open.science/r/pretrain-gnns-C81E/README.md) and
place it in the appropriate dir (see `constants.py` for the dir).

## Environment Setup
### Required Hardware
This code (and the associated environment files) are built for systems with access to a GPU capable of running CUDA 10.1. It is likely that alternative versions of this environment can be produced for CPU only systems, but these would need to be manually investigated. 

### Installing Packages
  1. Navigate to the root directory. Decide where you want to store your output dir: `export
     OUTPUT_ENV_PATH=[INSERT YOUR PATH HERE]`.
  2. Install the base conda environment: `conda env create -f conda.yml -p $OUTPUT_ENV_PATH`
  3. If the process completes successfully, something weird had happened but just go with it. If the process
     complains about a non-pip-related issue and rolls back the transaction, something else is wrong, and I
     don't know how to solve it. If the process fails with a pip-installation-error, continue below.
  3. To install the broken pip dependencies, first activate the partial conda env: `conda activate
     $OUTPUT_ENV_PATH`
  4. Next, install the pip dependencies:
```
$OUTPUT_ENV_PATH/bin/pip install tape_proteins
$OUTPUT_ENV_PATH/bin/pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
export TORCH="1.7.0"; export CUDA="cu101"
$OUTPUT_ENV_PATH/bin/pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
$OUTPUT_ENV_PATH/bin/pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
$OUTPUT_ENV_PATH/bin/pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
$OUTPUT_ENV_PATH/bin/pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
$OUTPUT_ENV_PATH/bin/pip install torch-geometric
$OUTPUT_ENV_PATH/bin/pip install transformers
$OUTPUT_ENV_PATH/bin/pip install torchmetrics
$OUTPUT_ENV_PATH/bin/pip install pygraphviz --global-option=build_ext --global-option="-I$OUTPUT_ENV_PATH/include/" --global-option="-L$OUTPUT_ENV_PATH/lib/"
```

## Synthetic Experiment Reproduction Instructions
  1. Follow the instructions above to obtain and pre-process the synthetic data.
  2. Follow the various `'Reproduction *.ipynb'` notebooks in the `synthetic_experiments` directory.
  
## Pre-training & Fine-tuning Runs
  1. To run these experiments, first make a directory for your run and add an argument configuration json file in line with `graph_augmented_pt/args.py`
  2. Then, run the `run_pretraining.py` script pointing at those arguments.
  3. To fine-tune, run the `run_finetuning.py` script.

## For running via scripts
For some reason, scripts continue to lookup in the system shared library locations rather than the conda envs.
To circumvent this (and for example allow tape to be imported without getting hit with a
`/usr/lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.22'` error, you can run 
`export LD_LIBRARY_PATH="$OUTPUT_ENV_PATH/lib"` before in the shell before launching
anything.

# License
This code is licensed under an MIT license. See [LICENSE.md](https://github.com/mmcdermott/structure_inducing_pre-training/blob/main/LICENSE.md) for more details.
