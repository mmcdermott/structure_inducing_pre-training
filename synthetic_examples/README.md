# SPPT Synthetic Experiments
Datasets can be pre-processed via the `'Generate Synthetic Data Node Features & Topics.ipynb'` notebook.
For the manifolds experiments, you must additionally run `Preprocessing Topics for Simplicial
Alignment.ipynb`.  Experiments can then be run via the `Reproduction*.ipynb` notebooks. Note that these do not
perfectly reproduce the precise numbers in the version of the paper submitted for review, as they run
additional samples for fewer epochs which will offer slightly reduced performance but provide estimates of
variance as well, and a minor bug was discovered with how synthetic datasets were re-sampled in between runs
for the cliques with noise experiments which was corrected.
