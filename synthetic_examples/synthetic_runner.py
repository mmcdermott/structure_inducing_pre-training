""" This file contains utilities for running synthetic experiments.

    TODO:
      1. Comments & Docstrings
"""

import sys
sys.path.append('..')

from graph_augmented_pt.datasets.pretraining_dataset import *
from graph_augmented_pt.run.run_singleton import *
from graph_augmented_pt.utils.eval_utils import *
from graph_augmented_pt.constants import *

import json, pickle, torch, matplotlib.pyplot as plt, matplotlib.lines as mlines, networkx as nx, numpy as np, time

from networkx.drawing.nx_agraph import graphviz_layout
from pathlib import Path

from pytorch_lightning.callbacks import Callback

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, label_ranking_average_precision_score

def plot_clusters(
    synthetic_dataset, embeddings, tsne_perplexity=5, label_sets={}, save_path=None,
    show_legend=True, do_show_plots=False, W = 5,
):
    G = synthetic_dataset.G
    if not synthetic_dataset.has_pos: synthetic_dataset._compute_pos()
    pos = synthetic_dataset._pos

    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2, perplexity=tsne_perplexity)
    embeddings_tsne = tsne.fit_transform(embeddings)

    fig, axes_rows = plt.subplots(nrows=len(label_sets), ncols=3, figsize=(W*3, W*len(label_sets)))
    if len(label_sets) == 1: axes_rows = [axes_rows]

    COLORS = [
        'r', 'c', 'g', 'y', 'purple', 'b', 'brown', 'orange', 'gray', 'pink',
        '#88A97B', '#819283', '#F47A2D', '#41768D', '#412332'
    ]

    for (name, labels), (graph_ax, pca_ax, tsne_ax) in zip(label_sets.items(), axes_rows):
        colors = [COLORS[l] for l in labels]
        shapes = ['o'] * len(labels)

        for i, (c, s) in enumerate(zip(colors, shapes)):
            nx.draw_networkx_nodes(G, nodelist=[i], ax=graph_ax, pos=pos, node_color=c, node_shape=s)
        nx.draw_networkx_edges(G, pos=pos, ax=graph_ax)
        nx.draw_networkx_labels(G, pos=pos, labels={i: str(i) for i in range(len(G))}, ax=graph_ax)

        legend_dict = {l: (c, s) for l, c, s in zip(labels, colors, shapes)}

        pca_ax.set_title('PCA')
        pca_ax.set_xlabel("PCA 1")
        pca_ax.set_ylabel("PCA 2")
        pca_ax.get_xaxis().set_ticks([])
        pca_ax.get_yaxis().set_ticks([])
        pca_ax.get_xaxis().set_ticklabels([])
        pca_ax.get_yaxis().set_ticklabels([])

        X, Y = list(zip(*embeddings_pca))

        for i, (x, y, c, m) in enumerate(zip(X, Y, colors, shapes)):
            pca_ax.scatter([x], [y], c=c, marker=m)
            pca_ax.annotate(i, (x, y))

        tsne_ax.set_title('TSNE')
        tsne_ax.set_axis_off()
        X, Y = list(zip(*embeddings_tsne))
        for i, (x, y, c, m) in enumerate(zip(X, Y, colors, shapes)):
            tsne_ax.scatter([x], [y], c=c, marker=m)
            tsne_ax.annotate(i, (x, y))

        handles, labels = [], []
        for i, (c, s) in legend_dict.items():
            label = f"{name} {i}"
            handles.append(mlines.Line2D(
                [], [], color=c, marker=s, markersize=15, label=label, linestyle='None'
            ))
            labels.append(label)

        pca_ax.legend(
            handles=handles,
            labels=labels,
            loc='lower center',
            ncol=len(handles)
        )

    if do_show_plots: plt.show()

    plt.savefig(save_path / 'embeddings.png')
    with open(save_path / 'pca.pkl', mode='wb') as f:
        pickle.dump((pca, embeddings_pca), f)
    with open(save_path / 'tsne.pkl', mode='wb') as f:
        pickle.dump((tsne, embeddings_tsne), f)

def eval_clusters(
    synthetic_dataset, embeddings, label_sets={}, multilabel_sets={},
    do_print=True, do_plot=True, save_path=None, **plot_kwargs
):
    if type(embeddings) is torch.Tensor: embeddings = embeddings.cpu().data.numpy()
    quant_metrics = quantitative_representation_eval(embeddings, label_sets, multilabel_sets)

    if do_print:
        for t, td in quant_metrics.items():
            for name, (_, _, _, acc, auroc) in td.items():
                print(f"{t}/{name} Accuracy = {100*acc:.1f}%%, AUROC = {auroc:.2f}")

    res_dict = {t: {k: v[3:] for k, v in td.items()} for t, td in quant_metrics.items()}
    with open(save_path / 'results.json', mode='w') as f:
        f.write(json.dumps(res_dict, indent=2))
    with open(save_path / 'results.pkl', mode='wb') as f:
        pickle.dump(quant_metrics, f)
    with open(save_path / 'embeddings_labels.pkl', mode='wb') as f:
        pickle.dump((embeddings, label_sets), f)
    with open(save_path / 'embeddings_multilabels.pkl', mode='wb') as f:
        pickle.dump((embeddings, multilabel_sets), f)

    if do_plot: plot_clusters(synthetic_dataset, embeddings, label_sets, save_path, **plot_kwargs)

    return quant_metrics

def quantitative_representation_eval(embeddings, label_sets={}, multilabel_sets={}, k=3):
    if not (label_sets or multilabel_sets): return {}

    if type(embeddings) is torch.Tensor: embeddings = embeddings.detach().numpy()
    out = {'labels': {}, 'multilabels': {}}

    for name, labels in label_sets.items():
        probs, preds = [], []
        label_vals = sorted(list(set(labels)))
        label_idxmap = {v: i for i, v in enumerate(label_vals)}

        for i in range(len(embeddings)):
            knn = KNeighborsClassifier(
                n_neighbors = k,
                weights = 'distance',
            )

            e = np.concatenate((embeddings[:i], embeddings[i+1:]), axis=0)
            if isinstance(labels, list): l = labels[:i] + labels[i+1:]
            elif isinstance(labels, np.ndarray): l = np.concatenate((labels[:i], labels[i+1:]), axis=0)

            # TODO: Train test splits!!
            knn.fit(e, l)
            preds.extend(knn.predict(embeddings[i:i+1]))

            knn_probs = knn.predict_proba(embeddings[i:i+1])[0]
            aligned_probs = [0] * len(label_vals)

            for l, p in zip(knn.classes_, knn_probs):
                aligned_probs[label_idxmap[l]] = p

            probs.append(aligned_probs)

        try:
            acc = accuracy_score(labels, preds)
        except Exception as e:
            print(f"Caught {e} on {name} for acc. Skipping.")
            acc = np.NaN

        try:
            auroc = roc_auc_score(labels, probs, multi_class='ovr')
        except Exception as e:
            print(f"Caught {e} on {name} for auroc. Skipping.")
            print(knn_probs, aligned_probs, label_vals)

            print(probs, preds, labels)
            auroc = np.NaN

        out['labels'][name] = (probs, preds, labels, acc, auroc)

    for name, labels_raw in multilabel_sets.items():
        probs, preds = [], []
        label_vals = sorted(list(set(e for v in labels_raw for e in v)))
        label_idxmap = {v: i for i, v in enumerate(label_vals)}

        labels = np.zeros((len(labels_raw), len(label_vals)))
        for i, ls in enumerate(labels_raw):
            for l in ls: labels[i][label_idxmap[l]] = 1

        for i in range(len(embeddings)):
            ps = []
            prs = []
            for label_col in range(len(label_vals)):
                knn = KNeighborsClassifier(
                    n_neighbors = 3,
                    weights = 'distance',
                )

                e = np.concatenate((embeddings[:i], embeddings[i+1:]), axis=0)
                l = np.concatenate((labels[:i, label_col], labels[i+1:, label_col]), axis=0)

                s = set(l)
                if s in ({0}, {1}):
                    ps.append(np.NaN)
                    prs.append(list(s)[0])
                    continue
                else:
                    assert s == {0, 1}

                # TODO: Train test splits!!
                knn.fit(e, l)
                prs.extend(knn.predict(embeddings[i:i+1]))

                knn_prob = knn.predict_proba(embeddings[i:i+1])
                try:
                    knn_prob = knn_prob[0][1]
                except:
                    print(knn.classes_)
                    print(knn_prob)
                    raise

                ps.append(knn_prob)
            probs.append(ps)
            preds.append(prs)

        try:
            preds = np.array(preds)
            acc = accuracy_score(labels, preds)
        except Exception as e:
            print(f"Caught {e} on {name} for acc. Skipping.")
            print(labels)
            print(preds)
            acc = np.NaN

        try:
            non_nan_cols = ~(np.isnan(probs).any(axis=0))
            ls = labels[:, non_nan_cols]
            ps = np.array(probs)[:, non_nan_cols]

            auroc = roc_auc_score(ls, ps, multi_class='ovr')
        except Exception as e:
            print(f"Caught {e} on {name} for auroc. Skipping.")
            print(labels)
            print(non_nan_cols.shape)
            print(non_nan_cols)
            print(ls.shape)
            print(ps.shape)

            print(probs)
            print('\n\n')
            print(preds)
            print('\n\n')
            print(labels)

            print(probs, preds, labels)
            auroc = np.NaN

        out['multilabels'][name] = (probs, preds, labels, acc, auroc)

    return out

class SyntheticNeighborPredictionEvalCallback(Callback):
    def __init__(
        self,
        train_dataset,
        neighbor_prediction_eval_freq: int = 100,
        eval_batch_size:               int = 50,
    ):
        super().__init__()
        self.train_dataset                 = train_dataset
        self.neighbor_prediction_eval_freq = neighbor_prediction_eval_freq
        self.eval_batch_size               = eval_batch_size

    def on_train_epoch_start(self, trainer, pl_module):
        global_step, epoch = trainer.global_step, trainer.current_epoch
        if (epoch % self.neighbor_prediction_eval_freq) != 0: return

        pl_module.encoder_module.eval()
        eval_args = {'Eval': (self.train_dataset.tokenized_node_features, self.train_dataset.label_sets)}

        if all(
            hasattr(self.train_dataset, attr) for attr in ('FT_tokenized_node_features', 'FT_label_sets')
        ):
            eval_args['FT'] = (
                self.train_dataset.FT_tokenized_node_features, self.train_dataset.FT_label_sets
            )

        for name, args in eval_args.items():
            features, label_sets = args
            N = len(features)
            embeddings = []
            for batch_start in range(0, N, self.eval_batch_size):
                batch = features[batch_start:batch_start + self.eval_batch_size]
                batch_embeddings = get_embeddings(
                    pl_module.encoder_module,
                    batch,
                    tokenizer = None,
                    device = pl_module.device,
                )
                embeddings.append(batch_embeddings.numpy())
            embeddings = np.concatenate(embeddings, axis=0)
            eval_metrics = quantitative_representation_eval(embeddings, label_sets)

            for t, td in eval_metrics.items():
                for k, (_, _, _, acc, auroc) in td.items():
                    pl_module.logger.experiment.add_scalar(f'{name}/{t}/{k}/acc', acc, global_step)
                    pl_module.logger.experiment.add_scalar(f'{name}/{t}/{k}/auroc', auroc, global_step)

        pl_module.encoder_module.train()

def run_synthetic(
    output_dir:                     str,
    synthetic_dataset:              PretrainingDataset,
    gml_weight:                     float = 1.0,
    point_weight:                   float = 1.0,
    gml_head:                       str   = EUCLIDEAN_DISTANCE,
    gml_head_kwargs:                dict  = dict(negative_margin=1e1),
    do_gru:                         bool = False,
    encoder_hidden_size:            int   = 10,
    encoder_num_hidden_layers:      int   = 2,
    encoder_num_attention_heads:    int   = 2,
    encoder_intermediate_size:      int   = 8,
    lr:                             float = 3e-4,
    save_freq:                      int   = 100,
    num_epochs:                     int   = 2500,
    num_samples:                    int   = 25,
    do_log_on_epoch:                bool  = True,
    re_gen_data_in_between:         bool  = False,
    re_gen_data_kwargs:             dict  = None,
    print_first:                    int   = 0,
    do_plot:                        bool  = True,
    show_first:                     int   = 0,
    weights_summary:                str   = 'top',
    neighbor_prediction_eval_freq:  int   = 100,
    do_neighbor_prediction_eval:    bool  = True,
    do_simple_profiler:             bool  = False,
    do_advanced_profiler:           bool  = False,
    do_checkpointing:               bool  = True,
    tqdm                                  = None,
    progress_bar_refresh_rate:      int   = 1,
    do_progress_bar:                bool  = True,
    do_anomaly_detection:           bool  = False, # Useful for debugging, but slow.
    do_normalize_embeds:            bool  = False,
    do_overwrite:                   bool  = False,
    eval_batch_size:                int   = 100,
    gru_pooling_strategy:           str   = None,
    gru_do_bidirectional:           bool  = None,
    do_bow_ae:                      bool  = False,
):
    if re_gen_data_in_between and re_gen_data_kwargs is None:
        re_gen_data_kwargs = {}


    assert not (do_bow_ae and do_gru), f"Can't do both!"

    start = time.time()
    output_dir = Path(output_dir)
    accum = RunAccumulator()

    models = [None for _ in range(num_samples)]
    dataset_seeds = [None for _ in range(num_samples)]
    model_metadata = [None for _ in range(num_samples)]

    def do_sample(sample_num):
        synthetic_dataset._seed(sample_num, 'Sample')

        sample_dir = output_dir / str(sample_num)
        if not sample_dir.exists(): os.makedirs(sample_dir)

        if re_gen_data_in_between:
            local_seed = sample_num
            synthetic_dataset.generate(
                re_post_init = True, seed=local_seed, **re_gen_data_kwargs
            )
            while synthetic_dataset.min_degree() < 1:
                local_seed += num_samples
                synthetic_dataset.generate(seed=local_seed)

            if hasattr(synthetic_dataset, '_save'):
                synthetic_dataset._save(sample_dir / 'dataset.pkl')

            last_gen_idx, last_gen_seed = synthetic_dataset._last_seed()
            assert last_gen_seed == local_seed, f"Something is wrong! want {local_seed}, got {s}"

            dataset_seeds[sample_num] = (last_gen_idx, last_gen_seed)

        out_filepath = sample_dir / 'results.pkl'

        callbacks = []
        if do_neighbor_prediction_eval: callbacks.append(
            SyntheticNeighborPredictionEvalCallback(
                synthetic_dataset,
                neighbor_prediction_eval_freq  = neighbor_prediction_eval_freq,
                eval_batch_size = eval_batch_size,
            )
        )

        if save_freq > 0: callbacks.append(
            SaveCallback(
                save_freq                       = save_freq,
                sample_dir                      = sample_dir,
                start                           = start,
            )
        )

        label_sets = synthetic_dataset.label_sets
        if hasattr(synthetic_dataset, 'multilabel_sets'):
            multilabel_sets = synthetic_dataset.multilabel_sets
        else: multilabel_sets = {}

        just_get_model = False
        if out_filepath.exists() and not do_overwrite:
            with open(out_filepath, mode='rb') as f: quant_metrics = pickle.load(f)
            just_get_model = True

        out = run_singleton(
            sample_dir                  = sample_dir,
            data_module                 = synthetic_dataset,
            gml_weight                  = gml_weight,
            point_weight                = point_weight,
            gml_head                    = gml_head,
            do_gru                      = do_gru,
            gml_head_kwargs             = gml_head_kwargs,
            encoder_hidden_size         = encoder_hidden_size,
            encoder_num_hidden_layers   = encoder_num_hidden_layers,
            encoder_num_attention_heads = encoder_num_attention_heads,
            encoder_intermediate_size   = encoder_intermediate_size,
            lr                          = lr,
            do_log_on_epoch             = do_log_on_epoch,
            max_epochs                  = num_epochs,
            weights_summary             = weights_summary,
            do_simple_profiler          = do_simple_profiler,
            do_advanced_profiler        = do_advanced_profiler,
            callbacks                   = callbacks,
            do_checkpointing            = do_checkpointing,
            progress_bar_refresh_rate   = progress_bar_refresh_rate,
            do_progress_bar             = do_progress_bar,
            just_get_model              = just_get_model,
            do_anomaly_detection        = do_anomaly_detection,
            do_normalize_embeds         = do_normalize_embeds,
            gru_pooling_strategy        = gru_pooling_strategy,
            gru_do_bidirectional        = gru_do_bidirectional,
            do_bow_ae                   = do_bow_ae,
        )


        final_checkpoint_fp = sample_dir / 'final_model.ckpt'
        model_metadata[sample_num] = final_checkpoint_fp

        if just_get_model:
            model = out.cpu()
            try:
                model.load_from_checkpoint(str(final_checkpoint_fp))
            except RuntimeError as e:
                try:
                    ckpt_contents = torch.load(final_checkpoint_fp)
                    model.load_state_dict(ckpt_contents['state_dict'])
                    model_metadata[sample_num] = (final_checkpoint_fp, ckpt_contents)
                except Exception as ee:
                    print(f"Failed to reload model directly! {e}")
                    print(f"\nAlso failed to reload model manually! {ee}")
                    model = (model, final_checkpoint_fp, ckpt_contents)

            models[sample_num] = model
        else:
            model, trainer = out

            embeddings = []
            N = len(synthetic_dataset)
            for batch_start in range(0, N, eval_batch_size):
                batch = synthetic_dataset.tokenized_node_features[batch_start:batch_start + eval_batch_size]
                batch_embeddings = get_embeddings(
                    model.encoder_module,
                    batch,
                    tokenizer = None,
                    device=model.device,
                )
                embeddings.append(batch_embeddings.numpy())

            embeddings = np.concatenate(embeddings, axis=0)

            quant_metrics = eval_clusters(
                synthetic_dataset,
                embeddings,
                label_sets      = label_sets,
                multilabel_sets = multilabel_sets,
                do_print        = (sample_num < print_first),
                save_path       = sample_dir,
                do_plot         = do_plot,
                do_show_plots   = do_plot and (sample_num < show_first),
                show_legend     = True,
            )

            model = model.cpu()
            models[sample_num] = model

        for t, td in quant_metrics.items():
            for name, (_, _, _, acc, auroc) in td.items():
                accum.update(
                    {
                        f'{t}/{name} Accuracy': acc,
                        f'{t}/{name} AUROC': auroc,
                    }
                )

    samples_rng = range(num_samples)
    if tqdm is not None and num_samples > 2: samples_rng = tqdm(samples_rng)

    for sample_num in samples_rng:
        do_sample(sample_num)

    return accum, models, model_metadata, dataset_seeds
