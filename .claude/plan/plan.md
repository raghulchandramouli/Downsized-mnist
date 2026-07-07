# Plan: Recreate MNIST-1D paper experiments

## Context

Paper: "Scaling Down Deep Learning with MNIST-1D" (Greydanus & Kobak, ICML 2024,
arXiv:2011.14439). It introduces MNIST-1D, a 40-dimensional procedurally-generated
1D analogue of MNIST, and uses it to cheaply reproduce several "science of deep
learning" phenomena that normally require much larger compute budgets. The user
wants to recreate all of the paper's experiments locally.

Repo state today (`Downsized-mnist`, uv-managed, Python 3.12):
- Scaffold only: `main.py` is a "Hello World" stub, `pyproject.toml` had just
  `marimo` as a dependency.
- `ablations/Double_Descent.ipynb` exists but is a 0-byte empty stub (verified via
  `git show` — never had real content), so there is nothing to preserve there.
- No dataset/model/training code exists yet.
- Reference implementation: official repo `github.com/greydanus/mnist1d`
  (Apache-2.0), whose core files (`mnist1d/data.py`, `mnist1d/transform.py`,
  `mnist1d/utils.py`, `notebooks/models.py`, `notebooks/train.py`) were fetched
  during research and give the exact procedural-generation algorithm, default
  hyperparameters, and model architectures used in the paper's figures.
- Already added as deps via `uv add`: `numpy`, `scipy`, `matplotlib`,
  `scikit-learn`, `torch` (confirmed installed in `.venv`).
- Confirmed marimo notebooks (`.py` files using `@app.cell`) run standalone via
  `python notebook.py` and exit cleanly — no jupyter/nbconvert dependency needed,
  which fits the project's existing choice of `marimo` over Jupyter.
- Decision reversed per user request: experiment notebooks will be classic
  Jupyter `.ipynb` files instead of marimo, so each ablation renders as an
  interactive, cell-by-cell visual walkthrough. Needs `jupyter`/`ipykernel`
  (and optionally `ipywidgets`) added as deps before implementation.

Goal: faithfully port the dataset generator and reproduce every experiment/case
study from the paper, organized as runnable, self-contained Jupyter notebooks,
using shared, reusable core code (not one-off copy-pasted training loops).

## Approach

### 1. Core package — `mnist1d/`
Port directly from the official repo (small, load-bearing, do this by hand,
not via subagent):
- `mnist1d/utils.py`: `set_seed`, `to_pickle`/`from_pickle`, `ObjectView`,
  `plot_signals`.
- `mnist1d/transform.py`: `pad`, `shear`, `translate`, `corr_noise_like`,
  `iid_noise_like`, `interpolate`, `transform` — the exact random-transform
  pipeline (pad → dilate → scale → translate → noise → shear → subsample to 40).
- `mnist1d/data.py`: `get_dataset_args` (default: 5000 samples, 80/20 split,
  40-d final length, seed 42), `get_templates` (10 hand-crafted digit-like
  templates), `make_dataset` (generates directly, no network dependency —
  skip the paper's optional pickle-download path since procedural generation
  is the point and avoids network calls in this environment).

### 2. Shared model/training code — `common/`
- `common/models.py`: port `LinearBase` (logistic regression), `MLPBase`,
  `ConvBase` (1D CNN), `GRUBase` from `notebooks/models.py`, plus extend
  `ConvBase` to accept a `pool` argument (`none`, `stride_2`, `max_pool`,
  `avg_pool`, `l2_pool`) for the pooling-benchmark experiment, and a small
  SSL projection-head wrapper for the self-supervised experiment.
- `common/train.py`: port `get_model_args`, `accuracy`, `train_model` — a
  generic step-based training loop (Adam, cross-entropy, periodic eval) reused
  across baseline classification, double descent, lottery tickets, and pooling
  experiments instead of duplicating training logic per notebook.

### 3. Experiment notebooks — `ablations/` (Jupyter `.ipynb`, one per case study)
Replace the empty `Double_Descent.ipynb` stub with a rewritten
`ablations/double_descent.ipynb` and add the rest of the paper's case studies
as siblings:

1. `baseline_classification.ipynb` — train LogisticRegression/MLP/CNN/GRU on
   default MNIST-1D; reproduce the accuracy table (~32/68/94/91%); repeat with
   `shuffle_seq=True` to show CNN/GRU accuracy collapses ~35pp while
   MLP/logistic are unaffected (spatial inductive bias sanity check).
2. `tsne_comparison.ipynb` — t-SNE on raw MNIST-1D vectors vs. real MNIST (via
   `sklearn.datasets.fetch_openml('mnist_784')`, degrade gracefully if no
   network) to show MNIST forms tight clusters in pixel space and MNIST-1D
   does not.
3. `lottery_tickets.ipynb` — iterative magnitude pruning on an MLP (Frankle &
   Carbin-style): prune smallest weights, reset survivors to their original
   init, retrain; compare ticket vs. random-mask vs. dense accuracy across
   sparsity levels; test transfer to flipped/shuffled data; analyze adjacency
   of nonzero first-layer weights as evidence of a learned locality bias.
4. `double_descent.ipynb` — train MLP/CNN with 15% label noise across a sweep of
   hidden sizes/channels; plot test error vs. model size to show the
   interpolation-threshold peak; contrast against a no-label-noise run.
5. `metalearn_learning_rate.ipynb` — unrolled/differentiable optimization
   (`torch.autograd` through several SGD steps) to meta-learn a scalar
   learning rate for an MLP classifier; show convergence to ~0.62 regardless
   of init.
6. `metalearn_activation_function.ipynb` — parameterize an activation function
   with a small meta-MLP initialized to approximate ELU, meta-learn its
   weights via unrolled optimization to minimize downstream training loss;
   compare against ReLU/ELU/Swish baselines and plot the learned nonlinearity.
7. `self_supervised_learning.ipynb` — SimCLR-style contrastive pretraining using
   `ConvBase` + projection head, with 1D-specific augmentations (de-trend
   linear slope, circular shift, re-add random slope); evaluate via linear
   probe accuracy at each depth of the projection head to reproduce the
   "guillotine regularization" effect.
8. `benchmark_pooling.ipynb` — train CNNs with each pooling variant
   (`none`/`stride_2`/`max_pool`/`avg_pool`/`l2_pool`) at training-set sizes
   {1000, 5000, 50000}; plot accuracy vs. data size per pooling method to show
   pooling helps in low-data regimes and stops mattering at scale.

Each notebook: imports `mnist1d` + `common`, runs in minutes on CPU (matching
the paper's stated wall-clock times), prints final metrics, and renders
matplotlib plots inline as Jupyter cell outputs, with widgets/sliders where
interactive exploration adds value (e.g. sparsity level, pooling variant).

### 4. Delegation strategy
Steps 1–2 (core package, shared models/training) done directly, since every
notebook depends on them. Once those are in place, dispatch the 8 experiment
notebooks to parallel subagents (one per notebook) — each given the finished
`mnist1d`/`common` APIs, the relevant methodology section from the paper, and
(if needed) told to fetch the matching notebook from
`github.com/greydanus/mnist1d` for hyperparameter parity, then reimplement it
as a clean Jupyter notebook rather than copy-pasting.

### 5. Verification
- Run `uv run python -c "from mnist1d.data import make_dataset, get_dataset_args; d = make_dataset(get_dataset_args()); print(d['x'].shape, d['y'].shape)"` to confirm the dataset generator works.
- Execute every `ablations/*.ipynb` notebook headlessly
  (`uv run jupyter nbconvert --to notebook --execute ablations/<file>.ipynb`)
  and confirm it completes without
  error and prints/plots the expected metrics (accuracy table, double-descent
  curve, lottery-ticket sparsity plot, meta-learned LR/activation, SSL linear
  probe accuracy, pooling-vs-datasize plot).
- Spot-check reported numbers are in the right ballpark vs. the paper's Table
  1 (LR 32%, MLP 68%, CNN 94%, GRU 91%, human 96%) and shuffle-ablation
  numbers, allowing for reasonable variance from a smaller default sample size
  where used for speed.
