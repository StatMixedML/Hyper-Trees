# Hyper-Trees Experiments

This folder is the reproducibility section for the paper. It contains all data, configurations, code, and the pinned environment needed to reproduce the published experimental results.

**Package assembled:** May 28, 2026 with the pinned package versions in `experiments/requirements-experiments.txt`.

## Installation

First, clone the repository:

```bash
git clone https://github.com/StatMixedML/Hyper-Trees.git
cd Hyper-Trees
```

We use [`uv`](https://docs.astral.sh/uv/) as the package manager. Install it first if you don't already have it:

```bash
pip install uv
```

Our paper runs used **`uv 0.8.12`**. 

In the project's top-level folder (the `Hyper-Trees/` folder you cloned into), create a Python 3.11 venv and install the pinned experiments environment:

```bash
# Create a Python 3.11.0 venv (must be exactly 3.11.0 to match the pinned environment)
uv venv --python 3.11.0

# Activate the venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install all dependencies (including transitive) at exact pinned versions
uv pip install -r experiments/requirements-experiments.txt --index-strategy unsafe-best-match
```

`experiments/requirements-experiments.txt` pins every dependency, including transitive ones, to the exact versions used in our paper experiments, ensuring reproducibility.


### CUDA Installation

All experiments in the paper were run on GPU using PyTorch 2.1.1 with CUDA 11.8. GPU is required for the deep learning baselines (DeepAR, TFT) and was also used for training Hyper-TreeNet models. The `requirements-experiments.txt` file pins `torch==2.1.1+cu118` and declares `--extra-index-url https://download.pytorch.org/whl/cu118`, so the CUDA build is pulled in automatically.

**CPU-only environments:** the Hyper-Tree, Hyper-TreeNet, LightGBM, and classical baselines will still run on CPU. The deep-learning baselines (DeepAR, TFT) technically fall back to CPU via PyTorch but become impractically slow for the larger datasets; a CUDA-11.8-compatible GPU is strongly recommended for a full end-to-end reproducibility run.

### Important Note

> ⚠️ **Important:**
> - Using any versions other than those pinned in `requirements-experiments.txt` (including Python, CUDA and PyTorch) will produce different results.
> - Do **not** install `hypertrees-forecasting` from PyPI or GitHub alongside this environment. The pinned `requirements-experiments.txt` already includes the exact version of the package and all its dependencies at fixed versions. Installing from PyPI or GitHub would pull in newer (unpinned) transitive dependencies, breaking version consistency and reproducibility.
> - If you already have `hypertrees-forecasting` installed from PyPI or GitHub, uninstall it first before setting up the experiments environment:
>   ```bash
>   uv pip uninstall hypertrees-forecasting
>   ```
>   Then proceed with the reproducible installation above.

### Hardware Specifications

All experiments in the paper were conducted with the following specifications:
- **OS:** Windows 11
- **CPU:** 13th Gen Intel(R) Core(TM) i9-13900H (14 cores)
- **RAM:** 64 GB
- **GPU:** NVIDIA RTX 3500 Ada Generation Laptop GPU (12 GB memory)

## Running Experiments

> The single entry point is **_experiments/Reproduce.ipynb_**
>
Open `experiments/Reproduce.ipynb` in Jupyter, VS Code, or PyCharm and run all cells. This reproduces every table, figure, and ablation in the paper (global, local, Rossmann A1-A11, embedding-dimension ablation, paper figures, and the final metrics tables). When the run finishes, every metrics table and every paper figure is rendered inline at the bottom of the notebook.

A full run takes approximately **4 hours 21 minutes** on the paper hardware, broken down as:

| Stage | Runtime |
|---|---|
| Global Hyper-Trees | 21.09 min |
| Rossmann ablations (A1-A11) | 45.87 min |
| Embedding-dimension ablation | 22.12 min |
| Local Hyper-Trees | 14.97 min |
| Global LightGBM | 4.24 min |
| Local LightGBM | 8.64 min |
| Global Deep Learning | 104.75 min |
| Global ETS | 2.47 min |
| Local Classical | 15.99 min |
| Figure creation | 20.33 min |
| **Total** | **261.00 min** |

Outputs:

- forecast CSVs per dataset and model family in
  `experiments/runs/results/{global,local}/` (e.g., `rossmann_hypertrees_fcsts.csv`,
  `rossmann_lgbm_fcsts.csv`, `rossmann_deeplearning_fcsts.csv`, `rossmann_ets_fcsts.csv`) and
  `experiments/runs/results/ablation/{rossmann,embedding_evaluation}/`
- metrics tables at
  `experiments/runs/results/metrics/{global,local,ablation_rossmann,ablation_embeddings}_metrics.csv`
- paper figure PDFs + PNGs in `experiments/runs/results/plots/`

### Paper Artefact Map

Explicit mapping from paper element to the code and output that produces it.

| Paper element | Produced by | Output location |
|---|---|---|
| Table 1 (Air Passengers Results) | `Reproduce.ipynb` -> local stage (airpassengers) | `results/metrics/local_metrics.csv` |
| Table 2 (Local Model Results) | `Reproduce.ipynb` -> local stage (auselectricity, ausretail, tourism_monthly) | `results/metrics/local_metrics.csv` |
| Table 3 (Global Model Results) | `Reproduce.ipynb` -> global stage (all datasets) | `results/metrics/global_metrics.csv` |
| Table 4 (Rossmann Ablation A1-A11) | `Reproduce.ipynb` -> Rossmann ablation (`rossmann_A1.ipynb` ... `rossmann_A11.ipynb`) | `results/metrics/ablation_rossmann_metrics.csv` |
| Table G1 (Embedding-Dimension Analysis) | `Reproduce.ipynb` -> embedding ablation (`embedding_ablation.ipynb`) | `results/metrics/ablation_embeddings_metrics.csv` |
| Figure 4 (Runtime Scaling) | `runs/notebooks/scaling_comparison.ipynb` | `results/plots/runtime_scaling.{pdf,png}` |
| Figure 6 (Hyper-Tree-STL Decomposition) | `runs/notebooks/stl.ipynb` | `results/plots/STL_Trend.{pdf,png}`, `STL_Seasonality.{pdf,png}` |
| Figure 7 (Estimated Parameters of Hyper-Tree-STL) | `runs/notebooks/stl.ipynb` | `results/plots/STL_a0.{pdf,png}`, `STL_a1.{pdf,png}`, `STL_c1.{pdf,png}`, `STL_d1.{pdf,png}` |
| Figure 8 (Feature Importance of Hyper-Tree-STL) | `runs/notebooks/stl.ipynb` | `results/plots/shap_a0.{pdf,png}`, `shap_a1.{pdf,png}`, `shap_c1.{pdf,png}`, `shap_d1.{pdf,png}` |
| Figure D1 (Global Model Forecasts) | `runs/notebooks/example_forecasts.ipynb` | `results/plots/` |
| Figures F1, F2, F3 (Time-Varying AR Parameters) | `runs/notebooks/time_varying_params.ipynb` | `results/plots/` |
| Figures G1, G2 (Tree Embeddings) | `runs/notebooks/embedding_visualization.ipynb` | `results/plots/` |

### Evaluating Results

The evaluation section of `Reproduce.ipynb` calls `evaluate_fcsts()`, which runs four evaluators in `experiments/utils.py` (global, local, Rossmann ablation, embedding-dim ablation) and returns a dict of metrics DataFrames rendered inline as tables. Each evaluator computes MAPE, sMAPE, WAPE, RMSE, MAE, and (where applicable) MASE per series, averages across series, and saves the aggregated table to `experiments/runs/results/metrics/{global,local,ablation_rossmann,ablation_embeddings}_metrics.csv`. For a lightweight standalone variant, `experiments/runs/fcst_evaluation.ipynb` loads the global or local result CSVs and displays them in a single cell.

## Folder Structure

```
experiments/
├── datasets/                                   # Datasets with configs
│   ├── airpassengers/                          # Local only
│   ├── auselectricity/                         # Global + Local (with ETS padding)
│   ├── ausretail/                              # Global + Local
│   ├── m3_monthly/                             # Global only
│   ├── m3_yearly/                              # Global only (with ETS padding)
│   ├── m5_agg/                                 # Global only
│   ├── rossmann/                               # Global only
│   └── tourism_monthly/                        # Global + Local (with ETS padding)
├── models.py                                   # Forecast model wrappers (Hyper-Tree*, baselines)
├── README.md                                   # This file
├── repro.py                                    # Papermill orchestration used by Reproduce.ipynb
├── Reproduce.ipynb                             # Single entry point: runs all experiments
├── requirements-experiments.txt                # Pinned dependencies for paper reproduction
├── runs/
│   ├── fcst_evaluation.ipynb                   # Standalone metrics viewer (reads result CSVs)
│   ├── results/                                # Outputs from experiment runs (auto-generated)
│   │   ├── ablation/
│   │   │   ├── embedding_evaluation/           # Embedding-dim ablation forecast CSVs (+ tree
│   │   │   │                                   #   embeddings / AR parameters for airpassengers)
│   │   │   └── rossmann/                       # Rossmann ablation forecast CSVs
│   │   ├── global/                             # Global-stage forecast CSVs (+ AR parameters
│   │   │                                       #   for rossmann / m5_agg)
│   │   ├── local/                              # Local-stage forecast CSVs (+ AR parameters
│   │   │                                       #   / tree embeddings for airpassengers)
│   │   ├── metrics/                            # Aggregated metrics tables
│   │   │                                       #   (global, local, ablation_rossmann,
│   │   │                                       #    ablation_embeddings)_metrics.csv
│   │   └── plots/                              # PDF + PNG outputs from figure + STL notebooks
│   └── notebooks/                              # Parameterized templates executed by papermill
└── utils.py                                    # Data loading, metrics, plotting + CSV-loading helpers
```

### Dataset Folder Contents

Each dataset folder contains:
- `train.parquet` / `test.parquet`: training and test splits
- `meta.json`: dataset metadata (series IDs, lags, features, forecast horizon, frequency)
- `config_global.py` and/or `config_local.py`: hyperparameters for each experiment type
- `dataset_source.txt`: original data source reference

Datasets used for the global Hyper-Tree-ETS experiments (`auselectricity`, `m3_yearly`, `tourism_monthly`) additionally provide:
- `train_padded.parquet` / `test_padded.parquet`: back-appended series with uniform length
- `meta_ets.json`: ETS-specific metadata including the `mask` column for valid observations

### Data availability

All datasets used in this paper are publicly available and are included directly in the replication kit under `experiments/datasets/` as pre-processed `train.parquet` / `test.parquet` files. No dataset requires registration, payment, or NDA access. Upstream sources and citations are documented per dataset in `dataset_source.txt`. To run the experiments, one does not need to download any additional data to run the reproducibility check. The `{train,test}.parquet` and `{train,test}_padded.parquet` files in each `experiments/datasets/<name>/` folder are produced by one-time preprocessing from the upstream raw sources cited in each `dataset_source.txt`.


## Models Compared

**Hyper-Tree models**:
- `Hyper-Tree-AR`: Autoregressive model with tree-learned, time-varying AR(p) parameters
- `Hyper-TreeNet-AR`: Hybrid GBDT encoder + MLP decoder for AR(p) parameters
- `Hyper-Tree-ETS`: Exponential smoothing with tree-learned parameters

**Tree-based baselines:**
- `LightGBM`: Standard LightGBM
- `LightGBM-AR`: LightGBM with autoregressive lag features
- `LightGBM-STL`: LightGBM on STL residuals (local only)

**Deep learning baselines:**
- `DeepAR`: Autoregressive RNN
- `TFT`: Temporal Fusion Transformer
- `Chronos`: Pre-trained foundation model (`chronos-t5-base`)

**Classical baselines:**
- `AutoARIMA` / `AutoARIMA-X` (local only): ARIMA with automatic `(p,d,q)` selection; the `-X` variant adds features.
- `AR(p)` / `AR(p)-X` (local only): fixed-order `ARIMA(p, 0, 0)`; the `-X` variant adds features.
- `AutoETS` (local only): Automatic Exponential Smoothing (used as MASE reference).

## Contact

For questions about reproducing the results, please open an issue on the project's GitHub repository.

## License

Code in this repository is released under the Apache License 2.0 with Commons Clause License Condition v1.0. See the [LICENSE](../LICENSE) file at the repository root. 