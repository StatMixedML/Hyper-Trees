<h4 align="center">

|                      | **[Release Notes](https://github.com/StatMixedML/Hyper-Trees/releases)** |
|----------------------|---|
| **Open&#160;Source** | [![License: Apache 2.0 + Commons Clause](https://img.shields.io/badge/License-Apache_2.0_with_Commons_Clause-yellow.svg)](LICENSE) |
| **CI/CD**            | [![github-actions](https://img.shields.io/github/actions/workflow/status/StatMixedML/Hyper-Trees/unit-tests.yml?logo=github)](https://github.com/StatMixedML/Hyper-Trees/actions/workflows/unit-tests.yml) <img src="https://codecov.io/gh/StatMixedML/Hyper-Trees/branch/main/graph/badge.svg" alt="Code coverage status badge"> |
| **Package**             | [![!pypi](https://img.shields.io/pypi/v/hypertrees-forecasting?color=orange)](https://pypi.org/project/hypertrees-forecasting/) [![!python-versions](https://img.shields.io/pypi/pyversions/hypertrees-forecasting)](https://www.python.org/) |
| **Downloads**        | ![Pepy Total Downloads](https://img.shields.io/pepy/dt/hypertrees-forecasting?label=PyPI%20Downloads&color=green) |
| **Paper**            | [![Arxiv link](https://img.shields.io/badge/arXiv-Forecasting%20with%20Hyper--Trees-color=brightgreen)](https://arxiv.org/pdf/2405.07836) |

</h4>

---
# Overview

Hyper-Trees are a novel framework for modeling time series data with gradient boosted trees (GBDTs). Instead of forecasting time series directly, Hyper-Trees use GBDTs to learn the parameters of a classical time series model such as ARIMA or Exponential Smoothing as functions of features. The target time series model then generates the final forecasts. This naturally injects the inductive bias of forecasting models into tree-based learning. 
While our framework is built upon the well-established LightGBM model, it can in principle be used with any modern GBDT framework.



<div align="center">
  <img src="hypertrees/hyper_treenet.png" width="70%" alt="Hyper-Tree architecture">
</div>

Hyper-Trees offer several advantages:

- **Improved Extrapolation in Tree-Based Models.** Forecasts are generated via a parametric target time series model, rather than the piece-wise constant output of tree-models.
- **Cross-Series Learning with Local Adaptivity.** A global GBDT learns the feature-to-parameter mapping, so similar series share information while each still receives its own parameters.
- **Time-Varying Parameters.** Coefficients vary cross-sectionally (series-specific features such as store type or region) and temporally (day, week, month, year, ...), capturing effects such as distinct AR(p) dynamics on weekdays versus weekends.
- **Model Transparency and Interpretability.** Forecasts are produced by classical time series models whose parameters retain clear statistical meaning.
- **Full Functionality of GBDTs.** Core GBDT capabilities (missing-value handling, feature importance, categorical support, monotonicity constraints) carry over unchanged.

---

# News
[2026-06-01] v0.1.0 released on [PyPI](https://pypi.org/project/hypertrees-forecasting/).<br>
[2024-05-01] Create repository and initial commits.

---

## Available Models

| Model | Description | Scope |
| :--- | :--- | :---: |
| **`Hyper-Tree-AR`** | Autoregressive model with tree-learned, time-varying AR(p) parameters. | ![](https://img.shields.io/badge/Global-blue) ![](https://img.shields.io/badge/Local-green) |
| **`Hyper-TreeNet-AR`** | Hybrid model combining tree embeddings with a neural network to learn AR(p) parameters. | ![](https://img.shields.io/badge/Global-blue) ![](https://img.shields.io/badge/Local-green) |
| **`Hyper-Tree-ETS`** | Exponential smoothing model where ETS parameters are estimated by trees. | ![](https://img.shields.io/badge/Global-blue) ![](https://img.shields.io/badge/Local-green) |
| **`Hyper-Tree-STL`** | STL decomposition with tree-learned parameters for trend and seasonality. | ![](https://img.shields.io/badge/Local-green) |

`Global` means a single model is trained across multiple time series; `Local` means a separate model is trained for each individual series.
All models currently provide point forecasts only. Probabilistic forecasting is planned for future releases. Note on `Hyper-Tree-STL`: it is designed to decompose time series into trend and seasonal components and is not intended for forecasting. However, the STL-parameters can still be used to generate forecasts.

# Getting Started

We refer to the `examples/` notebooks for quick-start guides on using the Hyper-Tree models.

---



# Installation

To run the `Hyper-TreeNet-AR` model efficiently, we recommend installing PyTorch with CUDA support. While GPU is recommended for faster runtime, it is not strictly required. All models also run on CPU. We use `uv pip` for installs. If you don't have `uv`, consider installing it or simply replace `uv pip install` with `pip install`.

### Basic Installation (CPU)

Install the latest release from PyPI:

```bash
uv pip install hypertrees-forecasting
```

Or install the development version directly from GitHub:

```bash
uv pip install git+https://github.com/StatMixedML/Hyper-Trees.git
```

Or clone the repository and install in editable mode for development:

```bash
git clone https://github.com/StatMixedML/Hyper-Trees.git
cd Hyper-Trees
uv pip install -e .
```

This installs Hyper-Trees with the latest compatible versions of all dependencies, including a CPU-compatible version of PyTorch. All models will work, just without GPU acceleration.

### Optional: Extra Dependencies

The example notebooks under `examples/` use `matplotlib` (plotting), `shap` (feature-importance visualization), and `optuna` (hyper-parameter optimization). To install these alongside the package, use the `extras` option:

```bash
uv pip install "hypertrees-forecasting[extras]"     # from PyPI
uv pip install -e ".[extras]"                       # editable / development
```

These packages are not required to use the Hyper-Tree models themselves, only to run the example notebooks.

### GPU Support

For CUDA-enabled PyTorch, install Hyper-Trees first, then install PyTorch from its CUDA index:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121 --upgrade
```

Replace `cu121` with the variant matching your driver. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the current list.

# Reproducing Paper Results

The full reproducibility package, including the pinned environment, datasets, configurations, and experiment notebooks needed to reproduce all paper results, lives in the [`experiments/`](experiments/) folder. See the [Experiments README](experiments/README.md) for installation instructions and step-by-step guidance on running the experiments.

---

# Early-stage software
`hypertrees-forecasting` is in an early stage of development and is provided *“as is”*, without any warranty or guarantee. We welcome bug reports, feature requests, and pull requests, and encourage feedback by opening a [new discussion](https://github.com/StatMixedML/Hyper-Trees/discussions). We strongly recommend thorough testing and validation before using the package in production or other critical applications.

---

# Acknowledgments

This work draws on and integrates methods and implementations from the following key repositories:

- [**<u>LightGBM</u>**](https://github.com/microsoft/LightGBM) – Gradient boosting framework for efficient tree-based learning.  
- [**<u>PyTorch</u>**](https://github.com/pytorch/pytorch) – Deep learning framework for tensor computation and neural network modeling.  
- [**<u>Nixtla</u>**](https://github.com/Nixtla) – Open Source Time Series Ecosystem.  
- [**<u>sktime</u>**](https://github.com/sktime/sktime) – A unified framework for machine learning with time series.
- [**<u>GluonTS</u>**](https://github.com/awslabs/gluonts) – Probabilistic time series modeling and forecasting with deep learning.  
---

## License

This project is licensed under the Apache License 2.0 with Commons Clause License Condition v1.0. In short, the code is free for research, academic, testing, production, and internal commercial use; selling access to the Software's functionality as a primary offering (e.g., as an API service, managed service, or hosted offering) requires a separate commercial license. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use `Hyper-Trees` in your research, please cite our paper:

[![Arxiv link](https://img.shields.io/badge/arXiv-Forecasting%20with%20Hyper--Trees-color=brightgreen)](https://arxiv.org/pdf/2405.07836) <br/>

```bibtex
@article{maerz2024hypertrees,
  title   = {Forecasting with Hyper-Trees},
  author  = {März, Alexander and Rasul, Kashif},
  journal = {arXiv preprint arXiv:2405.07836},
  year    = {2024}
}
```
