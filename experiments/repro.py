# Orchestration helpers for running Hyper-Trees experiments via papermill.

from __future__ import annotations

from pathlib import Path


DATASETS_GLOBAL = [
    "rossmann",
    "auselectricity",
    "ausretail",
    "m3_monthly",
    "m3_yearly",
    "m5_agg",
    "tourism_monthly",
]

DATASETS_LOCAL = [
    "airpassengers",
    "auselectricity",
    "ausretail",
    "tourism_monthly",
]


def _find_repo_root(start: Path | None = None) -> Path:
    """
    Walk upward from ``start`` to find the repository root.

    The repository root is the nearest ancestor directory that contains
    a ``pyproject.toml`` file.

    Parameters
    ----------
    start : Path or None
        Directory to start searching from. Defaults to the current
        working directory when *None*.

    Returns
    -------
    Path
        Absolute path to the repository root directory.
    """
    root = (start or Path.cwd()).resolve()
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    return root


def run_papermill(
        template: Path,
        parameters: dict | None = None,
        cwd: Path | None = None,
        label: str | None = None,
) -> None:
    """
    Execute a notebook via papermill in a fresh kernel subprocess.

    Parameters
    ----------
    template : Path
        Path to the ``.ipynb`` template to execute.
    parameters : dict or None
        Papermill parameters injected into the notebook. Defaults to
        an empty dict when *None*.
    cwd : Path or None
        Working directory for the kernel subprocess. Uses the current
        working directory when *None*.
    label : str or None
        Optional label printed to stdout before execution starts,
        useful for tracking progress.
    """
    import papermill as pm

    if label:
        print(f"--- {label} ---", flush=True)
    pm.execute_notebook(
        str(template),
        None,
        parameters=parameters or {},
        cwd=str(cwd) if cwd is not None else None,
        engine_kwargs={"iopub_timeout": 600},
    )


def run_global_hypertrees(
        datasets: list[str],
        repo_root: Path | None = None,
) -> None:
    """
    Run the global Hyper-Tree models notebook once per dataset.

    Parameters
    ----------
    datasets : list of str
        Dataset names to iterate over (e.g. ``["rossmann", "m3_yearly"]``).
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    template = root / "experiments/runs/notebooks/global_hypertrees.ipynb"
    for data in datasets:
        run_papermill(
            template,
            parameters={"dataset": data},
            cwd=root,
            label=f"global hypertrees: {data}",
        )


def run_global_lgbm(
        datasets: list[str],
        repo_root: Path | None = None,
) -> None:
    """
    Run the global LightGBM models notebook once per dataset.

    Parameters
    ----------
    datasets : list of str
        Dataset names to iterate over.
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    template = root / "experiments/runs/notebooks/global_lgbm.ipynb"
    for data in datasets:
        run_papermill(
            template,
            parameters={"dataset": data},
            cwd=root,
            label=f"global lgbm: {data}",
        )


def run_global_deeplearning(
        datasets: list[str],
        repo_root: Path | None = None,
) -> None:
    """
    Run the global deep learning models notebook once per dataset.

    Parameters
    ----------
    datasets : list of str
        Dataset names to iterate over.
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    template = root / "experiments/runs/notebooks/global_deeplearning.ipynb"
    for data in datasets:
        run_papermill(
            template,
            parameters={"dataset": data},
            cwd=root,
            label=f"global deeplearning: {data}",
        )


def run_global_ets(
        datasets: list[str],
        repo_root: Path | None = None,
) -> None:
    """
    Run the global AutoETS notebook once per dataset.

    Parameters
    ----------
    datasets : list of str
        Dataset names to iterate over.
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    template = root / "experiments/runs/notebooks/global_ets.ipynb"
    for data in datasets:
        run_papermill(
            template,
            parameters={"dataset": data},
            cwd=root,
            label=f"global ets: {data}",
        )


def run_local_hypertrees(
        datasets: list[str],
        repo_root: Path | None = None,
) -> None:
    """
    Run the local Hyper-Tree models notebook once per dataset.

    Parameters
    ----------
    datasets : list of str
        Dataset names to iterate over.
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    template = root / "experiments/runs/notebooks/local_hypertrees.ipynb"
    for data in datasets:
        run_papermill(
            template,
            parameters={"dataset": data},
            cwd=root,
            label=f"local hypertrees: {data}",
        )


def run_local_lightgbm(
        datasets: list[str],
        repo_root: Path | None = None,
) -> None:
    """
    Run the local LightGBM models notebook once per dataset.

    Parameters
    ----------
    datasets : list of str
        Dataset names to iterate over.
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    template = root / "experiments/runs/notebooks/local_lightgbm.ipynb"
    for data in datasets:
        run_papermill(
            template,
            parameters={"dataset": data},
            cwd=root,
            label=f"local lightgbm: {data}",
        )


def run_local_classical(
        datasets: list[str],
        repo_root: Path | None = None,
) -> None:
    """
    Run the local classical models notebook once per dataset.

    Parameters
    ----------
    datasets : list of str
        Dataset names to iterate over.
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    template = root / "experiments/runs/notebooks/local_classical.ipynb"
    for data in datasets:
        run_papermill(
            template,
            parameters={"dataset": data},
            cwd=root,
            label=f"local classical: {data}",
        )


def run_rossmann_ablations(repo_root: Path | None = None) -> None:
    """
    Run every ``rossmann_A*.ipynb`` template in ``notebooks/``.

    Parameters
    ----------
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    in_dir = root / "experiments/runs/notebooks"
    templates = sorted(
        f for f in in_dir.iterdir()
        if f.name.startswith("rossmann_") and f.suffix == ".ipynb"
    )
    for tpl in templates:
        run_papermill(
            tpl,
            cwd=root,
            label=f"rossmann ablation: {tpl.stem}",
        )


def run_embedding_ablations(
        datasets: list[str],
        embedding_dims: list[int],
        repo_root: Path | None = None,
) -> None:
    """
    Run the embedding-dim ablation notebook for every (dataset, dim) pair.

    Parameters
    ----------
    datasets : list of str
        Dataset names to iterate over.
    embedding_dims : list of int
        Embedding dimensionalities to sweep for each dataset.
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    template = root / "experiments/runs/notebooks/embedding_ablation.ipynb"
    for data in datasets:
        for dim in embedding_dims:
            run_papermill(
                template,
                parameters={"data_run": data, "embedding_dim": dim},
                cwd=root,
                label=f"embedding: {data} (dim={dim})",
            )


def create_figures(repo_root: Path | None = None) -> None:
    """
    Execute the paper-figure plot notebooks.

    Runs the STL case study (trend, seasonality, parameter plots and SHAP bar
    charts) along with scaling comparison, example forecasts, time-varying
    parameters, and embedding visualization notebooks. PNG/PDF plot artefacts
    are written under ``results/plots/``.

    Parameters
    ----------
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    """
    root = repo_root or _find_repo_root()
    nb_dir = root / "experiments/runs/notebooks"
    for name in (
            "scaling_comparison.ipynb",
            "stl.ipynb",
            "example_forecasts.ipynb",
            "time_varying_params.ipynb",
            "embedding_visualization.ipynb",
    ):
        run_papermill(
            nb_dir / name,
            cwd=root,
            label=f"figures: {Path(name).stem}",
        )


def _display_as_2x2(paths: list[Path], caption: str | None = None) -> None:
    """
    Compose four PNGs into a single 2x2 composite image and display inline.

    Layout is row-major: ``paths[0]``, ``paths[1]`` on top; ``paths[2]``,
    ``paths[3]`` on the bottom. Falls back to sequential display if the list
    does not contain exactly four PNGs.

    Parameters
    ----------
    paths : list of Path
        PNG file paths. Exactly four are expected for the 2x2 grid;
        any other count triggers the sequential fallback.
    caption : str or None
        Optional Markdown caption rendered above the composite image.
    """
    from io import BytesIO
    from IPython.display import display, Image, Markdown
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    if len(paths) != 4:
        for p in paths:
            display(Markdown(f"**{p.stem}**"))
            display(Image(filename=str(p)))
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, p in zip(axes.flatten(), paths):
        ax.imshow(imread(str(p)))
        ax.set_title(p.stem, fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    if caption:
        display(Markdown(f"**{caption}**"))
    display(Image(data=buf.read()))


def _display_pdf(png_path: Path, width: str = "100%", height: str = "700px") -> None:
    """
    Embed the PDF sibling of a PNG inline in the caller notebook.

    Looks for a file with the same stem as ``png_path`` but a ``.pdf``
    extension and renders it via ``IPython.display.IFrame``. The PDF must
    be reachable by the browser through a path relative to the notebook
    server's working directory; silently skips if the PDF is missing.

    Parameters
    ----------
    png_path : Path
        Path to the PNG file whose PDF sibling should be displayed.
    width : str
        CSS width for the IFrame (default ``"100%"``).
    height : str
        CSS height for the IFrame (default ``"700px"``).
    """
    import os
    from IPython.display import display, IFrame, Markdown

    pdf_path = png_path.with_suffix(".pdf")
    if not pdf_path.exists():
        return
    try:
        rel = os.path.relpath(str(pdf_path), start=os.getcwd()).replace(os.sep, "/")
    except ValueError:
        rel = str(pdf_path)
    display(Markdown(f"*PDF: `{pdf_path.name}`*"))
    display(IFrame(src=rel, width=width, height=height))


def show_figures(repo_root: Path | None = None, show_pdfs: bool = False) -> None:
    """
    Display already-generated figures without re-running the figure notebooks.

    Parameters
    ----------
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    show_pdfs : bool
        When *True*, embed PDF versions of the plots via IFrame instead
        of displaying PNGs. Defaults to *False*.
    """
    from IPython.display import display, Image, Markdown

    root = repo_root or _find_repo_root()
    plots_dir = root / "experiments/runs/results/plots"
    all_pngs = sorted(plots_dir.glob("*.png"))
    by_stem = {p.stem: p for p in all_pngs}

    # Explicit 2x2 orderings: (row 0, row 0, row 1, row 1).
    stl_param_order = ["STL_a0", "STL_a1", "STL_c1", "STL_d1"]
    shap_order      = ["shap_a0", "shap_a1", "shap_c1", "shap_d1"]

    stl_param_paths = [by_stem[s] for s in stl_param_order if s in by_stem]
    shap_paths      = [by_stem[s] for s in shap_order      if s in by_stem]

    # Explicit order: trend first, seasonality second.
    decomposition_order = ["STL_Trend", "STL_Seasonality"]
    decomposition_paths = [by_stem[s] for s in decomposition_order if s in by_stem]

    runtime_scaling_paths = [by_stem[s] for s in ["runtime_scaling"] if s in by_stem]

    linear_groups: list[tuple[str, list[Path]]] = [
        ("Runtime Scaling", runtime_scaling_paths),
        ("Hyper-Tree-STL Decomposition", decomposition_paths),
    ]

    grid_groups: list[tuple[str, list[Path]]] = [
        ("Hyper-Tree-STL Estimated Parameters", stl_param_paths),
        ("Hyper-Tree-STL SHAP Values",          shap_paths),
    ]

    # Explicit order: dim=1 before dim=10 (alphabetical puts "dim10" before "dim1").
    embedding_order = [
        "embedding_visualization_dim1_embeddings",
        "embedding_visualization_dim10_embeddings",
    ]
    embedding_paths = [by_stem[s] for s in embedding_order if s in by_stem]

    # Explicit order: Air Passengers, Rossmann, M5 (paper figure order).
    time_varying_order = [
        "time_varying_params_airpassengers",
        "time_varying_params_rossmann",
        "time_varying_params_m5_agg",
    ]
    time_varying_paths = [by_stem[s] for s in time_varying_order if s in by_stem]

    tail_groups: list[tuple[str, list[Path]]] = [
        ("Global Model Forecasts",
         [p for p in all_pngs if p.stem.startswith("example_forecasts")]),
        ("Time-Varying Parameters", time_varying_paths),
        ("Tree Embeddings", embedding_paths),
    ]

    def _render(path: Path) -> None:
        """Render a single figure as PNG (default) or inline PDF."""
        display(Markdown(f"**{path.stem}**"))
        if show_pdfs:
            _display_pdf(path)
        else:
            display(Image(filename=str(path)))

    shown: set[Path] = set()

    # 1. Linear (stacked) groups.
    for heading, paths in linear_groups:
        if not paths:
            continue
        display(Markdown(f"## {heading}"))
        for path in paths:
            _render(path)
            shown.add(path)

    # 2. 2x2 composite groups (PNG only). In PDF mode, fall back to one
    #    iframe per figure in the same order.
    for heading, paths in grid_groups:
        if not paths:
            continue
        display(Markdown(f"## {heading}"))
        if show_pdfs:
            for path in paths:
                _render(path)
        else:
            _display_as_2x2(paths)
        shown.update(paths)

    # 3. Remaining linear groups.
    for heading, paths in tail_groups:
        if not paths:
            continue
        display(Markdown(f"## {heading}"))
        for path in paths:
            _render(path)
            shown.add(path)

    # Anything that didn't match a known group - show it so nothing is
    # silently dropped if new plot families are added later.
    leftover = [p for p in all_pngs if p not in shown]
    if leftover:
        display(Markdown("## Other"))
        for path in leftover:
            _render(path)


def display_figures(repo_root: Path | None = None, show_pdfs: bool = False) -> None:
    """
    Run the figure notebooks and display the results.

    Parameters
    ----------
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.
    show_pdfs : bool
        When *True*, embed PDF versions of the plots via IFrame instead
        of displaying PNGs. Defaults to *False*.
    """
    root = repo_root or _find_repo_root()
    create_figures(repo_root=root)
    show_figures(repo_root=root, show_pdfs=show_pdfs)


def evaluate_fcsts(repo_root: Path | None = None) -> dict:
    """
    Compute the four forecast metrics tables and save them as CSVs.

    Evaluates global, local, Rossmann ablation, and embedding ablation
    forecasts. Each metrics table is written to ``results/metrics/`` as a
    CSV file for later reference.

    Parameters
    ----------
    repo_root : Path or None
        Repository root directory. Discovered automatically when *None*.

    Returns
    -------
    dict
        Mapping of category name (``"global"``, ``"local"``,
        ``"ablation_rossmann"``, ``"ablation_embeddings"``) to its
        metrics DataFrame, or *None* when evaluation failed for that
        category.
    """
    import pandas as pd
    from experiments.utils import (
        evaluate_forecasts,
        evaluate_ablation_rossmann,
        evaluate_ablation_embeddings,
    )

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None

    root = repo_root or _find_repo_root()
    results = root / "experiments/runs/results"
    metrics_dir = results / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_dir = str(results) + "/"

    categories = [
        ("global",              lambda: evaluate_forecasts(results_dir, train_type="global")),
        ("local",               lambda: evaluate_forecasts(results_dir, train_type="local")),
        ("ablation_rossmann",   lambda: evaluate_ablation_rossmann(results_dir)),
        ("ablation_embeddings", lambda: evaluate_ablation_embeddings(results_dir)),
    ]
    metrics: dict = {}
    for name, fn in categories:
        try:
            df = fn()
            df.to_csv(metrics_dir / f"{name}_metrics.csv")
            metrics[name] = df
        except Exception as e:
            print(f"  [skipped: {name}] {e}", flush=True)
            metrics[name] = None
    return metrics
