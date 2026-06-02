"""Centralized output paths.

All experiment, training, and visualization code should route their writes
through the helpers here so that:

  - output directories are created on demand (no more `FileNotFoundError:
    No such file or directory: 'results/plots/...'` for new clones);
  - filenames for sweeps, models, and figures are derived in one place;
  - `results/` can live outside the working tree without callers caring.

Layout:

    <project_root>/
        data/                      input collision datasets (.npy/.csv)
        results/
            h2/
                models/
                    mdn/...        H2 Gaussian MDN weights
                    beta_mdn/...   H2 Beta MDN weights
            o2/
                models/
                    mdn/...        O2 Gaussian MDN weights
                    beta_mdn/...   O2 Beta MDN weights
            plots/                 figures from experiments
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"

H2_MODELS_DIR = RESULTS_DIR / "h2" / "models"
O2_MODELS_DIR = RESULTS_DIR / "o2" / "models"

# Backward-compat aliases — point at H2 since all pre-existing models are H2.
MDN_DIR = H2_MODELS_DIR / "mdn"
BETA_MDN_DIR = H2_MODELS_DIR / "beta_mdn"

H2_MDN_DIR = H2_MODELS_DIR / "mdn"
H2_BETA_MDN_DIR = H2_MODELS_DIR / "beta_mdn"
O2_MDN_DIR = O2_MODELS_DIR / "mdn"
O2_BETA_MDN_DIR = O2_MODELS_DIR / "beta_mdn"


def ensure_dir(path: str | Path) -> Path:
    """Create the directory at *path* (and parents) if it doesn't yet exist.

    Returns the path as a `Path` for convenience.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    """Make sure the parent dir of *path* exists, then return *path* as a Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def model_dir(kind: str, molecule: str | None = None) -> Path:
    """Return the standard directory for a given model kind (creates it).

    Pass *molecule* (e.g. ``"h2"`` or ``"o2"``) to place models under the
    per-molecule subdirectory.  Omitting it falls back to the H2 aliases for
    backward compatibility with callers that pre-date the per-molecule layout.
    """
    kind = kind.lower()
    if molecule is not None:
        base = RESULTS_DIR / molecule.lower() / "models"
        if kind == "mdn":
            return ensure_dir(base / "mdn")
        if kind in ("beta_mdn", "betamdn", "beta"):
            return ensure_dir(base / "beta_mdn")
        return ensure_dir(base / kind)
    # Legacy fallback (molecule not specified → H2 aliases).
    if kind == "mdn":
        return ensure_dir(MDN_DIR)
    if kind in ("beta_mdn", "betamdn", "beta"):
        return ensure_dir(BETA_MDN_DIR)
    return ensure_dir(MODELS_DIR / kind)


def model_path(kind: str, name: str, molecule: str | None = None) -> Path:
    """Full path for a saved model file, ensuring parent dirs exist."""
    if not name.endswith(".pth"):
        name = name + ".pth"
    return ensure_parent(model_dir(kind, molecule) / name)


def wf_sweep_dir(
    kind: str, tag: str, trainseed: int | None = None, molecule: str | None = None
) -> Path:
    """Standard dir for a weighting-factor sweep.

    When `trainseed` is given, models live under a `trainseed<N>/` subdirectory
    so multiple training-seed runs of the same dataset can coexist, e.g.
    `wf_sweep_dir("mdn", "H2_400000_dataseed42", trainseed=41)` ->
    `.../h2/mdn/weightsensitivity/H2_400000_dataseed42/trainseed41`.
    """
    base = model_dir(kind, molecule) / "weightsensitivity" / tag
    if trainseed is not None:
        base = base / f"trainseed{trainseed}"
    return ensure_dir(base)


def wf_sweep_model_path(
    kind: str,
    tag: str,
    wf: float,
    trainseed: int | None = None,
    molecule: str | None = None,
) -> Path:
    """Standard model filename inside a wf sweep."""
    prefix = "beta_mdn" if kind.startswith("beta") else "mdn"
    fname = f"{prefix}_{tag.split('_')[0]}_wf{str(wf).replace('.', '_')}.pth"
    return ensure_parent(wf_sweep_dir(kind, tag, trainseed, molecule) / fname)


def plot_path(name: str, subdir: str | None = None) -> Path:
    """Full path for a saved figure, ensuring parent dirs exist."""
    target = PLOTS_DIR if subdir is None else PLOTS_DIR / subdir
    return ensure_parent(target / name)


def log_path(name: str, subdir: str | None = None) -> Path:
    """Full path for a simulation log file, ensuring parent dirs exist."""
    target = LOGS_DIR if subdir is None else LOGS_DIR / subdir
    return ensure_parent(target / name)
