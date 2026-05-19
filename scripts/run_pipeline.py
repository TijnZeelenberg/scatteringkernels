"""Single command-line entry point for the most common experiment workflows.

The previous workflow required running 3-5 different scripts in the right
order and remembering which output path corresponds to which run. This script
collects everything into one CLI so future users can:

    python scripts/run_pipeline.py train --kind beta_mdn --dataset data/...
    python scripts/run_pipeline.py wf-sweep --kind beta_mdn --dataset data/...
    python scripts/run_pipeline.py relaxation --species H2 --model-kind beta_mdn --model results/...
    python scripts/run_pipeline.py viscosity --model results/...
    python scripts/run_pipeline.py wf-sweep-eval --kind beta_mdn --tag H2_400000

All sub-commands route their output through `paths.py`, so the required
directories are created on the fly — fresh clones don't need to mkdir anything.
"""

from __future__ import annotations

import argparse
import sys

# Make sure project-root imports work whether you run this as
#   python scripts/run_pipeline.py ...
# or from any other directory.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import paths  # noqa: E402
from experiments.energy_relaxation import (  # noqa: E402
    SimulationParams,
    load_beta_mdn,
    load_mdn,
    load_sparta_reference,
    plot_relaxation_comparison,
    print_relaxation_table,
    run_relaxation_comparison,
)
from experiments.viscosity import green_kubo_viscosity  # noqa: E402
from physics.borgnakkelarssen_model import borgnakke_larssen_model  # noqa: E402
from physics.species import Species  # noqa: E402
from training.core import train_collision_model  # noqa: E402
from training.wfsweep import run_wf_sweep  # noqa: E402
from visualization.wfsweep import run_wf_sweep_experiments  # noqa: E402


SPECIES_REGISTRY = {"H2": Species.H2, "O2": Species.O2}


def _species_from_arg(name: str) -> Species:
    if name not in SPECIES_REGISTRY:
        raise SystemExit(f"Unknown species: {name!r}. Known: {list(SPECIES_REGISTRY)}")
    return SPECIES_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_train(args):
    output = args.output or paths.model_path(args.kind, f"{args.kind}_run")
    train_collision_model(
        kind=args.kind,
        datapath=args.dataset,
        outputpath=str(output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wf=args.wf,
        T_eq=args.T_eq,
        patience=args.patience,
        showplots=args.showplots,
    )


def cmd_wf_sweep(args):
    run_wf_sweep(
        kind=args.kind,
        datapath=args.dataset,
        tag=args.tag,
        trainseed=args.trainseed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )


def cmd_wf_sweep_eval(args):
    species = _species_from_arg(args.species)
    run_wf_sweep_experiments(
        kind=args.kind,
        tag=args.tag,
        species=species,
        sparta_path=args.sparta,
        trainseed=args.trainseed,
    )


def cmd_relaxation(args):
    species = _species_from_arg(args.species)
    randomseed = args.randomseed
    params = SimulationParams(
        nr_steps=args.nr_steps,
        trans_temperature=args.trans_T,
        rot_temperature=args.rot_T,
        randomseed=randomseed,
    )

    models: dict[str, object] = {}
    if args.model is not None:
        loader = load_beta_mdn if args.model_kind.startswith("beta") else load_mdn
        models[args.model_kind.upper()] = loader(args.model, randomseed=randomseed)
    if args.include_bl:
        models["BL"] = borgnakke_larssen_model(randomseed=randomseed)
    if not models:
        raise SystemExit("Pick at least one of --model or --include-bl.")

    results = run_relaxation_comparison(species, models, params=params)
    sparta = load_sparta_reference(args.sparta) if args.sparta else None

    print_relaxation_table(results, sparta, rot_temperature_initial=params.rot_temperature)

    if args.output:
        plot_relaxation_comparison(results, sparta, output_path=args.output)


def cmd_viscosity(args):
    species = _species_from_arg(args.species)
    randomseed = args.randomseed
    params = SimulationParams(
        nr_steps=args.nr_steps,
        trans_temperature=args.T, rot_temperature=args.T,
        randomseed=randomseed,
    )

    loader = load_beta_mdn if args.model_kind.startswith("beta") else load_mdn
    model = loader(args.model, randomseed=randomseed)

    from experiments.energy_relaxation import run_relaxation

    stats = run_relaxation(species, model, params=params)
    visc = green_kubo_viscosity(
        stats,
        dt=params.dt,
        volume=params.box_size ** 3,
        equilibration_steps=args.equilibration,
        max_lag=args.max_lag,
    )
    print(f"T_eq:      {visc.T_eq:.2f} K")
    print(f"Viscosity: {visc.viscosity:.6e} Pa·s")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_pipeline", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train a single MDN or Beta MDN.")
    t.add_argument("--kind", choices=["mdn", "beta_mdn"], required=True)
    t.add_argument("--dataset", required=True)
    t.add_argument("--output", default=None)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--wf", type=float, default=1.0)
    t.add_argument("--T-eq", dest="T_eq", type=float, default=None,
                   help="Equilibrium temperature [K] for NTC importance weighting (overrides --wf).")
    t.add_argument("--patience", type=int, default=30)
    t.add_argument("--showplots", action="store_true")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("wf-sweep", help="Train a weighting-factor sweep.")
    s.add_argument("--kind", choices=["mdn", "beta_mdn"], required=True)
    s.add_argument("--dataset", required=True)
    s.add_argument("--tag", required=True, help="Output directory tag, e.g. H2_400000.")
    s.add_argument(
        "--trainseed",
        type=int,
        default=None,
        help="Training random seed; when set, models go under trainseed<N>/ in the sweep dir.",
    )
    s.add_argument("--epochs", type=int, default=100)
    s.add_argument("--batch-size", type=int, default=128)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--patience", type=int, default=100)
    s.set_defaults(func=cmd_wf_sweep)

    e = sub.add_parser("wf-sweep-eval", help="Run DSMC experiments for a wf sweep.")
    e.add_argument("--kind", choices=["mdn", "beta_mdn"], required=True)
    e.add_argument("--tag", required=True)
    e.add_argument(
        "--trainseed",
        type=int,
        default=None,
        help="If the sweep was trained with a trainseed, load from trainseed<N>/ subdir.",
    )
    e.add_argument("--species", default="H2")
    e.add_argument("--sparta", required=True)
    e.set_defaults(func=cmd_wf_sweep_eval)

    r = sub.add_parser("relaxation", help="Run an energy-relaxation experiment.")
    r.add_argument("--species", default="H2")
    r.add_argument("--model", default=None)
    r.add_argument("--model-kind", choices=["mdn", "beta_mdn"], default="mdn")
    r.add_argument("--sparta", default=None)
    r.add_argument("--nr-steps", type=int, default=100)
    r.add_argument("--trans-T", type=float, default=300.0)
    r.add_argument("--rot-T", type=float, default=100.0)
    r.add_argument("--randomseed", type=int, default=1)
    r.add_argument("--include-bl", action="store_true")
    r.add_argument("--output", default=None)
    r.set_defaults(func=cmd_relaxation)

    v = sub.add_parser("viscosity", help="Compute Green-Kubo viscosity for a model.")
    v.add_argument("--species", default="H2")
    v.add_argument("--model", required=True)
    v.add_argument("--model-kind", choices=["mdn", "beta_mdn"], default="mdn")
    v.add_argument("--nr-steps", type=int, default=200)
    v.add_argument("-T", "--T", type=float, default=220.0)
    v.add_argument("--equilibration", type=int, default=50)
    v.add_argument("--max-lag", type=int, default=100)
    v.add_argument("--randomseed", type=int, default=1)
    v.set_defaults(func=cmd_viscosity)

    return p


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
