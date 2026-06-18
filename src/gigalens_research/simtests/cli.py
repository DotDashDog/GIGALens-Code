"""Command-line interface for the simtests framework.

Sub-commands
------------
``generate``   Generate (or load/adapt) the system dataset for a campaign.
``run``        Run the inference loop (with optional sharding for Slurm arrays).
``aggregate``  Aggregate completed runs into figures and a refreshed index.csv.
``plot``       Render per-run diagnostic plots from saved arrays (CPU, shardable).
``status``     Print a quick progress summary without touching any results.

Usage examples::

    # Generate 100 GL2 Sérsic systems
    python -m gigalens_research.simtests generate experiments/hundred_systems_GL2/campaign.yaml

    # Run inference (single process, all systems)
    python -m gigalens_research.simtests run experiments/hundred_systems_GL2/campaign.yaml

    # Slurm array: task i of N
    python -m gigalens_research.simtests run campaign.yaml --shard 3/64

    # Aggregate figures
    python -m gigalens_research.simtests aggregate campaign.yaml

    # Progress check
    python -m gigalens_research.simtests status campaign.yaml
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import List, Optional


def _load_plugins(campaign_spec: Any) -> None:  # type: ignore[misc]
    """Import all plugin modules listed in the campaign spec."""
    for mod_path in getattr(campaign_spec, "plugins", []) or []:
        try:
            importlib.import_module(mod_path)
        except ImportError as exc:
            print(f"[cli] warning: could not import plugin {mod_path!r}: {exc}", file=sys.stderr)


def _load_default_experiments() -> None:
    """Import the built-in experiment modules so their builders are registered."""
    for mod in (
        "gigalens_research.simtests.experiments.gl2_sersic",
        "gigalens_research.simtests.experiments.vela_shapelets",
    ):
        try:
            importlib.import_module(mod)
        except ImportError:
            pass


def _load_campaign(yaml_path: str) -> "Any":  # type: ignore[misc]
    from .config import CampaignSpec
    spec = CampaignSpec.from_yaml(yaml_path)
    _load_default_experiments()
    _load_plugins(spec)
    return spec


def cmd_generate(args: argparse.Namespace) -> None:
    from .generate import generate_campaign

    spec = _load_campaign(args.campaign)
    base_dir = os.path.expanduser(args.output_dir or spec.output_dir)
    generate_campaign(spec, base_dir, force=args.force, verbose=not args.quiet)


def cmd_run(args: argparse.Namespace) -> None:
    from .run import run_campaign

    if getattr(args, "distributed", False):
        import jax
        jax.distributed.initialize()

    spec = _load_campaign(args.campaign)
    base_dir = os.path.expanduser(args.output_dir or spec.output_dir)

    shard_i, shard_n = _parse_shard(args.shard)

    run_campaign(
        spec,
        base_dir,
        shard_i=shard_i,
        shard_n=shard_n,
        verbose=not args.quiet,
        skip_existing=not args.rerun,
    )


def cmd_aggregate(args: argparse.Namespace) -> None:
    from .aggregate import aggregate_campaign

    spec = _load_campaign(args.campaign)
    base_dir = os.path.expanduser(args.output_dir or spec.output_dir)
    aggregate_campaign(spec, base_dir, verbose=not args.quiet)


def cmd_plot(args: argparse.Namespace) -> None:
    from .plot import plot_campaign, ALL_PANELS

    spec = _load_campaign(args.campaign)
    base_dir = os.path.expanduser(args.output_dir or spec.output_dir)

    shard_i, shard_n = _parse_shard(args.shard)
    panels = ([p.strip() for p in args.panels.split(",") if p.strip()]
              if args.panels else list(ALL_PANELS))

    plot_campaign(
        spec,
        base_dir,
        panels=panels,
        stage=args.stage,
        z_score_group=args.z_score_group,
        shard_i=shard_i,
        shard_n=shard_n,
        include_failed=args.include_failed,
        overwrite=args.overwrite,
        verbose=not args.quiet,
    )


def cmd_status(args: argparse.Namespace) -> None:
    from .config import CampaignSpec
    from .run import enumerate_runs
    from .system import load_manifest
    from .index import load_run_json

    spec = _load_campaign(args.campaign)
    base_dir = os.path.expanduser(args.output_dir or spec.output_dir)
    dataset_dir = os.path.join(base_dir, "dataset")
    runs_dir = os.path.join(base_dir, "runs")

    try:
        manifest = load_manifest(dataset_dir)
        n_sys = manifest["n_systems"]
    except FileNotFoundError:
        print(f"[status] dataset not found at {dataset_dir}. Run 'generate' first.")
        return

    all_runs = enumerate_runs(spec, dataset_dir)
    n_total = len(all_runs)
    n_ok = n_failed = n_missing = 0
    for system_id, sweep_point in all_runs:
        sweep_name = spec.sweep_dir_name(sweep_point)
        out_dir = os.path.join(runs_dir, system_id, sweep_name)
        rec = load_run_json(out_dir)
        if rec is None:
            n_missing += 1
        elif rec.get("status") == "ok":
            n_ok += 1
        else:
            n_failed += 1

    n_sweep = len(spec.sweep_points)
    print(f"Campaign : {spec.name!r}")
    print(f"Base dir : {base_dir}")
    print(f"Systems  : {n_sys}  ×  {n_sweep} sweep point(s)  =  {n_total} runs total")
    print(f"OK       : {n_ok}")
    print(f"Failed   : {n_failed}")
    print(f"Missing  : {n_missing}")
    if n_total > 0:
        print(f"Progress : {100 * n_ok / n_total:.1f}%")


def _parse_shard(shard_str: str) -> tuple:
    """Parse ``'i/N'`` → ``(i, N)``."""
    if not shard_str or shard_str == "0/1":
        return 0, 1
    parts = shard_str.split("/")
    if len(parts) != 2:
        raise ValueError(f"--shard must be 'i/N' (e.g. '3/64'); got {shard_str!r}")
    return int(parts[0]), int(parts[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gigalens_research.simtests",
        description="GIGALens simulated-system test framework.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Generate the system dataset.")
    p_gen.add_argument("campaign", help="Path to campaign YAML.")
    p_gen.add_argument("--output-dir", default=None,
                       help="Override campaign output_dir.")
    p_gen.add_argument("--force", action="store_true",
                       help="Regenerate even if dataset already exists.")
    p_gen.add_argument("--quiet", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    # --- run ---
    p_run = sub.add_parser("run", help="Run the inference loop.")
    p_run.add_argument("campaign", help="Path to campaign YAML.")
    p_run.add_argument("--output-dir", default=None)
    p_run.add_argument("--shard", default="0/1", metavar="i/N",
                       help="Run only every N-th job, starting at i (0-based). "
                            "Used for Slurm array jobs: set to $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT.")
    p_run.add_argument("--rerun", action="store_true",
                       help="Re-run even if a run.json with status=ok already exists.")
    p_run.add_argument("--distributed", action="store_true",
                       help="Call jax.distributed.initialize() before running "
                            "(required for multi-GPU MAP/HMC on multi-node jobs).")
    p_run.add_argument("--quiet", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # --- aggregate ---
    p_agg = sub.add_parser("aggregate", help="Aggregate results and produce figures.")
    p_agg.add_argument("campaign", help="Path to campaign YAML.")
    p_agg.add_argument("--output-dir", default=None)
    p_agg.add_argument("--quiet", action="store_true")
    p_agg.set_defaults(func=cmd_aggregate)

    # --- plot ---
    p_plot = sub.add_parser(
        "plot", help="Render per-run diagnostic plots from saved arrays (CPU).")
    p_plot.add_argument("campaign", help="Path to campaign YAML.")
    p_plot.add_argument("--output-dir", default=None)
    p_plot.add_argument("--shard", default="0/1", metavar="i/N",
                        help="Render only the strided slice runs[i::N]. Use for "
                             "Slurm arrays: $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT.")
    p_plot.add_argument("--panels", default=None,
                        help="Comma-separated subset of: image,convergence,source,"
                             "corner,z_scores,source_comparison,diagnostics. "
                             "Default: all. Drop 'corner' for fast lightweight plots.")
    p_plot.add_argument("--stage", default=None,
                        help="Stage to report on (default: auto-pick the sampler).")
    p_plot.add_argument("--z-score-group", default="mass",
                        help="Parameter group for the z-score panel "
                             "(mass/lens_light/src_light/all). Default: mass.")
    p_plot.add_argument("--include-failed", action="store_true",
                        help="Also plot runs whose run.json status != ok.")
    p_plot.add_argument("--overwrite", action="store_true",
                        help="Re-render panels even if their PNGs already exist.")
    p_plot.add_argument("--quiet", action="store_true")
    p_plot.set_defaults(func=cmd_plot)

    # --- status ---
    p_st = sub.add_parser("status", help="Print a progress summary.")
    p_st.add_argument("campaign", help="Path to campaign YAML.")
    p_st.add_argument("--output-dir", default=None)
    p_st.set_defaults(func=cmd_status)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


# Allow running as `python gigalens_research/simtests/cli.py` too.
if __name__ == "__main__":
    main()


# Type alias to avoid circular import in type annotations above.
Any = object
