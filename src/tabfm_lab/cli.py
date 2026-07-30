"""Command line entry point: ``tabfm-lab``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .data import TASK_BUILDERS, DatasetUnavailableError
from .pipeline import run_benchmark
from .reporting import render_report


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabfm-lab",
        description=(
            "Benchmark Google TabFM against tuned baselines on e-commerce and "
            "sports-betting classification and regression tasks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        choices=sorted(TASK_BUILDERS),
        help="Run only this task. Repeatable. Defaults to all four.",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List tasks and exit.")
    parser.add_argument(
        "--no-tabfm",
        action="store_true",
        help="Skip TabFM and run only the baselines.",
    )
    parser.add_argument(
        "--no-baselines", action="store_true", help="Skip the baselines and run only TabFM."
    )
    parser.add_argument(
        "--backend",
        default="pytorch",
        choices=["pytorch", "jax"],
        help="TabFM backend to load the pretrained weights with.",
    )
    parser.add_argument(
        "--context-rows",
        type=int,
        default=None,
        help="Training rows TabFM reads as in-context examples (library default if unset).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of sampled contexts TabFM averages over.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="TabFM query batch size. Lower it if inference runs out of memory.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help=(
            "Local TabFM checkpoint. Use this when the machine cannot reach "
            "Hugging Face to download the weights."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for TabFM, e.g. 'cpu' or 'cuda'. Defaults to TabFM's own choice.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for results.json and results.md (default: <repo>/reports).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Cache directory for downloaded datasets (default: <repo>/data/raw).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Keep third-party chatter out of the run log.
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if args.list_tasks:
        for name in sorted(TASK_BUILDERS):
            print(name)
        return 0

    if args.no_tabfm and args.no_baselines:
        print("Nothing to run: --no-tabfm and --no-baselines are mutually exclusive.",
              file=sys.stderr)
        return 2

    names = args.tasks or sorted(TASK_BUILDERS)
    tasks = []
    for name in names:
        try:
            task = TASK_BUILDERS[name](args.data_dir)
        except DatasetUnavailableError as exc:
            print(f"\nCould not build task {name!r}:\n{exc}\n", file=sys.stderr)
            continue
        print(task.summary(), flush=True)
        tasks.append(task)

    if not tasks:
        print("No tasks could be built — see the download errors above.", file=sys.stderr)
        return 1

    tabfm_kwargs: dict[str, object] = {"backend": args.backend, "random_state": args.seed}
    for flag, name in (
        (args.context_rows, "context_rows"),
        (args.n_estimators, "n_estimators"),
        (args.batch_size, "batch_size"),
        (args.checkpoint_path, "checkpoint_path"),
        (args.device, "device"),
    ):
        if flag is not None:
            tabfm_kwargs[name] = flag

    report = run_benchmark(
        tasks,
        include_tabfm=not args.no_tabfm,
        include_baselines=not args.no_baselines,
        tabfm_kwargs=tabfm_kwargs,
        seed=args.seed,
    )

    output_dir = args.output or (_repo_root() / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2, default=str))
    markdown = render_report(report)
    (output_dir / "results.md").write_text(markdown)

    print("\n" + markdown)
    print(f"Wrote {output_dir / 'results.json'} and {output_dir / 'results.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
