"""Fit models and write them out as pickled artifacts for the API to serve.

Separating training from serving keeps the request path free of data downloads
and model fitting: the container that answers predictions only ever unpickles.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..data import TASK_BUILDERS
from ..data.base import TabularTask
from ..data.sources import DatasetUnavailableError
from ..evaluation.metrics import evaluate
from ..models.baselines import build_baselines
from ..models.tabfm_backend import TabFMModel, TabFMUnavailableError
from .artifacts import ARTIFACT_SUFFIX, ModelArtifact, build_artifact

LOGGER = logging.getLogger(__name__)

#: --model choice -> substring identifying the baseline to use.
_BASELINE_CHOICES = {"boosting": "HistGradientBoosting", "linear": "Regression"}

MODEL_CHOICES = ("auto", "tabfm", "boosting", "linear")


def default_artifact_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "artifacts"


def _baseline(task: TabularTask, kind: str, seed: int):
    wanted = _BASELINE_CHOICES[kind]
    for model in build_baselines(task, seed=seed):
        if wanted in model.name:
            return model
    raise ValueError(f"No baseline matching {kind!r} for {task.task_type}")


def select_model(task: TabularTask, choice: str, *, seed: int = 42, **tabfm_kwargs):
    """Return an unfitted model for ``choice``.

    ``auto`` prefers TabFM and silently falls back to gradient boosting when the
    weights cannot be loaded, so a deployment without Hugging Face access still
    produces a complete, servable set of artifacts.
    """
    if choice in _BASELINE_CHOICES:
        return _baseline(task, choice, seed)
    if choice == "tabfm":
        return TabFMModel(task.task_type, **tabfm_kwargs)
    if choice != "auto":
        raise ValueError(f"Unknown model choice {choice!r}")

    candidate = TabFMModel(task.task_type, **tabfm_kwargs)
    try:
        # Touch the weights now so the fallback happens before any fitting work.
        candidate._build_estimator()  # noqa: SLF001 - deliberate availability probe
    except TabFMUnavailableError as exc:
        LOGGER.warning("TabFM unavailable, falling back to gradient boosting: %s",
                       str(exc).splitlines()[0])
        return _baseline(task, "boosting", seed)
    return candidate


def train_task(
    task: TabularTask,
    output_dir: Path,
    *,
    model_choice: str = "auto",
    seed: int = 42,
    **tabfm_kwargs,
) -> Path:
    """Fit one task, score it on the held-out split, and pickle the result."""
    model = select_model(task, model_choice, seed=seed, **tabfm_kwargs)
    LOGGER.info("Fitting %s on %s", getattr(model, "name", model), task.name)
    model.fit(task.X_train, task.y_train, task.categorical_features)

    predictions = model.predict(task.X_test)
    probabilities = None
    classes = getattr(model, "classes_", None)
    if task.task_type == "classification":
        try:
            probabilities = model.predict_proba(task.X_test)
        except (AttributeError, NotImplementedError):
            probabilities = None
    metrics = evaluate(task, predictions, probabilities, classes)

    artifact = build_artifact(task, model, metrics)
    return artifact.save(output_dir / f"{task.name}{ARTIFACT_SUFFIX}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabfm-lab-train",
        description="Fit models and write pickled artifacts for the serving API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", dest="tasks", action="append", choices=sorted(TASK_BUILDERS),
        help="Task to train. Repeatable. Defaults to all.",
    )
    parser.add_argument(
        "--model", default="auto", choices=MODEL_CHOICES,
        help="Which model to fit. 'auto' prefers TabFM and falls back to boosting.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Artifact directory (default: <repo>/artifacts).",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset cache directory.")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "jax"])
    parser.add_argument("--checkpoint-path", default=None, help="Local TabFM checkpoint.")
    parser.add_argument("--device", default=None, help="Device for TabFM, e.g. cpu or cuda.")
    parser.add_argument("--context-rows", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    tabfm_kwargs = {"backend": args.backend, "random_state": args.seed}
    for value, name in (
        (args.checkpoint_path, "checkpoint_path"),
        (args.device, "device"),
        (args.context_rows, "context_rows"),
        (args.n_estimators, "n_estimators"),
    ):
        if value is not None:
            tabfm_kwargs[name] = value

    output_dir = args.output or default_artifact_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name in args.tasks or sorted(TASK_BUILDERS):
        try:
            task = TASK_BUILDERS[name](args.data_dir)
        except DatasetUnavailableError as exc:
            print(f"\nSkipping {name}: {exc}\n", file=sys.stderr)
            continue
        written.append(
            train_task(
                task, output_dir, model_choice=args.model, seed=args.seed, **tabfm_kwargs
            )
        )

    if not written:
        print("No artifacts written.", file=sys.stderr)
        return 1

    print(f"\nWrote {len(written)} artifact(s) to {output_dir}:")
    for path in written:
        artifact = ModelArtifact.load(path)
        summary = ", ".join(f"{k}={v:.4f}" for k, v in list(artifact.metadata.metrics.items())[:3])
        print(f"  {path.name:34s} {artifact.metadata.model_name:22s} {summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
