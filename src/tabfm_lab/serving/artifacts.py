"""Pickled model artifacts and the registry that serves them.

An artifact bundles a fitted model with everything needed to score a request
without reloading the training data: the feature order, which columns are
categorical, class labels, any target transform, sensible per-field defaults for
a UI to prefill, and — for the sports tasks — the fitted
:class:`~tabfm_lab.data.sports.MatchFeaturizer` so a fixture can be scored from
two team names.

Security note: unpickling executes arbitrary code from the file. The registry
therefore loads only from a configured directory that the operator controls, and
never from request data. Treat artifacts as trusted deployable code, not as user
input.
"""

from __future__ import annotations

import logging
import pickle
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.base import TabularTask

LOGGER = logging.getLogger(__name__)

ARTIFACT_SUFFIX = ".pkl"
#: Bumped when the artifact layout changes incompatibly.
ARTIFACT_FORMAT_VERSION = 1
#: Protocol 5 (Python 3.8+) supports out-of-band buffers for large arrays.
PICKLE_PROTOCOL = 5


def _json_safe(value: Any) -> Any:
    """Coerce numpy/pandas scalars into something JSON can represent."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.ndarray, pd.Series)):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


@dataclass
class FieldSpec:
    """Describes one input field, so a client can render a form for it."""

    name: str
    dtype: str
    categorical: bool
    default: Any = None
    choices: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: _json_safe(v) for k, v in asdict(self).items()}


@dataclass
class ArtifactMetadata:
    """Everything about a served model except the model itself."""

    task: str
    industry: str
    task_type: str
    target: str
    model_name: str
    feature_columns: list[str]
    categorical_features: list[str]
    fields: list[FieldSpec] = field(default_factory=list)
    classes: list[Any] | None = None
    target_transform: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    description: str = ""
    teams: list[str] | None = None
    trained_at: str = ""
    format_version: int = ARTIFACT_FORMAT_VERSION
    versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fields"] = [f.to_dict() for f in self.fields]
        payload["classes"] = _json_safe(self.classes) if self.classes is not None else None
        payload["metrics"] = {k: _json_safe(v) for k, v in self.metrics.items()}
        return payload


def _environment_versions() -> dict[str, str]:
    """Record the versions an artifact was produced with.

    A pickle is only loadable against compatible libraries, so the mismatch that
    breaks it later is worth capturing at write time.

    Every value is coerced with ``str()``, which is load-bearing rather than
    cosmetic: ``torch.__version__`` is a ``torch.torch_version.TorchVersion``
    instance, not a plain string, so storing it as-is pickles a reference to a
    torch class. That made every artifact — including ones holding nothing but a
    scikit-learn model — refuse to unpickle anywhere torch was not installed.
    """
    versions = {"python": platform.python_version()}
    for module in ("numpy", "pandas", "sklearn", "tabfm", "torch"):
        try:
            version = __import__(module).__version__
        except Exception:  # noqa: BLE001 - absence is informative, not fatal
            continue
        versions[module] = str(version)
    return versions


def _describe_fields(
    frame: pd.DataFrame, categorical: list[str], max_choices: int = 40
) -> list[FieldSpec]:
    """Summarise each training column into a form-renderable spec.

    Defaults come from the training distribution — median for numerics, mode for
    categoricals — so a client can prefill a realistic request instead of
    demanding all 35 engineered features from a human.
    """
    specs: list[FieldSpec] = []
    for column in frame.columns:
        series = frame[column]
        is_categorical = column in categorical

        if is_categorical:
            values = series.astype(str)
            counts = values.value_counts()
            choices = list(counts.index[:max_choices])
            specs.append(
                FieldSpec(
                    name=column,
                    dtype="string",
                    categorical=True,
                    default=choices[0] if choices else None,
                    choices=choices,
                )
            )
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            specs.append(
                FieldSpec(
                    name=column,
                    dtype="boolean" if series.dtype == bool else "number",
                    categorical=False,
                    default=float(numeric.median()),
                    minimum=float(numeric.min()),
                    maximum=float(numeric.max()),
                )
            )
        else:
            specs.append(FieldSpec(name=column, dtype="string", categorical=False))
    return specs


@dataclass
class ModelArtifact:
    """A fitted model plus its metadata, persisted as a single pickle."""

    metadata: ArtifactMetadata
    model: Any
    featurizer: Any = None

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and rename, so a reader never sees a half-written
        # artifact if the process dies mid-dump.
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(self, handle, protocol=PICKLE_PROTOCOL)
        tmp.replace(path)
        LOGGER.info("Wrote artifact %s (%.1f KiB)", path, path.stat().st_size / 1024)
        return path

    @staticmethod
    def load(path: Path) -> ModelArtifact:
        with Path(path).open("rb") as handle:
            artifact = pickle.load(handle)
        if not isinstance(artifact, ModelArtifact):
            raise TypeError(f"{path} does not contain a ModelArtifact")
        if artifact.metadata.format_version != ARTIFACT_FORMAT_VERSION:
            LOGGER.warning(
                "Artifact %s has format version %s; this build expects %s",
                path,
                artifact.metadata.format_version,
                ARTIFACT_FORMAT_VERSION,
            )
        return artifact

    # -- inference --------------------------------------------------------
    def frame_from_records(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a model-ready frame, filling absent fields with defaults.

        Column order is restored from the artifact rather than trusted from the
        request, because a silently reordered frame would still predict — just
        wrongly.
        """
        defaults = {f.name: f.default for f in self.metadata.fields}
        rows = []
        for record in records:
            unknown = set(record) - set(self.metadata.feature_columns)
            if unknown:
                raise ValueError(f"Unknown feature(s): {sorted(unknown)}")
            rows.append({name: record.get(name, defaults.get(name)) for name in
                         self.metadata.feature_columns})
        return pd.DataFrame(rows, columns=self.metadata.feature_columns)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray | None:
        if self.metadata.task_type != "classification":
            return None
        try:
            return np.asarray(self.model.predict_proba(frame))
        except (AttributeError, NotImplementedError):
            return None


def build_artifact(
    task: TabularTask, model: Any, metrics: dict[str, float] | None = None
) -> ModelArtifact:
    """Package a model fitted on ``task`` for serving."""
    classes = getattr(model, "classes_", None)
    metadata = ArtifactMetadata(
        task=task.name,
        industry=task.industry,
        task_type=task.task_type,
        target=task.target,
        model_name=getattr(model, "name", type(model).__name__),
        feature_columns=list(task.X_train.columns),
        categorical_features=list(task.categorical_features),
        fields=_describe_fields(task.X_train, task.categorical_features),
        classes=list(classes) if classes is not None else None,
        target_transform=task.target_transform,
        metrics=dict(metrics or {}),
        n_train=len(task.X_train),
        n_test=len(task.X_test),
        description=task.description,
        teams=task.extras.get("teams"),
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        versions=_environment_versions(),
    )
    return ModelArtifact(
        metadata=metadata, model=model, featurizer=task.extras.get("featurizer")
    )


class ArtifactRegistry:
    """Loads every artifact in a directory and serves them by task name."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._artifacts: dict[str, ModelArtifact] = {}
        self._errors: dict[str, str] = {}

    def load(self) -> ArtifactRegistry:
        self._artifacts.clear()
        self._errors.clear()

        if not self.directory.is_dir():
            LOGGER.warning("Artifact directory %s does not exist", self.directory)
            return self

        for path in sorted(self.directory.glob(f"*{ARTIFACT_SUFFIX}")):
            try:
                artifact = ModelArtifact.load(path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not blank the service
                self._errors[path.name] = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Failed to load artifact %s", path)
                continue
            self._artifacts[artifact.metadata.task] = artifact
            LOGGER.info("Loaded artifact %s (%s)", artifact.metadata.task,
                        artifact.metadata.model_name)
        return self

    def __len__(self) -> int:
        return len(self._artifacts)

    def __contains__(self, task: str) -> bool:
        return task in self._artifacts

    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def names(self) -> list[str]:
        return sorted(self._artifacts)

    def get(self, task: str) -> ModelArtifact:
        try:
            return self._artifacts[task]
        except KeyError:
            raise KeyError(
                f"No model named {task!r}. Available: {self.names() or 'none'}"
            ) from None

    def all(self) -> list[ModelArtifact]:
        return [self._artifacts[name] for name in self.names()]
