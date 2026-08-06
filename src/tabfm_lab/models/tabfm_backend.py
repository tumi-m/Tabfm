"""Adapter around Google's TabFM tabular foundation model.

TabFM predicts by in-context learning: the training rows are handed to a frozen
pretrained network as context and the query rows are answered in a single
forward pass, with no gradient step on our data. That makes "fitting" here
little more than preparing encoders and stashing the context.

Two practical consequences shape this wrapper:

* The context window is bounded (``max_num_rows``), so on a large training split
  TabFM reads a sample rather than the whole table. ``n_estimators`` re-samples
  the context and averages, trading latency for variance.
* The pretrained weights are fetched from Hugging Face on first use. When that
  download is unavailable the failure is reported as
  :class:`TabFMUnavailableError` so a run can degrade to the baselines instead
  of dying.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any

import numpy as np
import pandas as pd

from ..data.base import TabularTask
from .preprocess import as_sklearn_frame, ordinal_transformer, to_dense_float

LOGGER = logging.getLogger(__name__)

BACKENDS = ("pytorch", "jax")

#: Attribute on a fitted TabFM estimator holding the frozen foundation model.
#: Excluded from pickles and restored on load — see TabFMModel.__getstate__.
_WEIGHTS_ATTRIBUTE = "model"


class TabFMUnavailableError(RuntimeError):
    """TabFM could not be imported, or its pretrained weights could not load."""


def _load_backend(
    backend: str,
    model_type: str,
    *,
    checkpoint_path: str | None = None,
    device: str | None = None,
):
    """Import TabFM and load the pretrained weights for ``model_type``.

    ``checkpoint_path`` points at an already-downloaded checkpoint, which is the
    way to run on a machine that cannot reach Hugging Face.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

    try:
        import tabfm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TabFMUnavailableError(
            "The 'tabfm' package is not installed. Install it with:\n"
            '    pip install "tabfm[pytorch]"\n'
            "See https://github.com/google-research/tabfm"
        ) from exc

    module_name = f"tabfm_v1_0_0_{backend}"
    try:
        weights_module = getattr(tabfm, module_name)
    except AttributeError as exc:
        raise TabFMUnavailableError(
            f"TabFM has no {module_name!r} module. The {backend} backend is probably "
            f'not installed; try: pip install "tabfm[{backend}]"'
        ) from exc

    options: dict[str, Any] = {"model_type": model_type}
    if checkpoint_path:
        options["checkpoint_path"] = str(checkpoint_path)
    if device:
        options["device"] = device

    try:
        return weights_module.load(**_supported_kwargs(weights_module.load, options))
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the caller
        raise TabFMUnavailableError(
            f"Could not load TabFM pretrained weights ({model_type}).\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            "The weights download from huggingface.co on first use. If this machine "
            "cannot reach Hugging Face: download the checkpoint where it can be "
            "reached and pass --checkpoint-path, or point HF_HOME at a populated "
            "cache and set HF_HUB_OFFLINE=1."
        ) from exc


def _supported_kwargs(target: Any, requested: dict[str, Any]) -> dict[str, Any]:
    """Keep only the kwargs ``target`` actually accepts.

    TabFM's tuning knobs have moved between releases; filtering against the live
    signature means a version that renames one degrades to its default instead
    of raising a TypeError mid-run.
    """
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return requested

    accepted = {k: v for k, v in requested.items() if k in parameters}
    for name in set(requested) - set(accepted):
        LOGGER.debug("TabFM does not accept %r on this version; using its default", name)
    return accepted


class TabFMModel:
    """Uniform ``fit``/``predict`` surface over TabFMClassifier and TabFMRegressor."""

    def __init__(
        self,
        task_type: str,
        *,
        backend: str = "pytorch",
        context_rows: int | None = None,
        n_estimators: int | None = None,
        batch_size: int | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.task_type = task_type
        self.backend = backend
        self.context_rows = context_rows
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.random_state = random_state

        self.name = f"TabFM ({backend})"
        self._estimator = None
        self._transformer = None
        self._categorical: list[str] = []
        self.classes_: np.ndarray | None = None

    # -- construction -----------------------------------------------------
    def _build_estimator(self):
        try:
            from tabfm import TabFMClassifier, TabFMRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise TabFMUnavailableError(
                'The "tabfm" package is not installed. Install it with:\n'
                '    pip install "tabfm[pytorch]"'
            ) from exc

        model_type = "regression" if self.task_type == "regression" else "classification"
        weights = _load_backend(
            self.backend,
            model_type,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        cls = TabFMRegressor if self.task_type == "regression" else TabFMClassifier
        requested = {
            # TabFM's own names: context size is max_num_rows, and the query
            # batch is batch_size.
            "max_num_rows": self.context_rows,
            "n_estimators": self.n_estimators,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }
        requested = {k: v for k, v in requested.items() if v is not None}
        return cls(model=weights, **_supported_kwargs(cls, requested))

    # -- sklearn-ish API --------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, categorical: list[str] | None = None):
        self._categorical = list(categorical or [])
        frame = as_sklearn_frame(X, self._categorical)
        self._transformer = ordinal_transformer(frame, self._categorical)
        matrix = to_dense_float(self._transformer.fit_transform(frame))

        target = np.asarray(y)
        if self.task_type == "classification":
            self.classes_ = np.unique(target)

        self._estimator = self._build_estimator()
        self._estimator.fit(matrix, target)
        return self

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        if self._transformer is None:
            raise RuntimeError("fit() must be called before predicting")
        return to_dense_float(self._transformer.transform(as_sklearn_frame(X, self._categorical)))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_estimator()
        return np.asarray(self._estimator.predict(self._matrix(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only defined for classification")
        self._ensure_estimator()
        return np.asarray(self._estimator.predict_proba(self._matrix(X)))

    # -- pickling ---------------------------------------------------------
    # A fitted TabFM estimator holds two very different things: the in-context
    # training rows (small, and genuinely part of the fitted state) and a
    # reference to the frozen foundation model (hundreds of megabytes, identical
    # for every task). Pickling naively would copy the weights into every
    # artifact. Instead the weights are dropped on the way out and reloaded on
    # first use, which TabFM's own process-wide load cache makes nearly free
    # when several artifacts are served from one process.

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        estimator = state.get("_estimator")
        if estimator is not None:
            inner = estimator.__dict__.copy()
            inner.pop(_WEIGHTS_ATTRIBUTE, None)
            state["_estimator_blueprint"] = {
                "module": type(estimator).__module__,
                "qualname": type(estimator).__qualname__,
                "state": inner,
            }
        state["_estimator"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Rebuilt on first prediction so unpickling stays cheap and cannot fail
        # on a machine that is only inspecting artifact metadata.
        self._estimator = None

    def _ensure_estimator(self) -> None:
        """Rebuild the fitted estimator from a blueprint after unpickling."""
        if self._estimator is not None:
            return

        blueprint = self.__dict__.pop("_estimator_blueprint", None)
        if blueprint is None:
            raise RuntimeError("fit() must be called before predicting")

        module = importlib.import_module(blueprint["module"])
        estimator_class = getattr(module, blueprint["qualname"])

        model_type = "regression" if self.task_type == "regression" else "classification"
        weights = _load_backend(
            self.backend,
            model_type,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        estimator = estimator_class.__new__(estimator_class)
        estimator.__dict__.update(blueprint["state"])
        setattr(estimator, _WEIGHTS_ATTRIBUTE, weights)
        self._estimator = estimator


def build_tabfm(task: TabularTask, **kwargs) -> TabFMModel:
    """Construct a TabFM model matching ``task``'s objective."""
    return TabFMModel(task.task_type, **kwargs)


# --------------------------------------------------------------------------
# Introspection, for the UI's "how TabFM is used" view
# --------------------------------------------------------------------------
#: Hugging Face repository the pretrained weights come from.
WEIGHTS_REPO = "google/tabfm-1.0.0-pytorch"


def availability(
    backend: str = "pytorch",
    *,
    probe_weights: bool = False,
    checkpoint_path: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Report whether TabFM can actually run here, and why not if it cannot.

    ``probe_weights`` is off by default because loading them downloads hundreds
    of megabytes on first use — far too costly for a page that merely wants to
    display status. Pass it only in response to an explicit user action.
    """
    info: dict[str, Any] = {
        "package_installed": False,
        "package_version": None,
        "backend": backend,
        "backend_available": False,
        "weights_probed": probe_weights,
        "weights_loadable": False,
        "weights_repo": WEIGHTS_REPO,
        "detail": "",
    }

    try:
        import tabfm
    except ImportError as exc:
        info["detail"] = f"{exc}. Install with: pip install \"tabfm[{backend}]\""
        return info

    info["package_installed"] = True
    try:
        from importlib.metadata import version

        info["package_version"] = version("tabfm")
    except Exception:  # noqa: BLE001 - a missing version is cosmetic
        info["package_version"] = "unknown"

    module_name = f"tabfm_v1_0_0_{backend}"
    if not hasattr(tabfm, module_name):
        info["detail"] = (
            f"The {backend} backend is not installed "
            f'(tabfm has no {module_name!r}). Try: pip install "tabfm[{backend}]"'
        )
        return info
    info["backend_available"] = True

    if not probe_weights:
        info["detail"] = "Package and backend present; weights not checked."
        return info

    try:
        _load_backend(backend, "classification", checkpoint_path=checkpoint_path, device=device)
    except TabFMUnavailableError as exc:
        info["detail"] = str(exc)
        return info

    info["weights_loadable"] = True
    info["detail"] = "TabFM is ready: package, backend and pretrained weights all load."
    return info


def constructor_parameters(task_type: str = "classification") -> dict[str, Any]:
    """Defaults of the installed TabFM estimator, read from its live signature.

    Reading them rather than documenting them means this cannot drift from the
    version actually installed.
    """
    from tabfm import TabFMClassifier, TabFMRegressor

    estimator = TabFMRegressor if task_type == "regression" else TabFMClassifier
    return {
        name: parameter.default
        for name, parameter in inspect.signature(estimator).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def wrapper_source() -> str:
    """This module's own source, so the UI can show the real integration."""
    from pathlib import Path

    return Path(__file__).read_text(encoding="utf-8")
