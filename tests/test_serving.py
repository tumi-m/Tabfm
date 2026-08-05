"""Tests for artifact persistence and the HTTP API.

Everything here builds its own tiny models on synthetic data, so the suite stays
offline and fast — no dataset downloads, no foundation-model weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# tests/ is not a package, so pytest puts this directory on sys.path directly.
from test_sports_features import synthetic_matches

from tabfm_lab.data.base import TabularTask
from tabfm_lab.data.sports import MatchFeaturizer, build_match_features
from tabfm_lab.models.baselines import build_baselines
from tabfm_lab.serving.artifacts import (
    ArtifactRegistry,
    ModelArtifact,
    build_artifact,
)


def _classification_task(name: str = "demo-classification") -> TabularTask:
    rng = np.random.default_rng(0)
    n = 240
    frame = pd.DataFrame(
        {
            "score": rng.normal(size=n),
            "tier": rng.choice(["bronze", "silver", "gold"], size=n),
        }
    )
    target = pd.Series(np.where(frame["score"] > 0, 1, 0), name="converted")
    half = n // 2
    return TabularTask(
        name=name,
        industry="ecommerce",
        task_type="classification",
        target="converted",
        X_train=frame.iloc[:half].reset_index(drop=True),
        y_train=target.iloc[:half].reset_index(drop=True),
        X_test=frame.iloc[half:].reset_index(drop=True),
        y_test=target.iloc[half:].reset_index(drop=True),
        categorical_features=["tier"],
        description="synthetic",
    )


def _fitted_artifact(task: TabularTask) -> ModelArtifact:
    model = next(m for m in build_baselines(task) if "Boosting" in m.name)
    model.fit(task.X_train, task.y_train, task.categorical_features)
    return build_artifact(task, model, {"accuracy": 0.9})


# ---------------------------------------------------------------- artifacts
def test_artifact_round_trips_through_pickle(tmp_path) -> None:
    task = _classification_task()
    artifact = _fitted_artifact(task)
    path = artifact.save(tmp_path / "demo.pkl")

    loaded = ModelArtifact.load(path)
    assert loaded.metadata.task == task.name
    assert loaded.metadata.feature_columns == list(task.X_train.columns)

    before = artifact.predict(task.X_test)
    after = loaded.predict(task.X_test)
    np.testing.assert_array_equal(before, after)


def test_saved_artifact_records_field_defaults(tmp_path) -> None:
    task = _classification_task()
    artifact = _fitted_artifact(task)

    specs = {f.name: f for f in artifact.metadata.fields}
    assert specs["tier"].categorical and set(specs["tier"].choices) <= {"bronze", "silver", "gold"}
    assert specs["score"].categorical is False
    assert specs["score"].default == pytest.approx(task.X_train["score"].median())


def test_recorded_versions_are_plain_strings() -> None:
    """Metadata must not smuggle a library's own version type into the pickle.

    `torch.__version__` is a TorchVersion instance, not a str. Storing it
    directly pickled a reference to a torch class, so every artifact — even one
    holding only a scikit-learn model — failed to unpickle without torch
    installed. A str subclass must be flattened, not merely stringy.
    """
    from tabfm_lab.serving import artifacts as artifacts_module

    versions = artifacts_module._environment_versions()
    assert versions, "expected at least the Python version"
    for name, value in versions.items():
        assert type(value) is str, f"{name} version is {type(value)}, not a plain str"


def test_artifact_pickle_does_not_reference_optional_dependencies(tmp_path) -> None:
    """A baseline artifact must carry no class from an optional dependency.

    The plain strings "torch" and "tabfm" may legitimately appear — they are
    dict keys in the recorded versions. What must not appear is a *class path*,
    because that is what forces an import at load time. Neither the serving
    image nor the hosted app installs these, so a stray reference makes the
    artifact unloadable there.
    """
    artifact = _fitted_artifact(_classification_task())
    path = artifact.save(tmp_path / "portable.pkl")
    payload = path.read_bytes()

    for reference in (b"torch.torch_version", b"TorchVersion", b"tabfm.src"):
        assert reference not in payload, f"artifact pickle references {reference!r}"


def test_missing_fields_fall_back_to_defaults(tmp_path) -> None:
    artifact = _fitted_artifact(_classification_task())
    frame = artifact.frame_from_records([{}])

    assert list(frame.columns) == artifact.metadata.feature_columns
    assert frame.notna().all().all()


def test_unknown_feature_is_rejected() -> None:
    artifact = _fitted_artifact(_classification_task())
    with pytest.raises(ValueError, match="Unknown feature"):
        artifact.frame_from_records([{"not_a_column": 1}])


def test_record_order_is_taken_from_the_artifact() -> None:
    """A client sending columns in a different order must still score correctly."""
    artifact = _fitted_artifact(_classification_task())
    frame = artifact.frame_from_records([{"tier": "gold", "score": 1.5}])
    assert list(frame.columns) == artifact.metadata.feature_columns


def test_registry_skips_a_corrupt_artifact(tmp_path) -> None:
    _fitted_artifact(_classification_task()).save(tmp_path / "good.pkl")
    (tmp_path / "broken.pkl").write_bytes(b"not a pickle at all")

    registry = ArtifactRegistry(tmp_path).load()
    assert registry.names() == ["demo-classification"]
    assert "broken.pkl" in registry.errors


def test_registry_on_missing_directory_is_empty(tmp_path) -> None:
    registry = ArtifactRegistry(tmp_path / "nope").load()
    assert len(registry) == 0
    assert registry.names() == []


def test_featurizer_survives_pickling(tmp_path) -> None:
    """The sports featuriser must reload intact — it has no lambdas by design."""
    matches = synthetic_matches()
    featurizer = MatchFeaturizer()
    build_match_features(matches, featurizer)

    task = _classification_task("sports-demo")
    task.extras["featurizer"] = featurizer
    task.extras["teams"] = featurizer.teams()
    artifact = _fitted_artifact(task)
    artifact.featurizer = featurizer
    artifact.metadata.teams = featurizer.teams()

    path = artifact.save(tmp_path / "sports.pkl")
    reloaded = ModelArtifact.load(path)

    assert reloaded.featurizer.teams() == featurizer.teams()
    home, away = featurizer.teams()[:2]
    original = featurizer.features_for(home, away)
    restored = reloaded.featurizer.features_for(home, away)
    assert original["elo_diff"] == pytest.approx(restored["elo_diff"])


def test_features_for_does_not_mutate_state() -> None:
    """Scoring an unplayed fixture must not alter the ratings it reads."""
    matches = synthetic_matches()
    featurizer = MatchFeaturizer()
    build_match_features(matches, featurizer)

    home, away = featurizer.teams()[:2]
    before = featurizer.features_for(home, away)
    featurizer.features_for(home, away)
    featurizer.features_for(home, away)
    after = featurizer.features_for(home, away)

    assert before == pytest.approx(after, nan_ok=True)


# ---------------------------------------------------------------------- API
@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    _fitted_artifact(_classification_task()).save(tmp_path / "demo-classification.pkl")
    monkeypatch.setenv("TABFM_ARTIFACT_DIR", str(tmp_path))

    from tabfm_lab.serving.app import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def test_health_is_independent_of_the_models(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_loaded_models(client: TestClient) -> None:
    response = client.get("/api/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["models_loaded"] == 1
    assert body["models"] == ["demo-classification"]


def test_ready_is_503_without_artifacts(tmp_path, monkeypatch) -> None:
    """An empty volume must keep the pod out of the Service, not serve 404s."""
    monkeypatch.setenv("TABFM_ARTIFACT_DIR", str(tmp_path / "empty"))
    from tabfm_lab.serving.app import create_app

    with TestClient(create_app()) as empty_client:
        assert empty_client.get("/api/ready").status_code == 503
        assert empty_client.get("/api/health").status_code == 200


def test_list_and_detail_expose_the_input_schema(client: TestClient) -> None:
    listing = client.get("/api/models").json()
    assert [m["task"] for m in listing] == ["demo-classification"]

    detail = client.get("/api/models/demo-classification").json()
    assert detail["feature_columns"] == ["score", "tier"]
    assert {f["name"] for f in detail["fields"]} == {"score", "tier"}
    assert detail["versions"]["python"]


def test_predict_returns_probabilities(client: TestClient) -> None:
    response = client.post(
        "/api/models/demo-classification/predict",
        json={"records": [{"score": 2.0, "tier": "gold"}, {"score": -2.0, "tier": "bronze"}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2

    for prediction in body["predictions"]:
        total = sum(prediction["probabilities"].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    # The synthetic label is score > 0, so the two rows must separate.
    assert body["predictions"][0]["probabilities"]["1"] > 0.5
    assert body["predictions"][1]["probabilities"]["1"] < 0.5


def test_predict_with_partial_record_uses_defaults(client: TestClient) -> None:
    response = client.post(
        "/api/models/demo-classification/predict", json={"records": [{}]}
    )
    assert response.status_code == 200


def test_predict_rejects_unknown_feature(client: TestClient) -> None:
    response = client.post(
        "/api/models/demo-classification/predict",
        json={"records": [{"bogus": 1}]},
    )
    assert response.status_code == 422


def test_predict_rejects_oversized_batch(client: TestClient) -> None:
    response = client.post(
        "/api/models/demo-classification/predict",
        json={"records": [{"score": 0.0}] * 1001},
    )
    assert response.status_code == 422


def test_unknown_model_is_404(client: TestClient) -> None:
    assert client.get("/api/models/nope").status_code == 404
    assert client.post("/api/models/nope/predict", json={"records": [{}]}).status_code == 404


def test_fixture_endpoint_503_without_football_models(client: TestClient) -> None:
    """Only the demo classifier is loaded here, so fixtures are unavailable."""
    response = client.post(
        "/api/fixtures/predict", json={"home_team": "A", "away_team": "B"}
    )
    assert response.status_code == 503


def test_fixture_rejects_identical_teams(client: TestClient) -> None:
    response = client.post(
        "/api/fixtures/predict", json={"home_team": "A", "away_team": "A"}
    )
    assert response.status_code == 422


def test_frontend_is_served_at_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "TabFM Lab" in response.text
