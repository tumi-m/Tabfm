"""Request and response models for the prediction API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

MAX_BATCH_ROWS = 1000


class FieldSchema(BaseModel):
    """One input field, enough for a client to render a control for it."""

    name: str
    dtype: str
    categorical: bool
    default: Any = None
    choices: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None


class ModelSummary(BaseModel):
    task: str
    industry: str
    task_type: str
    target: str
    model_name: str
    description: str = ""
    n_train: int = 0
    n_test: int = 0
    metrics: dict[str, float | None] = Field(default_factory=dict)
    classes: list[Any] | None = None
    trained_at: str = ""


class ModelDetail(ModelSummary):
    feature_columns: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    fields: list[FieldSchema] = Field(default_factory=list)
    teams: list[str] | None = None
    target_transform: str | None = None
    versions: dict[str, str] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    """A batch of feature records. Missing fields fall back to training defaults."""

    records: list[dict[str, Any]] = Field(
        ..., description="Feature records. Omitted fields use the training default.",
        min_length=1,
    )

    @field_validator("records")
    @classmethod
    def _limit_batch(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Bounded so one request cannot pin a worker for an unbounded time.
        if len(records) > MAX_BATCH_ROWS:
            raise ValueError(f"At most {MAX_BATCH_ROWS} records per request")
        return records


class Prediction(BaseModel):
    prediction: Any
    probabilities: dict[str, float] | None = None
    # Regression targets modelled on a transformed scale also report the
    # value in the units the target was originally measured in.
    prediction_original_units: float | None = None


class PredictResponse(BaseModel):
    task: str
    model_name: str
    task_type: str
    predictions: list[Prediction]
    elapsed_ms: float


class MatchOdds(BaseModel):
    """Decimal bookmaker odds for the 1X2 market."""

    home: float = Field(..., gt=1.0)
    draw: float = Field(..., gt=1.0)
    away: float = Field(..., gt=1.0)


class FixtureRequest(BaseModel):
    """Score a fixture from team names; the server derives the features."""

    home_team: str
    away_team: str
    date: str | None = Field(
        default=None, description="ISO date. Defaults to the last date in the training history."
    )
    odds: MatchOdds | None = Field(
        default=None, description="Optional decimal odds; enables the value-bet analysis."
    )

    @field_validator("away_team")
    @classmethod
    def _distinct(cls, away: str, info) -> str:
        home = info.data.get("home_team")
        if home is not None and home == away:
            raise ValueError("home_team and away_team must differ")
        return away


class ValueBet(BaseModel):
    outcome: str
    model_probability: float
    market_probability: float
    decimal_odds: float
    # Expected profit per unit staked: p_model * odds - 1.
    edge: float
    recommended: bool


class FixtureResponse(BaseModel):
    home_team: str
    away_team: str
    date: str
    result_probabilities: dict[str, float] | None = None
    predicted_result: str | None = None
    expected_total_goals: float | None = None
    over_2_5_probability: float | None = None
    value_bets: list[ValueBet] | None = None
    market_overround: float | None = None
    # False when either side lacks enough history for the features to be real.
    reliable: bool = True
    notes: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    models_loaded: int
    models: list[str]
    errors: dict[str, str] = Field(default_factory=dict)
