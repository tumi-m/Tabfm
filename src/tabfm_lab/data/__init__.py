"""Dataset loading and feature engineering for both industries."""

from __future__ import annotations

from pathlib import Path

from .base import TabularTask
from .sources import DatasetUnavailableError

__all__ = ["TabularTask", "DatasetUnavailableError", "build_all_tasks", "TASK_BUILDERS"]


def build_all_tasks(cache_dir: Path | None = None) -> list[TabularTask]:
    """Every benchmark task across both industries."""
    from . import ecommerce, sports

    return ecommerce.build_tasks(cache_dir) + sports.build_tasks(cache_dir)


def _builder(module: str, function: str):
    def build(cache_dir: Path | None = None) -> TabularTask:
        import importlib

        loaded = importlib.import_module(f".{module}", __package__)
        # Keyword, not positional: the sports builders take `leagues` first, so
        # a positional cache_dir would arrive as the league list.
        return getattr(loaded, function)(cache_dir=cache_dir)

    return build


#: Task name -> zero-argument builder, used by the CLI's ``--task`` filter.
TASK_BUILDERS = {
    "ecommerce-conversion": _builder("ecommerce", "conversion_task"),
    "ecommerce-page-value": _builder("ecommerce", "page_value_task"),
    "ecommerce-campaign-response": _builder("ecommerce", "campaign_response_task"),
    "ecommerce-customer-value": _builder("ecommerce", "customer_value_task"),
    "sports-match-result": _builder("sports", "match_result_task"),
    "sports-total-goals": _builder("sports", "total_goals_task"),
    "sports-multi-league": _builder("sports", "multi_league_result_task"),
}
