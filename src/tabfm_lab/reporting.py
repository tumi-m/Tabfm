"""Render benchmark reports as Markdown."""

from __future__ import annotations

from typing import Any

from .evaluation.metrics import PRIMARY_METRIC

#: Preferred column order; anything else is appended alphabetically.
_METRIC_ORDER = [
    "log_loss",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "roc_auc_ovr",
    "pr_auc",
    "f1_macro",
    "brier",
    "calibration_error",
    "rmse",
    "mae",
    "r2",
    "rmse_original_units",
    "mae_original_units",
]


def _ordered_metrics(names: set[str]) -> list[str]:
    known = [m for m in _METRIC_ORDER if m in names]
    return known + sorted(names - set(known))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No results._\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), separator, *(line(r) for r in rows)]) + "\n"


def _format(value: Any) -> str:
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def _best_model(runs: list[dict[str, Any]], task_type: str) -> str | None:
    metric, higher_is_better = PRIMARY_METRIC[task_type]
    scored = [
        (run["metrics"][metric], run["model"])
        for run in runs
        if run["status"] == "ok" and metric in run.get("metrics", {})
    ]
    if not scored:
        return None
    return (max if higher_is_better else min)(scored)[1]


def render_task(task: dict[str, Any]) -> str:
    lines = [
        f"### `{task['name']}`",
        "",
        task["description"],
        "",
        f"- **Industry:** {task['industry']}  ",
        f"- **Objective:** {task['task_type']} on `{task['target']}`  ",
        f"- **Rows:** {task['n_train']:,} train / {task['n_test']:,} test  ",
        f"- **Features:** {task['n_features']}",
        "",
    ]

    runs = task["runs"]
    ok_runs = [r for r in runs if r["status"] == "ok"]
    metric_names = _ordered_metrics({m for r in ok_runs for m in r["metrics"]})

    headers = ["Model", *metric_names, "Fit (s)", "Predict (s)"]
    rows: list[list[str]] = []
    for run in runs:
        if run["status"] != "ok":
            rows.append([run["model"], *["–"] * len(metric_names), "–", "–"])
            continue
        rows.append(
            [
                run["model"],
                *[_format(run["metrics"].get(m, "–")) for m in metric_names],
                f"{run['fit_seconds']:.2f}",
                f"{run['predict_seconds']:.2f}",
            ]
        )
    lines.append(_table(headers, rows))

    best = _best_model(runs, task["task_type"])
    if best:
        metric, _ = PRIMARY_METRIC[task["task_type"]]
        lines.append(f"**Best by {metric}:** {best}\n")

    for run in runs:
        if run["status"] != "ok":
            first_line = run["note"].splitlines()[0] if run["note"] else run["status"]
            lines.append(f"> **{run['model']}** — {run['status']}: {first_line}\n")

    if task.get("market_benchmark"):
        benchmark = task["market_benchmark"]
        lines.append("**Bookmaker benchmark** (Bet365 closing prices, margin removed)\n")
        keys = _ordered_metrics(set(benchmark) - {"n_matches", "mean_overround"})
        lines.append(
            _table(
                [*keys, "mean overround", "matches"],
                [
                    [
                        *[_format(benchmark[k]) for k in keys],
                        _format(benchmark.get("mean_overround", "–")),
                        str(benchmark.get("n_matches", "–")),
                    ]
                ],
            )
        )

    betting_rows = [
        [run["model"], bet["strategy"], str(bet["n_bets"]), _format(bet["roi"]),
         _format(bet["profit"]), _format(bet["hit_rate"]), _format(bet["average_odds"]),
         _format(bet["max_drawdown"])]
        for run in runs
        for bet in run.get("betting", [])
    ]
    if betting_rows:
        lines.append("**Value-betting simulation** (1 unit flat stake; Kelly on a 1,000 bankroll)\n")
        lines.append(
            _table(
                ["Model", "Strategy", "Bets", "ROI", "Profit", "Hit rate", "Avg odds", "Max DD"],
                betting_rows,
            )
        )
        lines.append(
            "> A negative ROI is the expected result against efficient closing prices; "
            "the number to watch is the gap to the bookmaker benchmark above.\n"
        )

    return "\n".join(lines)


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# TabFM benchmark results",
        "",
        f"_Generated {report.get('generated_at', 'n/a')}._",
        "",
        "Zero-shot TabFM measured against tuned baselines on e-commerce and "
        "sports-betting tasks. Lower is better for `log_loss`, `brier`, "
        "`calibration_error`, `rmse` and `mae`.",
        "",
    ]
    for task in report.get("tasks", []):
        lines.append(render_task(task))
        lines.append("")
    return "\n".join(lines)
