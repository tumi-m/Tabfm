"""Download and cache the public datasets used by the benchmarks.

Every dataset is declared as an ordered chain of candidate URLs. The first
reachable URL wins, which lets the pipeline prefer a canonical upstream source
(for example football-data.co.uk, which ships bookmaker odds) while still
working on machines whose egress policy only permits a mirror.

Downloads are cached on disk, so a second run is offline and deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60
_USER_AGENT = "tabfm-lab/0.1 (+https://github.com/tumi-m/Tabfm)"


def default_cache_dir() -> Path:
    """Repository-local cache directory for raw downloads."""
    return Path(__file__).resolve().parents[3] / "data" / "raw"


@dataclass(frozen=True)
class RemoteFile:
    """A single logical file with one or more interchangeable source URLs."""

    name: str
    urls: tuple[str, ...]
    description: str = ""
    # Columns that must be present for the file to be considered usable.
    required_columns: tuple[str, ...] = field(default=())


class DatasetUnavailableError(RuntimeError):
    """Raised when no candidate URL for a dataset could be retrieved."""


def fetch(
    remote: RemoteFile,
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path:
    """Return a local path to ``remote``, downloading it if necessary.

    Tries each candidate URL in order and caches the first success. Raises
    :class:`DatasetUnavailableError` if every candidate fails, with the reason
    for each attempt, since the usual cause is a restrictive egress policy
    rather than a bug.
    """
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / remote.name

    if target.exists() and not force:
        LOGGER.debug("Using cached %s", target)
        return target

    failures: list[str] = []
    for url in remote.urls:
        try:
            LOGGER.info("Downloading %s from %s", remote.name, url)
            response = requests.get(
                url, timeout=timeout, headers={"User-Agent": _USER_AGENT}
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            failures.append(f"  {url}\n    -> {type(exc).__name__}: {exc}")
            continue

        if not response.content:
            failures.append(f"  {url}\n    -> empty response body")
            continue

        # Write atomically so an interrupted run never leaves a truncated cache
        # entry that later looks valid.
        tmp = target.with_suffix(target.suffix + ".part")
        tmp.write_bytes(response.content)
        tmp.replace(target)
        LOGGER.info("Cached %s (%.1f KiB)", target.name, len(response.content) / 1024)
        return target

    raise DatasetUnavailableError(
        f"Could not download {remote.name!r} from any known source.\n"
        + "\n".join(failures)
        + "\n\nIf this machine restricts outbound network access, download the file "
        f"manually and place it at: {target}"
    )


# --------------------------------------------------------------------------
# E-commerce: UCI Online Shoppers Purchasing Intention
# --------------------------------------------------------------------------
# 12,330 anonymised browsing sessions from an online retailer over one year.
# Each session belongs to a distinct user. Sakar et al. (2018), UCI ML Repository.
ONLINE_SHOPPERS = RemoteFile(
    name="online_shoppers_intention.csv",
    urls=(
        # Canonical UCI copy (served as a zip; preferred when reachable).
        "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip",
        # Plain-CSV mirrors of the same file.
        "https://raw.githubusercontent.com/sharmaroshan/Online-Shoppers-Purchasing-Intention/master/online_shoppers_intention.csv",
    ),
    description="Online shopper sessions labelled with whether they ended in a purchase.",
    required_columns=("Revenue", "PageValues", "ProductRelated_Duration"),
)


# --------------------------------------------------------------------------
# Sports betting: football-data.co.uk match results (and odds where available)
# --------------------------------------------------------------------------
# football-data.co.uk publishes one CSV per league per season, including
# closing bookmaker odds (B365H/B365D/B365A and friends). The datahub mirror is
# regenerated daily from the same upstream but ships the match-statistics subset
# without odds, so we always try upstream first.

#: League code -> (football-data.co.uk code, datahub mirror slug)
FOOTBALL_LEAGUES: dict[str, tuple[str, str]] = {
    "premier-league": ("E0", "premier-league"),
    "championship": ("E1", "championship"),
    "la-liga": ("SP1", "la-liga"),
    "serie-a": ("I1", "serie-a"),
    "bundesliga": ("D1", "bundesliga"),
    "ligue-1": ("F1", "ligue-1"),
}

#: Columns football-data.co.uk uses for Bet365 1X2 odds.
ODDS_COLUMNS = ("B365H", "B365D", "B365A")

#: Bet365 over/under 2.5 goals odds, priced against the total-goals model.
OVER_UNDER_COLUMNS = ("B365>2.5", "B365<2.5")


def football_season(league: str, season: str) -> RemoteFile:
    """Build the source chain for one league-season.

    ``season`` is the football-data.co.uk four-digit form, e.g. ``"2324"`` for
    the 2023/24 season.
    """
    if league not in FOOTBALL_LEAGUES:
        raise KeyError(
            f"Unknown league {league!r}. Known leagues: {sorted(FOOTBALL_LEAGUES)}"
        )
    upstream_code, mirror_slug = FOOTBALL_LEAGUES[league]
    return RemoteFile(
        name=f"football_{league}_{season}.csv",
        urls=(
            # Canonical source: includes bookmaker odds.
            f"https://www.football-data.co.uk/mmz4281/{season}/{upstream_code}.csv",
            # Daily mirror: match statistics only, no odds.
            f"https://raw.githubusercontent.com/datasets/football-datasets/master/datasets/{mirror_slug}/season-{season}.csv",
        ),
        description=f"{league} {season} match results from football-data.co.uk.",
        required_columns=("HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"),
    )
