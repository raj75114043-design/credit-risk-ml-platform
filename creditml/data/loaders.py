"""Load the generated data (generating it on first use)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from creditml.data.generator import ROOT, write

DATA = ROOT / "data"


def _ensure() -> None:
    if not (DATA / "applications.csv.gz").exists():
        write(DATA)


@lru_cache(maxsize=1)
def applications() -> pd.DataFrame:
    _ensure()
    return pd.read_csv(DATA / "applications.csv.gz", parse_dates=["disbursal_month"])


@lru_cache(maxsize=1)
def losses() -> pd.DataFrame:
    _ensure()
    return pd.read_csv(DATA / "losses.csv.gz")


@lru_cache(maxsize=1)
def cards() -> pd.DataFrame:
    _ensure()
    return pd.read_csv(DATA / "card_behaviour.csv.gz")
