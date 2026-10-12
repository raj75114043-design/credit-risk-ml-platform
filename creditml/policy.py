"""Credit policy: from PD to approve / decline, priced by expected profit, checked for fairness.

Expected profit of a loan = net interest margin earned if it performs − PD × LGD × EAD.
The approval cut-off on PD is chosen on the calibration period to maximise realised profit, then applied out of time.
Baselines: approve everyone, and a typical bureau rule (score ≥ 700, no new-to-credit).

Fairness (gender is never a model input; it is used only here): approval-rate ratio (four-fifths rule) and the gap in
approval rates among applicants who did **not** default (equal opportunity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COST_OF_FUNDS = 9.0     # % p.a.
EAD_SHARE = 0.75        # average exposure at default as a share of the amount
LGD_BY_PRODUCT = {"personal": 0.55, "two_wheeler": 0.25, "gold": 0.06}   # long-run averages from the loss data


def margin(d: pd.DataFrame) -> np.ndarray:
    """Net interest earned over the loan if it performs (average balance ≈ half the amount)."""
    return d["loan_amount"].to_numpy() * (d["interest_rate"].to_numpy() - COST_OF_FUNDS) / 100 * d["tenure_months"].to_numpy() / 12 / 2


def realised_profit(d: pd.DataFrame, approve: np.ndarray, lgd: np.ndarray | None = None) -> float:
    lgd = lgd if lgd is not None else d["product"].map(LGD_BY_PRODUCT).to_numpy()
    y = d["default_12m"].to_numpy()
    p = np.where(y == 1, -lgd * EAD_SHARE * d["loan_amount"].to_numpy(), margin(d))
    return float(p[approve].sum())


def expected_profit(d: pd.DataFrame, pd_: np.ndarray) -> np.ndarray:
    lgd = d["product"].map(LGD_BY_PRODUCT).to_numpy()
    return (1 - pd_) * margin(d) - pd_ * lgd * EAD_SHARE * d["loan_amount"].to_numpy()


def choose_cutoff(d: pd.DataFrame, pd_: np.ndarray) -> float:
    grid = np.unique(np.quantile(pd_, np.linspace(0.3, 1.0, 141)))
    prof = [realised_profit(d, pd_ <= c) for c in grid]
    return float(grid[int(np.argmax(prof))])


def bureau_rule(d: pd.DataFrame) -> np.ndarray:
    return (d["bureau_score"].fillna(0) >= 700).to_numpy()


def summary(d: pd.DataFrame, approve: np.ndarray) -> dict:
    y = d["default_12m"].to_numpy()
    return {"approval_rate": float(approve.mean()), "bad_rate_approved": float(y[approve].mean()) if approve.any() else 0.0,
            "profit_crore": realised_profit(d, approve) / 1e7, "approved": int(approve.sum())}


def fairness(d: pd.DataFrame, approve: np.ndarray, pd_: np.ndarray) -> dict:
    g = d["gender"].to_numpy()
    y = d["default_12m"].to_numpy()
    rate = {k: float(approve[g == k].mean()) for k in ("F", "M")}
    good = {k: float(approve[(g == k) & (y == 0)].mean()) for k in ("F", "M")}
    calib = {k: {"predicted": float(pd_[g == k].mean()), "observed": float(y[g == k].mean())} for k in ("F", "M")}
    return {"approval_rate": rate, "disparate_impact_ratio": min(rate.values()) / max(rate.values()),
            "equal_opportunity_gap": abs(good["F"] - good["M"]), "approval_rate_among_good": good, "calibration_by_group": calib}
