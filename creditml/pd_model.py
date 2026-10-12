"""Probability of default (classification).

Feature governance comes first: every raw field has an *availability* tag, and only fields known at application time
and allowed for credit decisions enter a model. A leakage audit (univariate AUC) catches anything that looks too good.

Two models are trained on Jan 2023 – Jun 2024 disbursals, calibrated on Jul – Dec 2024 and judged out of time on
Jan – Jun 2025 (the latest vintages with a full 12-month outcome):

* **WoE scorecard** — quantile binning with a missing bin, weight-of-evidence encoding, L2 logistic regression,
  points scaled to 600 at 50:1 odds with 20 points to double the odds; per-applicant reason codes.
* **Gradient boosting** — monotonic constraints where credit policy demands them (higher bureau score never raises
  risk; higher FOIR never lowers it), isotonic calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

AVAILABILITY = {  # field: when it is known / whether it may be used
    "product": "application", "channel": "application", "age": "application", "employment": "application", "city_tier": "application",
    "monthly_income": "application", "bureau_score": "application", "bureau_vintage_months": "application", "utilisation_pct": "application",
    "enquiries_6m": "application", "max_dpd_12m": "application", "active_loans": "application", "loan_amount": "application",
    "tenure_months": "application", "foir": "application", "avg_bank_balance": "application", "ltv": "application",
    "interest_rate": "pricing_decision (set from the risk grade — circular)", "collection_calls_90d": "post_disbursal (leakage)",
    "gender": "protected (fairness monitoring only)", "device_id": "identifier", "phone_age_days": "fraud_only", "email_age_days": "fraud_only",
    "application_hour": "fraud_only", "is_fraud": "label (fraud)",
}
CATEGORICAL = ["product", "channel", "employment", "city_tier"]
NUMERIC = [k for k, v in AVAILABILITY.items() if v == "application" and k not in CATEGORICAL]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "default_12m"
SPLITS = {"dev": ("2023-01-01", "2024-06-01"), "calibration": ("2024-07-01", "2024-12-01"), "oot": ("2025-01-01", "2025-06-01")}
MONOTONE = {"bureau_score": -1, "foir": 1, "enquiries_6m": 1, "max_dpd_12m": 1, "avg_bank_balance": -1}


def split(apps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    m = apps[apps[TARGET].notna()]
    out = {k: m[(m["disbursal_month"] >= lo) & (m["disbursal_month"] <= hi)] for k, (lo, hi) in SPLITS.items()}
    out["dev"] = out["dev"][~out["dev"]["is_fraud"]]   # confirmed fraud is not credit risk: excluded from development
    out["immature"] = apps[apps[TARGET].isna()]
    return out


def leakage_audit(dev: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    rows = []
    for c, tag in AVAILABILITY.items():
        if c in ("device_id", "is_fraud") or c not in dev:
            continue
        x = dev[c]
        if not pd.api.types.is_numeric_dtype(x):
            x = x.map(dev.groupby(c)[TARGET].mean())
        x = x.fillna(x.median())
        auc = roc_auc_score(dev[TARGET], x)
        auc = max(auc, 1 - auc)
        rows.append({"feature": c, "availability": tag, "univariate_auc": round(float(auc), 3), "used": c in FEATURES,
                     "flag": "LEAKAGE SUSPECT" if auc >= threshold else ("excluded" if c not in FEATURES else "")})
    return pd.DataFrame(rows).sort_values("univariate_auc", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ WoE scorecard
@dataclass
class Scorecard:
    bins: dict = field(default_factory=dict)          # numeric feature -> edges
    woe: dict = field(default_factory=dict)           # feature -> {bin label: woe}
    lr: LogisticRegression | None = None
    intercept_shift: float = 0.0
    factor: float = 20 / np.log(2)
    offset: float = 600 - 20 / np.log(2) * np.log(50)

    def _bin(self, c: str, x: pd.Series) -> pd.Series:
        if c in CATEGORICAL:
            return x.astype(str)
        x = pd.to_numeric(x, errors="coerce")
        b = pd.cut(x, self.bins[c], include_lowest=True).astype(str)
        return b.where(x.notna(), "MISSING")

    def fit(self, d: pd.DataFrame) -> Scorecard:
        y = d[TARGET].to_numpy()
        goods, bads = (y == 0).sum(), (y == 1).sum()
        for c in FEATURES:
            if c not in CATEGORICAL:
                q = np.unique(np.nanquantile(d[c], np.linspace(0, 1, 7)))
                self.bins[c] = np.r_[-np.inf, q[1:-1], np.inf]
            b = self._bin(c, d[c])
            t = pd.crosstab(b, y)
            t = t.reindex(columns=[0, 1], fill_value=0) + 0.5
            self.woe[c] = np.log((t[0] / goods) / (t[1] / bads)).to_dict()
        x = self.transform(d)
        self.lr = LogisticRegression(C=0.5, max_iter=2000).fit(x, y)
        return self

    def transform(self, d: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self._bin(c, d[c]).map(self.woe[c]).fillna(0).to_numpy() for c in FEATURES])

    def logit(self, d: pd.DataFrame) -> np.ndarray:
        return self.lr.decision_function(self.transform(d)) + self.intercept_shift

    def recalibrate(self, d: pd.DataFrame) -> None:
        """Shift the intercept so the mean PD matches the calibration period's default rate (drift in base rate)."""
        lo, hi = -3.0, 3.0
        target = d[TARGET].mean()
        for _ in range(60):
            mid = (lo + hi) / 2
            self.intercept_shift = mid
            if (1 / (1 + np.exp(-self.logit(d)))).mean() > target:
                hi = mid
            else:
                lo = mid

    def pd_(self, d: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit(d)))

    def points(self, d: pd.DataFrame) -> np.ndarray:
        return np.round(self.offset - self.factor * self.logit(d)).astype(int)   # higher = safer

    def reasons(self, row: pd.DataFrame, k: int = 3) -> list[dict]:
        """Adverse-action reason codes: features costing the most points versus their best bin."""
        out = []
        for i, c in enumerate(FEATURES):
            w = self.lr.coef_[0][i]
            b = self._bin(c, row[c]).iloc[0]
            cur = self.woe[c].get(b, 0.0)
            best = min(self.woe[c].values(), key=lambda v: w * v)
            lost = self.factor * w * (cur - best)
            v = row[c].iloc[0]
            v = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
            out.append({"feature": c, "value": v, "bin": b, "points_lost": round(float(lost), 1)})
        return sorted(out, key=lambda r: -r["points_lost"])[:k]


# ------------------------------------------------------------------ gradient boosting
@dataclass
class GBMModel:
    clf: HistGradientBoostingClassifier
    cats: dict
    iso: IsotonicRegression | None = None

    def _x(self, d: pd.DataFrame) -> pd.DataFrame:
        x = d[FEATURES].copy()
        for c in NUMERIC:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        for c in CATEGORICAL:
            x[c] = pd.Categorical(x[c], categories=self.cats[c]).codes
        return x

    def raw(self, d: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(self._x(d))[:, 1]

    def pd_(self, d: pd.DataFrame) -> np.ndarray:
        p = self.raw(d)
        return self.iso.predict(p) if self.iso is not None else p


def fit_gbm(dev: pd.DataFrame, cal: pd.DataFrame) -> GBMModel:
    cats = {c: sorted(dev[c].dropna().unique()) for c in CATEGORICAL}
    mono = [MONOTONE.get(c, 0) for c in FEATURES]
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=1.0,
                                         categorical_features=[FEATURES.index(c) for c in CATEGORICAL], monotonic_cst=mono, early_stopping=True,
                                         validation_fraction=0.15, random_state=0)
    m = GBMModel(clf, cats)
    clf.fit(m._x(dev), dev[TARGET].astype(int))
    m.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.999).fit(m.raw(cal), cal[TARGET])
    return m


# ------------------------------------------------------------------ metrics
def ks(y, p) -> float:
    o = np.argsort(p)
    y = np.asarray(y)[o]
    cb, cg = np.cumsum(y) / y.sum(), np.cumsum(1 - y) / (1 - y).sum()
    return float(np.max(np.abs(cb - cg)))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False) + 1
    return d.groupby("decile").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def scores(y, p) -> dict:
    auc = roc_auc_score(y, p)
    cal = calibration_table(y, p)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks(y, p), "brier": float(np.mean((np.asarray(p) - np.asarray(y)) ** 2)),
            "mean_pd": float(np.mean(p)), "default_rate": float(np.mean(y)),
            "calibration_mae": float((cal["predicted"] - cal["observed"]).abs().mean())}
