"""Loss given default (regression).

LGD is bimodal — many defaulted borrowers cure (LGD ≈ 0) while unsecured write-offs lose most of the exposure — so a
single regression predicts the middle, where almost nobody is. Compared models:

* **product mean** — the common simple approach
* **direct GBM** — gradient boosting regression on LGD
* **two-stage** — P(cure) classifier × E[LGD | no cure] regressor, combined as (1 − p) · E[LGD | no cure]

Only information available at default is used (product, collateral LTV, months on book, employment, bank balance at
application, exposure). Accuracy is reported per loan (MAE) and for the portfolio's total loss by product, which is
what provisioning (expected credit loss) depends on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

FEATURES = ["product", "ltv", "months_on_book_at_default", "employment", "avg_bank_balance", "ead", "loan_amount", "monthly_income", "bureau_score", "channel"]
CATS = ["product", "employment", "channel"]


def frame(apps: pd.DataFrame, losses: pd.DataFrame) -> pd.DataFrame:
    return losses.merge(apps, on="application_id")


def _x(d: pd.DataFrame) -> pd.DataFrame:
    x = d[FEATURES].copy()
    for c in FEATURES:
        if c not in CATS:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in CATS:
        x[c] = x[c].astype("category").cat.set_categories(sorted(["personal", "two_wheeler", "gold", "salaried", "self_employed", "gig", "branch", "dsa",
                                                                  "digital_partner"])).cat.codes
    return x


@dataclass
class TwoStageLGD:
    cure: HistGradientBoostingClassifier
    severity: HistGradientBoostingRegressor

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        x = _x(d)
        p = self.cure.predict_proba(x)[:, 1]
        return np.clip((1 - p) * self.severity.predict(x) + p * 0.03, 0, 1)

    def cure_probability(self, d: pd.DataFrame) -> np.ndarray:
        return self.cure.predict_proba(_x(d))[:, 1]


def _cat_idx():
    return [FEATURES.index(c) for c in CATS]


def fit_two_stage(tr: pd.DataFrame) -> TwoStageLGD:
    x = _x(tr)
    cure = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, categorical_features=_cat_idx(),
                                          random_state=0).fit(x, tr["cured"])
    nc = ~tr["cured"].to_numpy()
    sev = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, categorical_features=_cat_idx(),
                                        random_state=0).fit(x[nc], tr["lgd"][nc])
    return TwoStageLGD(cure, sev)


def fit_direct(tr: pd.DataFrame) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, categorical_features=_cat_idx(),
                                         random_state=0).fit(_x(tr), tr["lgd"])


def predict_direct(m, d) -> np.ndarray:
    return np.clip(m.predict(_x(d)), 0, 1)


def portfolio_error(d: pd.DataFrame, pred: np.ndarray) -> dict:
    out = {}
    for prod, g in d.assign(pred=pred).groupby("product"):
        actual = (g["lgd"] * g["ead"]).sum()
        out[prod] = float(((g["pred"] * g["ead"]).sum() - actual) / actual)
    return out
