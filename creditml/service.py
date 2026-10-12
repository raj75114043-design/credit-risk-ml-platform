"""Serving layer: trains the production models once (≈ 20 s) and exposes credit decisions, LGD, segments and fraud."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from creditml import lgd_model as L
from creditml import pd_model as P
from creditml import policy as Po
from creditml import unsupervised as U
from creditml.data import loaders

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "eval.json"


@dataclass
class Models:
    scorecard: P.Scorecard
    gbm: P.GBMModel
    cutoff: float
    lgd: L.TwoStageLGD
    kmeans: object
    seg_scaler: StandardScaler
    seg_names: dict
    iforest: IsolationForest
    fraud_scaler: StandardScaler
    fraud_cut: float
    fraud_ref: np.ndarray
    card: dict


@lru_cache(maxsize=1)
def models() -> Models:
    apps = loaders.applications()
    s = P.split(apps)
    dev = pd.concat([s["dev"], s["calibration"].iloc[: len(s["calibration"]) // 2]])   # production: use more recent data
    cal = pd.concat([s["calibration"].iloc[len(s["calibration"]) // 2:], s["oot"]])
    cal = cal[~cal["is_fraud"]]
    sc = P.Scorecard().fit(dev)
    sc.recalibrate(cal)
    gbm = P.fit_gbm(dev, cal)
    cutoff = Po.choose_cutoff(cal, gbm.pd_(cal))
    lgd = L.fit_two_stage(L.frame(apps, loaders.losses()))
    seg, km, scaler = U.segment(loaders.cards(), 6)
    names = seg.groupby("cluster")["segment"].first().to_dict()
    f = U.fraud_features(apps)
    fs = StandardScaler().fit(f)
    iso = IsolationForest(n_estimators=300, random_state=0).fit(fs.transform(f))
    ref = -iso.score_samples(fs.transform(f))
    card = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    return Models(sc, gbm, cutoff, lgd, km, scaler, names, iso, fs, float(np.quantile(ref, 0.99)), np.sort(ref),
                  {k: card.get(k) for k in ("pd_auc_gbm", "pd_auc_scorecard", "fraud_auroc", "segmentation_ari", "fairness_di_ratio")})


def _fraud_score(app: pd.DataFrame) -> tuple[float, float]:
    m = models()
    apps = loaders.applications()
    both = pd.concat([apps, app], ignore_index=True)
    f = U.fraud_features(both).iloc[[-1]]
    s = float(-m.iforest.score_samples(m.fraud_scaler.transform(f))[0])
    return s, float(np.searchsorted(m.fraud_ref, s) / len(m.fraud_ref))


def decide(application: dict) -> dict:
    m = models()
    d = pd.DataFrame([application])
    for c in P.FEATURES + ["interest_rate", "device_id", "phone_age_days", "email_age_days", "application_hour"]:
        if c not in d:
            d[c] = np.nan
    pd_gb = float(m.gbm.pd_(d)[0])
    lgd = Po.LGD_BY_PRODUCT[d["product"].iloc[0]]
    ead = Po.EAD_SHARE * float(d["loan_amount"].iloc[0])
    fscore, fpct = _fraud_score(d.assign(application_id="NEW", disbursal_month=pd.Timestamp("2026-10-01"), is_fraud=False, default_12m=np.nan))
    decision = "refer_to_fraud_review" if fscore >= m.fraud_cut else ("approve" if pd_gb <= m.cutoff else "decline")
    out = {"decision": decision, "pd_12m": round(pd_gb, 4), "scorecard_points": int(m.scorecard.points(d)[0]), "pd_cutoff": round(m.cutoff, 4),
           "expected_loss_inr": round(pd_gb * lgd * ead),
           "expected_profit_inr": round(float(Po.expected_profit(d, np.array([pd_gb]))[0])) if pd.notna(d["interest_rate"].iloc[0]) else None,
           "fraud_anomaly_percentile": round(fpct, 4), "reason_codes": m.scorecard.reasons(d)}
    return out


def predict_lgd(loan: dict) -> dict:
    d = pd.DataFrame([loan])
    for c in L.FEATURES:
        if c not in d:
            d[c] = np.nan
    m = models()
    return {"expected_lgd": round(float(m.lgd.predict(d)[0]), 4), "cure_probability": round(float(m.lgd.cure_probability(d)[0]), 4)}


def segment_customer(customer: dict) -> dict:
    m = models()
    x = pd.DataFrame([customer])[U.SEG_FEATURES].copy()
    x["txns_per_month"] = np.log1p(x["txns_per_month"])
    x["credit_limit"] = np.log(x["credit_limit"])
    z = m.seg_scaler.transform(x)
    dist = np.linalg.norm(m.kmeans.cluster_centers_ - z, axis=1)
    order = np.argsort(dist)
    return {"segment": m.seg_names[int(order[0])], "runner_up": m.seg_names[int(order[1])], "margin": round(float(dist[order[1]] - dist[order[0]]), 3)}


def fraud_queue(top: int = 25) -> pd.DataFrame:
    m = models()
    apps = loaders.applications()
    s = -m.iforest.score_samples(m.fraud_scaler.transform(U.fraud_features(apps)))
    q = apps.assign(anomaly=s).nlargest(top, "anomaly")
    return q[["application_id", "disbursal_month", "channel", "monthly_income", "avg_bank_balance", "phone_age_days", "application_hour", "device_id", "anomaly",
              "is_fraud"]]


DEFAULT_APPLICATION = {"product": "personal", "channel": "digital_partner", "age": 27, "employment": "gig", "city_tier": "tier2", "monthly_income": 32000,
                       "bureau_score": 640, "bureau_vintage_months": 30, "utilisation_pct": 78, "enquiries_6m": 4, "max_dpd_12m": 30, "active_loans": 3,
                       "loan_amount": 200000, "tenure_months": 36, "interest_rate": 19.5, "foir": 0.74, "avg_bank_balance": 4000, "ltv": None,
                       "device_id": "D123456", "phone_age_days": 400, "email_age_days": 900, "application_hour": 14, "gender": "M"}
