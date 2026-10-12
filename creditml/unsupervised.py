"""Unsupervised learning: customer segmentation and application-fraud detection.

**Segmentation** — 8,000 credit-card customers described by six months of behaviour (utilisation, repayment, cash /
travel / online spend shares, activity, limit). Features are log-transformed and standardised; k-means for k = 2…10
is scored by silhouette and a Gaussian mixture by BIC; the chosen k is profiled and each segment named by rules on its
centroid. Segments are checked for business value: do they separate default risk?

**Fraud detection** — no fraud labels exist at application time. Isolation Forest and Local Outlier Factor score
engineered features (declared income vs bank balance, income vs peers, phone / e-mail age, night-time applications,
applications per device). Device sharing is also analysed as a graph: devices used by ≥ 3 applicants form rings.
Labels (confirmed fraud) are used only to evaluate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = ["utilisation", "payment_ratio", "cash_share", "travel_share", "online_share", "txns_per_month", "active_months", "credit_limit"]


def seg_matrix(cards: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    x = cards[SEG_FEATURES].copy()
    x["txns_per_month"] = np.log1p(x["txns_per_month"])
    x["credit_limit"] = np.log(x["credit_limit"])
    sc = StandardScaler().fit(x)
    return sc.transform(x), sc


def choose_k(x: np.ndarray, ks=range(2, 11), seed: int = 0) -> pd.DataFrame:
    rows = []
    sample = np.random.default_rng(seed).choice(len(x), min(3000, len(x)), replace=False)
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit(x)
        gm = GaussianMixture(k, covariance_type="full", random_state=seed).fit(x)
        rows.append({"k": k, "silhouette": float(silhouette_score(x[sample], km.labels_[sample])), "inertia": float(km.inertia_), "gmm_bic": float(gm.bic(x))})
    return pd.DataFrame(rows)


def name_segment(c: pd.Series) -> str:
    """Rule-based names from a centroid in original units — readable by a product manager."""
    if c["active_months"] < 2.5:
        return "Dormant"
    if c["travel_share"] > 0.2:
        return "Premium travellers"
    if c["cash_share"] > 0.15 and c["utilisation"] > 0.8:
        return "Cash-stressed"
    if c["payment_ratio"] > 0.85:
        return "Transactors (pay in full)"
    if c["payment_ratio"] < 0.35:
        return "Revolvers"
    return "New / building"


def segment(cards: pd.DataFrame, k: int, seed: int = 0) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    x, sc = seg_matrix(cards)
    km = KMeans(k, n_init=10, random_state=seed).fit(x)
    out = cards.assign(cluster=km.labels_)
    prof = out.groupby("cluster")[SEG_FEATURES].mean()
    names = {cl: name_segment(prof.loc[cl]) for cl in prof.index}
    out["segment"] = out["cluster"].map(names)
    return out, km, sc


def profile(seg: pd.DataFrame) -> pd.DataFrame:
    p = seg.groupby("segment").agg(customers=("customer_id", "size"), utilisation=("utilisation", "mean"), payment_ratio=("payment_ratio", "mean"),
                                   cash_share=("cash_share", "mean"), travel_share=("travel_share", "mean"), txns_per_month=("txns_per_month", "mean"),
                                   default_rate=("default_next_12m", "mean"))
    return p.sort_values("default_rate", ascending=False).round(3)


def segmentation_quality(seg: pd.DataFrame) -> dict:
    return {"ari_vs_latent": float(adjusted_rand_score(seg["segment_true"], seg["cluster"])), "segments": int(seg["cluster"].nunique()),
            "risk_spread": float(seg.groupby("segment")["default_next_12m"].mean().max() / max(seg.groupby("segment")["default_next_12m"].mean().min(), 1e-3))}


# ------------------------------------------------------------------ fraud
FRAUD_FEATURES = ["income_to_balance", "income_vs_peers", "phone_age_log", "email_age_log", "night_application", "device_applications"]


def fraud_features(apps: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=apps.index)
    d["income_to_balance"] = np.log1p(apps["monthly_income"]) - np.log1p(apps["avg_bank_balance"])
    peer = apps.groupby(["employment", "city_tier"])["monthly_income"].transform("median")
    d["income_vs_peers"] = np.log(apps["monthly_income"] / peer)
    d["phone_age_log"] = -np.log1p(apps["phone_age_days"])
    d["email_age_log"] = -np.log1p(apps["email_age_days"])
    d["night_application"] = apps["application_hour"].between(0, 5).astype(float)
    d["device_applications"] = np.log(apps.groupby("device_id")["device_id"].transform("size"))
    return d


def fraud_scores(apps: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    f = fraud_features(apps)
    x = StandardScaler().fit_transform(f)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed).fit(x)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=False).fit(x)
    out = pd.DataFrame({"isolation_forest": -iso.score_samples(x), "lof": -lof.negative_outlier_factor_,
                        "rule": ((apps["phone_age_days"] < 60) & (apps.groupby("device_id")["device_id"].transform("size") > 1)).astype(float)
                        + (apps["application_hour"].between(0, 5)).astype(float) * 0.5}, index=apps.index)
    return out


def device_rings(apps: pd.DataFrame, anomaly: pd.Series | None = None, min_size: int = 3, min_score_pct: float = 0.98) -> pd.DataFrame:
    """Devices shared by ≥ min_size applications (stars in the applicant–device graph). Families share devices too,
    so a ring is flagged only when its members are also unusual: mean anomaly score above the given percentile."""
    a = apps.assign(score=anomaly if anomaly is not None else 0.0)
    g = a.groupby("device_id").agg(applications=("application_id", "size"), first=("disbursal_month", "min"), last=("disbursal_month", "max"),
                                   channels=("channel", lambda s: ",".join(sorted(set(s)))), mean_anomaly=("score", "mean"), confirmed_fraud=("is_fraud", "mean"))
    g = g[g["applications"] >= min_size]
    if anomaly is not None:
        g["flagged"] = g["mean_anomaly"] >= anomaly.quantile(min_score_pct)
    return g.sort_values(["applications", "mean_anomaly"], ascending=False).reset_index()


def fraud_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    o = np.argsort(-score)
    n = len(y)
    top1, top3 = o[: max(1, n // 100)], o[: max(1, 3 * n // 100)]
    return {"auroc": float(roc_auc_score(y, score)), "precision_top1pct": float(y[top1].mean()), "recall_top3pct": float(y[top3].sum() / y.sum())}
