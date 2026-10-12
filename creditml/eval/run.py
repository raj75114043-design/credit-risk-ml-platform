"""Evaluation and quality gate.

    python -m creditml.eval.run --gate

Classification (PD) · regression (LGD) · unsupervised (segmentation, fraud anomalies, device rings) · credit policy,
fairness and drift. Results are written to reports/eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from creditml import lgd_model as L
from creditml import pd_model as P
from creditml import policy as Po
from creditml import unsupervised as U
from creditml.data import loaders

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = {
    "leakage_caught": (1, "min"), "pd_auc_gbm": (0.78, "min"), "pd_auc_scorecard": (0.77, "min"), "pd_lift_vs_bureau": (0.05, "min"), "pd_ks_gbm": (0.40, "min"),
    "pd_calibration_mae_gbm": (0.03, "max"), "gbm_monotone_in_bureau": (1, "min"), "drift_alert_raised": (1, "min"), "score_psi": (0.25, "max"),
    "policy_beats_approve_all": (1, "min"), "policy_beats_bureau_rule": (1, "min"), "fairness_di_ratio": (0.8, "min"), "fairness_eo_gap": (0.05, "max"),
    "lgd_beats_product_mean": (1, "min"), "lgd_portfolio_error": (0.05, "max"),
    "segmentation_ari": (0.7, "min"), "segment_risk_spread": (5.0, "min"),
    "fraud_auroc": (0.85, "min"), "fraud_precision_top1pct": (0.4, "min"), "fraud_iforest_beats_baselines": (1, "min"), "ring_precision": (0.7, "min"),
}


def classification(m: dict) -> None:
    apps = loaders.applications()
    s = P.split(apps)
    m["split_sizes"] = {k: len(v) for k, v in s.items()}
    dev, cal, oot = s["dev"], s["calibration"], s["oot"]
    audit = P.leakage_audit(dev)
    m["leakage_audit"] = audit.to_dict("records")
    flagged = set(audit.loc[audit["flag"] == "LEAKAGE SUSPECT", "feature"])
    m["leakage_caught"] = float("collection_calls_90d" in flagged and not (flagged & set(P.FEATURES)))
    # what a leaky model would have reported
    leak_x = dev[P.NUMERIC + ["collection_calls_90d"]]
    leak = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(leak_x, dev[P.TARGET])
    m["leaky_model_auc_oot"] = float(roc_auc_score(oot[P.TARGET], leak.predict_proba(oot[P.NUMERIC + ["collection_calls_90d"]])[:, 1]))

    sc = P.Scorecard().fit(dev)
    sc.recalibrate(cal)
    gbm = P.fit_gbm(dev, cal)
    y = oot[P.TARGET].to_numpy()
    p_sc, p_gb = sc.pd_(oot), gbm.pd_(oot)
    bureau = -oot["bureau_score"].fillna(oot["bureau_score"].median()).to_numpy()
    m["pd_oot"] = {"scorecard": P.scores(y, p_sc), "gbm": P.scores(y, p_gb), "bureau_score_only": {"auc": float(roc_auc_score(y, bureau))}}
    m["pd_auc_gbm"], m["pd_auc_scorecard"] = m["pd_oot"]["gbm"]["auc"], m["pd_oot"]["scorecard"]["auc"]
    m["pd_lift_vs_bureau"] = m["pd_auc_gbm"] - m["pd_oot"]["bureau_score_only"]["auc"]
    m["pd_ks_gbm"], m["pd_calibration_mae_gbm"] = m["pd_oot"]["gbm"]["ks"], m["pd_oot"]["gbm"]["calibration_mae"]
    m["pd_calibration_table_gbm"] = P.calibration_table(y, p_gb).round(4).to_dict("records")
    m["thin_file_auc"] = {k: float(roc_auc_score(y[mask], p_gb[mask])) for k, mask in
                          (("new_to_credit", oot["bureau_score"].isna().to_numpy()), ("with_bureau", oot["bureau_score"].notna().to_numpy()))}
    grid = oot.sample(300, random_state=0)
    curves = np.array([gbm.raw(grid.assign(bureau_score=v)) for v in range(550, 851, 25)])
    m["gbm_monotone_in_bureau"] = float((np.diff(curves, axis=0) <= 1e-9).all())
    # drift: score PSI and characteristic stability per feature
    m["score_psi"] = P.psi(sc.points(dev), sc.points(oot))
    csi = {c: P.psi(dev[c].fillna(-1).to_numpy(), oot[c].fillna(-1).to_numpy()) for c in P.NUMERIC}
    for c in P.CATEGORICAL:
        e, a = dev[c].value_counts(normalize=True), oot[c].value_counts(normalize=True)
        e, a = e.reindex(a.index.union(e.index)).fillna(1e-4).clip(lower=1e-4), a.reindex(a.index.union(e.index)).fillna(1e-4).clip(lower=1e-4)
        csi[c] = float(((a - e) * np.log(a / e)).sum())
    m["csi"] = {k: round(v, 4) for k, v in sorted(csi.items(), key=lambda kv: -kv[1])}
    m["drift_alerts"] = [k for k, v in csi.items() if v > 0.1]
    m["drift_alert_raised"] = float("channel" in m["drift_alerts"])
    m["pd_underprediction_oot"] = m["pd_oot"]["gbm"]["default_rate"] - m["pd_oot"]["gbm"]["mean_pd"]
    m["default_rate_by_channel_oot"] = oot.groupby("channel")[P.TARGET].mean().round(4).to_dict()
    # policy
    cut_gb = Po.choose_cutoff(cal, gbm.pd_(cal))
    cut_sc = Po.choose_cutoff(cal, sc.pd_(cal))
    m["policy"] = {"gbm_cutoff_pd": cut_gb, "scorecard_cutoff_pd": cut_sc, "gbm": Po.summary(oot, p_gb <= cut_gb), "scorecard": Po.summary(oot, p_sc <= cut_sc),
                   "approve_all": Po.summary(oot, np.ones(len(oot), bool)), "bureau_rule_700": Po.summary(oot, Po.bureau_rule(oot))}
    pol = m["policy"]
    m["policy_beats_approve_all"] = float(pol["gbm"]["profit_crore"] > pol["approve_all"]["profit_crore"])
    m["policy_beats_bureau_rule"] = float(pol["gbm"]["profit_crore"] > pol["bureau_rule_700"]["profit_crore"])
    fair = Po.fairness(oot, p_gb <= cut_gb, p_gb)
    m["fairness"] = fair
    m["fairness_di_ratio"], m["fairness_eo_gap"] = fair["disparate_impact_ratio"], fair["equal_opportunity_gap"]
    m["example_reasons"] = sc.reasons(oot.sort_values("bureau_score").iloc[[5]])


def regression(m: dict) -> None:
    f = L.frame(loaders.applications(), loaders.losses())
    idx = np.random.default_rng(0).permutation(len(f))
    k = int(0.7 * len(f))
    tr, te = f.iloc[idx[:k]], f.iloc[idx[k:]]
    preds = {"product_mean": te["product"].map(tr.groupby("product")["lgd"].mean()).to_numpy(), "direct_gbm": L.predict_direct(L.fit_direct(tr), te)}
    two = L.fit_two_stage(tr)
    preds["two_stage"] = two.predict(te)
    res = {}
    for name, p in preds.items():
        tot = float(((p * te["ead"]).sum() - (te["lgd"] * te["ead"]).sum()) / (te["lgd"] * te["ead"]).sum())
        res[name] = {"mae": float(np.abs(p - te["lgd"]).mean()), "rmse": float(np.sqrt(((p - te["lgd"]) ** 2).mean())), "portfolio_loss_error": tot,
                     "by_product": L.portfolio_error(te, p)}
    m["lgd"] = res
    m["lgd_cure_auc"] = float(roc_auc_score(te["cured"], two.cure_probability(te)))
    m["lgd_distribution"] = {"cured_share": float(f["cured"].mean()), "mean": float(f["lgd"].mean()), "share_below_0.1": float((f["lgd"] < 0.1).mean()),
                             "share_above_0.7": float((f["lgd"] > 0.7).mean())}
    m["lgd_beats_product_mean"] = float(res["two_stage"]["mae"] < res["product_mean"]["mae"])
    m["lgd_portfolio_error"] = abs(res["two_stage"]["portfolio_loss_error"])


def unsupervised(m: dict) -> None:
    cards = loaders.cards()
    x, _ = U.seg_matrix(cards)
    ks = U.choose_k(x)
    k = int(ks.loc[ks["silhouette"].idxmax(), "k"])
    seg, _, _ = U.segment(cards, k)
    q = U.segmentation_quality(seg)
    m["segmentation"] = {"k_selected": k, "k_search": ks.round(4).to_dict("records"), **q, "profile": U.profile(seg).reset_index().to_dict("records")}
    m["segmentation_ari"], m["segment_risk_spread"] = q["ari_vs_latent"], q["risk_spread"]
    apps = loaders.applications()
    s = U.fraud_scores(apps)
    y = apps["is_fraud"].to_numpy()
    m["fraud"] = {c: U.fraud_metrics(y, s[c].to_numpy()) for c in s}
    m["fraud_base_rate"] = float(y.mean())
    m["fraud_auroc"], m["fraud_precision_top1pct"] = m["fraud"]["isolation_forest"]["auroc"], m["fraud"]["isolation_forest"]["precision_top1pct"]
    m["fraud_iforest_beats_baselines"] = float(all(m["fraud_auroc"] > m["fraud"][b]["auroc"] for b in ("lof", "rule")))
    rings = U.device_rings(apps, s["isolation_forest"])
    fl = rings[rings["flagged"]]
    m["rings"] = {"shared_devices_3plus": len(rings), "flagged": len(fl), "precision": float((fl["confirmed_fraud"] > 0.5).mean()),
                  "fraud_captured": float(apps[apps["device_id"].isin(fl["device_id"])]["is_fraud"].sum() / apps["is_fraud"].sum())}
    m["ring_precision"] = m["rings"]["precision"]


def evaluate() -> dict:
    m: dict = {}
    classification(m)
    regression(m)
    unsupervised(m)
    return m


def gate(m: dict) -> list[str]:
    out = []
    for k, (thr, kind) in THRESHOLDS.items():
        v = m[k]
        if (kind == "min" and v < thr) or (kind == "max" and v > thr):
            out.append(f"{k}={v:.4f} (gate {'>=' if kind == 'min' else '<='} {thr})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval.json"))
    a = ap.parse_args()
    m = evaluate()
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(m, indent=2, default=str))
    skip = {"leakage_audit", "pd_calibration_table_gbm", "segmentation", "csi", "split_sizes"}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k not in skip}, indent=1, default=str))
    br = gate(m)
    if a.gate:
        print("QUALITY GATE " + ("FAILED: " + "; ".join(br) if br else "PASSED"))
        sys.exit(1 if br else 0)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
