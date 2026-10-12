import numpy as np
import pandas as pd

from creditml import pd_model as P
from creditml import policy as Po
from creditml import unsupervised as U
from creditml.data import loaders


def test_feature_governance():
    assert "collection_calls_90d" not in P.FEATURES and "gender" not in P.FEATURES and "interest_rate" not in P.FEATURES
    s = P.split(loaders.applications())
    audit = P.leakage_audit(s["dev"])
    assert audit.iloc[0]["feature"] == "collection_calls_90d" and audit.iloc[0]["flag"] == "LEAKAGE SUSPECT"
    assert not s["dev"]["is_fraud"].any()


def test_scorecard_points_and_reasons():
    s = P.split(loaders.applications())
    sc = P.Scorecard().fit(s["dev"])
    o = s["oot"].head(200)
    pts, pd_ = sc.points(o), sc.pd_(o)
    assert pd.Series(pts).corr(pd.Series(pd_), method="spearman") < -0.99   # more points = safer
    r = sc.reasons(o.iloc[[0]])
    assert len(r) == 3 and r[0]["points_lost"] >= r[1]["points_lost"] >= r[2]["points_lost"]


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert P.ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    x = np.random.default_rng(0).normal(size=5000)
    assert P.psi(x, x) < 1e-3 and P.psi(x, x + 1) > 0.25


def test_policy_and_fairness():
    d = pd.DataFrame({"loan_amount": [100000.0] * 4, "interest_rate": [19.0] * 4, "tenure_months": [24] * 4, "product": ["personal"] * 4,
                      "default_12m": [0, 0, 1, 0], "gender": ["F", "M", "F", "M"]})
    approve = np.array([True, True, False, True])
    assert Po.realised_profit(d, approve) > Po.realised_profit(d, np.ones(4, bool))
    f = Po.fairness(d, approve, np.array([0.05, 0.05, 0.5, 0.05]))
    assert f["approval_rate"] == {"F": 0.5, "M": 1.0} and f["equal_opportunity_gap"] == 0.0


def test_fraud_features_and_metrics():
    a = loaders.applications().head(3000)
    f = U.fraud_features(a)
    assert list(f.columns) == U.FRAUD_FEATURES and f.notna().all().all()
    m = U.fraud_metrics(np.r_[np.ones(10), np.zeros(990)], np.r_[np.ones(10), np.zeros(990)])
    assert m["auroc"] == 1.0 and m["precision_top1pct"] == 1.0


def test_segment_names():
    c = pd.Series({"active_months": 6, "travel_share": 0.3, "cash_share": 0.0, "utilisation": 0.3, "payment_ratio": 0.9})
    assert U.name_segment(c) == "Premium travellers"
    assert U.name_segment(c.assign(active_months=1) if hasattr(c, "assign") else pd.Series({**c, "active_months": 1})) == "Dormant"
