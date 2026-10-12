# Architecture

| Component | Module | Notes |
|---|---|---|
| Data | `data/generator.py` | Applications, defaulted-loan recoveries, card behaviour |
| Governance | `pd_model.AVAILABILITY`, `leakage_audit` | Availability tags; univariate-AUC audit |
| PD | `pd_model.py` | WoE scorecard (points, reason codes), monotonic GBM + isotonic calibration |
| Policy | `policy.py` | Expected profit, cut-off, baselines, fairness |
| LGD | `lgd_model.py` | Two-stage cure × severity |
| Segmentation | `unsupervised.py` | k-means (silhouette), GMM BIC for comparison, rule-based names |
| Fraud | `unsupervised.py` | Isolation Forest, LOF, device rings |
| Serving | `service.py`, `api.py`, `streamlit_app.py` | Models trained once at start-up (~25 s) |
