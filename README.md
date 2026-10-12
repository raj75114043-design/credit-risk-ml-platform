# Credit Risk ML Platform

[![CI](https://github.com/raj75114043-design/credit-risk-ml-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/raj75114043-design/credit-risk-ml-platform/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://raj-credit-risk-ml-platform.streamlit.app)
[![GitLab](https://img.shields.io/badge/GitLab-mirror-FC6D26?style=flat-square&logo=gitlab&logoColor=white)](https://gitlab.com/raj75114043/credit-risk-ml-platform)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/ML-classification_·_regression_·_unsupervised-0A7E8C?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-FF3B3B?style=flat-square)

Classical machine learning for the retail-lending lifecycle of a fictional Indian NBFC, **Kestrel Finance** —
the three families of ML each doing the job it is actually used for in a lender:

| Family | Problem | Approach |
|---|---|---|
| **Classification** | Probability of default (PD) for every application | WoE scorecard with reason codes + monotonic gradient boosting, out-of-time validation, calibration, profit-based cut-off, fairness checks |
| **Regression** | Loss given default (LGD) for provisioning and pricing | Two-stage model (cure probability × severity) for a bimodal target, vs direct regression and product averages |
| **Unsupervised** | Who are our card customers? Which applications look like fraud — with no fraud labels? | k-means segmentation chosen by silhouette and named by rules; Isolation Forest anomalies; device-sharing rings |

> **Live console:** https://raj-credit-risk-ml-platform.streamlit.app — all applicants, loans and customers are synthetic.

---

## Real-world traps built into the data (and how each is handled)

| Trap | Why it matters | Handling |
|---|---|---|
| **Outcome window** — default = 90+ DPD within 12 months, so loans from the last 12 months have no outcome yet | Training on immature loans labels future defaulters as "good" | 13,965 immature applications excluded; out-of-time test on the latest mature vintages |
| **Target leakage** — collection calls are recorded *after* disbursal | A leaky model scores AUC 0.9998 in testing and is useless at application time | Every field carries an availability tag; a univariate-AUC audit flags it |
| **Circular pricing** — interest rate is set *from* the risk grade | Using it teaches the model the old policy | Excluded from PD; used only for profit |
| **Protected attribute** — gender | Must not drive decisions; income carries a structural gap | Never a model input; disparate-impact and equal-opportunity checks on decisions |
| **New-to-credit applicants** — no bureau score | Missing is information, not zero | Explicit missing bin in the scorecard, native missing handling in GBM; thin-file AUC reported |
| **Population drift** — a digital partner channel grows from 5 % to 35 % in 2025 with 2× the default rate | Score is stable but PD is under-estimated | Characteristic stability (CSI) alert on channel; calibration gap reported |
| **Fraud** (1.7 %) — inflated income, new phone numbers, shared devices | Fraud is not credit risk and has no labels at decision time | Excluded from PD development; unsupervised detection routes cases to review |
| **Bimodal LGD** — cures near 0, write-offs near 1 | A single regression predicts the empty middle | Two-stage model |

## Results

`python -m creditml.eval.run --gate` (≈ 30 s) recomputes everything; CI fails if any gate is missed.

**Classification — PD, out of time (6,339 loans disbursed Jan–Jun 2025)**

| Model | AUC | Gini | KS | Brier | Mean PD vs default rate |
|---|---|---|---|---|---|
| **Gradient boosting (monotonic, isotonic-calibrated)** | **0.825** | **0.649** | **0.493** | **0.073** | 8.9 % vs 10.0 % |
| WoE scorecard (logistic) | 0.808 | 0.616 | 0.471 | 0.076 | 9.6 % vs 10.0 % |
| Bureau score alone | 0.734 | 0.468 | | | |
| *Leaky model with post-disbursal collection calls* | *0.9998* | | | | *— rejected by the audit* |

New-to-credit applicants: AUC 0.70 (vs 0.85 with a bureau record). GBM monotonic in bureau score: verified on 300 applicants.

**Policy and fairness (cut-off chosen on Jul–Dec 2024, applied to Jan–Jun 2025)**

| Policy | Approval rate | Bad rate among approved | Profit (₹ crore) |
|---|---|---|---|
| **GBM, profit-optimal cut-off (PD ≤ 22.5 %)** | 89.5 % | **6.3 %** | **7.97** |
| WoE scorecard, profit-optimal cut-off | 89.5 % | 6.8 % | 7.81 |
| Approve everyone | 100 % | 10.0 % | 6.77 |
| Bureau rule (score ≥ 700, no new-to-credit) | 49.7 % | 4.1 % | 4.55 |

Approval-rate ratio women / men **0.98** (four-fifths rule ≥ 0.80); approval gap among borrowers who repaid **1.6 pts**.

**Regression — LGD (660 held-out defaulted loans)**

| Model | MAE | RMSE | Portfolio loss error |
|---|---|---|---|
| **Two-stage (cure × severity)** | **0.196** | **0.264** | +3.0 % |
| Direct gradient boosting | 0.201 | 0.264 | +1.3 % |
| Average by product | 0.223 | 0.282 | +4.1 % |

Cure classifier AUC 0.73. 32 % of defaults have LGD < 0.1 and 32 % > 0.7.

**Unsupervised**

| Check | Result | Gate |
|---|---|---|
| Segments: k chosen by silhouette | **6** (silhouette 0.30) | |
| Agreement with the generating segments (ARI) | **0.86** | ≥ 0.70 |
| Default-rate spread across segments | **21×** (Cash-stressed 22 % vs Dormant 1.1 %) | ≥ 5× |
| Fraud — Isolation Forest AUROC | **0.926** (LOF 0.717, rules 0.682) | ≥ 0.85 and beats baselines |
| Fraud — precision in the top 1 % (base rate 1.7 %) | **70 %** (40× lift) | ≥ 40 % |
| Device rings flagged (≥ 3 applicants + unusual members) | **61**, of which **84 %** are fraud rings; 38 % of all fraud | precision ≥ 70 % |

### Read these numbers honestly

- **The data is synthetic** with known mechanisms; real bureau and bank-statement data are noisier and drift in ways nobody planted. Validate on real vintages before any use; a production PD model also needs independent model validation and documentation under the lender's model-risk policy.
- **Both PD models under-estimate 2025 defaults (8.9 % vs 10.0 %).** The ranking holds (score PSI 0.04), but the channel mix shifted (CSI 0.68). The monitoring alert is the right outcome; the fix is re-calibrating on recent vintages, which a 12-month outcome window delays.
- **GBM beats the scorecard by 1.7 AUC points** and makes slightly more profit; the scorecard gives cleaner reason codes. Many lenders run the scorecard as the decision model and the GBM as a challenger — the platform supports either.
- **New-to-credit applicants are much harder (AUC 0.70).** Alternative data (bank-statement cash flow) carries most of the signal for them.
- **The fairness checks pass here, but income carries a structural gender gap**; proxies can reintroduce bias even when gender is excluded, so the check has to run on every model change.
- **Fraud labels are used only for evaluation.** Precision of 70 % in the top 1 % means three in ten reviews are false alarms; families sharing a device are the main legitimate look-alike.

## Quick start

```bash
pip install -r requirements-dev.txt
make data test eval       # generate data, tests, full evaluation with gate
make console              # Streamlit console on :8501
make api                  # REST API on :8000 (X-API-Key: dev-key)
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/decision` | approve / decline / refer to fraud review, PD, scorecard points, expected loss and profit, reason codes |
| `POST /v1/lgd` | expected LGD and cure probability for a defaulted loan |
| `POST /v1/segment` | segment for a card customer's behaviour |
| `GET /v1/fraud/queue` | most anomalous applications for the fraud team |
| `GET /v1/model` | cut-offs and evaluation card |
| `GET /health`, `/ready`, `/metrics` | probes and Prometheus metrics |

Docker: `make up` (API + console; `--profile observability` adds Prometheus).

## Repository layout

```
creditml/
  data/generator.py   applications, losses, card behaviour (reproducible; manifest checked in CI)
  pd_model.py         feature governance, leakage audit, WoE scorecard, monotonic GBM, metrics
  policy.py           profit-based cut-off, baselines, fairness
  lgd_model.py        product mean, direct GBM, two-stage LGD
  unsupervised.py     segmentation, Isolation Forest / LOF, device rings
  service.py          production models for the API and console
  api.py · config.py
  eval/run.py         evaluation and quality gate
streamlit_app.py      console: decision, PD, policy & fairness, LGD, segments, fraud
docs/                 architecture, evaluation, runbook, ADRs
```

## Design decisions

- [ADR-0001](docs/adr/0001-feature-governance.md) — Feature availability tags and a leakage audit before any modelling
- [ADR-0002](docs/adr/0002-scorecard-and-gbm.md) — WoE scorecard and monotonic GBM side by side, validated out of time
- [ADR-0003](docs/adr/0003-profit-cutoff-and-fairness.md) — Cut-off by expected profit, with fairness checks on the decision
- [ADR-0004](docs/adr/0004-two-stage-lgd.md) — Two-stage LGD for a bimodal target
- [ADR-0005](docs/adr/0005-unsupervised-fraud.md) — Unsupervised fraud detection with labels only for evaluation

## Limitations and roadmap

- Reject inference (declined applicants have no outcome) and through-the-door population estimates.
- Lifetime PD and IFRS 9 / Ind AS 109 staging; EAD modelling for revolving products.
- SHAP explanations for the GBM challenger; adverse-action notices in plain language.
- Graph-based fraud (shared addresses, bank accounts, employers) beyond device sharing.

## Licence

MIT
