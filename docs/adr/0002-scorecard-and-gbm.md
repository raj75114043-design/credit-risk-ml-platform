# ADR-0002: WoE scorecard and monotonic GBM side by side, validated out of time

**Status:** Accepted · 2026-10-12

## Context
Credit models must be accurate, stable over time, explainable to applicants (adverse-action reasons) and sensible
(more bureau score must never mean more risk).

## Decision
A weight-of-evidence logistic scorecard (quantile bins, a missing bin, points with PDO 20) and a gradient-boosting model
with monotonic constraints on bureau score, FOIR, enquiries, past delinquency and bank balance, calibrated with isotonic
regression. Development Jan 2023 – Jun 2024, calibration Jul – Dec 2024, out-of-time test Jan – Jun 2025.

## Consequences
+ GBM AUC 0.825 vs scorecard 0.808 vs bureau score alone 0.734; monotonicity verified.
+ Scorecard reason codes give the top three attributes costing points.
− Neither model anticipates the 2025 channel shift in its calibration; monitoring catches it.
