# ADR-0004: Two-stage LGD for a bimodal target

**Status:** Accepted · 2026-10-12

## Context
About a third of defaulted borrowers cure (LGD ≈ 0) and a third of losses exceed 70 %. A single regression averages
the two populations and predicts values that almost never occur.

## Decision
Model P(cure) with a classifier and E[LGD | no cure] with a regressor; combine as (1 − p) · severity + p · cure LGD.
Compare with product averages and direct regression on loan-level MAE and on portfolio loss by product.

## Consequences
+ MAE 0.196 vs 0.223 for product averages; portfolio loss within 3 %.
− Direct regression has a smaller portfolio-level error (1.3 %) — for provisioning totals, either is acceptable; the two-stage model gives an interpretable cure probability for collections strategy.
