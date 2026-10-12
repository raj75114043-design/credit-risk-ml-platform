# ADR-0001: Feature availability tags and a leakage audit before any modelling

**Status:** Accepted · 2026-10-12

## Context
Lending data mixes fields captured at application, fields created by the decision (price, limit) and fields created
after disbursal (collections). Any of the latter in a PD model makes back-testing look excellent and production fail.

## Decision
Every raw field is tagged: `application`, `pricing_decision`, `post_disbursal`, `protected`, `identifier`, `fraud_only`.
Only `application` fields are model inputs. A univariate-AUC audit runs on every training set and flags fields above
0.80 for review.

## Consequences
+ The leaky collection-calls field (AUC 1.00) is caught; the model it would have produced (AUC 0.9998) is reported as a warning.
+ The interest rate (circular) and gender (protected) never reach the model.
− New fields need a tag before use — a small process cost that is the point.
