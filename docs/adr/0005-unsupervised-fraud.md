# ADR-0005: Unsupervised fraud detection with labels only for evaluation

**Status:** Accepted · 2026-10-12

## Context
Confirmed fraud labels arrive months after disbursal and are biased towards what investigators already look for.

## Decision
Isolation Forest on engineered signals (income vs bank balance, income vs peers, phone and e-mail age, night-time
applications, applications per device), compared with Local Outlier Factor and a hand rule. Devices shared by three or
more applicants are flagged as rings only when their members are also anomalous (families share devices too).
Labels are used only to measure precision and recall.

## Consequences
+ AUROC 0.93; 70 % of the top 1 % are fraud (40× the base rate); 84 % of flagged rings are fraud rings.
− Fraudsters adapt; the features and thresholds need periodic review against new confirmed cases.
