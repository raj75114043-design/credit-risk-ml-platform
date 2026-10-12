# ADR-0003: Cut-off by expected profit, with fairness checks on the decision

**Status:** Accepted · 2026-10-12

## Context
AUC does not decide who gets a loan; a cut-off does. The best cut-off depends on margins and losses per product, and the
decision — not the score — is what fairness rules apply to.

## Decision
Choose the PD cut-off that maximises realised profit (net interest margin on performing loans minus PD × LGD × EAD) on
the calibration period; apply it out of time; compare with approve-all and a bureau-score rule. Measure the approval
ratio between women and men (four-fifths rule) and the approval gap among borrowers who repaid (equal opportunity).

## Consequences
+ Profit ₹7.97 crore vs ₹6.77 crore (approve all) and ₹4.55 crore (bureau rule) on the same applicants.
+ Approval ratio 0.98, equal-opportunity gap 1.6 points.
− The profit model uses long-run LGD by product and simple margin assumptions; treasury and collections inputs would refine it.
