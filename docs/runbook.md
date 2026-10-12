# Runbook

| Signal | First checks | Action |
|---|---|---|
| CSI > 0.1 on a feature | Channel / product mix, sourcing partner changes | Check calibration by segment; re-calibrate on recent vintages when outcomes mature |
| Score PSI > 0.25 | Data pipeline, bureau format changes | Stop automated decisions until resolved |
| Approval ratio < 0.8 or equal-opportunity gap > 5 pts | Which features moved? Proxies for protected attributes? | Escalate to model risk; review features and cut-off |
| Fraud queue growing | New channel or partner | Review top cases; update features with confirmed patterns |
| LGD portfolio error > 5 % | Collections or recovery-channel changes | Retrain the severity model on recent recoveries |
