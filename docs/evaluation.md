# Evaluation method

- **PD:** development Jan 2023 – Jun 2024 (confirmed fraud removed), calibration Jul – Dec 2024, out-of-time Jan – Jun 2025; later loans are immature. Metrics: AUC, Gini, KS, Brier, calibration by decile, score PSI and per-feature CSI.
- **Policy:** cut-off chosen on the calibration period, scored out of time; profit = margin on performing loans − LGD × EAD on defaults.
- **Fairness:** approval-rate ratio and approval gap among non-defaulters, by gender, at the chosen cut-off.
- **LGD:** random 70 / 30 split of defaulted loans; MAE, RMSE, portfolio loss error overall and by product.
- **Segmentation:** k = 2…10 by silhouette (GMM BIC reported); ARI against the generating segments; default-rate spread.
- **Fraud:** scores computed without labels; AUROC, precision in the top 1 %, recall in the top 3 %; ring precision.
