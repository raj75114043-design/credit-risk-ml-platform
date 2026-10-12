# Contributing

1. `pip install -r requirements-dev.txt`
2. Every new field needs an availability tag in `creditml/pd_model.py` before it can be used.
3. `make lint test eval` must pass. Never lower a gate threshold to make a change pass; explain metric changes in the PR.
4. Fairness, leakage and reason-code regressions block a merge. Never commit real applicant data.
