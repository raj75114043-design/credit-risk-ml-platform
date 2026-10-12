"""Synthetic retail-lending data for a fictional Indian NBFC, Kestrel Finance (Jan 2023 – Jun 2026).

* ``applications`` — 42,000 loan applications (personal, two-wheeler and gold loans) with application, bureau and
  bank-statement fields. Default = 90+ days past due within 12 months of disbursal, so loans disbursed after Jun 2025
  are *immature* (no outcome yet). Real-world traps:
    - new-to-credit applicants have no bureau score (missing, not zero)
    - risk-based pricing: the interest rate is set *from* risk, so it leaks risk information
    - ``collection_calls_90d`` is recorded after disbursal — target leakage that must be excluded
    - from Jan 2025 a new digital partner channel brings a riskier population (drift)
    - gender is recorded for fairness monitoring only and must not be a model input
    - ~2 % first-party / synthetic-identity fraud: inflated income, shared devices, new phone numbers
* ``losses`` — recovery outcomes for defaulted loans: loss given default (LGD) is bimodal (cures vs write-offs) and
  depends on collateral.
* ``card_behaviour`` — 8,000 credit-card customers' 6-month behaviour, generated from six latent segments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = {"personal": (300_000, 0.6), "two_wheeler": (110_000, 0.25), "gold": (150_000, 0.15)}   # typical amount, share
CHANNELS = ["branch", "dsa", "digital_partner"]
EMPLOYMENT = ["salaried", "self_employed", "gig"]
CITY_TIERS = ["tier1", "tier2", "tier3"]
SEGMENTS = ["transactor", "revolver", "cash_stressed", "premium_traveller", "dormant", "new_to_card"]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def applications(n: int = 42_000, seed: int = 5) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    months = pd.date_range("2023-01-01", "2026-06-01", freq="MS")
    w = np.linspace(1, 1.6, len(months))
    month = r.choice(months, n, p=w / w.sum())
    late = month >= np.datetime64("2025-01-01")
    channel = np.where(late, r.choice(CHANNELS, n, p=[0.35, 0.30, 0.35]), r.choice(CHANNELS, n, p=[0.55, 0.40, 0.05]))
    product = r.choice(list(PRODUCTS), n, p=[v[1] for v in PRODUCTS.values()])
    gender = r.choice(["F", "M"], n, p=[0.32, 0.68])
    emp = r.choice(EMPLOYMENT, n, p=[0.58, 0.30, 0.12])
    emp = np.where((channel == "digital_partner") & (r.random(n) < 0.35), "gig", emp)
    age = np.clip(r.normal(34, 9, n), 21, 65).round()
    city = r.choice(CITY_TIERS, n, p=[0.45, 0.35, 0.20])
    base_inc = {"salaried": 62_000, "self_employed": 70_000, "gig": 28_000}
    income = np.array([base_inc[e] for e in emp]) * np.exp(r.normal(0, 0.45, n)) * np.where(city == "tier1", 1.25, np.where(city == "tier3", 0.8, 1.0))
    income *= np.where(gender == "F", 0.9, 1.0)   # structural pay gap → a fairness risk through income
    income = income.round(-2)
    ntc = r.random(n) < np.where(channel == "digital_partner", 0.30, 0.12)
    bureau_vintage = np.where(ntc, 0, np.clip(r.gamma(2.5, 30, n), 6, 300)).round()
    latent = r.normal(0, 1, n) + np.where(emp == "gig", -0.4, 0) + np.where(channel == "digital_partner", -0.35, 0)   # unobserved credit quality
    bureau = np.where(ntc, np.nan, np.clip(720 + 55 * latent + 0.08 * bureau_vintage + r.normal(0, 25, n), 300, 900)).round()
    util = np.where(ntc, np.nan, np.clip(_sigmoid(-0.9 * latent + r.normal(0, 0.8, n)) * 110, 0, 140)).round(1)
    enq6 = r.poisson(np.where(ntc, 1.2, np.clip(1.5 - 0.6 * latent, 0.2, 6)))
    max_dpd = np.where(ntc, np.nan, np.where(r.random(n) < _sigmoid(-2.5 - 1.3 * latent), r.choice([30, 60, 90], n, p=[0.6, 0.3, 0.1]), 0))
    active = np.where(ntc, 0, r.poisson(np.clip(2 - 0.3 * latent, 0.3, 6)))
    amount = np.array([PRODUCTS[p][0] for p in product]) * np.exp(r.normal(0, 0.4, n))
    amount = np.minimum(amount, income * np.where(product == "personal", 10, 6)).round(-3)
    tenure = np.where(product == "gold", 12, r.choice([12, 24, 36, 48], n, p=[0.15, 0.35, 0.35, 0.15]))
    # risk-based pricing (leaks risk)
    rate = (np.where(product == "gold", 11.5, np.where(product == "two_wheeler", 14.0, 16.0)) \
        + np.where(ntc, 3.0, np.clip((760 - np.nan_to_num(bureau, nan=700)) / 40, -1, 6))
            + r.normal(0, 0.5, n)).round(2)
    emi = amount * (rate / 1200) / (1 - (1 + rate / 1200) ** -tenure)
    obligations = active * r.uniform(2_000, 9_000, n)
    foir = ((emi + obligations) / income).round(3)
    avg_balance = (income * np.exp(r.normal(-0.5, 0.6, n)) * _sigmoid(latent + 1)).round(-2)
    ltv = np.where(product == "gold", r.uniform(0.55, 0.75, n), np.where(product == "two_wheeler", r.uniform(0.7, 0.95, n), np.nan)).round(3)
    # fraud: inflated income, shared device, fresh phone — each signal present in only some cases, and legitimate
    # look-alikes exist (families share a device, people change numbers, gig workers apply at night)
    fraud = r.random(n) < np.where(channel == "digital_partner", 0.04, 0.012)
    inflate = fraud & (r.random(n) < 0.55)
    income = np.where(inflate, income * r.uniform(1.4, 2.6, n), income).round(-2)
    device = np.array([f"D{x:06d}" for x in r.integers(0, 10**6, n)])
    family = (~fraud) & (r.random(n) < 0.02)
    device = np.where(family, np.array([f"F{x:05d}" for x in r.integers(0, int(n * 0.025), n)]), device)
    ring_devices = [f"R{x:03d}" for x in range(60)]
    device = np.where(fraud & (r.random(n) < 0.4), r.choice(ring_devices, n), device)
    phone_age_days = np.where(fraud & (r.random(n) < 0.6), r.integers(1, 120, n), np.where(r.random(n) < 0.08, r.integers(1, 120, n), r.integers(120, 3650, n)))
    email_age_days = np.where(fraud & (r.random(n) < 0.5), r.integers(1, 150, n), np.where(r.random(n) < 0.08, r.integers(1, 150, n), r.integers(150, 4000, n)))
    hour = np.where((fraud & (r.random(n) < 0.3)) | ((emp == "gig") & (r.random(n) < 0.15)) | (r.random(n) < 0.02), r.integers(0, 5, n), r.integers(8, 23, n))
    # default: nonlinear in true drivers
    foir_true = (emi + obligations) / (income / np.where(inflate, 2.0, 1.0))
    logit = (-3.75 + np.where(ntc, 0.45, 0) - 0.85 * latent + 2.2 * np.clip(foir_true - 0.55, 0, None) + 0.18 * np.minimum(enq6, 8)
             + np.where(np.nan_to_num(max_dpd) >= 60, 0.7, 0)
             + np.where(emp == "gig", 0.35, 0) + np.where(product == "gold", -1.2, np.where(product == "two_wheeler", -0.2, 0)) + np.where(age < 25, 0.3, 0)
             + np.where(late, 0.15, 0) + np.where(fraud, 3.0, 0) - 0.15 * np.log1p(avg_balance / 10_000))
    default = r.random(n) < _sigmoid(logit)
    disb = pd.to_datetime(month)
    mature = disb <= pd.Timestamp("2025-06-01")
    calls = np.where(default, r.poisson(9, n), r.poisson(0.4, n))   # recorded after disbursal: LEAKAGE
    df = pd.DataFrame({
        "application_id": [f"KF{i:06d}" for i in range(n)], "disbursal_month": disb, "product": product, "channel": channel, "gender": gender,
        "age": age, "employment": emp, "city_tier": city, "monthly_income": income, "bureau_score": bureau, "bureau_vintage_months": bureau_vintage,
        "utilisation_pct": util, "enquiries_6m": enq6, "max_dpd_12m": max_dpd, "active_loans": active, "loan_amount": amount, "tenure_months": tenure,
        "interest_rate": rate, "foir": foir, "avg_bank_balance": avg_balance, "ltv": ltv, "device_id": device, "phone_age_days": phone_age_days,
        "email_age_days": email_age_days, "application_hour": hour, "collection_calls_90d": np.where(mature, calls, np.nan),
        "default_12m": np.where(mature, default.astype(float), np.nan), "is_fraud": fraud})
    return df.sort_values(["disbursal_month", "application_id"]).reset_index(drop=True)


def losses(apps: pd.DataFrame, seed: int = 6) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    d = apps[apps["default_12m"] == 1].copy()
    n = len(d)
    ead = d["loan_amount"] * r.uniform(0.45, 0.95, n)
    months_on_book = r.integers(3, 13, n)
    bal = np.log1p(d["avg_bank_balance"].to_numpy() / 1000)
    score = np.nan_to_num(d["bureau_score"].to_numpy(), nan=680)
    cure_logit = (-1.6 + np.where(d["product"] == "gold", 1.8, np.where(d["product"] == "two_wheeler", 0.3, 0)) + np.where(d["employment"] == "salaried", 0.9, 0)
                  + 0.45 * (bal - 3) + 0.012 * (score - 680) + 0.18 * (months_on_book - 6) - np.where(d["is_fraud"], 2.5, 0))
    cured = r.random(n) < _sigmoid(cure_logit)
    ltv = d["ltv"].fillna(1).to_numpy()
    collateral_recovery = np.where(d["product"] == "gold", np.clip(r.normal(0.97, 0.03, n) / ltv * 0.72, 0, 1),
                                   np.where(d["product"] == "two_wheeler", np.clip(r.normal(0.55, 0.12, n) * (1.2 - 0.0 * ltv) - 0.35 * (ltv - 0.8), 0, 1), 0))
    unsecured_recovery = np.where(d["product"] == "personal", np.clip(r.beta(1.5, 5, n) + 0.06 * (bal - 3)
                                                                            - np.where(d["channel"] == "digital_partner", 0.08, 0), 0, 1), 0)
    amortised = 1 - 0.03 * months_on_book
    lgd_wo = np.clip((1 - collateral_recovery - unsecured_recovery) * amortised + 0.08, 0, 1)   # + collection costs
    lgd = np.where(cured, np.clip(r.normal(0.03, 0.02, n), 0, 0.1), lgd_wo)
    return pd.DataFrame({"application_id": d["application_id"].to_numpy(), "ead": ead.round(-2), "months_on_book_at_default": months_on_book,
                         "cured": cured, "lgd": lgd.round(4),
                         "recovery_channel": np.where(cured, "cure", np.where(d["product"] == "personal", "agency", "repossession"))})


def card_behaviour(n: int = 8_000, seed: int = 7) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    seg = r.choice(SEGMENTS, n, p=[0.28, 0.22, 0.10, 0.08, 0.17, 0.15])
    prof = {  # utilisation, payment ratio, cash share, travel share, online share, monthly txns, months active
        "transactor": (0.25, 1.0, 0.01, 0.06, 0.35, 28, 6), "revolver": (0.70, 0.18, 0.04, 0.04, 0.30, 22, 6),
        "cash_stressed": (0.92, 0.07, 0.35, 0.01, 0.15, 14, 6), "premium_traveller": (0.35, 0.95, 0.01, 0.40, 0.25, 35, 6),
        "dormant": (0.03, 1.0, 0.0, 0.0, 0.2, 1, 1.5), "new_to_card": (0.45, 0.6, 0.03, 0.03, 0.55, 12, 3)}
    rows = []
    for s in seg:
        u, p, c, t, o, tx, act = prof[s]
        rows.append({"segment_true": s, "utilisation": np.clip(r.normal(u, 0.2), 0, 1.2), "payment_ratio": np.clip(r.normal(p, 0.22), 0, 1),
                     "cash_share": np.clip(r.normal(c, 0.09), 0, 1), "travel_share": np.clip(r.normal(t, 0.1), 0, 1), "online_share": np.clip(r.normal(o, 0.15), 0, 1),
                     "txns_per_month": max(0.0, r.normal(tx, tx * 0.45)), "active_months": float(np.clip(round(r.normal(act, 1.2)), 0, 6)),
                     "credit_limit": float(np.exp(r.normal({"premium_traveller": 12.9, "new_to_card": 11.5}.get(s, 12.0), 0.45)))})
    d = pd.DataFrame(rows)
    risk = {"transactor": 0.01, "revolver": 0.06, "cash_stressed": 0.22, "premium_traveller": 0.008, "dormant": 0.01, "new_to_card": 0.05}
    d["default_next_12m"] = r.random(n) < d["segment_true"].map(risk)
    d.insert(0, "customer_id", [f"CC{i:05d}" for i in range(n)])
    return d.round(4)


def write(out: Path = ROOT / "data") -> dict:
    out.mkdir(exist_ok=True)
    a = applications()
    lo = losses(a)
    c = card_behaviour()
    a.to_csv(out / "applications.csv.gz", index=False)
    lo.to_csv(out / "losses.csv.gz", index=False)
    c.to_csv(out / "card_behaviour.csv.gz", index=False)
    man = {"applications": len(a), "mature": int(a["default_12m"].notna().sum()), "default_rate": round(float(a["default_12m"].mean()), 4),
           "fraud_rate": round(float(a["is_fraud"].mean()), 4), "defaults_with_losses": len(lo), "card_customers": len(c),
           "fingerprint": str(pd.util.hash_pandas_object(a.drop(columns=["disbursal_month"]), index=False).sum() % 10**12)}
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    return man


if __name__ == "__main__":
    print(json.dumps(write(), indent=2))
