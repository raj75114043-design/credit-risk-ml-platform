"""Credit risk console (Streamlit)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from creditml import pd_model as P  # noqa: E402
from creditml import policy as Po  # noqa: E402
from creditml import service as S  # noqa: E402
from creditml import unsupervised as U  # noqa: E402
from creditml.data import loaders  # noqa: E402

logging.disable(logging.INFO)
st.set_page_config(page_title="Credit Risk ML Platform", page_icon="🏦", layout="wide")
EV = json.loads(Path("reports/eval.json").read_text()) if Path("reports/eval.json").exists() else {}


@st.cache_resource(show_spinner="Training PD, LGD, segmentation and fraud models (first start only, ~25 s)…")
def models():
    return S.models()


M = models()

with st.sidebar:
    st.markdown("### 🏦 Credit Risk ML Platform")
    st.caption("Kestrel Finance (fictional NBFC) · personal, two-wheeler and gold loans")
    st.markdown("**Classification** — probability of default  \n**Regression** — loss given default  \n**Unsupervised** — segments, fraud anomalies, device rings")
    if EV:
        st.markdown(f"Out-of-time AUC **{EV['pd_auc_gbm']:.3f}** · LGD MAE **{EV['lgd']['two_stage']['mae']:.3f}** · fraud AUROC **{EV['fraud_auroc']:.3f}**")
    st.caption("All applicants, loans and customers are synthetic.")

tabs = st.tabs(["🧾 Credit decision", "📈 PD model", "💰 Policy & fairness", "📉 LGD", "🧩 Segments", "🕵️ Fraud", "🏗️ Architecture"])

with tabs[0]:
    a = dict(S.DEFAULT_APPLICATION)
    c = st.columns(4)
    a["product"] = c[0].selectbox("Product", ["personal", "two_wheeler", "gold"])
    a["channel"] = c[1].selectbox("Channel", ["branch", "dsa", "digital_partner"], index=2)
    a["employment"] = c[2].selectbox("Employment", ["salaried", "self_employed", "gig"], index=2)
    a["city_tier"] = c[3].selectbox("City", ["tier1", "tier2", "tier3"], index=1)
    c = st.columns(4)
    ntc = c[0].checkbox("New to credit (no bureau record)")
    a["bureau_score"] = None if ntc else c[0].slider("Bureau score", 300, 900, 640)
    a["monthly_income"] = c[1].number_input("Monthly income (₹)", 5000, 1_000_000, 32000, 1000)
    a["loan_amount"] = c[2].number_input("Loan amount (₹)", 10000, 5_000_000, 200000, 10000)
    a["tenure_months"] = c[3].selectbox("Tenure (months)", [12, 24, 36, 48], index=2)
    c = st.columns(4)
    a["foir"] = c[0].slider("FOIR (EMIs ÷ income)", 0.05, 1.2, 0.74, 0.01)
    a["avg_bank_balance"] = c[1].number_input("Average bank balance (₹)", 0, 2_000_000, 4000, 1000)
    a["enquiries_6m"] = c[2].slider("Bureau enquiries (6 m)", 0, 12, 4)
    a["utilisation_pct"] = None if ntc else c[3].slider("Card utilisation (%)", 0, 140, 78)
    if ntc:
        a.update(bureau_vintage_months=0, max_dpd_12m=None, active_loans=0)
    if a["product"] != "personal":
        a["ltv"] = st.slider("Loan-to-value", 0.4, 1.0, 0.75 if a["product"] == "gold" else 0.85, 0.01)
    with st.expander("Application metadata (fraud signals)"):
        c = st.columns(3)
        a["phone_age_days"] = c[0].number_input("Phone number age (days)", 0, 10000, 400)
        a["email_age_days"] = c[1].number_input("E-mail age (days)", 0, 10000, 900)
        a["application_hour"] = c[2].slider("Hour of application", 0, 23, 14)
    r = S.decide(a)
    colour = {"approve": st.success, "decline": st.error, "refer_to_fraud_review": st.warning}[r["decision"]]
    colour(f"**Decision: {r['decision'].replace('_', ' ').upper()}** — PD {r['pd_12m']:.1%} (cut-off {r['pd_cutoff']:.1%}), score {r['scorecard_points']} points")
    k = st.columns(3)
    k[0].metric("Expected loss", f"₹{r['expected_loss_inr']:,}")
    k[1].metric("Expected profit", f"₹{r['expected_profit_inr']:,}" if r["expected_profit_inr"] is not None else "—")
    k[2].metric("Fraud anomaly percentile", f"{r['fraud_anomaly_percentile']:.1%}")
    st.markdown("**Reason codes** (points lost versus the best attribute — what an adverse-action notice would list)")
    st.dataframe(pd.DataFrame(r["reason_codes"]), hide_index=True, width="stretch")

with tabs[1]:
    if EV:
        st.markdown("**Out-of-time performance (Jan–Jun 2025 disbursals, 12-month outcome)**")
        t = pd.DataFrame({k: v for k, v in EV["pd_oot"].items() if k != "bureau_score_only"}).T[["auc", "gini", "ks", "brier", "mean_pd", "default_rate"]]
        t.loc["bureau score only", "auc"] = EV["pd_oot"]["bureau_score_only"]["auc"]
        st.dataframe(t.round(4), width="stretch")
        c1, c2 = st.columns(2)
        cal = pd.DataFrame(EV["pd_calibration_table_gbm"]).melt("decile", ["predicted", "observed"], var_name="series", value_name="default rate")
        c1.altair_chart(alt.Chart(cal).mark_line(point=True).encode(x="decile:O", y="default rate:Q", color="series:N").properties(
            height=280, title="Calibration by risk decile (GBM)"), width="stretch")
        csi = pd.Series(EV["csi"]).rename("CSI").reset_index().rename(columns={"index": "feature"}).head(10)
        c2.altair_chart(alt.Chart(csi).mark_bar().encode(x="CSI:Q", y=alt.Y("feature:N", sort="-x"), color=alt.condition("datum.CSI > 0.1", alt.value("#d62728"),
                        alt.value("#4c78a8"))).properties(height=280, title="Characteristic stability, development → out of time"), width="stretch")
        st.markdown("**Feature governance & leakage audit**")
        st.dataframe(pd.DataFrame(EV["leakage_audit"]), hide_index=True, width="stretch")
        st.caption(f"A model allowed to use post-disbursal collection calls would report AUC {EV['leaky_model_auc_oot']:.3f} — and be useless at application time.")

with tabs[2]:
    apps = loaders.applications()
    s = P.split(apps)
    oot = s["oot"]
    p = M.gbm.pd_(oot)
    grid = np.linspace(0.02, 0.5, 49)
    curve = pd.DataFrame([{"cut-off PD": c, **Po.summary(oot, p <= c)} for c in grid])
    c1, c2 = st.columns(2)
    c1.altair_chart(alt.Chart(curve).mark_line().encode(x="cut-off PD:Q", y=alt.Y("profit_crore:Q", title="profit (₹ crore)")).properties(
        height=280, title="Out-of-time profit by PD cut-off"), width="stretch")
    long = curve.melt("cut-off PD", ["approval_rate", "bad_rate_approved"])
    c2.altair_chart(alt.Chart(long).mark_line().encode(x="cut-off PD:Q", y="value:Q", color="variable:N").properties(
        height=280, title="Approval rate and bad rate"), width="stretch")
    if EV:
        st.dataframe(pd.DataFrame(EV["policy"]).drop(columns=["gbm_cutoff_pd", "scorecard_cutoff_pd"], errors="ignore").T.round(4), width="stretch")
        f = EV["fairness"]
        st.markdown(f"**Fairness (gender is not a model input):** approval-rate ratio **{f['disparate_impact_ratio']:.2f}** (four-fifths rule ≥ 0.80) · "
                    f"approval gap among good borrowers **{f['equal_opportunity_gap']:.1%}**")
        st.json(f["calibration_by_group"])

with tabs[3]:
    from creditml import lgd_model as L
    f = L.frame(loaders.applications(), loaders.losses())
    st.altair_chart(alt.Chart(f).mark_bar().encode(x=alt.X("lgd:Q", bin=alt.Bin(maxbins=40)), y="count()", color="product:N").properties(
        height=260, title="LGD is bimodal: cures near 0, unsecured write-offs near 1"), width="stretch")
    if EV:
        st.dataframe(pd.DataFrame({k: {"MAE": v["mae"], "RMSE": v["rmse"], "portfolio loss error": v["portfolio_loss_error"]} for k, v in EV["lgd"].items()}).T.round(4),
                     width="stretch")
    c = st.columns(4)
    prod = c[0].selectbox("Product ", ["personal", "two_wheeler", "gold"])
    emp = c[1].selectbox("Employment ", ["salaried", "self_employed", "gig"])
    mob = c[2].slider("Months on book at default", 1, 24, 6)
    bal = c[3].number_input("Bank balance at application (₹)", 0, 500000, 15000, 1000)
    st.json(S.predict_lgd({"product": prod, "employment": emp, "channel": "branch", "ltv": 0.75 if prod != "personal" else None, "months_on_book_at_default": mob,
                           "avg_bank_balance": bal, "ead": 150000, "loan_amount": 200000, "monthly_income": 40000, "bureau_score": 700}))

with tabs[4]:
    seg, _, _ = U.segment(loaders.cards(), 6)
    st.altair_chart(alt.Chart(seg.sample(2500, random_state=0)).mark_circle(size=18, opacity=0.6).encode(
        x="utilisation:Q", y="payment_ratio:Q", color="segment:N", tooltip=["segment", "cash_share", "travel_share"]).properties(height=340), width="stretch")
    st.dataframe(U.profile(seg), width="stretch")
    if EV:
        ks = pd.DataFrame(EV["segmentation"]["k_search"])
        st.altair_chart(alt.Chart(ks).mark_line(point=True).encode(x="k:O", y="silhouette:Q").properties(height=200, title="k chosen by silhouette"), width="stretch")
        st.caption(f"Agreement with the generating segments (ARI): {EV['segmentation_ari']:.2f} · default rate varies {EV['segment_risk_spread']:.0f}× across segments")

with tabs[5]:
    if EV:
        st.dataframe(pd.DataFrame(EV["fraud"]).T.round(3), width="stretch")
        st.caption(f"Base fraud rate {EV['fraud_base_rate']:.1%}: the top 1 % by Isolation Forest is {EV['fraud_precision_top1pct']:.0%} fraud "
                   f"({EV['fraud_precision_top1pct'] / EV['fraud_base_rate']:.0f}× lift). Device rings: {EV['rings']['flagged']} flagged, "
                   f"{EV['rings']['precision']:.0%} confirmed fraud rings, {EV['rings']['fraud_captured']:.0%} of all fraud.")
    st.markdown("**Fraud review queue** (labels shown only for this synthetic demo)")
    st.dataframe(S.fraud_queue(25), hide_index=True, width="stretch")

with tabs[6]:
    st.graphviz_chart("""
    digraph C { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f4f4f4", fontname=Helvetica];
      A [label="Application\\n+ bureau + bank statement"]; G [label="Feature governance\\nleakage audit"]; F [label="Isolation Forest\\n+ device rings"];
      S [label="WoE scorecard\\nreason codes"]; B [label="GBM (monotonic)\\nisotonic calibration"]; P [label="Policy\\nprofit cut-off · fairness"];
      D [label="Approve / decline /\\nfraud review"]; L [label="LGD two-stage\\ncure × severity"];
      K [label="Card behaviour\\nk-means segments"]; M [label="Monitoring\\nPSI · CSI · calibration"];
      A -> G -> S -> P; G -> B -> P -> D; A -> F -> D; L -> P; B -> M; K -> P [style=dashed, label="strategy"]; }""")
