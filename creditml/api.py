"""FastAPI service.

  POST /v1/decision      application → approve / decline / refer to fraud review, PD, score, expected loss, reason codes
  POST /v1/lgd           defaulted loan → expected LGD and cure probability
  POST /v1/segment       card customer behaviour → segment
  GET  /v1/fraud/queue   most anomalous applications for the fraud team
  GET  /v1/model         evaluation card
  GET  /health, /ready, /metrics
"""
from __future__ import annotations

import time
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from creditml import logging_setup
from creditml import service as S
from creditml.config import get_settings

logging_setup.configure()
app = FastAPI(title="Credit Risk ML Platform", version="1.0.0")
REQ = Counter("creditml_http_requests_total", "HTTP requests", ["path", "status"])
LAT = Histogram("creditml_request_seconds", "Request latency", ["path"])
DECISIONS = Counter("creditml_decisions_total", "Credit decisions", ["decision"])


@app.middleware("http")
async def ctx(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    tok = logging_setup.request_id.set(rid)
    t0 = time.perf_counter()
    try:
        resp = await call_next(request)
    finally:
        logging_setup.request_id.reset(tok)
    resp.headers["X-Request-ID"] = rid
    REQ.labels(request.url.path, resp.status_code).inc()
    LAT.labels(request.url.path).observe(time.perf_counter() - t0)
    return resp


def auth(x_api_key: str | None = Header(default=None)):
    if x_api_key not in get_settings().api_keys:
        raise HTTPException(401, "Invalid or missing API key")


class Application(BaseModel):
    product: Literal["personal", "two_wheeler", "gold"]
    channel: Literal["branch", "dsa", "digital_partner"]
    age: int = Field(ge=18, le=80)
    employment: Literal["salaried", "self_employed", "gig"]
    city_tier: Literal["tier1", "tier2", "tier3"]
    monthly_income: float = Field(gt=0)
    bureau_score: float | None = Field(None, ge=300, le=900, description="null for new-to-credit applicants")
    bureau_vintage_months: float = Field(0, ge=0)
    utilisation_pct: float | None = Field(None, ge=0, le=200)
    enquiries_6m: int = Field(0, ge=0)
    max_dpd_12m: float | None = Field(None, ge=0)
    active_loans: int = Field(0, ge=0)
    loan_amount: float = Field(gt=0)
    tenure_months: int = Field(ge=3, le=84)
    interest_rate: float | None = Field(None, ge=0, le=40, description="offered rate; used only for profit, never for PD")
    foir: float = Field(ge=0, le=5)
    avg_bank_balance: float = Field(ge=0)
    ltv: float | None = Field(None, ge=0, le=1.5)
    device_id: str = "unknown"
    phone_age_days: int = Field(365, ge=0)
    email_age_days: int = Field(365, ge=0)
    application_hour: int = Field(12, ge=0, le=23)


class DefaultedLoan(BaseModel):
    product: Literal["personal", "two_wheeler", "gold"]
    employment: Literal["salaried", "self_employed", "gig"]
    channel: Literal["branch", "dsa", "digital_partner"]
    ltv: float | None = None
    months_on_book_at_default: int = Field(ge=0, le=120)
    avg_bank_balance: float = Field(ge=0)
    ead: float = Field(gt=0)
    loan_amount: float = Field(gt=0)
    monthly_income: float = Field(gt=0)
    bureau_score: float | None = None


class CardCustomer(BaseModel):
    utilisation: float = Field(ge=0, le=2)
    payment_ratio: float = Field(ge=0, le=1)
    cash_share: float = Field(ge=0, le=1)
    travel_share: float = Field(ge=0, le=1)
    online_share: float = Field(ge=0, le=1)
    txns_per_month: float = Field(ge=0)
    active_months: float = Field(ge=0, le=6)
    credit_limit: float = Field(gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    m = S.models()
    return {"status": "ready", "pd_cutoff": m.cutoff, "segments": sorted(set(m.seg_names.values()))}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/decision", dependencies=[Depends(auth)])
def decision(a: Application):
    r = S.decide(a.model_dump())
    DECISIONS.labels(r["decision"]).inc()
    return r


@app.post("/v1/lgd", dependencies=[Depends(auth)])
def lgd(loan: DefaultedLoan):
    return S.predict_lgd(loan.model_dump())


@app.post("/v1/segment", dependencies=[Depends(auth)])
def segment(c: CardCustomer):
    return S.segment_customer(c.model_dump())


@app.get("/v1/fraud/queue", dependencies=[Depends(auth)])
def fraud_queue(top: int = Query(25, ge=1, le=200)):
    q = S.fraud_queue(top)
    q["disbursal_month"] = q["disbursal_month"].dt.strftime("%Y-%m")
    return q.drop(columns=["is_fraud"]).round(4).to_dict("records")


@app.get("/v1/model", dependencies=[Depends(auth)])
def model_card():
    m = S.models()
    return {"pd_cutoff": m.cutoff, "fraud_review_score": m.fraud_cut, "evaluation": m.card}
