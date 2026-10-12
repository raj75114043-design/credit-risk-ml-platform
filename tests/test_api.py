import pytest
from fastapi.testclient import TestClient

from creditml.service import DEFAULT_APPLICATION


@pytest.fixture(scope="module")
def client():
    from creditml.api import app
    return TestClient(app)


H = {"X-API-Key": "test-key"}
APP = {k: v for k, v in DEFAULT_APPLICATION.items() if k != "gender"}


def test_auth_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/v1/decision", json=APP).status_code == 401


def test_decisions(client):
    risky = client.post("/v1/decision", headers=H, json=APP).json()
    good = client.post("/v1/decision", headers=H, json={**APP, "bureau_score": 800, "employment": "salaried", "monthly_income": 120000, "foir": 0.25,
                                                       "avg_bank_balance": 150000, "enquiries_6m": 0, "max_dpd_12m": 0, "utilisation_pct": 15}).json()
    assert risky["decision"] == "decline" and good["decision"] == "approve" and good["pd_12m"] < risky["pd_12m"]
    assert len(risky["reason_codes"]) == 3
    fraud = client.post("/v1/decision", headers=H, json={**APP, "phone_age_days": 3, "email_age_days": 2, "application_hour": 2, "monthly_income": 250000,
                                                        "avg_bank_balance": 1000, "device_id": "R001"}).json()
    assert fraud["decision"] == "refer_to_fraud_review"
    ntc = client.post("/v1/decision", headers=H, json={**APP, "bureau_score": None, "utilisation_pct": None, "max_dpd_12m": None}).json()
    assert 0 < ntc["pd_12m"] < 1
    assert client.post("/v1/decision", headers=H, json={**APP, "product": "car"}).status_code == 422


def test_lgd_segment_fraud_queue(client):
    base = {"product": "personal", "employment": "gig", "channel": "digital_partner", "months_on_book_at_default": 3, "avg_bank_balance": 2000, "ead": 150000,
            "loan_amount": 200000, "monthly_income": 30000, "bureau_score": 640}
    gold = client.post("/v1/lgd", headers=H, json={**base, "product": "gold", "ltv": 0.7, "employment": "salaried", "avg_bank_balance": 60000}).json()
    pers = client.post("/v1/lgd", headers=H, json=base).json()
    assert gold["expected_lgd"] < pers["expected_lgd"]
    seg = client.post("/v1/segment", headers=H, json={"utilisation": 0.95, "payment_ratio": 0.05, "cash_share": 0.4, "travel_share": 0.0, "online_share": 0.1,
                                                    "txns_per_month": 12, "active_months": 6, "credit_limit": 150000}).json()
    assert seg["segment"] == "Cash-stressed"
    q = client.get("/v1/fraud/queue", headers=H, params={"top": 10}).json()
    assert len(q) == 10 and "is_fraud" not in q[0]
