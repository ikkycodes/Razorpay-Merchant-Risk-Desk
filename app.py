"""FastAPI triage service (Part D)."""
import time
import logging
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from train import build_final_model
from prepare import add_features
from casenote import run_case
_bundle = build_final_model()
MODEL = _bundle["model"]; COLUMNS = _bundle["columns"]; MEDIAN = _bundle["median"]; THRESH = _bundle["threshold"]
KNOWN_CATS = {"Grocery", "Food Delivery", "Fashion", "Electronics", "Utilities", "Travel"}
logging.basicConfig(filename="service.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("service")
app = FastAPI()
metrics = {"requests_total": 0, "success_total": 0, "error_total": 0, "human_review_total": 0, "latency_sum": 0.0}
class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: Optional[float] = None
    is_new_device: int
    account_age_days: int
class CaseIn(Txn):
    question: str
def _validate_business(d):
    mc = str(d["merchant_category"]).strip().title()
    if not (1 <= float(d["amount_inr"]) <= 1000000):
        return (400, {"error": "amount_inr must be between 1 and 1000000", "field": "amount_inr"})
    if not (0 <= int(d["hour_of_day"]) <= 23):
        return (400, {"error": "hour_of_day must be between 0 and 23", "field": "hour_of_day"})
    if int(d["txn_count_last_24h"]) < 0:
        return (400, {"error": "txn_count_last_24h must not be negative", "field": "txn_count_last_24h"})
    if float(d["avg_amount_30d_inr"]) <= 0:
        return (400, {"error": "avg_amount_30d_inr must be greater than zero", "field": "avg_amount_30d_inr"})
    if mc not in KNOWN_CATS:
        return (400, {"error": "unknown merchant_category", "field": "merchant_category"})
    return None
def _score_row(d):
    row = pd.DataFrame([{**d, "merchant_category": str(d["merchant_category"]).strip().title(), "city": str(d["city"]).strip().title()}])
    feat = add_features(row)
    Xenc = pd.get_dummies(feat, columns=["merchant_category", "city", "payment_mode"], dtype=int)
    for c in COLUMNS:
        if c not in Xenc.columns:
            Xenc[c] = 0
    Xenc = Xenc[COLUMNS]
    Xenc["distance_from_home_km"] = Xenc["distance_from_home_km"].fillna(MEDIAN)
    prob = float(MODEL.predict_proba(Xenc)[:, 1][0])
    flagged = prob >= THRESH
    return prob, flagged
def _log_and_count(endpoint, status, latency_ms, txn_id=None, prob=None, safe=False, override=False, large_hold=False):
    metrics["requests_total"] += 1
    metrics["latency_sum"] += latency_ms
    if status < 400:
        metrics["success_total"] += 1
    else:
        metrics["error_total"] += 1
    level = "INFO"
    if status >= 500:
        level = "ERROR"
    elif status >= 400 or safe or override or large_hold:
        level = "WARNING"
    msg = f"{endpoint} status={status} latency_ms={latency_ms:.2f}"
    if txn_id is not None:
        msg += f" txn_id={txn_id}"
    if prob is not None:
        msg += f" fraud_probability={prob:.4f}"
    if safe:
        msg += " safe_default=True"
    if override:
        msg += " guardrail_override=True"
    if large_hold:
        msg += " large_payment_hold=True"
    if level == "ERROR":
        logger.error(msg)
    elif level == "WARNING":
        logger.warning(msg)
    else:
        logger.info(msg)
@app.middleware("http")
async def _mw(request: Request, call_next):
    start = time.time()
    resp = await call_next(request)
    if request.method == "POST" and request.url.path in ("/score", "/case") and resp.status_code == 422:
        lat = (time.time() - start) * 1000.0
        try:
            body = await request.body()
            import json as _j
            _txn = _j.loads(body).get("txn_id") if body else None
        except Exception:
            _txn = None
        _log_and_count(request.url.path, 422, lat, _txn, None, False, False, False)
    return resp
def _finalize(endpoint, status, start, txn_id=None, prob=None, human=False, safe=False, override=False, large_hold=False):
    lat = (time.time() - start) * 1000.0
    _log_and_count(endpoint, status, lat, txn_id, prob, safe, override, large_hold)
    if human and status < 400:
        metrics["human_review_total"] += 1
@app.get("/health")
def health():
    return {"status": "ok"}
@app.post("/score")
def score(t: Txn):
    start = time.time()
    d = t.model_dump()
    bad = _validate_business(d)
    if bad:
        code, body = bad
        _finalize("/score", code, start, d.get("txn_id"), None, False)
        return JSONResponse(status_code=code, content=body)
    prob, flagged = _score_row(d)
    prob_r = round(prob, 4)
    large_hold = (not flagged) and float(d["amount_inr"]) >= 100000 and int(d["is_new_device"]) == 1
    if flagged:
        decision = "hold_for_review"; human = True
    elif large_hold:
        decision = "hold_for_review"; human = True
    else:
        decision = "approve"; human = False
    _finalize("/score", 200, start, d["txn_id"], prob_r, human, False, False, large_hold)
    return {"txn_id": d["txn_id"], "fraud_probability": prob_r, "flagged": bool(flagged), "decision": decision, "human_review": human}
@app.post("/case")
def case(c: CaseIn):
    start = time.time()
    d = c.model_dump()
    bad = _validate_business(d)
    if bad:
        code, body = bad
        _finalize("/case", code, start, d.get("txn_id"), None, False)
        return JSONResponse(status_code=code, content=body)
    prob, flagged = _score_row(d)
    prob_r = round(prob, 4)
    large_hold = (not flagged) and float(d["amount_inr"]) >= 100000 and int(d["is_new_device"]) == 1
    if not flagged:
        decision = "hold_for_review" if large_hold else "approve"
        human = bool(large_hold)
        _finalize("/case", 200, start, d["txn_id"], prob_r, human, False, False, large_hold)
        return {"txn_id": d["txn_id"], "fraud_probability": prob_r, "flagged": False, "decision": decision, "human_review": human, "case_note": None, "attempts": 0, "safe_default": False, "guardrail_override": False}
    try:
        r = run_case(d["txn_id"], prob, float(d["amount_inr"]), d["question"])
    except ConnectionError:
        _finalize("/case", 503, start, d["txn_id"], prob_r, True)
        return JSONResponse(status_code=503, content={"error": "case-note service temporarily unavailable, try again later"})
    if r["status"] == "no_policy_match":
        _finalize("/case", 200, start, d["txn_id"], prob_r, True, False, False, False)
        return {"txn_id": d["txn_id"], "fraud_probability": prob_r, "flagged": True, "decision": "hold_for_review", "human_review": True, "case_note": None, "attempts": 0, "safe_default": False, "guardrail_override": False}
    note = r["case_note"]; safe = bool(r["safe_default"]); go = bool(r["guardrail_override"])
    action = note["recommended_action"]
    if action == "approve":
        action = "hold_for_review"; note = dict(note); note["recommended_action"] = "hold_for_review"; go = True
    decision = action
    human = decision in ("hold_for_review", "block_and_notify") or safe
    if safe:
        human = True
    _finalize("/case", 200, start, d["txn_id"], prob_r, human, safe, go, False)
    return {"txn_id": d["txn_id"], "fraud_probability": prob_r, "flagged": True, "decision": decision, "human_review": human, "case_note": note, "attempts": r["attempts"], "safe_default": safe, "guardrail_override": go}
@app.get("/metrics")
def get_metrics():
    rt = metrics["requests_total"]
    st = metrics["success_total"]; et = metrics["error_total"]; hr = metrics["human_review_total"]
    avg = (metrics["latency_sum"] / rt) if rt else 0.0
    return {"requests_total": rt, "success_total": st, "error_total": et, "success_rate": round(st / rt, 4) if rt else 0.0, "error_rate": round(et / rt, 4) if rt else 0.0, "avg_latency_ms": avg, "human_review_total": hr, "human_review_rate": round(hr / rt, 4) if rt else 0.0}

