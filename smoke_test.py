"""Smoke tests Q1-Q11 against http://127.0.0.1:8000."""
import time
import requests
BASE = "http://127.0.0.1:8000"
def base_txn(txn_id="T900010", mc="Grocery", city="Mumbai", pay="UPI", amt=5000.0, hr=14, cnt=2, avg=6000.0, dist=8.0, nd=0, age=400):
    return {"txn_id": txn_id, "merchant_category": mc, "city": city, "payment_mode": pay, "amount_inr": amt, "hour_of_day": hr, "txn_count_last_24h": cnt, "avg_amount_30d_inr": avg, "distance_from_home_km": dist, "is_new_device": nd, "account_age_days": age}
REQS = [
    {"id": "Q1", "cat": "happy", "method": "POST", "path": "/score", "body": base_txn("T900010"), "exp_status": 200},
    {"id": "Q2", "cat": "happy", "method": "POST", "path": "/score", "body": base_txn("T900011", amt=8500.0, age=60, cnt=6, nd=1), "exp_status": 200},
    {"id": "Q3", "cat": "happy", "method": "POST", "path": "/score", "body": base_txn("T900020", amt=150000.0, nd=1, age=400, cnt=2), "exp_status": 200, "checks": {"decision": "hold_for_review", "human_review": True}},
    {"id": "Q4", "cat": "escalation", "method": "POST", "path": "/case", "body": {**base_txn("T900012", mc="Electronics", amt=62000.0, age=30, cnt=7, nd=1), "question": "high value purchase made from new device verification"}, "exp_status": 200},
    {"id": "Q5", "cat": "escalation", "method": "POST", "path": "/case", "body": {**base_txn("T900014", mc="Electronics", amt=62000.0, age=30, cnt=7, nd=1, avg=8000.0, dist=15.0), "question": "high value purchase made from new device verification"}, "exp_status": 200, "checks": {"safe_default": True}},
    {"id": "Q6", "cat": "escalation", "method": "POST", "path": "/case", "body": {**base_txn("T900025", mc="Electronics", amt=62000.0, age=30, cnt=7, nd=1, avg=8000.0, dist=15.0), "question": "my card was stolen and used without permission"}, "exp_status": 200, "checks": {"decision": "hold_for_review", "flagged": True, "guardrail_override": True}},
    {"id": "Q7", "cat": "escalation", "method": "POST", "path": "/case", "body": {**base_txn("T900099", amt=62000.0, age=30, cnt=7, nd=1, avg=8000.0), "question": "high value purchase made from new device verification"}, "exp_status": 503, "checks": {"error": "case-note service temporarily unavailable, try again later"}},
    {"id": "Q8", "cat": "edge", "method": "POST", "path": "/score", "body": base_txn("T900021", amt=0.5), "exp_status": 400, "checks": {"field": "amount_inr"}},
    {"id": "Q9", "cat": "edge", "method": "POST", "path": "/score", "body": {k: v for k, v in base_txn("T900023").items() if k != "amount_inr"}, "exp_status": 422},
    {"id": "Q10", "cat": "edge", "method": "POST", "path": "/score", "body": base_txn("T900024", amt=1.0, hr=0, cnt=1, avg=2000.0, dist=5.0, nd=0, age=500), "exp_status": 200, "checks": {"decision": "approve", "flagged": False}},
    {"id": "Q11", "cat": "edge", "method": "GET", "path": "/metrics", "exp_status": 200},
]
def wait_health():
    for _ in range(120):
        try:
            r = requests.get(BASE + "/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False
def main():
    assert wait_health(), "service not ready"
    passed = {"happy": 0, "escalation": 0, "edge": 0}
    total = {"happy": 0, "escalation": 0, "edge": 0}
    n_pass = 0
    for q in REQS:
        total[q["cat"]] += 1
        try:
            if q["method"] == "GET":
                r = requests.get(BASE + q["path"], timeout=30)
            else:
                r = requests.post(BASE + q["path"], json=q.get("body"), timeout=60)
            ok = (r.status_code == q["exp_status"])
            detail = ""
            if ok and "checks" in q:
                try:
                    j = r.json()
                    for k, v in q["checks"].items():
                        if j.get(k) != v:
                            ok = False
                            detail = f"field {k}: expected {v}, got {j.get(k)}"
                except Exception as e:
                    ok = False
                    detail = str(e)
            if ok and q["id"] in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6") and r.status_code == 200:
                j = r.json()
                for k in (["txn_id", "fraud_probability", "flagged", "decision", "human_review"] if q["path"] == "/score" else ["txn_id", "fraud_probability", "flagged", "decision", "human_review", "case_note", "attempts", "safe_default", "guardrail_override"]):
                    if k not in j:
                        ok = False
                        detail = f"missing {k}"
            res = "PASS" if ok else "FAIL"
            if ok:
                n_pass += 1
                passed[q["cat"]] += 1
            print(f"{q['id']} {q['cat']} expected={q['exp_status']} observed={r.status_code} {res} {detail}")
        except Exception as e:
            print(f"{q['id']} {q['cat']} expected={q['exp_status']} observed=ERROR FAIL {e}")
    print(f"TOTAL: {n_pass}/{len(REQS)} PASS")
    for c in ("happy", "escalation", "edge"):
        print(f"{c}: {passed[c]}/{total[c]}")
    try:
        m = requests.get(BASE + "/metrics", timeout=10).json()
        print("METRICS:")
        print(m)
    except Exception as e:
        print(f"metrics fetch failed: {e}")
if __name__ == "__main__":
    main()
