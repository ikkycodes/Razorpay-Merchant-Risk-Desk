"""Deterministic offline mock LLM (reconstructed). Do not modify behavior."""
import json
import re
_counts = {}
def reset():
    _counts.clear()
def _count(txn_id):
    _counts[txn_id] = _counts.get(txn_id, 0) + 1
    return _counts[txn_id]
def generate(prompt: str) -> str:
    m = re.search(r"^TXN_ID:\s*(\S+)\s*$", prompt, re.M)
    if not m:
        raise ValueError("prompt missing TXN_ID line")
    txn_id = m.group(1)
    mp = re.search(r"^FRAUD_PROBABILITY:\s*(\S+)\s*$", prompt, re.M)
    ma = re.search(r"^AMOUNT_INR:\s*(\S+)\s*$", prompt, re.M)
    mpl = re.search(r"^POLICY_ID:\s*(\S+)\s*$", prompt, re.M)
    if not (mp and ma and mpl):
        raise ValueError("prompt missing required lines")
    if txn_id == "T900099":
        raise ConnectionError("mock connection failed")
    call = _count(txn_id)
    prob = float(mp.group(1)); amt = float(ma.group(1)); pol = mpl.group(1)
    team = {"P1": "card_risk", "P2": "upi_risk", "P3": "merchant_ops", "P4": "customer_care", "P5": "merchant_ops", "P6": "merchant_ops"}.get(pol, "customer_care")
    action = "block_and_notify" if prob >= 0.75 else ("hold_for_review" if prob >= 0.4 else "approve")
    pri = "high" if prob >= 0.7 else ("medium" if prob >= 0.4 else "low")
    good = {"summary": f"Txn {txn_id} for INR {amt} matches {pol}; prob {prob}.", "priority": pri, "recommended_team": team, "recommended_action": action, "policy_id": pol}
    scen = {"T900010": "good", "T900011": "badjson_then_good", "T900012": "badjson_then_good", "T900013": "wrongkey_then_good", "T900014": "always_bad", "T900015": "approve_override", "T900016": "wrongpolicy_twice", "T900017": "good", "T900018": "badenum_then_good", "T900019": "badjson_then_good", "T900020": "good", "T900021": "good", "T900022": "good", "T900023": "good", "T900024": "good", "T900025": "approve_override"}.get(txn_id, "good")
    if scen == "good":
        return json.dumps(good)
    if scen == "always_bad":
        return "not json at all {{{"
    if scen == "approve_override":
        if call == 1:
            g = dict(good); g["recommended_action"] = "approve"
            return json.dumps(g)
        return json.dumps(good)
    if scen in ("badjson_then_good", "wrongkey_then_good", "wrongpolicy_then_good", "badenum_then_good") and call == 1:
        if scen == "badjson_then_good":
            return "{bad json"
        if scen == "wrongkey_then_good":
            g = dict(good); del g["priority"]
            return json.dumps(g)
        if scen == "wrongpolicy_then_good":
            g = dict(good); g["policy_id"] = "P1" if pol != "P1" else "P2"
            return json.dumps(g)
        if scen == "wrongpolicy_twice":
            g = dict(good); g["policy_id"] = "P1" if pol != "P1" else "P2"
            return json.dumps(g)
        if scen == "badenum_then_good":
            g = dict(good); g["priority"] = "urgent"
            return json.dumps(g)
    if scen == "wrongpolicy_twice" and call <= 2:
        g = dict(good); g["policy_id"] = "P1" if pol != "P1" else "P2"
        return json.dumps(g)
    return json.dumps(good)
