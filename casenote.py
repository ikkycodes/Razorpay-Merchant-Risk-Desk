"""Validated case notes (Part C)."""
import json
import mock_llm
from retrieval import retrieve, MIN_SCORE
SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "medium", "high"]}, "recommended_team": {"type": "string", "enum": ["card_risk", "upi_risk", "merchant_ops", "customer_care"]}, "recommended_action": {"type": "string", "enum": ["approve", "hold_for_review", "block_and_notify"]}, "policy_id": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5", "P6"]}}, "required": ["summary", "priority", "recommended_team", "recommended_action", "policy_id"]}
def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text):
    return f"""Role: You are a fraud triage assistant for a payments risk desk in India.
Task: Write a structured case note for the transaction below using ONLY the supplied policy text.
Constraints: Use only the supplied policy text. Return exactly one JSON object with no additional prose.
Output format: JSON object with keys summary (string), priority (low|medium|high), recommended_team (card_risk|upi_risk|merchant_ops|customer_care), recommended_action (approve|hold_for_review|block_and_notify), policy_id (P1|P2|P3|P4|P5|P6).
Policy text: {policy_text}
TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}
"""
def validate(text, retrieved_policy_id):
    try:
        note = json.loads(text)
    except Exception:
        return (None, 1)
    if not isinstance(note, dict):
        return (None, 1)
    for k in SCHEMA["required"]:
        if k not in note or not isinstance(note[k], str):
            return (None, 2)
    if note["priority"] not in ("low", "medium", "high"):
        return (None, 3)
    if note["recommended_team"] not in ("card_risk", "upi_risk", "merchant_ops", "customer_care"):
        return (None, 3)
    if note["recommended_action"] not in ("approve", "hold_for_review", "block_and_notify"):
        return (None, 3)
    if note["policy_id"] not in ("P1", "P2", "P3", "P4", "P5", "P6"):
        return (None, 3)
    if note["policy_id"] != retrieved_policy_id:
        return (None, 3)
    return (note, 0)
def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=3):
    pid, ptext, score = retrieve(question)
    if score < MIN_SCORE:
        return {"status": "no_policy_match", "attempts": 0, "failures": [], "case_note": None, "safe_default": False, "guardrail_override": False}
    prompt = build_prompt(txn_id, fraud_probability, amount_inr, pid, ptext)
    failures = []
    attempts = 0
    for _ in range(max_attempts):
        resp = mock_llm.generate(prompt)
        attempts += 1
        note, gate = validate(resp, pid)
        if gate == 0:
            go = False
            if note["recommended_action"] == "approve" and float(fraud_probability) >= 0.5:
                note["recommended_action"] = "hold_for_review"
                go = True
            return {"status": "ok", "attempts": attempts, "failures": failures, "case_note": note, "safe_default": False, "guardrail_override": go}
        failures.append(gate)
    safe = {"summary": "Automated case note unavailable; manual review required.", "priority": "high", "recommended_team": "customer_care", "recommended_action": "hold_for_review", "policy_id": pid}
    return {"status": "safe_default", "attempts": attempts, "failures": failures, "case_note": safe, "safe_default": True, "guardrail_override": False}
def main():
    mock_llm.reset()
    scenarios = json.load(open("case_scenarios.json"))
    total_calls = 0
    for s in scenarios:
        r = run_case(s["txn_id"], s["fraud_probability"], s["amount_inr"], s["question"])
        total_calls += r["attempts"]
        print(f"{s['scenario_id']} {s['txn_id']}: policy={r['case_note']['policy_id'] if r['case_note'] else 'none'} attempts={r['attempts']} failures={r['failures']} status={r['status']} safe_default={r['safe_default']} action={r['case_note']['recommended_action'] if r['case_note'] else None} guardrail={r['guardrail_override']}")
    print(f"Total model calls: {total_calls}")
    try:
        run_case("T900099", 0.9, 1000.0, "my card was stolen and used without permission")
        print("T900099 did NOT raise (unexpected)")
    except ConnectionError:
        print("T900099 raises ConnectionError as expected")
    n_calls_before = dict(mock_llm._counts)
    run_case("T900099X", 0.1, 100.0, "what is the weather today in Mumbai")
    print(f"OOS no-call check: counts unchanged for new txn = {'T900099X' not in mock_llm._counts}")
if __name__ == "__main__":
    main()
