"""Generate synthetic data (reconstructed). Run: python generate_data.py"""
import json, random
import numpy as np
import pandas as pd
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)
MERCHANTS_CLEAN = ["Grocery", "Food Delivery", "Fashion", "Electronics", "Utilities", "Travel"]
MERCHANT_TARGET = {"Grocery": 1406, "Food Delivery": 1326, "Fashion": 976, "Electronics": 845, "Utilities": 685, "Travel": 681}
CITIES_CLEAN = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata"]
PAY_TARGET = {"UPI": 3221, "Card": 1450, "NetBanking": 731, "Wallet": 517}
def raw_variants(clean):
    return [clean, f" {clean}", f"{clean.upper()} ", clean.lower()]
def build_cleaned_frame():
    n = 5919
    merch = []
    for m, c in MERCHANT_TARGET.items():
        merch.extend([m]*c)
    rng.shuffle(merch)
    pays = []
    for p, c in PAY_TARGET.items():
        pays.extend([p]*c)
    rng.shuffle(pays)
    city_counts = [987,987,987,986,986,986]
    cities = []
    for city, c in zip(CITIES_CLEAN, city_counts):
        cities.extend([city]*c)
    rng.shuffle(cities)
    is_fraud = np.array([1]*366 + [0]*(n-366))
    rng.shuffle(is_fraud)
    fraud_idx = np.where(is_fraud==1)[0]
    nonfraud_idx = np.where(is_fraud==0)[0]
    rng.shuffle(fraud_idx); rng.shuffle(nonfraud_idx)
    cb = np.zeros(n, dtype=int)
    cb[fraud_idx[:304]] = 1
    cb[nonfraud_idx[:69]] = 1
    df = pd.DataFrame({"merchant_clean": merch, "city_clean": cities, "payment_mode": pays, "is_fraud": is_fraud, "chargeback_filed": cb})
    night_idx = rng.choice(n, size=365, replace=False)
    hours = rng.integers(6, 24, size=n)
    hours[night_idx] = rng.integers(0, 6, size=365)
    df["hour_of_day"] = hours
    acct = np.zeros(n, dtype=int)
    acct[nonfraud_idx] = rng.integers(25, 1200, size=len(nonfraud_idx))
    acct[fraud_idx] = rng.integers(5, 650, size=len(fraud_idx))
    df["account_age_days"] = acct
    new_dev = np.zeros(n, dtype=int)
    new_dev[fraud_idx] = (rng.random(len(fraud_idx)) < 0.45).astype(int)
    new_dev[nonfraud_idx] = (rng.random(len(nonfraud_idx)) < 0.10).astype(int)
    df["is_new_device"] = new_dev
    txn_cnt = np.zeros(n, dtype=int)
    txn_cnt[nonfraud_idx] = rng.integers(1, 7, size=len(nonfraud_idx))
    txn_cnt[fraud_idx] = rng.integers(2, 11, size=len(fraud_idx))
    df["txn_count_last_24h"] = txn_cnt
    dist = np.round(rng.gamma(shape=2.0, scale=7.0, size=n) + 1.0, 2)
    # fraud slightly farther from home
    dist[fraud_idx] = np.round(dist[fraud_idx] + rng.uniform(0.0, 6.0, size=len(fraud_idx)), 2)
    miss_dist_idx = rng.choice(n, size=191, replace=False)
    distf = dist.astype(float)
    distf[miss_dist_idx] = np.nan
    df["distance_from_home_km"] = distf
    return df, fraud_idx, nonfraud_idx, night_idx
def add_amounts(df, fraud_idx, nonfraud_idx):
    n = len(df)
    amounts = np.zeros(n)
    high_idx = np.concatenate([fraud_idx[:12], nonfraud_idx[:5]])
    low_idx = np.setdiff1d(np.arange(n), high_idx)
    low_amt = np.clip(rng.lognormal(mean=8.0, sigma=1.0, size=len(low_idx)), 150, 49000)
    amounts[low_idx] = np.round(low_amt, 2)
    amounts[high_idx] = np.round(rng.uniform(51000, 100000, size=17), 2)
    amounts[low_idx[0]] = 141.37
    amounts[high_idx[0]] = 105883.95
    amounts = np.clip(amounts, 141.37, 105883.95)
    amounts[low_idx[0]] = 141.37
    amounts[high_idx[0]] = 105883.95
    df["amount_inr"] = np.round(amounts, 2)
    ratios = np.clip(rng.lognormal(mean=0.0, sigma=0.5, size=n), 0.2, 5.0)
    avgs = df["amount_inr"].values / ratios
    avgs = np.round(avgs, 2)
    ratio_row = int(fraud_idx[5])
    avgs[ratio_row] = round(float(df["amount_inr"].iloc[ratio_row]) / 13.11, 2)
    df["avg_amount_30d_inr"] = avgs
    return df
def main():
    df, fraud_idx, nonfraud_idx, night_idx = build_cleaned_frame()
    df = add_amounts(df, fraud_idx, nonfraud_idx)
    n_clean = len(df)
    df["txn_id"] = [f"T{900000+i}" for i in range(n_clean)]
    df["customer_id"] = [f"C{100000+int(x)}" for x in rng.integers(0, 50000, size=n_clean)]
    # Neutralize leakage from bad rows: make them nonfraud, cb=0 so cleaned leak table stays exact
    df_bad_fix = df
    oor_amounts = [0.5, 0.0, 1500000.0, 2000000.0, -10.0]
    bad_rows = []
    for i in range(81):
        amt = float("nan") if i < 76 else oor_amounts[i-76]
        dist = float(np.round(float(rng.gamma(2.0, 7.0)) + 1.0, 2))
        if i < 2:
            dist = float("nan")
        bad_rows.append({"txn_id": f"T{900000+n_clean+i}", "customer_id": f"C{100000+int(rng.integers(0,50000))}", "merchant_clean": str(rng.choice(MERCHANTS_CLEAN)), "city_clean": str(rng.choice(CITIES_CLEAN)), "payment_mode": str(rng.choice(list(PAY_TARGET.keys()))), "amount_inr": amt, "hour_of_day": int(rng.integers(0, 24)), "txn_count_last_24h": int(rng.integers(1, 8)), "avg_amount_30d_inr": float(np.round(float(rng.uniform(500, 20000)), 2)), "distance_from_home_km": dist, "is_new_device": int(rng.integers(0, 2)), "account_age_days": int(rng.integers(1, 1000)), "chargeback_filed": int(rng.integers(0, 2)), "is_fraud": int(rng.integers(0, 2))})
    base = pd.concat([df, pd.DataFrame(bad_rows)], ignore_index=True)
    base = base.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    base["merchant_category"] = [str(rng.choice(raw_variants(x))) for x in base["merchant_clean"]]
    base["city"] = [str(rng.choice(raw_variants(x))) for x in base["city_clean"]]
    all_m = []
    for m in MERCHANTS_CLEAN:
        all_m.extend(raw_variants(m))
    all_c = []
    for c in CITIES_CLEAN:
        all_c.extend(raw_variants(c))
    for i, v in enumerate(all_m[:24]):
        pass
    for i, v in enumerate(all_c[:24]):
        pass
    # Assign variants within matching clean groups so cleaned counts stay exact
    for m in MERCHANTS_CLEAN:
        idx = base.index[base["merchant_clean"] == m][:4].tolist()
        for j, k in enumerate(idx):
            base.loc[k, "merchant_category"] = raw_variants(m)[j]
    for c in CITIES_CLEAN:
        idx = base.index[base["city_clean"] == c][:4].tolist()
        for j, k in enumerate(idx):
            base.loc[k, "city"] = raw_variants(c)[j]
    dups = base.sample(n=90, random_state=SEED+1)
    raw = pd.concat([base, dups], ignore_index=True)
    raw = raw.sample(frac=1.0, random_state=SEED+2).reset_index(drop=True)
    cols = ["txn_id","customer_id","merchant_category","city","payment_mode","amount_inr","hour_of_day","txn_count_last_24h","avg_amount_30d_inr","distance_from_home_km","is_new_device","account_age_days","chargeback_filed","is_fraud"]
    raw[cols].to_csv("transactions_raw.csv", index=False)
    print(f"rows written: {len(raw)}")

def write_helpers():
    import json as _json
    policies = [{"policy_id":"P1","text":"Card stolen or lost policy: if a card payment is reported stolen or lost, block the card, notify the customer, and hold the transaction for card risk review."},{"policy_id":"P2","text":"UPI phishing and collect request fraud policy: unexpected UPI collect requests and phishing links require upi risk team verification and temporary hold."},{"policy_id":"P3","text":"High value new device purchase policy: purchases above threshold from a new device need merchant ops verification and account age check."},{"policy_id":"P4","text":"Account takeover night transaction policy: night time logins from new locations indicate possible account takeover; escalate to customer care and hold."},{"policy_id":"P5","text":"Merchant refund and chargeback abuse policy: repeated refunds and chargeback filings need merchant ops audit and transaction hold."},{"policy_id":"P6","text":"Travel ticket cancellation fraud policy: bulk travel bookings with quick cancellations require merchant ops review and block if fraudulent."}]
    open("fraud_policies.json","w").write(_json.dumps(policies, indent=2))
    queries = [{"query_id":"Q1","query":"my card was stolen and used without permission","expected_policy":"P1"},{"query_id":"Q2","query":"received fake upi collect request phishing link","expected_policy":"P2"},{"query_id":"Q3","query":"high value purchase made from new device verification","expected_policy":"P3"},{"query_id":"Q4","query":"night transaction from new location account takeover","expected_policy":"P4"},{"query_id":"Q5","query":"customer filed chargeback and refund abuse audit","expected_policy":"P5"},{"query_id":"Q6","query":"bulk travel ticket booking and cancellation fraud","expected_policy":"P6"},{"query_id":"Q7","query":"lost card blocked after stolen transaction","expected_policy":"P1"},{"query_id":"Q8","query":"upi phishing collect request verification needed","expected_policy":"P2"},{"query_id":"Q9","query":"what is the weather today in Mumbai","expected_policy":"no_policy_match"},{"query_id":"Q10","query":"xyzzy qwerty asdfgh zxcvbn","expected_policy":"no_policy_match"}]
    open("retrieval_queries.json","w").write(_json.dumps(queries, indent=2))
    scenarios = [{"scenario_id":"S1","txn_id":"T900010","fraud_probability":0.82,"amount_inr":12000.0,"question":"Card stolen, block immediately?"},{"scenario_id":"S2","txn_id":"T900011","fraud_probability":0.75,"amount_inr":8500.0,"question":"UPI phishing collect request?"},{"scenario_id":"S3","txn_id":"T900012","fraud_probability":0.68,"amount_inr":62000.0,"question":"High value new device purchase?"},{"scenario_id":"S4","txn_id":"T900013","fraud_probability":0.71,"amount_inr":15000.0,"question":"Night account takeover?"},{"scenario_id":"S5","txn_id":"T900014","fraud_probability":0.66,"amount_inr":9500.0,"question":"Chargeback abuse case?"},{"scenario_id":"S6","txn_id":"T900015","fraud_probability":0.78,"amount_inr":22000.0,"question":"Travel cancellation fraud?"},{"scenario_id":"S7","txn_id":"T900016","fraud_probability":0.59,"amount_inr":18000.0,"question":"Card stolen again?"},{"scenario_id":"S8","txn_id":"T900017","fraud_probability":0.81,"amount_inr":11000.0,"question":"UPI phishing again?"},{"scenario_id":"S9","txn_id":"T900018","fraud_probability":0.62,"amount_inr":54000.0,"question":"High value device check?"},{"scenario_id":"S10","txn_id":"T900019","fraud_probability":0.73,"amount_inr":13500.0,"question":"Night takeover again?"}]
    open("case_scenarios.json","w").write(_json.dumps(scenarios, indent=2))
    print("data files written.")
if __name__ == "__main__":
    main()
    write_helpers()

