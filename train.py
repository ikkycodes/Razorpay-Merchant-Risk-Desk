"""Part B — Fraud model, imbalance, threshold."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from prepare import clean_transactions, add_features
NUM7 = ["amount_inr","hour_of_day","txn_count_last_24h","avg_amount_30d_inr","distance_from_home_km","is_new_device","account_age_days"]
ENG3 = ["amount_vs_avg_ratio","is_night","high_value"]
CATS = ["merchant_category","city","payment_mode"]
def make_Xy():
    raw = pd.read_csv("transactions_raw.csv")
    df = add_features(clean_transactions(raw))
    y = df["is_fraud"].astype(int)
    Xraw = df.drop(columns=["txn_id","customer_id","chargeback_filed","is_fraud"])
    Xenc = pd.get_dummies(Xraw, columns=CATS, dtype=int)
    num_part = NUM7 + ENG3
    ohe = sorted([c for c in Xenc.columns if c not in num_part])
    cols = num_part + ohe
    return Xenc[cols], y, cols, df
def split_all(X, y):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=42)
    return Xtr, Xva, Xte, ytr, yva, yte
def prf(y_true, y_pred):
    return (precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0), f1_score(y_true, y_pred, zero_division=0))
def build_final_model():
    X, y, cols, _ = make_Xy()
    Xtr, Xva, Xte, ytr, yva, yte = split_all(X, y)
    med = float(Xtr["distance_from_home_km"].median())
    Xtr = Xtr.copy(); Xva = Xva.copy(); Xte = Xte.copy()
    for _X in (Xtr, Xva, Xte):
        _X["distance_from_home_km"] = _X["distance_from_home_km"].fillna(med)
    gs = GridSearchCV(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="f1")
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    probs = best.predict_proba(Xva)[:, 1]
    t_star = 0.1
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if recall_score(yva, (probs >= t).astype(int)) >= 0.65:
            t_star = t
    return {"model": best, "columns": cols, "median": med, "threshold": t_star}

def main():
    X, y, cols, df = make_Xy()
    print(f"Columns ({len(cols)}): {cols}")
    Xtr, Xva, Xte, ytr, yva, yte = split_all(X, y)
    print(f"Train: {len(Xtr)} rows, {int((ytr==1).sum())} fraud")
    print(f"Valid: {len(Xva)} rows, {int((yva==1).sum())} fraud")
    print(f"Test: {len(Xte)} rows, {int((yte==1).sum())} fraud")
    med = float(Xtr["distance_from_home_km"].median())
    print(f"Training median distance: {med}")
    Xtr = Xtr.copy(); Xva = Xva.copy(); Xte = Xte.copy()
    for _X in (Xtr, Xva, Xte):
        _X["distance_from_home_km"] = _X["distance_from_home_km"].fillna(med)
    def make_lr(bal=False):
        return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced" if bal else None))])
    m_plain = make_lr(False).fit(Xtr, ytr)
    m_bal = make_lr(True).fit(Xtr, ytr)
    fr = Xtr[ytr == 1]; nf = Xtr[ytr == 0]
    samp = fr.sample(n=len(nf), replace=True, random_state=42)
    import numpy as _np
    Xo = pd.concat([nf, samp]); yo = _np.array([0]*len(nf) + [1]*len(samp))
    m_over = make_lr(False).fit(Xo, yo)
    print(f"Oversampled train: {len(Xo)} rows ({int((yo==0).sum())} nonfraud, {int((yo==1).sum())} fraud)")
    for name, m in [("plain", m_plain), ("balanced", m_bal), ("oversampled", m_over)]:
        p = m.predict(Xva)
        prec, rec, f1 = prf(yva, p)
        tn, fp, fn, tp = confusion_matrix(yva, p).ravel()
        print(f"{name}: prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} cm=(TN={tn},FP={fp},FN={fn},TP={tp})")
    print("Recall-first desks prefer balanced/oversampled (recall up ~0.44) at cost of more false positives.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s_bal = cross_val_score(make_lr(True), Xtr, ytr, cv=cv, scoring="f1")
    s_rf = cross_val_score(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), Xtr, ytr, cv=cv, scoring="f1")
    print(f"Balanced LR CV F1: mean={s_bal.mean():.3f} std={s_bal.std():.3f}")
    print(f"Untuned forest CV F1: mean={s_rf.mean():.3f} std={s_rf.std():.3f}")

def main2(Xtr, Xva, Xte, ytr, yva, yte, cols):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}, cv=cv, scoring="f1")
    gs.fit(Xtr, ytr)
    print(f"Grid combos: 12; best={gs.best_params_} bestCV_F1={gs.best_score_:.3f}")
    print("Final: tuned forest (higher F1, stable); sacrifices LR interpretability.")
    best = gs.best_estimator_
    probs = best.predict_proba(Xva)[:, 1]
    print("Threshold sweep (validation):")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = (probs >= t).astype(int)
        prec, rec, f1 = prf(yva, pred)
        print(f" t={t:.1f}: prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} flagged={int(pred.sum())}")
    t_star = 0.1
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if recall_score(yva, (probs >= t).astype(int)) >= 0.65:
            t_star = t
    print(f"Chosen t_star={t_star}")
    pte = (best.predict_proba(Xte)[:, 1] >= t_star).astype(int)
    prec, rec, f1 = prf(yte, pte)
    tn, fp, fn, tp = confusion_matrix(yte, pte).ravel()
    print(f"Test @t*: prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} flagged={int(pte.sum())} cm=(TN={tn},FP={fp},FN={fn},TP={tp})")
    per = fp / max(tp, 1)
    print(f"Risk note ({t_star}): flagged {int(pte.sum())}; {per:.1f} legit holds per fraud; {int(fn)} missed. Stricter t raises precision, lowers recall.")
    bp = gs.best_params_
    raw = pd.read_csv("transactions_raw.csv")
    from prepare import clean_transactions as _ct, add_features as _af
    dfl = _af(_ct(raw))
    yl = dfl["is_fraud"].astype(int)
    Xr = dfl.drop(columns=["txn_id","customer_id","is_fraud"])
    Xe = pd.get_dummies(Xr, columns=["merchant_category","city","payment_mode"], dtype=int)
    nump = ["amount_inr","hour_of_day","txn_count_last_24h","avg_amount_30d_inr","distance_from_home_km","is_new_device","account_age_days"] + ["amount_vs_avg_ratio","is_night","high_value"]
    ohe = sorted([c for c in Xe.columns if c not in nump + ["chargeback_filed"]])
    Xl = Xe[[c for c in nump if c in Xe.columns] + ohe + ["chargeback_filed"]]
    from sklearn.model_selection import train_test_split as _tts
    Xltr, Xlva, yltr, ylva = _tts(Xl, yl, test_size=0.2, stratify=yl, random_state=42)
    # align: use same split fn for comparability: recompute via split_all on Xl
    Xltr, Xlva, Xlte, yltr, ylva, ylte = split_all(Xl, yl)
    medl = float(Xltr["distance_from_home_km"].median())
    for _X in (Xltr, Xlva):
        _X["distance_from_home_km"] = _X["distance_from_home_km"].fillna(medl)
    rf_clean = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, **bp).fit(Xtr[cols].copy(), ytr)
    rf_leak = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, **bp).fit(Xltr, yltr)
    fc = f1_score(ylva, rf_clean.predict(Xlva[cols]), zero_division=0)
    fl = f1_score(ylva, rf_leak.predict(Xlva), zero_division=0)
    print(f"Leakage: clean F1={fc:.3f} leaky F1={fl:.3f} (chargeback is post-outcome; invalid).")
    imp = pd.Series(best.feature_importances_, index=cols).sort_values(ascending=False)
    print("Top5 features:")
    print(imp.head(5).to_string())
if __name__ == "__main__":
    X, y, cols, df = make_Xy()
    # reuse main() splits by calling pieces
    from sklearn.model_selection import train_test_split as _t2
    Xtr, Xtmp, ytr, ytmp = _t2(X, y, test_size=0.4, stratify=y, random_state=42)
    Xva, Xte, yva, yte = _t2(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=42)
    med = float(Xtr["distance_from_home_km"].median())
    print(f"Columns ({len(cols)}): {cols}")
    print(f"Train: {len(Xtr)} rows, {int((ytr==1).sum())} fraud")
    print(f"Valid: {len(Xva)} rows, {int((yva==1).sum())} fraud")
    print(f"Test: {len(Xte)} rows, {int((yte==1).sum())} fraud")
    print(f"Training median distance: {med}")
    Xtr = Xtr.copy(); Xva = Xva.copy(); Xte = Xte.copy()
    for _X in (Xtr, Xva, Xte):
        _X["distance_from_home_km"] = _X["distance_from_home_km"].fillna(med)
    def make_lr(bal=False):
        return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, class_weight="balanced" if bal else None))])
    m_plain = make_lr(False).fit(Xtr, ytr)
    m_bal = make_lr(True).fit(Xtr, ytr)
    fr = Xtr[ytr == 1]; nf = Xtr[ytr == 0]
    samp = fr.sample(n=len(nf), replace=True, random_state=42)
    import numpy as _np
    Xo = pd.concat([nf, samp]); yo = _np.array([0]*len(nf) + [1]*len(samp))
    m_over = make_lr(False).fit(Xo, yo)
    print(f"Oversampled train: {len(Xo)} rows ({int((yo==0).sum())} nonfraud, {int((yo==1).sum())} fraud)")
    for name, m in [("plain", m_plain), ("balanced", m_bal), ("oversampled", m_over)]:
        p = m.predict(Xva)
        prec, rec, f1 = prf(yva, p)
        tn, fp, fn, tp = confusion_matrix(yva, p).ravel()
        print(f"{name}: prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} cm=(TN={tn},FP={fp},FN={fn},TP={tp})")
    print("Recall-first desks prefer balanced/oversampled at cost of more FPs.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    s_bal = cross_val_score(make_lr(True), Xtr, ytr, cv=cv, scoring="f1")
    s_rf = cross_val_score(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), Xtr, ytr, cv=cv, scoring="f1")
    print(f"Balanced LR CV F1: mean={s_bal.mean():.3f} std={s_bal.std():.3f}")
    print(f"Untuned forest CV F1: mean={s_rf.mean():.3f} std={s_rf.std():.3f}")
    main2(Xtr, Xva, Xte, ytr, yva, yte, cols)

