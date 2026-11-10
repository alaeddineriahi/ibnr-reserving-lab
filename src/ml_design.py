"""Choose the ML design on a validation backtest that uses only pre-1995 data.

Validation: cut-off at the end of 1992 (diagonal 4) and predict 1993-1994 (2 diagonals).
Both designs are trained only on pseudo cut-offs before 1992.
- direct:    one model predicts the paid amount over h = 1..H directly
- recursive: one model predicts the next year only, then is applied step by step
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P

FEAT = P.FEATURES


def rows_at(book, d, pooled_f, hmax, last_known):
    """Training rows for cut-off d, targets restricted to diagonals <= last_known."""
    cum, prem = book["cum"], book["prem"]
    own_f = P.factors([cum], P.known_mask(d))
    out = []
    for i in range(d + 1):
        l = d - i
        for h in range(1, hmax + 1):
            j = l + h
            if j > P.N - 1 or i + j > last_known:
                continue
            out.append(dict(lob=book["lob"], lag=l, horizon=h, paid_ratio=cum[i, l] / prem[i],
                            last_inc_ratio=(cum[i, l] - (cum[i, l - 1] if l > 0 else 0)) / prem[i],
                            log_premium=np.log(prem[i]),
                            cl_own=cum[i, l] * (np.prod(own_f[l:j]) - 1) / prem[i],
                            cl_pooled=cum[i, l] * (np.prod(pooled_f[l:j]) - 1) / prem[i],
                            prem=prem[i], target=(cum[i, j] - cum[i, l]) / prem[i]))
    return out


def train(books, cutoffs, hmax, last_known):
    rows = []
    for d in cutoffs:
        pooled = {lob: P.factors([b["cum"] for b in books if b["lob"] == lob], P.known_mask(d)) for lob in P.LOBS}
        for b in books:
            rows += rows_at(b, d, pooled[b["lob"]], hmax, last_known)
    tr = pd.DataFrame(rows)
    tr["lob"] = tr["lob"].astype(pd.CategoricalDtype(P.LOBS))
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    m.fit(tr[FEAT], tr["target"], sample_weight=tr["prem"])
    return m


def feat_row(lob, cum_i, prem_i, l, h, own_f, pooled_f):
    return dict(lob=lob, lag=l, horizon=h, paid_ratio=cum_i[l] / prem_i,
                last_inc_ratio=(cum_i[l] - (cum_i[l - 1] if l > 0 else 0)) / prem_i,
                log_premium=np.log(prem_i),
                cl_own=cum_i[l] * (np.prod(own_f[l:l + h]) - 1) / prem_i,
                cl_pooled=cum_i[l] * (np.prod(pooled_f[l:l + h]) - 1) / prem_i)


def predict(model, book, d, H, pooled_f, recursive):
    cum, prem = book["cum"].copy(), book["prem"]
    own_f = P.factors([book["cum"]], P.known_mask(d))
    out = {}
    for i in range(d + 1):
        l = d - i
        if l + H > P.N - 1:
            continue
        if not recursive:
            X = pd.DataFrame([feat_row(book["lob"], cum[i], prem[i], l, H, own_f, pooled_f)])
            X["lob"] = X["lob"].astype(pd.CategoricalDtype(P.LOBS))
            out[i] = model.predict(X[FEAT])[0] * prem[i]
        else:
            c = cum[i].copy(); total = 0.0
            for step in range(H):
                ll = l + step
                X = pd.DataFrame([feat_row(book["lob"], c, prem[i], ll, 1, own_f, pooled_f)])
                X["lob"] = X["lob"].astype(pd.CategoricalDtype(P.LOBS))
                inc = max(model.predict(X[FEAT])[0], 0.0) * prem[i]
                c[ll + 1] = c[ll] + inc; total += inc
            out[i] = total
    return out


def evaluate(books, d, H, preds):
    rows = []
    for b in books:
        cum = b["cum"]
        act = {i: cum[i, d - i + H] - cum[i, d - i] for i in range(d + 1) if d - i + H <= P.N - 1}
        for name, p in preds.items():
            rows.append(dict(lob=b["lob"], company=b["company"], method=name,
                             actual=sum(act.values()), predicted=sum(p[b["company"], b["lob"]][i] for i in act)))
    df = pd.DataFrame(rows)
    df["ape"] = (df.predicted - df.actual).abs() / df.actual.where(df.actual > 0)
    return (df.groupby("method").apply(lambda g: pd.Series({
        "aggregate_error_pct": 100 * (g.predicted.sum() / g.actual.sum() - 1),
        "median_ape_pct": 100 * g.ape.median(),
        "weighted_ape_pct": 100 * (g.predicted - g.actual).abs().sum() / g.actual.sum()}), include_groups=False))


def run_setting(books, d, H, label):
    last_known = d                                   # targets must be known at the cut-off
    cutoffs = list(range(2, d))
    direct = train(books, cutoffs, H, last_known)
    rec = train(books, cutoffs, 1, last_known)
    pooled = {lob: P.factors([b["cum"] for b in books if b["lob"] == lob], P.known_mask(d)) for lob in P.LOBS}
    preds = {"LightGBM direct": {}, "LightGBM recursive": {}, "Chain-Ladder": {}, "Cape Cod": {}}
    for b in books:
        key = (b["company"], b["lob"]); pf = pooled[b["lob"]]
        own_f = P.factors([b["cum"]], P.known_mask(d))
        preds["LightGBM direct"][key] = predict(direct, b, d, H, pf, recursive=False)
        preds["LightGBM recursive"][key] = predict(rec, b, d, H, pf, recursive=True)
        preds["Chain-Ladder"][key] = P.cl_future(b["cum"], d, own_f, h=H)
        preds["Cape Cod"][key] = P.capecod_future(b["cum"], b["prem"], d, own_f, h=H)
    s = evaluate(books, d, H, preds)
    print(f"\n== {label} ==\n" + s.round(1).to_string())
    return s, preds


if __name__ == "__main__":
    books, _ = P.load_books()
    val, _ = run_setting(books, d=4, H=2, label="VALIDATION: cut-off end 1992, predict 1993-1994")
    val.to_csv(P.REP / "ml_design_validation.csv")
    test, preds = run_setting(books, d=P.D, H=P.K, label="TEST: cut-off end 1994, predict 1995-1997")
    test.to_csv(P.REP / "ml_design_test.csv")
