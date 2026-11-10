"""IBNR reserving backtest on the CAS Schedule P loss-reserve database.

Question: from the paid triangles known at the end of 1994, how well does each
method predict the payments actually made in 1995-1997 (the next 3 calendar
diagonals)?

Methods
- Chain-Ladder (company's own development factors)
- Chain-Ladder on the industry pattern (factors pooled over all companies in the line)
- Cape Cod (Bornhuetter-Ferguson with an expected loss ratio estimated from premium)
- LightGBM, one-year-ahead model applied recursively, trained only on pseudo cut-offs
  inside the pre-1995 history (design chosen on a 1992 validation cut-off, see ml_design.py)
- ODP bootstrap + process variance around Chain-Ladder, giving a 90% interval per company

Run:  python src/pipeline.py
"""
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import chainladder as cl

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
REP.mkdir(parents=True, exist_ok=True)

N = 10            # accident years 1988-1997, development lags 1-10
K = 3             # calendar diagonals hidden (1995, 1996, 1997)
D = N - 1 - K     # last known calendar diagonal index at the cut-off (end of 1994)
LOBS = ["ppauto", "comauto", "wkcomp", "othliab"]
N_SIMS = 500
SEED = 0


# ----------------------------------------------------------------- data
def load_books():
    """Return a list of books with complete, usable triangles, and a log of exclusions."""
    tri = cl.load_sample("clrd")
    books, dropped = [], []
    for lob in LOBS:
        sub = tri[tri["LOB"] == lob]
        for g in sub.index["GRNAME"]:
            t = sub[sub["GRNAME"] == g]
            cum = t["CumPaidLoss"].values[0, 0]
            prem = t["EarnedPremNet"].latest_diagonal.values[0, 0][:, -1]
            upper = np.add.outer(np.arange(N), np.arange(N)) <= N - 1
            reason = None
            if np.isnan(cum[upper]).any():
                reason = "missing cells"
            elif np.isnan(prem).any() or (prem <= 0).any():
                reason = "missing or zero premium"
            elif (cum[:, 0] <= 0).any():
                reason = "zero first-year payments"
            if reason:
                dropped.append((lob, g, reason))
                continue
            books.append(dict(lob=lob, company=g, cum=np.nan_to_num(cum), prem=prem, tri=t["CumPaidLoss"]))
    return books, pd.DataFrame(dropped, columns=["lob", "company", "reason"])


def known_mask(d):
    return np.add.outer(np.arange(N), np.arange(N)) <= d


# ----------------------------------------------------------------- classical methods
def factors(cums, mask):
    """Volume-weighted age-to-age factors from one or several triangles (list of arrays)."""
    f = np.ones(N - 1)
    for j in range(N - 1):
        rows = mask[:, j] & mask[:, j + 1]
        num = sum(c[rows, j + 1].sum() for c in cums)
        den = sum(c[rows, j].sum() for c in cums)
        if rows.any() and den > 0:
            f[j] = num / den
    return f


def cl_future(cum, d, f, h=K):
    """Predicted paid over the next h diagonals per origin, from cut-off diagonal d."""
    out = {}
    for i in range(d + 1):
        l = d - i
        if l + h > N - 1:
            continue
        out[i] = cum[i, l] * (np.prod(f[l:l + h]) - 1)
    return out


def capecod_future(cum, prem, d, f, h=K):
    cdf = np.array([np.prod(f[j:]) for j in range(N)])          # to the end of the triangle (no tail)
    lags = {i: d - i for i in range(d + 1)}
    used = sum(prem[i] / cdf[l] for i, l in lags.items())
    elr = sum(cum[i, l] for i, l in lags.items()) / used
    pct = 1 / cdf
    return {i: elr * prem[i] * (pct[l + h] - pct[l]) for i, l in lags.items() if l + h <= N - 1}


# ----------------------------------------------------------------- ML (see ml_design.py)
FEATURES = ["lob", "lag", "horizon", "paid_ratio", "last_inc_ratio", "log_premium", "cl_own", "cl_pooled"]


# ----------------------------------------------------------------- bootstrap
def bootstrap_interval(book, rng):
    """ODP bootstrap of the Chain-Ladder on the end-1994 triangle + Gamma process variance.
    Returns the 5% and 95% quantiles of total paid over the next K diagonals."""
    t = book["tri"]
    tr = t[t.valuation < "1995"]
    b = cl.BootstrapODPSample(n_sims=N_SIMS, random_state=int(rng.integers(1e9))).fit(tr)
    sims = b.resampled_triangles_.values[:, 0]            # (sims, origins, lags) cumulative
    phi = float(np.asarray(b.scale_).ravel()[0])
    n_o = sims.shape[1]
    totals = np.zeros(N_SIMS)
    for s in range(N_SIMS):
        c = np.zeros((N, N))
        c[:n_o, :sims.shape[2]] = np.nan_to_num(sims[s])
        f = factors([c], known_mask(D))
        for i in range(D + 1):
            l = D - i
            if l + K > N - 1:
                continue
            base = book["cum"][i, l]                       # project from the real latest diagonal
            prev = base
            for j in range(l + 1, l + K + 1):
                m = max(prev * (f[j - 1] - 1), 0.0)        # expected incremental
                inc = rng.gamma(m / phi, phi) if (m > 0 and phi > 0) else m
                totals[s] += inc
                prev = prev + m
    return np.quantile(totals, [0.05, 0.95])


# ----------------------------------------------------------------- backtest
def run():
    books, dropped = load_books()
    dropped.to_csv(REP / "excluded_books.csv", index=False)
    print(f"usable books: {len(books)}, excluded: {len(dropped)}")
    print(dropped.groupby(["lob", "reason"]).size().to_string())

    import ml_design as MD                          # design chosen on the 1992 validation cut-off
    model = MD.train(books, list(range(2, D)), 1, D)   # one-year-ahead model, pseudo cut-offs 1990-1993
    pooled_D = {lob: factors([b["cum"] for b in books if b["lob"] == lob], known_mask(D)) for lob in LOBS}

    rng = np.random.default_rng(SEED)
    res = []
    for k, b in enumerate(books):
        cum, prem = b["cum"], b["prem"]
        actual = {i: cum[i, D - i + K] - cum[i, D - i] for i in range(D + 1) if D - i + K <= N - 1}
        own_f = factors([cum], known_mask(D))
        preds = {
            "Chain-Ladder": cl_future(cum, D, own_f),
            "Chain-Ladder (industry pattern)": cl_future(cum, D, pooled_D[b["lob"]]),
            "Cape Cod": capecod_future(cum, prem, D, own_f),
        }
        preds["LightGBM"] = MD.predict(model, b, D, K, pooled_D[b["lob"]], recursive=True)
        a = sum(actual.values())
        try:
            lo, hi = bootstrap_interval(b, rng)
        except Exception:
            lo, hi = np.nan, np.nan                        # bootstrap undefined (e.g. no residuals)
        for name, p in preds.items():
            res.append(dict(lob=b["lob"], company=b["company"], method=name, actual=a,
                            predicted=sum(p[i] for i in actual), premium=prem.sum(),
                            p05=lo, p95=hi))
        if k % 50 == 0:
            print(f"  {k}/{len(books)}")

    df = pd.DataFrame(res)
    df["error"] = df.predicted - df.actual
    df["ape"] = df.error.abs() / df.actual.where(df.actual > 0)
    df.to_csv(REP / "backtest_by_company.csv", index=False)

    def summarise(d):
        return pd.Series({
            "companies": d.company.nunique(),
            "actual": d.actual.sum(),
            "predicted": d.predicted.sum(),
            "aggregate_error_pct": 100 * (d.predicted.sum() / d.actual.sum() - 1),
            "median_ape_pct": 100 * d.ape.median(),
            "weighted_ape_pct": 100 * d.error.abs().sum() / d.actual.sum(),
        })
    by_lob = df.groupby(["lob", "method"]).apply(summarise, include_groups=False).reset_index()
    overall = df.groupby("method").apply(summarise, include_groups=False).reset_index()
    overall.insert(0, "lob", "all")
    summ = pd.concat([by_lob, overall], ignore_index=True)
    summ.to_csv(REP / "backtest_summary.csv", index=False)

    bc = df[(df.method == "Chain-Ladder") & df.p05.notna()].copy()
    bc["inside"] = bc.actual.between(bc.p05, bc.p95)
    print(f"bootstrap intervals: {len(bc)} books ({df[df.method=='Chain-Ladder'].p05.isna().sum()} undefined)")
    cov = bc.groupby("lob").inside.mean().mul(100).round(1)
    cov["all"] = round(100 * bc.inside.mean(), 1)
    cov.to_csv(REP / "bootstrap_coverage.csv", header=["pct_inside_90pct_interval"])

    pd.set_option("display.width", 200)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print("\n90% bootstrap interval coverage (target 90%):")
    print(cov.to_string())


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    run()
