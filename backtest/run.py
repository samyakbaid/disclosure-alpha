from pathlib import Path
import pandas as pd, numpy as np, yfinance as yf
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]

def load_prices(tickers, start="2023-01-01", end="2026-09-01"):
    px = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    return px.pct_change().dropna(how="all")

def main():
    sig = pd.read_parquet(ROOT/"data"/"processed"/"signals.parquet")
    manifest = pd.read_parquet(ROOT/"data"/"processed"/"manifest.parquet")

    # CRITICAL: use FILING date, not fiscal period end.
    # A signal is only usable after the filing is public.
    fdates = manifest[["ticker","reportDate","filingDate"]].rename(
        columns={"reportDate":"fiscal_period"})
    sig = sig.merge(fdates, on=["ticker","fiscal_period"], how="left")
    sig["filingDate"] = pd.to_datetime(sig["filingDate"])
    sig = sig.dropna(subset=["filingDate"])

    rets = load_prices(sorted(sig.ticker.unique()))

    rows = []
    for _, r in sig.iterrows():
        start = r["filingDate"] + pd.Timedelta(days=1)   # no look-ahead
        window = rets.loc[start:start + pd.Timedelta(days=21), r["ticker"]]
        if len(window) < 5:
            continue
        rows.append({"ticker": r["ticker"], "date": start,
                     "severity": r["severity"], "n_new": r["n_new"],
                     "tone_neg": 1 if r["tone_shift"]=="more_negative" else 0,
                     "fwd_ret": (1+window).prod() - 1})

    df = pd.DataFrame(rows)
    print(f"{len(df)} observations\n")

    # market control: equal-weighted mean return that period
    df["mkt"] = df.groupby("date")["fwd_ret"].transform("mean")

    X = sm.add_constant(df[["severity","n_new","tone_neg","mkt"]])
    model = sm.OLS(df["fwd_ret"], X).fit()
    print(model.summary())

    print("\n--- interpretation ---")
    for var in ["severity","n_new","tone_neg"]:
        p = model.pvalues[var]
        verdict = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"{var}: coef={model.params[var]:+.4f}, p={p:.3f} → {verdict}")

if __name__ == "__main__":
    main()