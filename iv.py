import argparse
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(S, K, T, r, q, sigma, call=True):
    """Black-Scholes price for European options."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if call else (K - S))
    vol_sqrt = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(price, S, K, T, r=0.0, q=0.0, call=True):
    """Return implied volatility via Brent root finder; NaN when not solvable."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    def f(sig):
        return bs_price(S, K, T, r, q, sig, call) - price

    try:
        # Wide bounds that work for most equity options; adjust if needed.
        return brentq(f, 1e-4, 5.0, maxiter=100, xtol=1e-6)
    except ValueError:
        return np.nan


REQUIRED_COLUMNS = {
    "quote_date",
    "expiration",
    "underlying_bid_1545",
    "underlying_ask_1545",
    "bid_1545",
    "ask_1545",
    "strike",
    "option_type",
}


RATE_COLS = ["1M", "2M", "3M", "4M", "6M", "12M"]
RATE_TERMS_YEARS = np.array([1, 2, 3, 4, 6, 12], dtype=float) / 12.0
TRADING_DAYS = 252.0


def interpolate_rates(rates: np.ndarray, terms: np.ndarray, T: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(terms, T, side="right")
    low_idx = np.clip(idx - 1, 0, len(terms) - 1)
    high_idx = np.clip(idx, 0, len(terms) - 1)
    low_rates = np.take_along_axis(rates, low_idx[:, None], axis=1).reshape(-1)
    high_rates = np.take_along_axis(rates, high_idx[:, None], axis=1).reshape(-1)
    low_terms = terms[low_idx]
    high_terms = terms[high_idx]
    span = high_terms - low_terms
    weight = np.zeros_like(T, dtype=float)
    np.divide(T - low_terms, span, out=weight, where=span != 0)
    return low_rates + (high_rates - low_rates) * weight


def compute_chunk_iv(chunk: pd.DataFrame, rates_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Compute IV for a single chunk."""
    if "spot" in chunk.columns:
        chunk["spot"] = pd.to_numeric(chunk["spot"], errors="coerce")
    else:
        chunk["spot"] = (chunk["underlying_bid_1545"] + chunk["underlying_ask_1545"]) / 2

    if "mid" in chunk.columns:
        chunk["mid"] = pd.to_numeric(chunk["mid"], errors="coerce")
    else:
        chunk["mid"] = (chunk["bid_1545"] + chunk["ask_1545"]) / 2

    chunk["quote_date"] = pd.to_datetime(
        chunk["quote_date"],
        dayfirst=True,
        errors="coerce",
        format="mixed",
    )
    chunk["expiration"] = pd.to_datetime(
        chunk["expiration"],
        dayfirst=True,
        errors="coerce",
        format="mixed",
    )
    chunk = chunk.dropna(subset=["quote_date", "expiration"])
    chunk["T"] = (chunk["expiration"] - chunk["quote_date"]).dt.days / 365.0

    has_rq = {"r_annual", "q_annual"}.issubset(chunk.columns)
    if has_rq:
        chunk["r_annual"] = pd.to_numeric(chunk["r_annual"], errors="coerce")
        chunk["q_annual"] = pd.to_numeric(chunk["q_annual"], errors="coerce")
    else:
        if rates_df is None:
            raise ValueError("Lipsesc r_annual/q_annual in input si nu exista fisierul de rates.")
        chunk = chunk.merge(
            rates_df,
            left_on="quote_date",
            right_on="Calendar Date",
            how="inner",
        )
        if chunk.empty:
            return chunk
        rate_matrix = chunk[RATE_COLS].to_numpy(dtype=float)
        T = chunk["T"].to_numpy(dtype=float)
        r_daily = interpolate_rates(rate_matrix, RATE_TERMS_YEARS, T)
        chunk["r_annual"] = (1.0 + r_daily) ** TRADING_DAYS - 1.0
        q_daily = chunk["Dividend Yield (Value-Weighted)"].to_numpy(dtype=float)
        chunk["q_annual"] = (1.0 + q_daily) ** TRADING_DAYS - 1.0

    chunk["strike"] = pd.to_numeric(chunk["strike"], errors="coerce")
    chunk["option_type"] = chunk["option_type"].astype(str).str.upper().str.strip()
    chunk = chunk.dropna(subset=["mid", "spot", "strike", "T", "r_annual", "q_annual"])
    if chunk.empty:
        return chunk

    calls = chunk["option_type"].eq("C").to_numpy()
    chunk["implied_vol"] = [
        implied_vol(p, s, k, t, r, q, call)
        for p, s, k, t, r, q, call in zip(
            chunk["mid"].to_numpy(dtype=float),
            chunk["spot"].to_numpy(dtype=float),
            chunk["strike"].to_numpy(dtype=float),
            chunk["T"].to_numpy(dtype=float),
            chunk["r_annual"].to_numpy(dtype=float),
            chunk["q_annual"].to_numpy(dtype=float),
            calls,
        )
    ]
    return chunk


def process_file(
    csv_path: pathlib.Path,
    out_path: pathlib.Path,
    rates_path: pathlib.Path,
    chunksize: int,
) -> int:
    def count_rows(path: pathlib.Path) -> int:
        total = 0
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                total += chunk.count(b"\n")
        return max(total - 1, 0)

    # Validate columns using only the header (cheap).
    header = pd.read_csv(csv_path, nrows=0)
    missing = REQUIRED_COLUMNS - set(header.columns)
    if missing:
        raise ValueError(f"Lipsesc coloanele obligatorii: {', '.join(sorted(missing))}")

    rates_df: Optional[pd.DataFrame] = None
    has_rq_in_input = {"r_annual", "q_annual"}.issubset(set(header.columns))
    if not has_rq_in_input:
        rates_df = pd.read_csv(rates_path)
        rates_df.columns = rates_df.columns.str.strip()
        required_rates = {"Calendar Date", "Dividend Yield (Value-Weighted)"} | set(RATE_COLS)
        rate_missing = required_rates - set(rates_df.columns)
        if rate_missing:
            raise ValueError(f"Lipsesc coloanele in rates: {', '.join(sorted(rate_missing))}")
        rates_df["Calendar Date"] = pd.to_datetime(
            rates_df["Calendar Date"],
            dayfirst=True,
            errors="coerce",
            format="mixed",
        )
        rates_df = rates_df.dropna(subset=["Calendar Date"])
        rates_df = rates_df.sort_values("Calendar Date")
        rates_df = rates_df.set_index("Calendar Date")
        rates_df[RATE_COLS] = rates_df[RATE_COLS].interpolate(method="time")
        rates_df = rates_df.reset_index()
        rates_df[RATE_COLS] = rates_df[RATE_COLS].ffill().bfill()
        rates_df["Dividend Yield (Value-Weighted)"] = (
            rates_df["Dividend Yield (Value-Weighted)"].ffill().bfill()
        )

    total_rows = count_rows(csv_path)
    rows_read = 0
    start_time = time.time()
    last_report = start_time
    first = True
    written_rows = 0

    if out_path.exists():
        out_path.unlink()

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        rows_read += len(chunk)
        chunk = compute_chunk_iv(chunk, rates_df=rates_df)
        if chunk.empty:
            continue
        written_rows += len(chunk)
        chunk.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
        if first:
            # Print a small preview from the first chunk.
            preview_cols = [
                "underlying_symbol",
                "quote_date",
                "expiration",
                "strike",
                "option_type",
                "mid",
                "spot",
                "T",
                "implied_vol",
            ]
            preview_cols = [c for c in preview_cols if c in chunk.columns]
            print(
                chunk[preview_cols].head()
            )
        first = False
        now = time.time()
        if now - last_report >= 10 and total_rows > 0 and rows_read > 0:
            elapsed = now - start_time
            rate = rows_read / elapsed
            remaining = max(total_rows - rows_read, 0)
            eta_sec = remaining / rate if rate > 0 else 0
            pct = 100.0 * rows_read / total_rows
            print(f"Progress: {pct:.2f}% | ETA: {eta_sec/60:.1f} min")
            last_report = now
    return written_rows


def main():
    parser = argparse.ArgumentParser(description="Calculeaza implied volatility pentru options_eod_all.csv.")
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("options_eod_all_clean.csv"), help="Fisierul de intrare")
    parser.add_argument("--rates", type=pathlib.Path, default=pathlib.Path("div yield and rfr.csv"), help="Fisierul cu dividend yield si rfr")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("options_eod_all_with_iv.csv"), help="Fisierul de iesire")
    parser.add_argument("--chunksize", type=int, default=50000, help="Cate randuri sa proceseze simultan (pentru fisiere mari)")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Nu gasesc fisierul de intrare: {args.csv}")
    input_header = pd.read_csv(args.csv, nrows=0)
    has_rq_in_input = {"r_annual", "q_annual"}.issubset(set(input_header.columns))
    if (not has_rq_in_input) and (not args.rates.exists()):
        sys.exit(f"Nu gasesc fisierul cu rate: {args.rates}")

    try:
        written_rows = process_file(args.csv, args.out, rates_path=args.rates, chunksize=args.chunksize)
    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare la procesare: {exc}")

    if written_rows > 0:
        print(f"Am scris rezultatul in {args.out}")
    else:
        print("Nu s-a scris output final: 0 randuri valide pentru IV.")


if __name__ == "__main__":
    main()
