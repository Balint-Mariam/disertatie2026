import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "quote_date",
    "expiration",
    "underlying_bid_1545",
    "underlying_ask_1545",
    "bid_1545",
    "ask_1545",
    "strike",
    "option_type",
    "trade_volume",
    "open_interest",
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


def prepare_rates_df(rates_path: pathlib.Path) -> pd.DataFrame:
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
    return rates_df


def lower_bound_price(spot, strike, T, r, q, is_call):
    """European no-arbitrage lower bound with carry and discounting."""
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    call_lb = np.maximum(0.0, spot * disc_q - strike * disc_r)
    put_lb = np.maximum(0.0, strike * disc_r - spot * disc_q)
    return np.where(is_call, call_lb, put_lb)

def apply_basic_filters(
    chunk: pd.DataFrame,
    rates_df: pd.DataFrame,
    t_min_days: int,
    t_max_years: float,
    spread_rel_max: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean option data using standard filters and lower-bound checks."""
    stats = {
        "input": len(chunk),
        "dropped_basic": 0,
        "dropped_bad_dates": 0,
        "dropped_t_window": 0,
        "dropped_no_rates": 0,
        "dropped_moneyness": 0,
        "dropped_spread": 0,
        "dropped_option_type": 0,
        "dropped_lower_bound": 0,
        "kept_basic": 0,
    }
    chunk["spot"] = (chunk["underlying_bid_1545"] + chunk["underlying_ask_1545"]) / 2
    chunk["mid"] = (chunk["bid_1545"] + chunk["ask_1545"]) / 2

    valid = (
        (chunk["bid_1545"] > 0)
        & (chunk["ask_1545"] > 0)
        & (chunk["ask_1545"] > chunk["bid_1545"])
        & (chunk["trade_volume"] > 0)
        & (chunk["open_interest"] > 0)
        & (chunk["spot"] > 0)
        & (chunk["strike"] > 0)
        & (chunk["mid"] > 0)
    )

    stats["dropped_basic"] = int((~valid).sum())
    chunk = chunk.loc[valid].copy()
    if chunk.empty:
        return chunk, stats
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
    before_dates = len(chunk)
    chunk = chunk.dropna(subset=["quote_date", "expiration"])
    stats["dropped_bad_dates"] = before_dates - len(chunk)
    if chunk.empty:
        return chunk, stats
    chunk["T"] = (chunk["expiration"] - chunk["quote_date"]).dt.days / 365.0
    before_t = len(chunk)
    chunk = chunk[(chunk["T"] >= (t_min_days / 365.0)) & (chunk["T"] <= t_max_years)]
    stats["dropped_t_window"] = before_t - len(chunk)
    if chunk.empty:
        return chunk, stats

    before_rates = len(chunk)
    chunk = chunk.merge(
        rates_df,
        left_on="quote_date",
        right_on="Calendar Date",
        how="inner",
    )
    stats["dropped_no_rates"] = before_rates - len(chunk)
    if chunk.empty:
        return chunk, stats
    rate_matrix = chunk[RATE_COLS].to_numpy(dtype=float)
    T = chunk["T"].to_numpy(dtype=float)
    r_daily = interpolate_rates(rate_matrix, RATE_TERMS_YEARS, T)
    chunk["r_annual"] = (1.0 + r_daily) ** TRADING_DAYS - 1.0
    q_daily = chunk["Dividend Yield (Value-Weighted)"].to_numpy(dtype=float)
    chunk["q_annual"] = (1.0 + q_daily) ** TRADING_DAYS - 1.0

    chunk["moneyness"] = chunk["strike"] / chunk["spot"]
    before_mny = len(chunk)
    chunk = chunk[chunk["moneyness"].between(0.8, 1.2)]
    stats["dropped_moneyness"] = before_mny - len(chunk)
    if chunk.empty:
        return chunk, stats

    spread_rel = (chunk["ask_1545"] - chunk["bid_1545"]) / chunk["mid"]
    before_spread = len(chunk)
    chunk = chunk[spread_rel <= spread_rel_max]
    stats["dropped_spread"] = before_spread - len(chunk)
    if chunk.empty:
        return chunk, stats

    chunk["option_type"] = chunk["option_type"].astype(str).str.upper().str.strip()
    before_opt_type = len(chunk)
    chunk = chunk[chunk["option_type"].isin(["C", "P"])]
    stats["dropped_option_type"] = before_opt_type - len(chunk)
    if chunk.empty:
        return chunk, stats

    is_call = chunk["option_type"] == "C"
    lb = lower_bound_price(
        spot=chunk["spot"].to_numpy(dtype=float),
        strike=chunk["strike"].to_numpy(dtype=float),
        T=chunk["T"].to_numpy(dtype=float),
        r=chunk["r_annual"].to_numpy(dtype=float),
        q=chunk["q_annual"].to_numpy(dtype=float),
        is_call=is_call.to_numpy(),
    )
    tol = 1e-4
    before_lb = len(chunk)
    chunk = chunk[chunk["mid"] + tol >= lb]
    stats["dropped_lower_bound"] = before_lb - len(chunk)
    stats["kept_basic"] = len(chunk)
    if chunk.empty:
        return chunk, stats

    return chunk, stats


def process_file(
    csv_path: pathlib.Path,
    out_path: pathlib.Path,
    rates_path: pathlib.Path,
    chunksize: int,
    t_min_days: int,
    t_max_years: float,
    spread_rel_max: float,
) -> dict[str, int]:
    def count_rows(path: pathlib.Path) -> int:
        # Fast line count using binary reads; subtract header row.
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

    if not rates_path.exists():
        raise ValueError(f"Nu gasesc fisierul cu rate: {rates_path}")
    rates_df = prepare_rates_df(rates_path)

    total_rows = count_rows(csv_path)
    rows_read = 0
    start_time = time.time()
    last_report = start_time
    first_out = True

    totals = {
        "input": 0,
        "dropped_basic": 0,
        "dropped_bad_dates": 0,
        "dropped_t_window": 0,
        "dropped_no_rates": 0,
        "dropped_moneyness": 0,
        "dropped_spread": 0,
        "dropped_option_type": 0,
        "dropped_lower_bound": 0,
        "kept_basic": 0,
        "kept_final": 0,
    }

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        rows_read += len(chunk)
        chunk, stats = apply_basic_filters(
            chunk,
            rates_df=rates_df,
            t_min_days=t_min_days,
            t_max_years=t_max_years,
            spread_rel_max=spread_rel_max,
        )
        for k in stats:
            totals[k] += stats[k]
        if chunk.empty:
            continue
        totals["kept_final"] += len(chunk)
        chunk.to_csv(out_path, mode="w" if first_out else "a", index=False, header=first_out)
        if first_out:
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
            ]
            preview_cols = [col for col in preview_cols if col in chunk.columns]
            print(
                chunk[preview_cols].head()
            )
            first_out = False

        now = time.time()
        if now - last_report >= 10 and total_rows > 0 and rows_read > 0:
            elapsed = now - start_time
            rate = rows_read / elapsed
            remaining = max(total_rows - rows_read, 0)
            eta_sec = remaining / rate if rate > 0 else 0
            pct = 100.0 * rows_read / total_rows
            print(f"Progress: {pct:.2f}% | ETA: {eta_sec/60:.1f} min")
            last_report = now

    if totals["kept_final"] == 0:
        if out_path.exists():
            out_path.unlink()
        print("Nu a ramas niciun rand dupa filtrele de baza.")
        print_stats(totals)
        return totals

    print_stats(totals)
    return totals


def print_stats(totals: dict[str, int]):
    total_in = max(totals.get("input", 0), 1)

    def pct(x: int) -> float:
        return 100.0 * x / total_in

    print("\n=== Cleaning Stats ===")
    print(f"Input total: {totals['input']}")
    print(f"Dropped basic quality: {totals['dropped_basic']} ({pct(totals['dropped_basic']):.2f}%)")
    print(f"Dropped bad dates: {totals['dropped_bad_dates']} ({pct(totals['dropped_bad_dates']):.2f}%)")
    print(f"Dropped T window: {totals['dropped_t_window']} ({pct(totals['dropped_t_window']):.2f}%)")
    print(f"Dropped no rates match: {totals['dropped_no_rates']} ({pct(totals['dropped_no_rates']):.2f}%)")
    print(f"Dropped moneyness: {totals['dropped_moneyness']} ({pct(totals['dropped_moneyness']):.2f}%)")
    print(f"Dropped spread: {totals['dropped_spread']} ({pct(totals['dropped_spread']):.2f}%)")
    print(f"Dropped option_type invalid: {totals['dropped_option_type']} ({pct(totals['dropped_option_type']):.2f}%)")
    print(f"Dropped lower bound: {totals['dropped_lower_bound']} ({pct(totals['dropped_lower_bound']):.2f}%)")
    print(f"Kept after basic filters: {totals['kept_basic']} ({pct(totals['kept_basic']):.2f}%)")
    print("Parity filter: dezactivat")
    print(f"Final kept: {totals['kept_final']} ({pct(totals['kept_final']):.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Curata options_eod_all.csv si salveaza randurile valide.")
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("options_eod_all.csv"), help="Fisierul de intrare")
    parser.add_argument("--rates", type=pathlib.Path, default=pathlib.Path("div yield and rfr.csv"), help="Fisierul cu dividend yield si rfr")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("options_eod_all_clean.csv"), help="Fisierul de iesire")
    parser.add_argument("--chunksize", type=int, default=200000, help="Cate randuri sa proceseze simultan (pentru fisiere mari)")
    parser.add_argument("--t_min_days", type=int, default=14, help="T minim in zile (default 14)")
    parser.add_argument("--t_max_years", type=float, default=1.0, help="T maxim in ani (default 1)")
    parser.add_argument("--spread_rel_max", type=float, default=0.2, help="Prag maxim pentru spread relativ (ask-bid)/mid")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Nu gasesc fisierul de intrare: {args.csv}")

    try:
        totals = process_file(
            args.csv,
            args.out,
            rates_path=args.rates,
            chunksize=args.chunksize,
            t_min_days=args.t_min_days,
            t_max_years=args.t_max_years,
            spread_rel_max=args.spread_rel_max,
        )
    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare la procesare: {exc}")

    if totals["kept_final"] > 0:
        print(f"Am scris rezultatul in {args.out}")
    else:
        print("Nu s-a scris output final: 0 randuri dupa filtre.")

if __name__ == "__main__":
    main()
