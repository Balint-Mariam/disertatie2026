import argparse
import pathlib
import sys

import numpy as np
import pandas as pd


def describe_implied_vol(csv_path: pathlib.Path, chunksize: int):
    header = pd.read_csv(csv_path, nrows=0)
    if "implied_vol" not in header.columns:
        raise ValueError("Lipseste coloana 'implied_vol' in fisierul de intrare.")

    cols = set(header.columns)
    usecols = ["implied_vol"]
    for c in [
        "quote_date",
        "T",
        "expiration",
        "moneyness",
        "strike",
        "spot",
        "underlying_bid_1545",
        "underlying_ask_1545",
    ]:
        if c in cols:
            usecols.append(c)

    total_rows = 0
    nan_count = 0
    parts = []
    by_day_counts: dict[str, int] = {}

    maturity_labels = ["<=30d", "31-90d", "91-180d", "181-365d", ">365d"]
    maturity_counts = {k: 0 for k in maturity_labels}

    mny_labels = ["<=0.80", "0.80-0.90", "0.90-1.00", "1.00-1.10", "1.10-1.20", ">1.20"]
    mny_counts = {k: 0 for k in mny_labels}

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        vals = pd.to_numeric(chunk["implied_vol"], errors="coerce")
        total_rows += len(vals)
        nan_count += int(vals.isna().sum())
        non_nan = vals.dropna()
        if not non_nan.empty:
            parts.append(non_nan.to_numpy(dtype=float))

        if "quote_date" in chunk.columns:
            qd = pd.to_datetime(chunk["quote_date"], errors="coerce", dayfirst=True, format="mixed")
            cts = qd.dt.strftime("%Y-%m-%d").value_counts(dropna=True)
            for d, c in cts.items():
                by_day_counts[d] = by_day_counts.get(d, 0) + int(c)

        if "T" in chunk.columns:
            t_vals = pd.to_numeric(chunk["T"], errors="coerce")
        elif {"quote_date", "expiration"}.issubset(chunk.columns):
            qd = pd.to_datetime(chunk["quote_date"], errors="coerce", dayfirst=True, format="mixed")
            ex = pd.to_datetime(chunk["expiration"], errors="coerce", dayfirst=True, format="mixed")
            t_vals = (ex - qd).dt.days / 365.0
        else:
            t_vals = pd.Series(dtype=float)
        if not t_vals.empty:
            t_buckets = pd.cut(
                t_vals,
                bins=[0.0, 30 / 365.0, 90 / 365.0, 180 / 365.0, 365 / 365.0, np.inf],
                labels=maturity_labels,
                include_lowest=True,
            )
            cts = t_buckets.value_counts(dropna=True)
            for k in maturity_labels:
                maturity_counts[k] += int(cts.get(k, 0))

        if "moneyness" in chunk.columns:
            m_vals = pd.to_numeric(chunk["moneyness"], errors="coerce")
        elif {"strike", "spot"}.issubset(chunk.columns):
            m_vals = pd.to_numeric(chunk["strike"], errors="coerce") / pd.to_numeric(chunk["spot"], errors="coerce")
        elif {"strike", "underlying_bid_1545", "underlying_ask_1545"}.issubset(chunk.columns):
            spot = (
                pd.to_numeric(chunk["underlying_bid_1545"], errors="coerce")
                + pd.to_numeric(chunk["underlying_ask_1545"], errors="coerce")
            ) / 2.0
            m_vals = pd.to_numeric(chunk["strike"], errors="coerce") / spot
        else:
            m_vals = pd.Series(dtype=float)
        if not m_vals.empty:
            m_buckets = pd.cut(
                m_vals,
                bins=[0.0, 0.8, 0.9, 1.0, 1.1, 1.2, np.inf],
                labels=mny_labels,
                include_lowest=True,
            )
            cts = m_buckets.value_counts(dropna=True)
            for k in mny_labels:
                mny_counts[k] += int(cts.get(k, 0))

    print("=== Descriptive Stats: implied_vol ===")
    print(f"Numar total randuri: {total_rows}")
    print(f"NaN implied_vol: {nan_count}")

    if not parts:
        print("Nu exista valori finite pentru implied_vol.")
        return

    all_vals = np.concatenate(parts)
    print(f"Minim: {np.min(all_vals):.6f}")
    print(f"Mediana: {np.median(all_vals):.6f}")
    print(f"Percentila 95: {np.percentile(all_vals, 95):.6f}")
    print(f"Percentila 99: {np.percentile(all_vals, 99):.6f}")
    print(f"Maxim: {np.max(all_vals):.6f}")

    print("\n=== Observatii pe zi ===")
    if by_day_counts:
        day_series = pd.Series(by_day_counts).sort_index()
        print(f"Zile unice: {len(day_series)}")
        print(f"Min/Mediana/P95/Max observatii pe zi: {int(day_series.min())} / {int(day_series.median())} / {int(day_series.quantile(0.95))} / {int(day_series.max())}")
    else:
        print("Nu am putut calcula (lipseste quote_date).")

    print("\n=== Observatii pe bucket maturitate ===")
    if sum(maturity_counts.values()) > 0:
        for k in maturity_labels:
            print(f"{k}: {maturity_counts[k]}")
    else:
        print("Nu am putut calcula (lipsesc T sau datele pentru calcul T).")

    print("\n=== Observatii pe bucket moneyness ===")
    if sum(mny_counts.values()) > 0:
        for k in mny_labels:
            print(f"{k}: {mny_counts[k]}")
    else:
        print("Nu am putut calcula (lipsesc moneyness/strike/spot).")


def main():
    parser = argparse.ArgumentParser(description="Statistici descriptive pentru implied_vol.")
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path("options_eod_all_with_iv.csv"),
        help="Fisierul de intrare care contine coloana implied_vol",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="Cate randuri sa proceseze pe chunk",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Nu gasesc fisierul de intrare: {args.csv}")

    try:
        describe_implied_vol(args.csv, chunksize=args.chunksize)
    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare la procesare: {exc}")


if __name__ == "__main__":
    main()
