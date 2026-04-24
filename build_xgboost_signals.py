import argparse
import pathlib
import sys

import numpy as np
import pandas as pd


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def load_forecast_xgboost(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"quote_date", "node", "actual", "forecast"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane in forecast_xgboost_test.csv: {', '.join(sorted(missing))}")

    out = df[["quote_date", "node", "actual", "forecast"]].copy()
    out["quote_date"] = parse_dates(out["quote_date"])
    out = out.dropna(subset=["quote_date", "node"])
    out["realized_iv"] = pd.to_numeric(out["actual"], errors="coerce")
    out["forecast_iv"] = pd.to_numeric(out["forecast"], errors="coerce")
    out = out.drop(columns=["actual", "forecast"])
    return out


def build_prev_observed_from_wide(path: pathlib.Path) -> pd.DataFrame:
    wide = pd.read_csv(path)
    if "quote_date" not in wide.columns:
        raise ValueError("Lipseste coloana quote_date in iv_grid_wide.csv")
    node_cols = [c for c in wide.columns if c != "quote_date"]
    if not node_cols:
        raise ValueError("Nu exista coloane de noduri in iv_grid_wide.csv")

    wide["quote_date"] = parse_dates(wide["quote_date"])
    wide = wide.dropna(subset=["quote_date"]).sort_values("quote_date").reset_index(drop=True)

    long_df = wide.melt(
        id_vars=["quote_date"],
        value_vars=node_cols,
        var_name="node",
        value_name="observed_iv",
    )
    long_df["observed_iv"] = pd.to_numeric(long_df["observed_iv"], errors="coerce")
    long_df = long_df.sort_values(["node", "quote_date"]).reset_index(drop=True)

    long_df["decision_date"] = long_df.groupby("node")["quote_date"].shift(1)
    long_df["observed_iv_prev"] = long_df.groupby("node")["observed_iv"].shift(1)

    out = long_df[["quote_date", "node", "decision_date", "observed_iv_prev"]].copy()
    return out


def attach_grid_coordinates(signals: pd.DataFrame, grid_map_path: pathlib.Path) -> pd.DataFrame:
    grid_map = pd.read_csv(grid_map_path)
    required = {"feature", "log_moneyness", "T"}
    missing = required - set(grid_map.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane in iv_grid_map.csv: {', '.join(sorted(missing))}")

    grid_map = grid_map.rename(columns={"feature": "node"})
    out = signals.merge(
        grid_map[["node", "log_moneyness", "T"]],
        on="node",
        how="left",
    )
    return out


def compute_cross_sectional_strength(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    out["forecast_minus_observed"] = out["forecast_iv"] - out["observed_iv"]

    grp = out.groupby("quote_date")["forecast_minus_observed"]
    mean_cs = grp.transform("mean")
    std_cs = grp.transform("std").replace(0.0, np.nan)
    out["signal_strength"] = (out["forecast_minus_observed"] - mean_cs) / std_cs
    out["signal_strength"] = out["signal_strength"].fillna(0.0)
    return out


def select_positions(
    signals: pd.DataFrame,
    z_threshold: float,
    top_k_per_side: int,
) -> pd.DataFrame:
    out = signals.copy()
    out["signal_direction"] = "FLAT"
    out["signal"] = 0
    out["selected"] = 0

    for day, idx in out.groupby("quote_date").groups.items():
        day_df = out.loc[idx].copy()

        long_candidates = day_df[day_df["signal_strength"] >= z_threshold].sort_values(
            "signal_strength", ascending=False
        )
        short_candidates = day_df[day_df["signal_strength"] <= -z_threshold].sort_values(
            "signal_strength", ascending=True
        )

        if top_k_per_side > 0:
            long_idx = long_candidates.head(top_k_per_side).index
            short_idx = short_candidates.head(top_k_per_side).index
        else:
            long_idx = long_candidates.index
            short_idx = short_candidates.index

        out.loc[long_idx, "signal_direction"] = "LONG_VOL"
        out.loc[short_idx, "signal_direction"] = "SHORT_VOL"
        out.loc[long_idx, "signal"] = 1
        out.loc[short_idx, "signal"] = -1
        out.loc[long_idx.union(short_idx), "selected"] = 1

    return out


def build_signal_summary(signals: pd.DataFrame) -> pd.DataFrame:
    day_summary = (
        signals.groupby("quote_date", as_index=False)
        .agg(
            n_nodes=("node", "count"),
            n_selected=("selected", "sum"),
            n_long=("signal", lambda x: int((x == 1).sum())),
            n_short=("signal", lambda x: int((x == -1).sum())),
            n_flat=("signal", lambda x: int((x == 0).sum())),
            avg_abs_strength=("signal_strength", lambda x: float(np.mean(np.abs(x)))),
            avg_abs_strength_selected=(
                "signal_strength",
                lambda x: float(np.mean(np.abs(x[signals.loc[x.index, "selected"] == 1])))
                if int(signals.loc[x.index, "selected"].sum()) > 0
                else np.nan,
            ),
        )
    )
    return day_summary


def main():
    parser = argparse.ArgumentParser(
        description="Construieste semnale economice din forecasturile XGBoost pentru etapa de backtesting."
    )
    parser.add_argument("--forecast", type=pathlib.Path, default=pathlib.Path("forecast_xgboost_test.csv"))
    parser.add_argument("--wide", type=pathlib.Path, default=pathlib.Path("iv_grid_wide.csv"))
    parser.add_argument("--grid-map", type=pathlib.Path, default=pathlib.Path("iv_grid_map.csv"))
    parser.add_argument("--out-signals", type=pathlib.Path, default=pathlib.Path("signals_xgboost.csv"))
    parser.add_argument(
        "--out-summary",
        type=pathlib.Path,
        default=pathlib.Path("signals_xgboost_summary.csv"),
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=0.5,
        help="Prag minim pe signal_strength (z-score cross-sectional).",
    )
    parser.add_argument(
        "--top-k-per-side",
        type=int,
        default=10,
        help="Numar maxim de pozitii LONG si SHORT selectate pe zi. 0 = fara limita.",
    )
    args = parser.parse_args()

    if not args.forecast.exists():
        sys.exit(f"Nu gasesc fisierul forecast: {args.forecast}")
    if not args.wide.exists():
        sys.exit(f"Nu gasesc fisierul wide: {args.wide}")
    if not args.grid_map.exists():
        sys.exit(f"Nu gasesc fisierul grid map: {args.grid_map}")
    if args.top_k_per_side < 0:
        sys.exit("top-k-per-side trebuie sa fie >= 0.")

    try:
        print("Pas 1/4: citesc forecastul XGBoost si observatiile istorice...")
        fc = load_forecast_xgboost(args.forecast)
        prev_obs = build_prev_observed_from_wide(args.wide)

        print("Pas 2/4: construiesc deviatiile forecast vs observed (fara leakage)...")
        signals = fc.merge(prev_obs, on=["quote_date", "node"], how="left")
        signals = signals.rename(columns={"observed_iv_prev": "observed_iv"})
        signals = signals.dropna(subset=["forecast_iv", "observed_iv"])
        signals = attach_grid_coordinates(signals, args.grid_map)
        signals = compute_cross_sectional_strength(signals)

        print("Pas 3/4: aplic regula simpla de selectie LONG/SHORT volatility...")
        signals = select_positions(
            signals=signals,
            z_threshold=args.z_threshold,
            top_k_per_side=args.top_k_per_side,
        )

        signals = signals[
            [
                "quote_date",
                "decision_date",
                "node",
                "log_moneyness",
                "T",
                "observed_iv",
                "forecast_iv",
                "realized_iv",
                "forecast_minus_observed",
                "signal_strength",
                "signal_direction",
                "signal",
                "selected",
            ]
        ].sort_values(["quote_date", "node"])

        print("Pas 4/4: salvez semnalele si sumarul lor...")
        summary = build_signal_summary(signals)

        args.out_signals.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(args.out_signals, index=False)
        summary.to_csv(args.out_summary, index=False)

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare: {exc}")

    sel = signals["selected"].sum()
    print("\n=== Summary ===")
    print(f"Rows in signals: {len(signals)}")
    print(f"Rows selected for positions: {int(sel)}")
    print(f"Unique days: {signals['quote_date'].nunique()}")
    print(f"Unique nodes: {signals['node'].nunique()}")
    print(f"Signals file: {args.out_signals}")
    print(f"Summary file: {args.out_summary}")


if __name__ == "__main__":
    main()
