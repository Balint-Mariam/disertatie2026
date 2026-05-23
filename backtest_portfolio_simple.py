import argparse
import pathlib
import sys

import numpy as np
import pandas as pd


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def load_signals(signals_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(signals_path)
    required = {
        "quote_date",
        "node",
        "signal",
        "selected",
        "observed_iv",
        "realized_iv",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane in signals_xgboost.csv: {', '.join(sorted(missing))}")

    out = df.copy()
    out["quote_date"] = parse_dates(out["quote_date"])
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce")
    out["selected"] = pd.to_numeric(out["selected"], errors="coerce")
    out["observed_iv"] = pd.to_numeric(out["observed_iv"], errors="coerce")
    out["realized_iv"] = pd.to_numeric(out["realized_iv"], errors="coerce")

    out = out.dropna(subset=["quote_date", "node", "signal", "selected", "observed_iv", "realized_iv"])
    out = out[out["selected"] == 1].copy()
    out = out[out["signal"].isin([1, -1])].copy()
    out = out.sort_values(["quote_date", "node"]).reset_index(drop=True)
    return out


def validate_with_summary(signals: pd.DataFrame, summary_path: pathlib.Path) -> None:
    if not summary_path.exists():
        print("Info: signals_xgboost_summary.csv nu exista; continui fara verificare suplimentara.")
        return
    summary = pd.read_csv(summary_path)
    if "quote_date" not in summary.columns:
        print("Info: summary fara quote_date; sar verificarea.")
        return
    summary["quote_date"] = parse_dates(summary["quote_date"])
    s_days = set(signals["quote_date"].dropna().unique())
    sum_days = set(summary["quote_date"].dropna().unique())
    only_in_signals = len(s_days - sum_days)
    only_in_summary = len(sum_days - s_days)
    if only_in_signals or only_in_summary:
        print(
            "Info: mismatch zile intre signals si summary | "
            f"only_in_signals={only_in_signals}, only_in_summary={only_in_summary}"
        )


def compute_weights_and_position_pnl(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    out["iv_change"] = out["realized_iv"] - out["observed_iv"]
    out["weight"] = 0.0
    out["position_pnl"] = 0.0
    out["day_status"] = "unprocessed"

    for day, idx in out.groupby("quote_date").groups.items():
        day_df = out.loc[idx]
        long_idx = day_df[day_df["signal"] == 1].index
        short_idx = day_df[day_df["signal"] == -1].index
        n_long = len(long_idx)
        n_short = len(short_idx)

        if n_long > 0 and n_short > 0:
            # Side-neutral: +0.5 pe long, -0.5 pe short.
            w_long = 0.5 / n_long
            w_short = -0.5 / n_short
            out.loc[long_idx, "weight"] = w_long
            out.loc[short_idx, "weight"] = w_short
            out.loc[idx, "day_status"] = "tradeable_side_neutral"
        elif n_long > 0:
            # Lipseste short side -> evitam expunerea directionala in acest backtest brut.
            out.loc[idx, "weight"] = 0.0
            out.loc[idx, "day_status"] = "skipped_no_short_side"
        elif n_short > 0:
            out.loc[idx, "weight"] = 0.0
            out.loc[idx, "day_status"] = "skipped_no_long_side"
        else:
            out.loc[idx, "weight"] = 0.0
            out.loc[idx, "day_status"] = "skipped_no_positions"

    out["position_pnl"] = out["weight"] * out["iv_change"]
    return out


def build_daily_pnl(positions: pd.DataFrame) -> pd.DataFrame:
    g = positions.groupby("quote_date", as_index=False)
    daily = g.agg(
        n_long=("signal", lambda x: int((x == 1).sum())),
        n_short=("signal", lambda x: int((x == -1).sum())),
        gross_exposure=("weight", lambda x: float(np.abs(x).sum())),
        net_exposure=("weight", "sum"),
        daily_pnl=("position_pnl", "sum"),
    )
    daily = daily.sort_values("quote_date").reset_index(drop=True)
    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()

    day_status = (
        positions.groupby("quote_date")["day_status"]
        .first()
        .rename("day_status")
        .reset_index()
    )
    daily = daily.merge(day_status, on="quote_date", how="left")
    daily["tradeable_day"] = (daily["gross_exposure"] > 0).astype(int)
    return daily


def max_drawdown_from_cum_pnl(cum_pnl: pd.Series) -> float:
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    return float(drawdown.min()) if len(drawdown) > 0 else np.nan


def build_performance_summary(positions: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    daily_pnl = daily["daily_pnl"].to_numpy(dtype=float)
    total_pnl = float(np.nansum(daily_pnl))
    mean_daily = float(np.nanmean(daily_pnl)) if len(daily_pnl) > 0 else np.nan
    daily_vol = float(np.nanstd(daily_pnl, ddof=1)) if len(daily_pnl) > 1 else np.nan
    sharpe = np.nan
    if np.isfinite(daily_vol) and daily_vol > 0:
        sharpe = float(np.sqrt(252.0) * mean_daily / daily_vol)

    mdd = max_drawdown_from_cum_pnl(daily["cumulative_pnl"])
    out = pd.DataFrame(
        [
            {
                "strategy": "xgboost_signal_simple_vol_backtest",
                "total_pnl": total_pnl,
                "mean_daily_pnl": mean_daily,
                "daily_volatility": daily_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": mdd,
                "num_total_positions_selected": int(len(positions)),
                "num_positions_traded_nonzero_weight": int((positions["weight"] != 0).sum()),
                "num_days_total": int(daily["quote_date"].nunique()),
                "num_days_traded": int((daily["gross_exposure"] > 0).sum()),
                "num_days_skipped_no_short_side": int((daily["day_status"] == "skipped_no_short_side").sum()),
                "num_days_skipped_no_long_side": int((daily["day_status"] == "skipped_no_long_side").sum()),
                "avg_gross_exposure": float(daily["gross_exposure"].mean()),
                "avg_net_exposure": float(daily["net_exposure"].mean()),
            }
        ]
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Backtest economic simplu (brut) pe semnale XGBoost, fara hedging."
    )
    parser.add_argument("--signals", type=pathlib.Path, default=pathlib.Path("signals_xgboost.csv"))
    parser.add_argument(
        "--signals-summary",
        type=pathlib.Path,
        default=pathlib.Path("signals_xgboost_summary.csv"),
    )
    parser.add_argument(
        "--out-positions",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_positions_simple.csv"),
    )
    parser.add_argument(
        "--out-daily",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_daily_pnl_simple.csv"),
    )
    parser.add_argument(
        "--out-performance",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_performance_simple.csv"),
    )
    args = parser.parse_args()

    if not args.signals.exists():
        sys.exit(f"Nu gasesc fisierul de semnale: {args.signals}")

    try:
        print("Pas 1/4: incarc semnalele selectate...")
        signals = load_signals(args.signals)
        if signals.empty:
            raise ValueError("Nu exista pozitii selectate (selected=1) valide.")
        validate_with_summary(signals, args.signals_summary)

        print("Pas 2/4: construiesc ponderile zilnice side-neutral si pnl pe pozitie...")
        positions = compute_weights_and_position_pnl(signals)

        print("Pas 3/4: agreg daily pnl si cumulative pnl...")
        daily = build_daily_pnl(positions)

        print("Pas 4/4: calculez performanta agregata...")
        performance = build_performance_summary(positions, daily)

        keep_pos_cols = [
            "quote_date",
            "node",
            "signal",
            "selected",
            "observed_iv",
            "realized_iv",
            "iv_change",
            "weight",
            "position_pnl",
            "day_status",
        ]
        positions_out = positions[keep_pos_cols].copy()

        keep_day_cols = [
            "quote_date",
            "n_long",
            "n_short",
            "gross_exposure",
            "net_exposure",
            "daily_pnl",
            "cumulative_pnl",
            "tradeable_day",
            "day_status",
        ]
        daily_out = daily[keep_day_cols].copy()

        for p in [args.out_positions, args.out_daily, args.out_performance]:
            p.parent.mkdir(parents=True, exist_ok=True)
        positions_out.to_csv(args.out_positions, index=False)
        daily_out.to_csv(args.out_daily, index=False)
        performance.to_csv(args.out_performance, index=False)

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare: {exc}")

    print("\n=== Summary ===")
    print(f"Positions rows: {len(positions_out)}")
    print(f"Daily rows: {len(daily_out)}")
    print(f"Total PnL: {performance.loc[0, 'total_pnl']:.6f}")
    print(f"Sharpe: {performance.loc[0, 'sharpe_ratio']:.6f}" if np.isfinite(performance.loc[0, "sharpe_ratio"]) else "Sharpe: NaN")
    print(f"Output positions: {args.out_positions}")
    print(f"Output daily pnl: {args.out_daily}")
    print(f"Output performance: {args.out_performance}")


if __name__ == "__main__":
    main()
