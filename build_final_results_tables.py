import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def ensure_exists(paths: list[pathlib.Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Lipsesc fisiere de input: " + ", ".join(missing))


def load_forecast_metrics(paths: list[pathlib.Path]) -> pd.DataFrame:
    parts = []
    for p in paths:
        df = pd.read_csv(p)
        required = {"model", "rmse", "mae"}
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"Lipsesc coloane in {p.name}: {', '.join(sorted(miss))}")
        row = df.iloc[0].copy()
        model = str(row["model"]).lower().strip()
        model_map = {
            "naive_persistence": "Naive",
            "naive": "Naive",
            "arima": "ARIMA",
            "xgboost": "XGBoost",
        }
        row_out = {
            "model": model_map.get(model, model),
            "RMSE": float(row["rmse"]),
            "MAE": float(row["mae"]),
            "n_predictions": int(row["n_predictions"]) if "n_predictions" in df.columns else np.nan,
            "n_days": int(row["n_days"]) if "n_days" in df.columns else np.nan,
            "n_nodes": int(row["n_nodes"]) if "n_nodes" in df.columns else np.nan,
        }
        parts.append(row_out)
    out = pd.DataFrame(parts).sort_values("RMSE").reset_index(drop=True)
    return out


def extract_mapping_quality(mapping_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(mapping_path)
    if not {"metric", "value", "section"}.issubset(df.columns):
        raise ValueError("portfolio_mapping_quality_strict.csv trebuie sa aiba coloanele metric,value,section.")
    overall = df[df["section"] == "overall"].set_index("metric")["value"]
    wanted = [
        "total_selected_signals",
        "positions_mapped_entry_day",
        "positions_traded_with_tplus1_price",
        "traded_rate_vs_selected",
        "avg_mapping_distance_mapped_only",
    ]
    row = {k: float(overall.get(k, np.nan)) for k in wanted}
    out = pd.DataFrame([row])
    return out


def build_backtest_comparison(
    simple_path: pathlib.Path,
    realistic_strict_path: pathlib.Path,
    hedged_perf_path: pathlib.Path,
) -> pd.DataFrame:
    simple = pd.read_csv(simple_path).iloc[0]
    strict = pd.read_csv(realistic_strict_path).iloc[0]
    hedged = pd.read_csv(hedged_perf_path)

    rows = []
    rows.append(
        {
            "strategy_label": "simple_synthetic_iv_backtest",
            "strategy_key": "simple_synthetic",
            "total_pnl": float(simple["total_pnl"]),
            "mean_daily_pnl": float(simple["mean_daily_pnl"]),
            "daily_volatility": float(simple["daily_volatility"]),
            "sharpe_ratio": float(simple["sharpe_ratio"]),
            "max_drawdown": float(simple["max_drawdown"]),
            "num_days": int(simple["num_days_total"]) if "num_days_total" in simple.index else np.nan,
            "num_days_hedged": np.nan,
        }
    )

    # Unhedged: prefer varianta din hedged_performance (aliniata pe acelasi set de zile cu hedging).
    unhedged = hedged[hedged["strategy"] == "unhedged"]
    if not unhedged.empty:
        r = unhedged.iloc[0]
        rows.append(
            {
                "strategy_label": "realistic_unhedged_option_portfolio",
                "strategy_key": "unhedged",
                "total_pnl": float(r["total_pnl"]),
                "mean_daily_pnl": float(r["mean_daily_pnl"]),
                "daily_volatility": float(r["daily_volatility"]),
                "sharpe_ratio": float(r["sharpe_ratio"]),
                "max_drawdown": float(r["max_drawdown"]),
                "num_days": int(r["num_days"]) if "num_days" in r.index else np.nan,
                "num_days_hedged": int(r["num_days_hedged"]) if "num_days_hedged" in r.index else np.nan,
            }
        )
    else:
        rows.append(
            {
                "strategy_label": "realistic_unhedged_option_portfolio",
                "strategy_key": "unhedged",
                "total_pnl": float(strict["total_pnl"]),
                "mean_daily_pnl": float(strict["mean_daily_pnl"]),
                "daily_volatility": float(strict["daily_volatility"]),
                "sharpe_ratio": float(strict["sharpe_ratio"]),
                "max_drawdown": float(strict["max_drawdown"]),
                "num_days": int(strict["num_days_traded"]) if "num_days_traded" in strict.index else np.nan,
                "num_days_hedged": np.nan,
            }
        )

    for key, label in [
        ("delta_hedged", "delta_hedged_portfolio"),
        ("delta_gamma_hedged", "delta_gamma_hedged_portfolio"),
    ]:
        r = hedged[hedged["strategy"] == key]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(
            {
                "strategy_label": label,
                "strategy_key": key,
                "total_pnl": float(r["total_pnl"]),
                "mean_daily_pnl": float(r["mean_daily_pnl"]),
                "daily_volatility": float(r["daily_volatility"]),
                "sharpe_ratio": float(r["sharpe_ratio"]),
                "max_drawdown": float(r["max_drawdown"]),
                "num_days": int(r["num_days"]) if "num_days" in r.index else np.nan,
                "num_days_hedged": int(r["num_days_hedged"]) if "num_days_hedged" in r.index else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    order = [
        "simple_synthetic",
        "unhedged",
        "delta_hedged",
        "delta_gamma_hedged",
    ]
    out["order"] = out["strategy_key"].map({k: i for i, k in enumerate(order)})
    out = out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    return out


def build_hedge_effectiveness_table(
    hedge_effectiveness_path: pathlib.Path,
    hedged_daily_path: pathlib.Path,
    hedge_trades_path: pathlib.Path | None,
) -> pd.DataFrame:
    eff = pd.read_csv(hedge_effectiveness_path)
    if not {"metric", "value"}.issubset(eff.columns):
        raise ValueError("hedge_effectiveness_summary.csv trebuie sa aiba coloanele metric,value.")
    s = eff.set_index("metric")["value"]

    base = {
        "mean_abs_delta_before_hedge": float(s.get("mean_abs_delta_before_hedge", np.nan)),
        "mean_abs_delta_after_delta_only": float(s.get("mean_abs_delta_after_delta_only", np.nan)),
        "mean_abs_delta_after_delta_gamma": float(s.get("mean_abs_delta_after_delta_gamma", np.nan)),
        "mean_abs_gamma_before_hedge": float(s.get("mean_abs_gamma_before_hedge", np.nan)),
        "mean_abs_gamma_after_delta_gamma": float(s.get("mean_abs_gamma_after_delta_gamma", np.nan)),
        "pct_days_delta_only_hedged": float(s.get("pct_days_delta_only_hedged", np.nan)),
        "pct_days_delta_gamma_hedged": float(s.get("pct_days_delta_gamma_hedged", np.nan)),
    }

    # Optional: metrici pe zile efectiv hedged.
    if hedge_trades_path is not None and hedge_trades_path.exists():
        trades = pd.read_csv(hedge_trades_path)
        req = {
            "hedge_status",
            "total_delta_before",
            "total_gamma_before",
            "hedge_underlying_units_delta_only",
            "residual_delta_after_hedge",
            "residual_gamma_after_hedge",
        }
        if req.issubset(trades.columns):
            dg_hedged = trades[trades["hedge_status"] == "hedged"].copy()
            do_hedged = trades[trades["hedge_underlying_units_delta_only"].notna()].copy()

            if not do_hedged.empty:
                residual_do = (
                    pd.to_numeric(do_hedged["total_delta_before"], errors="coerce")
                    + pd.to_numeric(do_hedged["hedge_underlying_units_delta_only"], errors="coerce")
                )
                base["mean_abs_delta_before_hedged_days_only"] = float(
                    np.nanmean(np.abs(pd.to_numeric(do_hedged["total_delta_before"], errors="coerce")))
                )
                base["mean_abs_delta_after_delta_only_hedged_days_only"] = float(
                    np.nanmean(np.abs(residual_do))
                )
            else:
                base["mean_abs_delta_before_hedged_days_only"] = np.nan
                base["mean_abs_delta_after_delta_only_hedged_days_only"] = np.nan

            if not dg_hedged.empty:
                base["mean_abs_delta_after_delta_gamma_hedged_days_only"] = float(
                    np.nanmean(np.abs(pd.to_numeric(dg_hedged["residual_delta_after_hedge"], errors="coerce")))
                )
                base["mean_abs_gamma_before_hedged_days_only"] = float(
                    np.nanmean(np.abs(pd.to_numeric(dg_hedged["total_gamma_before"], errors="coerce")))
                )
                base["mean_abs_gamma_after_delta_gamma_hedged_days_only"] = float(
                    np.nanmean(np.abs(pd.to_numeric(dg_hedged["residual_gamma_after_hedge"], errors="coerce")))
                )
            else:
                base["mean_abs_delta_after_delta_gamma_hedged_days_only"] = np.nan
                base["mean_abs_gamma_before_hedged_days_only"] = np.nan
                base["mean_abs_gamma_after_delta_gamma_hedged_days_only"] = np.nan
        else:
            base["mean_abs_delta_before_hedged_days_only"] = np.nan
            base["mean_abs_delta_after_delta_only_hedged_days_only"] = np.nan
            base["mean_abs_delta_after_delta_gamma_hedged_days_only"] = np.nan
            base["mean_abs_gamma_before_hedged_days_only"] = np.nan
            base["mean_abs_gamma_after_delta_gamma_hedged_days_only"] = np.nan
    else:
        base["mean_abs_delta_before_hedged_days_only"] = np.nan
        base["mean_abs_delta_after_delta_only_hedged_days_only"] = np.nan
        base["mean_abs_delta_after_delta_gamma_hedged_days_only"] = np.nan
        base["mean_abs_gamma_before_hedged_days_only"] = np.nan
        base["mean_abs_gamma_after_delta_gamma_hedged_days_only"] = np.nan

    daily = pd.read_csv(hedged_daily_path)
    if {"hedge_status_delta_only", "hedge_status_delta_gamma"}.issubset(daily.columns):
        n_days = len(daily)
        n_do = int((daily["hedge_status_delta_only"] == "hedged").sum())
        n_dg = int((daily["hedge_status_delta_gamma"] == "hedged").sum())
        base["num_days_total"] = float(n_days)
        base["num_days_delta_only_hedged"] = float(n_do)
        base["num_days_delta_gamma_hedged"] = float(n_dg)
        base["num_days_without_valid_delta_gamma_hedge"] = float(n_days - n_dg)

    return pd.DataFrame([base])


def plot_cumulative_pnl(hedged_daily: pd.DataFrame, out_path: pathlib.Path) -> None:
    d = hedged_daily.copy()
    d["quote_date"] = parse_dates(d["quote_date"])
    d = d.sort_values("quote_date")
    d["cum_unhedged"] = pd.to_numeric(d["portfolio_daily_pnl_unhedged"], errors="coerce").fillna(0.0).cumsum()
    d["cum_delta"] = pd.to_numeric(d["daily_pnl_delta_hedged"], errors="coerce").fillna(0.0).cumsum()
    d["cum_delta_gamma"] = pd.to_numeric(d["daily_pnl_delta_gamma_hedged"], errors="coerce").fillna(0.0).cumsum()

    plt.figure(figsize=(10, 5.5))
    plt.plot(d["quote_date"], d["cum_unhedged"], label="Unhedged", linewidth=1.8)
    plt.plot(d["quote_date"], d["cum_delta"], label="Delta Hedged", linewidth=1.8)
    plt.plot(d["quote_date"], d["cum_delta_gamma"], label="Delta-Gamma Hedged", linewidth=1.8)
    plt.title("Cumulative PnL Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_daily_pnl_distribution(hedged_daily: pd.DataFrame, out_path: pathlib.Path) -> None:
    d = hedged_daily.copy()
    s1 = pd.to_numeric(d["portfolio_daily_pnl_unhedged"], errors="coerce").dropna()
    s2 = pd.to_numeric(d["daily_pnl_delta_hedged"], errors="coerce").dropna()
    s3 = pd.to_numeric(d["daily_pnl_delta_gamma_hedged"], errors="coerce").dropna()

    all_vals = np.concatenate([s1.to_numpy(), s2.to_numpy(), s3.to_numpy()]) if len(s1) and len(s2) and len(s3) else np.array([])
    if all_vals.size == 0:
        return
    bins = np.linspace(np.nanpercentile(all_vals, 1), np.nanpercentile(all_vals, 99), 50)

    plt.figure(figsize=(10, 5.5))
    plt.hist(s1, bins=bins, alpha=0.45, label="Unhedged", density=True)
    plt.hist(s2, bins=bins, alpha=0.45, label="Delta Hedged", density=True)
    plt.hist(s3, bins=bins, alpha=0.45, label="Delta-Gamma Hedged", density=True)
    plt.title("Daily PnL Distribution")
    plt.xlabel("Daily PnL")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_forecast_error_bars(forecast_df: pd.DataFrame, out_path: pathlib.Path) -> None:
    d = forecast_df.copy()
    order = ["Naive", "ARIMA", "XGBoost"]
    d["order"] = d["model"].map({k: i for i, k in enumerate(order)})
    d = d.sort_values("order")

    x = np.arange(len(d))
    width = 0.36
    plt.figure(figsize=(8, 5.2))
    plt.bar(x - width / 2, d["RMSE"], width=width, label="RMSE")
    plt.bar(x + width / 2, d["MAE"], width=width, label="MAE")
    plt.xticks(x, d["model"])
    plt.title("Forecast Error Comparison")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Construieste tabele si grafice finale pentru raportarea proiectului."
    )
    parser.add_argument("--metrics-naive", type=pathlib.Path, default=pathlib.Path("metrics_naive_overall.csv"))
    parser.add_argument("--metrics-arima", type=pathlib.Path, default=pathlib.Path("metrics_arima_overall.csv"))
    parser.add_argument("--metrics-xgboost", type=pathlib.Path, default=pathlib.Path("metrics_xgboost_overall.csv"))
    parser.add_argument("--perf-simple", type=pathlib.Path, default=pathlib.Path("portfolio_performance_simple.csv"))
    parser.add_argument("--perf-realistic-strict", type=pathlib.Path, default=pathlib.Path("portfolio_performance_realistic_strict.csv"))
    parser.add_argument("--mapping-quality-strict", type=pathlib.Path, default=pathlib.Path("portfolio_mapping_quality_strict.csv"))
    parser.add_argument("--greeks-summary-strict", type=pathlib.Path, default=pathlib.Path("portfolio_greeks_summary_strict.csv"))
    parser.add_argument("--hedged-performance", type=pathlib.Path, default=pathlib.Path("hedged_performance_summary.csv"))
    parser.add_argument("--hedge-effectiveness", type=pathlib.Path, default=pathlib.Path("hedge_effectiveness_summary.csv"))
    parser.add_argument("--hedged-daily", type=pathlib.Path, default=pathlib.Path("hedged_daily_pnl.csv"))
    parser.add_argument("--hedge-trades", type=pathlib.Path, default=pathlib.Path("hedge_trades_daily.csv"))
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("final_results"))
    args = parser.parse_args()

    ensure_exists(
        [
            args.metrics_naive,
            args.metrics_arima,
            args.metrics_xgboost,
            args.perf_simple,
            args.perf_realistic_strict,
            args.mapping_quality_strict,
            args.greeks_summary_strict,
            args.hedged_performance,
            args.hedge_effectiveness,
            args.hedged_daily,
        ]
    )

    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        print("Pas 1/5: construiesc tabelul comparativ forecast...")
        final_forecast = load_forecast_metrics(
            [args.metrics_naive, args.metrics_arima, args.metrics_xgboost]
        )
        final_forecast_path = args.outdir / "final_forecast_model_comparison.csv"
        final_forecast.to_csv(final_forecast_path, index=False)

        print("Pas 2/5: construiesc comparatia finala de backtest...")
        final_backtest = build_backtest_comparison(
            simple_path=args.perf_simple,
            realistic_strict_path=args.perf_realistic_strict,
            hedged_perf_path=args.hedged_performance,
        )
        final_backtest_path = args.outdir / "final_backtest_performance_comparison.csv"
        final_backtest.to_csv(final_backtest_path, index=False)

        print("Pas 3/5: construiesc sumarul final hedge + mapping...")
        final_hedge_eff = build_hedge_effectiveness_table(
            hedge_effectiveness_path=args.hedge_effectiveness,
            hedged_daily_path=args.hedged_daily,
            hedge_trades_path=args.hedge_trades if args.hedge_trades.exists() else None,
        )
        final_hedge_eff_path = args.outdir / "final_hedge_effectiveness.csv"
        final_hedge_eff.to_csv(final_hedge_eff_path, index=False)

        final_mapping = extract_mapping_quality(args.mapping_quality_strict)
        final_mapping_path = args.outdir / "final_mapping_quality.csv"
        final_mapping.to_csv(final_mapping_path, index=False)

        print("Pas 4/5: generez graficele pentru lucrare...")
        hedged_daily = pd.read_csv(args.hedged_daily)
        plot_cumulative_pnl(hedged_daily, args.outdir / "final_cumulative_pnl_comparison.png")
        plot_daily_pnl_distribution(hedged_daily, args.outdir / "final_daily_pnl_distribution.png")
        plot_forecast_error_bars(final_forecast, args.outdir / "final_forecast_error_bars.png")

        print("Pas 5/5: salvez un snapshot Greeks summary (copie utila in final_results)...")
        greeks_summary = pd.read_csv(args.greeks_summary_strict)
        greeks_summary.to_csv(args.outdir / "final_greeks_summary_strict_snapshot.csv", index=False)

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare la consolidare rezultate finale: {exc}")

    print("\n=== Final Outputs ===")
    print(final_forecast_path)
    print(final_backtest_path)
    print(final_hedge_eff_path)
    print(final_mapping_path)
    print(args.outdir / "final_cumulative_pnl_comparison.png")
    print(args.outdir / "final_daily_pnl_distribution.png")
    print(args.outdir / "final_forecast_error_bars.png")
    print(args.outdir / "final_greeks_summary_strict_snapshot.csv")


if __name__ == "__main__":
    main()

