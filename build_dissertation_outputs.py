import argparse
import math
import pathlib
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dissertation_plot_style import (
    MODEL_COLORS,
    PALETTE,
    SIGNAL_COLORS,
    STRATEGY_COLORS,
    apply_theme,
    save_figure,
)


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def count_csv_rows(path: pathlib.Path) -> int:
    total = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            total += chunk.count(b"\n")
    return max(total - 1, 0)


def count_valid_implied_vol(path: pathlib.Path, chunksize: int) -> tuple[int, int]:
    total = 0
    valid = 0
    for chunk in pd.read_csv(path, usecols=["implied_vol"], chunksize=chunksize):
        iv = pd.to_numeric(chunk["implied_vol"], errors="coerce")
        total += len(iv)
        valid += int((np.isfinite(iv) & (iv > 0)).sum())
    return total, valid


def save_csv_and_xlsx(
    df: pd.DataFrame,
    csv_path: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
    xlsx_sheet_name: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if xlsx_writer is not None:
        safe_sheet = xlsx_sheet_name[:31]
        df.to_excel(xlsx_writer, sheet_name=safe_sheet, index=False)


def build_data_cleaning_outputs(
    options_raw_path: pathlib.Path,
    options_clean_path: pathlib.Path,
    options_iv_path: pathlib.Path,
    iv_grid_long_path: pathlib.Path,
    iv_grid_day_stats_path: pathlib.Path,
    signals_path: pathlib.Path,
    positions_strict_path: pathlib.Path,
    tables_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
    chunksize: int,
) -> dict[str, float]:
    raw_rows = count_csv_rows(options_raw_path)
    clean_rows = count_csv_rows(options_clean_path)
    iv_rows = count_csv_rows(options_iv_path)
    iv_total_check, iv_valid_rows = count_valid_implied_vol(options_iv_path, chunksize=chunksize)
    grid_long_rows = count_csv_rows(iv_grid_long_path)

    day_stats = pd.read_csv(iv_grid_day_stats_path)
    day_stats["quote_date"] = parse_dates(day_stats["quote_date"])
    n_grid_days = int(day_stats["quote_date"].nunique())

    sig = pd.read_csv(signals_path, usecols=["selected", "signal"])
    sig["selected"] = pd.to_numeric(sig["selected"], errors="coerce")
    sig["signal"] = pd.to_numeric(sig["signal"], errors="coerce")
    selected_signals = int((sig["selected"] == 1).sum())

    pos = pd.read_csv(positions_strict_path, usecols=["trade_status", "weight"])
    pos["weight"] = pd.to_numeric(pos["weight"], errors="coerce")
    strict_traded = int(((pos["trade_status"] == "traded") & np.isfinite(pos["weight"]) & (pos["weight"] != 0)).sum())

    stages = pd.DataFrame(
        [
            {"stage": "raw_options_rows", "n_observations": raw_rows, "notes": "Initial raw options sample"},
            {"stage": "after_cleaning_filters", "n_observations": clean_rows, "notes": "After cleaning filters"},
            {"stage": "after_iv_pipeline_rows", "n_observations": iv_rows, "notes": "Rows in IV output file"},
            {"stage": "with_valid_implied_vol", "n_observations": iv_valid_rows, "notes": "implied_vol > 0 and finite"},
            {"stage": "grid_long_observations", "n_observations": grid_long_rows, "notes": "Standardized grid long format"},
            {"stage": "signals_selected", "n_observations": selected_signals, "notes": "Selected trading signals"},
            {
                "stage": "strict_realistic_positions_traded",
                "n_observations": strict_traded,
                "notes": "Strict exact-contract positions traded (weight != 0)",
            },
        ]
    )
    stages["dropped_vs_prev"] = stages["n_observations"].shift(1) - stages["n_observations"]
    stages["retention_vs_prev_pct"] = 100.0 * stages["n_observations"] / stages["n_observations"].shift(1)
    stages.loc[0, "retention_vs_prev_pct"] = 100.0
    stages["retention_vs_raw_pct"] = 100.0 * stages["n_observations"] / raw_rows

    summary = pd.DataFrame(
        [
            {"metric": "raw_rows", "value": float(raw_rows)},
            {"metric": "rows_after_cleaning", "value": float(clean_rows)},
            {"metric": "rows_in_iv_output", "value": float(iv_rows)},
            {"metric": "rows_with_valid_implied_vol", "value": float(iv_valid_rows)},
            {"metric": "grid_long_rows", "value": float(grid_long_rows)},
            {"metric": "grid_days", "value": float(n_grid_days)},
            {"metric": "selected_signals", "value": float(selected_signals)},
            {"metric": "strict_realistic_positions_traded", "value": float(strict_traded)},
            {"metric": "iv_total_rows_check", "value": float(iv_total_check)},
            {"metric": "valid_iv_rate_vs_iv_output_pct", "value": 100.0 * iv_valid_rows / iv_rows if iv_rows > 0 else np.nan},
        ]
    )

    save_csv_and_xlsx(summary, tables_dir / "data_cleaning_summary.csv", xlsx_writer, "data_cleaning_summary")
    save_csv_and_xlsx(stages, tables_dir / "data_cleaning_retention_by_step.csv", xlsx_writer, "data_cleaning_retention")

    # Figure: sample retention funnel/bar
    fig, ax = plt.subplots(figsize=(10, 5.6))
    plot_df = stages.iloc[:4].copy()  # main pipeline stages
    ax.bar(
        plot_df["stage"],
        plot_df["n_observations"],
        color=[PALETTE["navy"], PALETTE["teal"], PALETTE["gray"], PALETTE["orange"]],
        alpha=0.92,
    )
    ax.set_title("Data Cleaning and Sample Retention")
    ax.set_ylabel("Number of Observations")
    ax.set_xlabel("Pipeline Stage")
    ax.tick_params(axis="x", rotation=18)
    for i, v in enumerate(plot_df["n_observations"]):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    save_figure(fig, figures_dir / "figure_01_data_cleaning_retention.png", dpi=300, save_pdf=True)

    return {
        "raw_rows": float(raw_rows),
        "clean_rows": float(clean_rows),
        "iv_rows": float(iv_rows),
        "valid_iv_rows": float(iv_valid_rows),
    }


def load_grid_long_for_days(iv_grid_long_path: pathlib.Path, target_days: set[pd.Timestamp], chunksize: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    target_days = {pd.Timestamp(d).normalize() for d in target_days}
    for chunk in pd.read_csv(iv_grid_long_path, chunksize=chunksize):
        chunk["quote_date"] = parse_dates(chunk["quote_date"]).dt.normalize()
        chunk = chunk[chunk["quote_date"].isin(target_days)]
        if chunk.empty:
            continue
        chunk["log_moneyness"] = pd.to_numeric(chunk["log_moneyness"], errors="coerce")
        chunk["T"] = pd.to_numeric(chunk["T"], errors="coerce")
        chunk["iv_grid"] = pd.to_numeric(chunk["iv_grid"], errors="coerce")
        parts.append(chunk[["quote_date", "log_moneyness", "T", "iv_grid"]].copy())
    if not parts:
        return pd.DataFrame(columns=["quote_date", "log_moneyness", "T", "iv_grid"])
    return pd.concat(parts, ignore_index=True)


def plot_iv_surface_for_day(day_df: pd.DataFrame, day: pd.Timestamp, out_path: pathlib.Path, title_suffix: str) -> None:
    if day_df.empty:
        return
    pvt = day_df.pivot_table(index="T", columns="log_moneyness", values="iv_grid", aggfunc="mean")
    if pvt.empty:
        return
    x_vals = pvt.columns.to_numpy(dtype=float)
    t_vals = pvt.index.to_numpy(dtype=float)
    X, Y = np.meshgrid(x_vals, t_vals)
    Z = pvt.to_numpy(dtype=float)
    Zm = np.ma.masked_invalid(Z)

    fig = plt.figure(figsize=(9.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        Y,
        X,
        Zm,
        cmap="viridis",
        linewidth=0.15,
        edgecolor=(0, 0, 0, 0.15),
        antialiased=True,
        alpha=0.97,
    )
    ax.set_xlabel("T (years)")
    ax.set_ylabel("log(K/S)")
    ax.set_zlabel("Implied Volatility")
    ax.set_title(f"Representative IV Surface ({title_suffix}) | {day.strftime('%Y-%m-%d')}")
    ax.view_init(elev=24, azim=35)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08, label="IV")
    save_figure(fig, out_path, dpi=300, save_pdf=True)


def build_grid_outputs(
    iv_grid_map_path: pathlib.Path,
    iv_grid_wide_path: pathlib.Path,
    iv_grid_long_path: pathlib.Path,
    iv_grid_day_stats_path: pathlib.Path,
    tables_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    appendix_dir: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
    chunksize: int,
) -> dict[str, float]:
    gmap = pd.read_csv(iv_grid_map_path)
    wide = pd.read_csv(iv_grid_wide_path)
    day_stats = pd.read_csv(iv_grid_day_stats_path)
    day_stats["quote_date"] = parse_dates(day_stats["quote_date"]).dt.normalize()
    wide["quote_date"] = parse_dates(wide["quote_date"]).dt.normalize()

    x_nodes = int(gmap["x_idx"].nunique())
    t_nodes = int(gmap["t_idx"].nunique())
    total_nodes = int(len(gmap))
    grid_features = [c for c in wide.columns if c != "quote_date"]

    wide_cov = wide.copy()
    wide_cov["coverage_final"] = wide_cov[grid_features].notna().mean(axis=1)
    day_level_cov = wide_cov[["quote_date", "coverage_final"]].copy()
    day_level_cov = day_level_cov.merge(day_stats, on="quote_date", how="left")
    day_level_cov = day_level_cov.sort_values("quote_date").reset_index(drop=True)

    non_null_values = int(wide[grid_features].notna().sum().sum())
    total_values = int(wide.shape[0] * len(grid_features))
    overall_cov = non_null_values / total_values if total_values > 0 else np.nan

    grid_summary = pd.DataFrame(
        [
            {"metric": "x_nodes", "value": float(x_nodes)},
            {"metric": "t_nodes", "value": float(t_nodes)},
            {"metric": "total_grid_nodes", "value": float(total_nodes)},
            {"metric": "grid_days_total", "value": float(wide["quote_date"].nunique())},
            {"metric": "grid_values_total", "value": float(total_values)},
            {"metric": "grid_values_non_null", "value": float(non_null_values)},
            {"metric": "overall_grid_coverage", "value": float(overall_cov)},
            {"metric": "mean_daily_coverage_final", "value": float(day_level_cov["coverage_final"].mean())},
            {"metric": "median_daily_coverage_final", "value": float(day_level_cov["coverage_final"].median())},
            {"metric": "min_daily_coverage_final", "value": float(day_level_cov["coverage_final"].min())},
            {"metric": "max_daily_coverage_final", "value": float(day_level_cov["coverage_final"].max())},
            {
                "metric": "sample_start_date",
                "value": day_level_cov["quote_date"].min().strftime("%Y-%m-%d")
                if day_level_cov["quote_date"].notna().any()
                else "",
            },
            {
                "metric": "sample_end_date",
                "value": day_level_cov["quote_date"].max().strftime("%Y-%m-%d")
                if day_level_cov["quote_date"].notna().any()
                else "",
            },
        ]
    )

    coverage_summary = pd.DataFrame(
        [
            {"metric": "n_days", "value": float(len(day_level_cov))},
            {"metric": "coverage_mean", "value": float(day_level_cov["coverage_final"].mean())},
            {"metric": "coverage_std", "value": float(day_level_cov["coverage_final"].std(ddof=1))},
            {"metric": "coverage_p10", "value": float(day_level_cov["coverage_final"].quantile(0.10))},
            {"metric": "coverage_p25", "value": float(day_level_cov["coverage_final"].quantile(0.25))},
            {"metric": "coverage_median", "value": float(day_level_cov["coverage_final"].median())},
            {"metric": "coverage_p75", "value": float(day_level_cov["coverage_final"].quantile(0.75))},
            {"metric": "coverage_p90", "value": float(day_level_cov["coverage_final"].quantile(0.90))},
            {"metric": "coverage_ge_50pct_days", "value": float((day_level_cov["coverage_final"] >= 0.50).sum())},
            {"metric": "coverage_ge_70pct_days", "value": float((day_level_cov["coverage_final"] >= 0.70).sum())},
            {"metric": "coverage_ge_80pct_days", "value": float((day_level_cov["coverage_final"] >= 0.80).sum())},
        ]
    )

    iv_stats = []
    for chunk in pd.read_csv(iv_grid_long_path, usecols=["iv_grid"], chunksize=chunksize):
        v = pd.to_numeric(chunk["iv_grid"], errors="coerce").dropna()
        if not v.empty:
            iv_stats.append(v.to_numpy(dtype=float))
    if iv_stats:
        iv_all = np.concatenate(iv_stats)
        iv_desc = pd.DataFrame(
            [
                {"metric": "iv_grid_min", "value": float(np.min(iv_all))},
                {"metric": "iv_grid_median", "value": float(np.median(iv_all))},
                {"metric": "iv_grid_p95", "value": float(np.percentile(iv_all, 95))},
                {"metric": "iv_grid_p99", "value": float(np.percentile(iv_all, 99))},
                {"metric": "iv_grid_max", "value": float(np.max(iv_all))},
                {"metric": "iv_grid_mean", "value": float(np.mean(iv_all))},
                {"metric": "iv_grid_std", "value": float(np.std(iv_all, ddof=1)) if len(iv_all) > 1 else np.nan},
                {"metric": "iv_grid_non_null_count", "value": float(len(iv_all))},
            ]
        )
    else:
        iv_desc = pd.DataFrame([{"metric": "iv_grid_non_null_count", "value": 0.0}])

    save_csv_and_xlsx(grid_summary, tables_dir / "grid_dataset_summary.csv", xlsx_writer, "grid_dataset_summary")
    save_csv_and_xlsx(coverage_summary, tables_dir / "daily_grid_coverage_summary.csv", xlsx_writer, "grid_coverage_summary")
    save_csv_and_xlsx(iv_desc, tables_dir / "grid_iv_descriptive_summary.csv", xlsx_writer, "grid_iv_descriptive")
    save_csv_and_xlsx(day_level_cov, appendix_dir / "daily_grid_coverage_detail.csv", xlsx_writer, "appendix_grid_cov")

    # Figure: node coverage heatmap
    node_cov = wide[grid_features].notna().mean(axis=0).rename("coverage")
    cov_map = gmap.merge(node_cov.reset_index().rename(columns={"index": "feature"}), on="feature", how="left")
    mat = cov_map.pivot_table(index="t_idx", columns="x_idx", values="coverage", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_title("Grid Node Coverage Across Sample")
    ax.set_xlabel("Moneyness Node Index")
    ax.set_ylabel("Maturity Node Index")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Coverage Ratio")
    save_figure(fig, figures_dir / "figure_02_grid_node_coverage_heatmap.png", dpi=300, save_pdf=True)

    # Representative IV surfaces
    cov_non_na = day_level_cov.dropna(subset=["coverage_final"])
    if not cov_non_na.empty:
        day_high = cov_non_na.sort_values("coverage_final", ascending=False).iloc[0]["quote_date"]
        med_target = cov_non_na["coverage_final"].median()
        day_med = cov_non_na.iloc[(cov_non_na["coverage_final"] - med_target).abs().argmin()]["quote_date"]
        target_days = {pd.Timestamp(day_high).normalize(), pd.Timestamp(day_med).normalize()}
        long_sel = load_grid_long_for_days(iv_grid_long_path, target_days=target_days, chunksize=chunksize)

        for day, suffix, fig_name in [
            (pd.Timestamp(day_high).normalize(), "High Coverage Day", "figure_03_iv_surface_high_coverage.png"),
            (pd.Timestamp(day_med).normalize(), "Median Coverage Day", "figure_04_iv_surface_median_coverage.png"),
        ]:
            day_df = long_sel[long_sel["quote_date"] == day]
            plot_iv_surface_for_day(day_df, day, figures_dir / fig_name, suffix)

    return {
        "x_nodes": float(x_nodes),
        "t_nodes": float(t_nodes),
        "total_nodes": float(total_nodes),
        "grid_days": float(wide["quote_date"].nunique()),
        "overall_grid_coverage": float(overall_cov),
    }


def build_forecasting_outputs(
    metrics_naive_path: pathlib.Path,
    metrics_arima_path: pathlib.Path,
    metrics_xgb_path: pathlib.Path,
    metrics_naive_by_node_path: pathlib.Path,
    metrics_arima_by_node_path: pathlib.Path,
    metrics_xgb_by_node_path: pathlib.Path,
    forecast_naive_test_path: pathlib.Path,
    forecast_arima_test_path: pathlib.Path,
    forecast_xgb_test_path: pathlib.Path,
    tables_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    appendix_dir: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
) -> pd.DataFrame:
    def read_one(path: pathlib.Path, label: str) -> dict:
        df = pd.read_csv(path)
        r = df.iloc[0]
        return {
            "model": label,
            "RMSE": float(r["rmse"]),
            "MAE": float(r["mae"]),
            "n_predictions": int(r["n_predictions"]) if "n_predictions" in df.columns else np.nan,
            "n_days": int(r["n_days"]) if "n_days" in df.columns else np.nan,
            "n_nodes": int(r["n_nodes"]) if "n_nodes" in df.columns else np.nan,
        }

    comp = pd.DataFrame(
        [
            read_one(metrics_naive_path, "Naive"),
            read_one(metrics_arima_path, "ARIMA"),
            read_one(metrics_xgb_path, "XGBoost"),
        ]
    )
    comp = comp.sort_values("RMSE").reset_index(drop=True)

    rank = comp.copy()
    rank["rank_rmse"] = rank["RMSE"].rank(method="min")
    rank["rank_mae"] = rank["MAE"].rank(method="min")
    rank["rank_avg"] = (rank["rank_rmse"] + rank["rank_mae"]) / 2.0
    rank = rank.sort_values("rank_avg").reset_index(drop=True)

    save_csv_and_xlsx(comp, tables_dir / "final_forecast_model_comparison.csv", xlsx_writer, "forecast_comparison")
    save_csv_and_xlsx(rank, tables_dir / "forecast_model_ranking.csv", xlsx_writer, "forecast_ranking")

    # Appendix: by-node metrics
    for src, dst, sheet in [
        (metrics_naive_by_node_path, appendix_dir / "forecast_metrics_by_node_naive.csv", "app_node_naive"),
        (metrics_arima_by_node_path, appendix_dir / "forecast_metrics_by_node_arima.csv", "app_node_arima"),
        (metrics_xgb_by_node_path, appendix_dir / "forecast_metrics_by_node_xgboost.csv", "app_node_xgb"),
    ]:
        if src.exists():
            df = pd.read_csv(src)
            save_csv_and_xlsx(df, dst, xlsx_writer, sheet)

    # Figure: RMSE/MAE bars
    plot_df = comp.copy()
    order = ["Naive", "ARIMA", "XGBoost"]
    plot_df["order"] = plot_df["model"].map({k: i for i, k in enumerate(order)})
    plot_df = plot_df.sort_values("order")

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    x = np.arange(len(plot_df))
    w = 0.34
    ax.bar(x - w / 2, plot_df["RMSE"], width=w, label="RMSE", color=PALETTE["navy"])
    ax.bar(x + w / 2, plot_df["MAE"], width=w, label="MAE", color=PALETTE["teal"])
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"])
    ax.set_title("Forecast Models: RMSE and MAE")
    ax.set_ylabel("Error")
    ax.legend()
    save_figure(fig, figures_dir / "figure_05_forecast_rmse_mae_comparison.png", dpi=300, save_pdf=True)

    # Figure: actual vs forecast for representative nodes
    xgb = pd.read_csv(forecast_xgb_test_path, usecols=["quote_date", "node", "actual", "forecast"])
    arima = pd.read_csv(forecast_arima_test_path, usecols=["quote_date", "node", "forecast"])
    naive = pd.read_csv(forecast_naive_test_path, usecols=["quote_date", "node", "forecast"])

    for df in [xgb, arima, naive]:
        df["quote_date"] = parse_dates(df["quote_date"])

    top_nodes = xgb["node"].value_counts().head(2).index.tolist()
    for i, node in enumerate(top_nodes, start=1):
        base = xgb[xgb["node"] == node][["quote_date", "actual", "forecast"]].rename(columns={"forecast": "forecast_xgboost"})
        tmp_a = arima[arima["node"] == node][["quote_date", "forecast"]].rename(columns={"forecast": "forecast_arima"})
        tmp_n = naive[naive["node"] == node][["quote_date", "forecast"]].rename(columns={"forecast": "forecast_naive"})
        m = base.merge(tmp_a, on="quote_date", how="inner").merge(tmp_n, on="quote_date", how="inner")
        m = m.sort_values("quote_date").reset_index(drop=True)
        if m.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.plot(m["quote_date"], m["actual"], label="Actual", color=PALETTE["gray"], linewidth=2.1)
        ax.plot(m["quote_date"], m["forecast_naive"], label="Naive", color=MODEL_COLORS["Naive"])
        ax.plot(m["quote_date"], m["forecast_arima"], label="ARIMA", color=MODEL_COLORS["ARIMA"])
        ax.plot(m["quote_date"], m["forecast_xgboost"], label="XGBoost", color=MODEL_COLORS["XGBoost"])
        ax.set_title(f"Actual vs Forecast ({node})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Implied Volatility")
        ax.legend(ncol=2)
        save_figure(fig, figures_dir / f"figure_0{5+i}_forecast_actual_vs_pred_{node}.png", dpi=300, save_pdf=True)

    return comp


def build_signal_outputs(
    signals_path: pathlib.Path,
    signals_summary_path: pathlib.Path,
    tables_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
) -> None:
    sig = pd.read_csv(signals_path)
    ssum = pd.read_csv(signals_summary_path)
    sig["quote_date"] = parse_dates(sig["quote_date"])
    ssum["quote_date"] = parse_dates(ssum["quote_date"])

    sig["selected"] = pd.to_numeric(sig["selected"], errors="coerce")
    sig["signal_strength"] = pd.to_numeric(sig["signal_strength"], errors="coerce")
    sig["signal"] = pd.to_numeric(sig["signal"], errors="coerce")

    if "signal_direction" in sig.columns:
        dir_counts = sig["signal_direction"].fillna("UNKNOWN").value_counts()
        n_long = int(dir_counts.get("LONG_VOL", 0))
        n_short = int(dir_counts.get("SHORT_VOL", 0))
        n_flat = int(dir_counts.get("FLAT", 0))
    else:
        n_long = int((sig["signal"] == 1).sum())
        n_short = int((sig["signal"] == -1).sum())
        n_flat = int((sig["signal"] == 0).sum())

    total_signals = int(len(sig))
    n_selected = int((sig["selected"] == 1).sum())

    signals_summary = pd.DataFrame(
        [
            {"metric": "total_signal_observations", "value": float(total_signals)},
            {"metric": "selected_signals", "value": float(n_selected)},
            {"metric": "selected_ratio", "value": float(n_selected / total_signals) if total_signals > 0 else np.nan},
            {"metric": "long_signals_total", "value": float(n_long)},
            {"metric": "short_signals_total", "value": float(n_short)},
            {"metric": "flat_signals_total", "value": float(n_flat)},
            {"metric": "mean_abs_signal_strength_all", "value": float(sig["signal_strength"].abs().mean())},
            {
                "metric": "mean_abs_signal_strength_selected",
                "value": float(sig.loc[sig["selected"] == 1, "signal_strength"].abs().mean()),
            },
        ]
    )

    positions_selection_summary = pd.DataFrame(
        [
            {"metric": "n_days", "value": float(ssum["quote_date"].nunique())},
            {"metric": "avg_nodes_per_day", "value": float(ssum["n_nodes"].mean())},
            {"metric": "avg_selected_per_day", "value": float(ssum["n_selected"].mean())},
            {"metric": "median_selected_per_day", "value": float(ssum["n_selected"].median())},
            {"metric": "avg_long_per_day", "value": float(ssum["n_long"].mean())},
            {"metric": "avg_short_per_day", "value": float(ssum["n_short"].mean())},
            {"metric": "avg_flat_per_day", "value": float(ssum["n_flat"].mean())},
            {"metric": "avg_abs_strength_selected", "value": float(ssum["avg_abs_strength_selected"].mean())},
        ]
    )

    save_csv_and_xlsx(signals_summary, tables_dir / "signals_summary.csv", xlsx_writer, "signals_summary")
    save_csv_and_xlsx(
        positions_selection_summary,
        tables_dir / "positions_selection_summary.csv",
        xlsx_writer,
        "positions_selection_summary",
    )

    # Figure: signal strength distribution
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    all_vals = sig["signal_strength"].abs().dropna()
    sel_vals = sig.loc[sig["selected"] == 1, "signal_strength"].abs().dropna()
    if len(all_vals) > 0:
        bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 45)
        ax.hist(all_vals, bins=bins, alpha=0.45, label="All signals", color=PALETTE["gray"], density=True)
        if len(sel_vals) > 0:
            ax.hist(sel_vals, bins=bins, alpha=0.55, label="Selected signals", color=PALETTE["teal"], density=True)
    ax.set_title("Distribution of Absolute Signal Strength")
    ax.set_xlabel("|signal_strength|")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(fig, figures_dir / "figure_08_signal_strength_distribution.png", dpi=300, save_pdf=True)

    # Figure: long/short/flat counts by month
    month_df = ssum.copy()
    month_df["month"] = month_df["quote_date"].dt.to_period("M").astype(str)
    agg = month_df.groupby("month", as_index=False)[["n_long", "n_short", "n_flat"]].sum()
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x, agg["n_long"], color=SIGNAL_COLORS["LONG"], label="Long")
    ax.bar(x, agg["n_short"], bottom=agg["n_long"], color=SIGNAL_COLORS["SHORT"], label="Short")
    ax.bar(x, agg["n_flat"], bottom=agg["n_long"] + agg["n_short"], color=SIGNAL_COLORS["FLAT"], label="Flat")
    ticks = x[:: max(1, len(x) // 12)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(agg["month"].iloc[ticks], rotation=45, ha="right")
    ax.set_title("Signal Direction Counts by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend(ncol=3)
    save_figure(fig, figures_dir / "figure_09_signal_direction_stacked_counts.png", dpi=300, save_pdf=True)


def build_backtest_outputs(
    perf_simple_path: pathlib.Path,
    daily_simple_path: pathlib.Path,
    perf_realistic_path: pathlib.Path,
    daily_realistic_path: pathlib.Path,
    mapping_quality_path: pathlib.Path,
    tables_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
) -> None:
    perf_simple = pd.read_csv(perf_simple_path)
    perf_real = pd.read_csv(perf_realistic_path)
    daily_simple = pd.read_csv(daily_simple_path)
    daily_real = pd.read_csv(daily_realistic_path)

    save_csv_and_xlsx(perf_simple, tables_dir / "backtest_simple_summary.csv", xlsx_writer, "backtest_simple")
    save_csv_and_xlsx(perf_real, tables_dir / "realistic_backtest_summary.csv", xlsx_writer, "backtest_realistic")

    mapping = pd.read_csv(mapping_quality_path)
    if {"metric", "value", "section"}.issubset(mapping.columns):
        overall = mapping[mapping["section"] == "overall"].copy()
    else:
        overall = mapping.copy()
    save_csv_and_xlsx(overall, tables_dir / "final_mapping_quality.csv", xlsx_writer, "mapping_quality")

    for df in [daily_simple, daily_real]:
        df["quote_date"] = parse_dates(df["quote_date"])

    # Simple cumulative PnL
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(daily_simple["quote_date"], pd.to_numeric(daily_simple["cumulative_pnl"], errors="coerce"), color=STRATEGY_COLORS["Simple"])
    ax.set_title("Cumulative PnL - Simple IV Backtest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    save_figure(fig, figures_dir / "figure_10_simple_backtest_cumulative_pnl.png", dpi=300, save_pdf=True)

    # Simple daily pnl dist
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    vals = pd.to_numeric(daily_simple["daily_pnl"], errors="coerce").dropna()
    ax.hist(vals, bins=40, color=STRATEGY_COLORS["Simple"], alpha=0.75, density=True)
    ax.set_title("Daily PnL Distribution - Simple IV Backtest")
    ax.set_xlabel("Daily PnL")
    ax.set_ylabel("Density")
    save_figure(fig, figures_dir / "figure_11_simple_backtest_daily_pnl_distribution.png", dpi=300, save_pdf=True)

    # Realistic cumulative PnL
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(
        daily_real["quote_date"],
        pd.to_numeric(daily_real["cumulative_pnl"], errors="coerce"),
        color=STRATEGY_COLORS["Unhedged"],
    )
    ax.set_title("Cumulative PnL - Realistic Strict Unhedged")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    save_figure(fig, figures_dir / "figure_12_realistic_unhedged_cumulative_pnl.png", dpi=300, save_pdf=True)

    # Realistic daily pnl dist
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    vals = pd.to_numeric(daily_real["daily_pnl"], errors="coerce").dropna()
    ax.hist(vals, bins=40, color=STRATEGY_COLORS["Unhedged"], alpha=0.75, density=True)
    ax.set_title("Daily PnL Distribution - Realistic Strict Unhedged")
    ax.set_xlabel("Daily PnL")
    ax.set_ylabel("Density")
    save_figure(fig, figures_dir / "figure_13_realistic_unhedged_daily_pnl_distribution.png", dpi=300, save_pdf=True)


def build_greeks_outputs(
    greeks_summary_path: pathlib.Path,
    daily_greeks_path: pathlib.Path,
    tables_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    appendix_dir: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
) -> None:
    gsum = pd.read_csv(greeks_summary_path)
    gdaily = pd.read_csv(daily_greeks_path)
    gdaily["quote_date"] = parse_dates(gdaily["quote_date"])

    save_csv_and_xlsx(gsum, tables_dir / "greeks_summary.csv", xlsx_writer, "greeks_summary")
    save_csv_and_xlsx(gdaily, appendix_dir / "portfolio_daily_greeks_detail.csv", xlsx_writer, "appendix_daily_greeks")

    exposure_summary_rows = []
    for col in ["total_delta", "total_gamma", "total_vega", "total_theta", "gross_delta", "gross_gamma", "gross_vega", "gross_theta"]:
        vals = pd.to_numeric(gdaily[col], errors="coerce")
        exposure_summary_rows.extend(
            [
                {"metric": f"{col}_mean", "value": float(vals.mean())},
                {"metric": f"{col}_std", "value": float(vals.std(ddof=1))},
                {"metric": f"{col}_min", "value": float(vals.min())},
                {"metric": f"{col}_max", "value": float(vals.max())},
            ]
        )
    exposure_summary = pd.DataFrame(exposure_summary_rows)
    save_csv_and_xlsx(exposure_summary, tables_dir / "portfolio_exposures_summary.csv", xlsx_writer, "exposure_summary")

    # Time series exposures
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)
    axes[0].plot(gdaily["quote_date"], gdaily["total_delta"], color=PALETTE["navy"])
    axes[0].set_ylabel("Total Delta")
    axes[0].set_title("Daily Portfolio Exposures (Unhedged Realistic Strict)")

    axes[1].plot(gdaily["quote_date"], gdaily["total_gamma"], color=PALETTE["teal"])
    axes[1].set_ylabel("Total Gamma")

    axes[2].plot(gdaily["quote_date"], gdaily["total_vega"], color=PALETTE["burgundy"])
    axes[2].set_ylabel("Total Vega")
    axes[2].set_xlabel("Date")
    save_figure(fig, figures_dir / "figure_14_portfolio_exposures_timeseries.png", dpi=300, save_pdf=True)

    # Exposure distribution (box)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    box_data = [
        pd.to_numeric(gdaily["total_delta"], errors="coerce").dropna(),
        pd.to_numeric(gdaily["total_gamma"], errors="coerce").dropna(),
        pd.to_numeric(gdaily["total_vega"], errors="coerce").dropna(),
    ]
    ax.boxplot(
        box_data,
        tick_labels=["Delta", "Gamma", "Vega"],
        patch_artist=True,
        boxprops={"facecolor": PALETTE["light_gray"]},
    )
    ax.set_title("Distribution of Daily Portfolio Exposures")
    save_figure(fig, figures_dir / "figure_15_portfolio_exposures_boxplot.png", dpi=300, save_pdf=True)


def build_hedging_outputs(
    hedged_perf_path: pathlib.Path,
    hedge_eff_path: pathlib.Path,
    hedged_daily_path: pathlib.Path,
    hedge_trades_path: pathlib.Path,
    tables_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    appendix_dir: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf = pd.read_csv(hedged_perf_path)
    eff = pd.read_csv(hedge_eff_path)
    daily = pd.read_csv(hedged_daily_path)
    trades = pd.read_csv(hedge_trades_path)
    daily["quote_date"] = parse_dates(daily["quote_date"])
    trades["quote_date"] = parse_dates(trades["quote_date"])

    save_csv_and_xlsx(perf, tables_dir / "hedged_performance_summary_final.csv", xlsx_writer, "hedged_perf_only")
    save_csv_and_xlsx(eff, tables_dir / "final_hedge_effectiveness.csv", xlsx_writer, "hedge_effectiveness")
    save_csv_and_xlsx(daily, appendix_dir / "hedged_daily_pnl_detail.csv", xlsx_writer, "appendix_hedged_daily")
    save_csv_and_xlsx(trades, appendix_dir / "hedge_trades_daily_detail.csv", xlsx_writer, "appendix_hedge_trades")

    # Cumulative pnl comparison
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    d = daily.sort_values("quote_date").copy()
    d["cum_unhedged"] = pd.to_numeric(d["portfolio_daily_pnl_unhedged"], errors="coerce").fillna(0.0).cumsum()
    d["cum_delta"] = pd.to_numeric(d["daily_pnl_delta_hedged"], errors="coerce").fillna(0.0).cumsum()
    d["cum_delta_gamma"] = pd.to_numeric(d["daily_pnl_delta_gamma_hedged"], errors="coerce").fillna(0.0).cumsum()
    ax.plot(d["quote_date"], d["cum_unhedged"], label="Unhedged", color=STRATEGY_COLORS["Unhedged"])
    ax.plot(d["quote_date"], d["cum_delta"], label="Delta Hedged", color=STRATEGY_COLORS["Delta Hedged"])
    ax.plot(d["quote_date"], d["cum_delta_gamma"], label="Delta-Gamma Hedged", color=STRATEGY_COLORS["Delta-Gamma Hedged"])
    ax.set_title("Cumulative PnL Comparison: Unhedged vs Delta vs Delta-Gamma")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    ax.legend()
    save_figure(fig, figures_dir / "figure_16_hedging_cumulative_pnl_comparison.png", dpi=300, save_pdf=True)

    # Daily pnl distribution comparison
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    s1 = pd.to_numeric(daily["portfolio_daily_pnl_unhedged"], errors="coerce").dropna()
    s2 = pd.to_numeric(daily["daily_pnl_delta_hedged"], errors="coerce").dropna()
    s3 = pd.to_numeric(daily["daily_pnl_delta_gamma_hedged"], errors="coerce").dropna()
    if len(s1) > 0 and len(s2) > 0 and len(s3) > 0:
        all_vals = np.concatenate([s1.to_numpy(), s2.to_numpy(), s3.to_numpy()])
        bins = np.linspace(np.nanpercentile(all_vals, 1), np.nanpercentile(all_vals, 99), 45)
        ax.hist(s1, bins=bins, alpha=0.45, density=True, label="Unhedged", color=STRATEGY_COLORS["Unhedged"])
        ax.hist(s2, bins=bins, alpha=0.45, density=True, label="Delta Hedged", color=STRATEGY_COLORS["Delta Hedged"])
        ax.hist(
            s3,
            bins=bins,
            alpha=0.45,
            density=True,
            label="Delta-Gamma Hedged",
            color=STRATEGY_COLORS["Delta-Gamma Hedged"],
        )
    ax.set_title("Daily PnL Distribution: Hedging Comparison")
    ax.set_xlabel("Daily PnL")
    ax.set_ylabel("Density")
    ax.legend()
    save_figure(fig, figures_dir / "figure_17_hedging_daily_pnl_distribution.png", dpi=300, save_pdf=True)

    # Mean abs delta/gamma before/after
    eff_s = eff.set_index("metric")["value"]
    bars = pd.DataFrame(
        [
            {
                "group": "Delta",
                "before": float(eff_s.get("mean_abs_delta_before_hedge", np.nan)),
                "after_delta_only": float(eff_s.get("mean_abs_delta_after_delta_only", np.nan)),
                "after_delta_gamma": float(eff_s.get("mean_abs_delta_after_delta_gamma", np.nan)),
            },
            {
                "group": "Gamma",
                "before": float(eff_s.get("mean_abs_gamma_before_hedge", np.nan)),
                "after_delta_only": np.nan,
                "after_delta_gamma": float(eff_s.get("mean_abs_gamma_after_delta_gamma", np.nan)),
            },
        ]
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(bars))
    w = 0.25
    ax.bar(x - w, bars["before"], width=w, label="Before Hedge", color=PALETTE["navy"])
    ax.bar(x, bars["after_delta_only"], width=w, label="After Delta-Only", color=PALETTE["teal"])
    ax.bar(x + w, bars["after_delta_gamma"], width=w, label="After Delta-Gamma", color=PALETTE["burgundy"])
    ax.set_xticks(x)
    ax.set_xticklabels(bars["group"])
    ax.set_title("Mean Absolute Exposure Before/After Hedging")
    ax.set_ylabel("Mean Absolute Exposure")
    ax.legend()
    save_figure(fig, figures_dir / "figure_18_hedge_effectiveness_before_after.png", dpi=300, save_pdf=True)

    # Delta/Gamma before vs after (time series)
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=True)
    axes[0].plot(trades["quote_date"], trades["total_delta_before"], label="Delta Before", color=PALETTE["navy"])
    axes[0].plot(trades["quote_date"], trades["residual_delta_after_hedge"], label="Delta After DG", color=PALETTE["teal"])
    axes[0].set_ylabel("Delta")
    axes[0].set_title("Delta and Gamma Before vs After Delta-Gamma Hedging")
    axes[0].legend()
    axes[1].plot(trades["quote_date"], trades["total_gamma_before"], label="Gamma Before", color=PALETTE["navy"])
    axes[1].plot(trades["quote_date"], trades["residual_gamma_after_hedge"], label="Gamma After DG", color=PALETTE["teal"])
    axes[1].set_ylabel("Gamma")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    save_figure(fig, figures_dir / "figure_19_delta_gamma_before_after_timeseries.png", dpi=300, save_pdf=True)

    return perf, eff


def build_master_results_summary(
    forecast_comparison: pd.DataFrame,
    mapping_quality_table: pd.DataFrame,
    backtest_comp: pd.DataFrame,
    hedge_eff_table: pd.DataFrame,
    tables_dir: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
) -> pd.DataFrame:
    best = forecast_comparison.sort_values("RMSE").iloc[0]
    map_row = mapping_quality_table.iloc[0]

    bt = backtest_comp.set_index("strategy_key")
    hedge_eff = hedge_eff_table.iloc[0]

    row = {
        "forecast_model_selected": best["model"],
        "forecast_rmse": float(best["RMSE"]),
        "forecast_mae": float(best["MAE"]),
        "mapping_traded_rate_vs_selected": float(map_row.get("traded_rate_vs_selected", np.nan)),
        "mapping_avg_distance": float(map_row.get("avg_mapping_distance_mapped_only", np.nan)),
        "realistic_unhedged_total_pnl": float(bt.loc["unhedged", "total_pnl"]) if "unhedged" in bt.index else np.nan,
        "realistic_unhedged_sharpe": float(bt.loc["unhedged", "sharpe_ratio"]) if "unhedged" in bt.index else np.nan,
        "delta_hedged_total_pnl": float(bt.loc["delta_hedged", "total_pnl"]) if "delta_hedged" in bt.index else np.nan,
        "delta_hedged_sharpe": float(bt.loc["delta_hedged", "sharpe_ratio"]) if "delta_hedged" in bt.index else np.nan,
        "delta_gamma_hedged_total_pnl": float(bt.loc["delta_gamma_hedged", "total_pnl"]) if "delta_gamma_hedged" in bt.index else np.nan,
        "delta_gamma_hedged_sharpe": float(bt.loc["delta_gamma_hedged", "sharpe_ratio"]) if "delta_gamma_hedged" in bt.index else np.nan,
        "mean_abs_delta_before": float(hedge_eff.get("mean_abs_delta_before_hedge", np.nan)),
        "mean_abs_delta_after_delta_only": float(hedge_eff.get("mean_abs_delta_after_delta_only", np.nan)),
        "mean_abs_gamma_before": float(hedge_eff.get("mean_abs_gamma_before_hedge", np.nan)),
        "mean_abs_gamma_after_delta_gamma": float(hedge_eff.get("mean_abs_gamma_after_delta_gamma", np.nan)),
        "pct_days_delta_only_hedged": float(hedge_eff.get("pct_days_delta_only_hedged", np.nan)),
        "pct_days_delta_gamma_hedged": float(hedge_eff.get("pct_days_delta_gamma_hedged", np.nan)),
    }
    out = pd.DataFrame([row])
    save_csv_and_xlsx(out, tables_dir / "thesis_main_results_summary.csv", xlsx_writer, "master_summary")
    return out


def build_final_backtest_performance_comparison(
    perf_simple_path: pathlib.Path,
    perf_realistic_path: pathlib.Path,
    hedged_perf: pd.DataFrame,
    tables_dir: pathlib.Path,
    xlsx_writer: pd.ExcelWriter | None,
) -> pd.DataFrame:
    ps = pd.read_csv(perf_simple_path).iloc[0]
    pr = pd.read_csv(perf_realistic_path).iloc[0]
    hp = hedged_perf.copy()

    rows = [
        {
            "strategy_label": "simple_synthetic_iv_backtest",
            "strategy_key": "simple_synthetic",
            "total_pnl": float(ps["total_pnl"]),
            "mean_daily_pnl": float(ps["mean_daily_pnl"]),
            "daily_volatility": float(ps["daily_volatility"]),
            "sharpe_ratio": float(ps["sharpe_ratio"]),
            "max_drawdown": float(ps["max_drawdown"]),
            "num_days": int(ps["num_days_total"]) if "num_days_total" in ps.index else np.nan,
            "num_days_hedged": np.nan,
        },
        {
            "strategy_label": "realistic_unhedged_option_portfolio",
            "strategy_key": "unhedged",
            "total_pnl": float(pr["total_pnl"]),
            "mean_daily_pnl": float(pr["mean_daily_pnl"]),
            "daily_volatility": float(pr["daily_volatility"]),
            "sharpe_ratio": float(pr["sharpe_ratio"]),
            "max_drawdown": float(pr["max_drawdown"]),
            "num_days": int(pr["num_days_traded"]) if "num_days_traded" in pr.index else np.nan,
            "num_days_hedged": np.nan,
        },
    ]

    for skey, slabel in [("delta_hedged", "delta_hedged_portfolio"), ("delta_gamma_hedged", "delta_gamma_hedged_portfolio")]:
        r = hp[hp["strategy"] == skey]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append(
            {
                "strategy_label": slabel,
                "strategy_key": skey,
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
    order = ["simple_synthetic", "unhedged", "delta_hedged", "delta_gamma_hedged"]
    out["order"] = out["strategy_key"].map({k: i for i, k in enumerate(order)})
    out = out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    save_csv_and_xlsx(out, tables_dir / "final_backtest_performance_comparison.csv", xlsx_writer, "backtest_comp")
    return out


def write_manifest(
    out_root: pathlib.Path,
    table_files: list[tuple[str, str, str]],
    figure_files: list[tuple[str, str, str]],
) -> None:
    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    readme_path = manifest_dir / "README_dissertation_outputs.md"

    lines = []
    lines.append("# Dissertation Outputs Manifest")
    lines.append("")
    lines.append("This folder contains final tables and figures prepared for dissertation reporting.")
    lines.append("")
    lines.append("## Tables")
    lines.append("")
    for fname, desc, section in table_files:
        lines.append(f"- `{fname}`: {desc} (Suggested section: **{section}**)")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fname, desc, section in figure_files:
        lines.append(f"- `{fname}`: {desc} (Suggested section: **{section}**)")
    lines.append("")
    lines.append("## Folder Guide")
    lines.append("")
    lines.append("- `tables/`: final CSV tables for main text.")
    lines.append("- `figures/`: final PNG/PDF figures for main text.")
    lines.append("- `appendix/`: detailed supporting tables for appendix.")
    lines.append("- `manifests/`: manifest and documentation files.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Methodology and model outputs are not recomputed in this manifest step.")
    lines.append("- Files are consolidated from existing pipeline outputs.")
    lines.append("- Figure style is standardized via `dissertation_plot_style.py`.")
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build final dissertation-ready tables and figures from existing outputs.")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("dissertation_outputs"))
    parser.add_argument("--chunksize", type=int, default=200000, help="Chunk size for large CSV scans.")
    args = parser.parse_args()

    root = args.outdir
    tables_dir = root / "tables"
    figures_dir = root / "figures"
    appendix_dir = root / "appendix"
    manifests_dir = root / "manifests"
    for d in [tables_dir, figures_dir, appendix_dir, manifests_dir]:
        d.mkdir(parents=True, exist_ok=True)

    apply_theme()

    # Input paths
    p = {
        "options_raw": pathlib.Path("options_eod_all.csv"),
        "options_clean": pathlib.Path("options_eod_all_clean.csv"),
        "options_iv": pathlib.Path("options_eod_all_with_iv.csv"),
        "iv_grid_long": pathlib.Path("iv_grid_long.csv"),
        "iv_grid_wide": pathlib.Path("iv_grid_wide.csv"),
        "iv_grid_map": pathlib.Path("iv_grid_map.csv"),
        "iv_grid_day_stats": pathlib.Path("iv_grid_day_stats.csv"),
        "metrics_naive_overall": pathlib.Path("metrics_naive_overall.csv"),
        "metrics_arima_overall": pathlib.Path("metrics_arima_overall.csv"),
        "metrics_xgb_overall": pathlib.Path("metrics_xgboost_overall.csv"),
        "metrics_naive_by_node": pathlib.Path("metrics_naive_by_node.csv"),
        "metrics_arima_by_node": pathlib.Path("metrics_arima_by_node.csv"),
        "metrics_xgb_by_node": pathlib.Path("metrics_xgboost_by_node.csv"),
        "forecast_naive_test": pathlib.Path("forecast_naive_test.csv"),
        "forecast_arima_test": pathlib.Path("forecast_arima_test.csv"),
        "forecast_xgb_test": pathlib.Path("forecast_xgboost_test.csv"),
        "signals": pathlib.Path("signals_xgboost.csv"),
        "signals_summary": pathlib.Path("signals_xgboost_summary.csv"),
        "perf_simple": pathlib.Path("portfolio_performance_simple.csv"),
        "daily_simple": pathlib.Path("portfolio_daily_pnl_simple.csv"),
        "perf_realistic_strict": pathlib.Path("portfolio_performance_realistic_strict.csv"),
        "daily_realistic_strict": pathlib.Path("portfolio_daily_pnl_realistic_strict.csv"),
        "mapping_quality_strict": pathlib.Path("portfolio_mapping_quality_strict.csv"),
        "greeks_summary_strict": pathlib.Path("portfolio_greeks_summary_strict.csv"),
        "daily_greeks_strict": pathlib.Path("portfolio_daily_greeks_strict.csv"),
        "positions_strict": pathlib.Path("portfolio_positions_realistic_strict.csv"),
        "hedged_perf": pathlib.Path("hedged_performance_summary.csv"),
        "hedge_eff": pathlib.Path("hedge_effectiveness_summary.csv"),
        "hedged_daily": pathlib.Path("hedged_daily_pnl.csv"),
        "hedge_trades": pathlib.Path("hedge_trades_daily.csv"),
    }

    missing = [str(path) for path in p.values() if not path.exists()]
    if missing:
        sys.exit("Lipsesc fisiere de input necesare:\n- " + "\n- ".join(missing))

    xlsx_path = tables_dir / "dissertation_tables.xlsx"
    try:
        xlsx_writer = pd.ExcelWriter(xlsx_path, engine="openpyxl")
    except Exception:
        xlsx_writer = None

    try:
        print("A) Building data and cleaning outputs...")
        build_data_cleaning_outputs(
            options_raw_path=p["options_raw"],
            options_clean_path=p["options_clean"],
            options_iv_path=p["options_iv"],
            iv_grid_long_path=p["iv_grid_long"],
            iv_grid_day_stats_path=p["iv_grid_day_stats"],
            signals_path=p["signals"],
            positions_strict_path=p["positions_strict"],
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            xlsx_writer=xlsx_writer,
            chunksize=args.chunksize,
        )

        print("B) Building grid outputs...")
        build_grid_outputs(
            iv_grid_map_path=p["iv_grid_map"],
            iv_grid_wide_path=p["iv_grid_wide"],
            iv_grid_long_path=p["iv_grid_long"],
            iv_grid_day_stats_path=p["iv_grid_day_stats"],
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            appendix_dir=appendix_dir,
            xlsx_writer=xlsx_writer,
            chunksize=args.chunksize,
        )

        print("C) Building forecasting outputs...")
        forecast_comp = build_forecasting_outputs(
            metrics_naive_path=p["metrics_naive_overall"],
            metrics_arima_path=p["metrics_arima_overall"],
            metrics_xgb_path=p["metrics_xgb_overall"],
            metrics_naive_by_node_path=p["metrics_naive_by_node"],
            metrics_arima_by_node_path=p["metrics_arima_by_node"],
            metrics_xgb_by_node_path=p["metrics_xgb_by_node"],
            forecast_naive_test_path=p["forecast_naive_test"],
            forecast_arima_test_path=p["forecast_arima_test"],
            forecast_xgb_test_path=p["forecast_xgb_test"],
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            appendix_dir=appendix_dir,
            xlsx_writer=xlsx_writer,
        )

        print("D) Building signal outputs...")
        build_signal_outputs(
            signals_path=p["signals"],
            signals_summary_path=p["signals_summary"],
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            xlsx_writer=xlsx_writer,
        )

        print("E/F) Building backtest outputs...")
        build_backtest_outputs(
            perf_simple_path=p["perf_simple"],
            daily_simple_path=p["daily_simple"],
            perf_realistic_path=p["perf_realistic_strict"],
            daily_realistic_path=p["daily_realistic_strict"],
            mapping_quality_path=p["mapping_quality_strict"],
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            xlsx_writer=xlsx_writer,
        )

        print("G) Building Greeks/exposures outputs...")
        build_greeks_outputs(
            greeks_summary_path=p["greeks_summary_strict"],
            daily_greeks_path=p["daily_greeks_strict"],
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            appendix_dir=appendix_dir,
            xlsx_writer=xlsx_writer,
        )

        print("H) Building hedging outputs...")
        hedged_perf, hedge_eff = build_hedging_outputs(
            hedged_perf_path=p["hedged_perf"],
            hedge_eff_path=p["hedge_eff"],
            hedged_daily_path=p["hedged_daily"],
            hedge_trades_path=p["hedge_trades"],
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            appendix_dir=appendix_dir,
            xlsx_writer=xlsx_writer,
        )

        backtest_comp = build_final_backtest_performance_comparison(
            perf_simple_path=p["perf_simple"],
            perf_realistic_path=p["perf_realistic_strict"],
            hedged_perf=hedged_perf,
            tables_dir=tables_dir,
            xlsx_writer=xlsx_writer,
        )

        print("I) Building master summary...")
        mapping_quality_table = pd.read_csv(tables_dir / "final_mapping_quality.csv")
        hedge_eff_table = pd.read_csv(tables_dir / "final_hedge_effectiveness.csv")
        build_master_results_summary(
            forecast_comparison=forecast_comp,
            mapping_quality_table=mapping_quality_table,
            backtest_comp=backtest_comp,
            hedge_eff_table=hedge_eff_table,
            tables_dir=tables_dir,
            xlsx_writer=xlsx_writer,
        )

        # Manifest
        table_files = [
            ("tables/data_cleaning_summary.csv", "Sample sizes along data cleaning and IV steps", "Methodology"),
            ("tables/data_cleaning_retention_by_step.csv", "Retention by major pipeline stage", "Methodology"),
            ("tables/grid_dataset_summary.csv", "Grid design and final dataset structure", "Methodology"),
            ("tables/daily_grid_coverage_summary.csv", "Daily coverage statistics on grid", "Methodology"),
            ("tables/final_forecast_model_comparison.csv", "Naive vs ARIMA vs XGBoost forecast metrics", "Results"),
            ("tables/forecast_model_ranking.csv", "Model ranking by RMSE/MAE", "Results"),
            ("tables/signals_summary.csv", "Economic signal generation overview", "Methodology"),
            ("tables/positions_selection_summary.csv", "Selected positions statistics", "Methodology"),
            ("tables/backtest_simple_summary.csv", "Simple IV-space backtest performance", "Results"),
            ("tables/final_mapping_quality.csv", "Strict realistic mapping quality", "Results"),
            ("tables/realistic_backtest_summary.csv", "Realistic strict unhedged performance", "Results"),
            ("tables/greeks_summary.csv", "Greeks computation summary", "Methodology"),
            ("tables/portfolio_exposures_summary.csv", "Portfolio exposure descriptive statistics", "Results"),
            ("tables/final_backtest_performance_comparison.csv", "Unhedged vs delta vs delta-gamma performance", "Results"),
            ("tables/final_hedge_effectiveness.csv", "Hedge effectiveness metrics", "Results"),
            ("tables/thesis_main_results_summary.csv", "Master summary table for thesis main text", "Results"),
        ]
        figure_files = [
            ("figures/figure_01_data_cleaning_retention.png", "Data cleaning and sample retention funnel/bar", "Methodology"),
            ("figures/figure_02_grid_node_coverage_heatmap.png", "Grid node coverage heatmap", "Methodology"),
            ("figures/figure_03_iv_surface_high_coverage.png", "Representative IV surface (high coverage day)", "Results"),
            ("figures/figure_04_iv_surface_median_coverage.png", "Representative IV surface (median coverage day)", "Results"),
            ("figures/figure_05_forecast_rmse_mae_comparison.png", "RMSE/MAE comparison across forecast models", "Results"),
            ("figures/figure_06_forecast_actual_vs_pred_*.png", "Actual vs forecast for representative node(s)", "Appendix"),
            ("figures/figure_08_signal_strength_distribution.png", "Signal strength distribution", "Methodology"),
            ("figures/figure_09_signal_direction_stacked_counts.png", "Long/Short/Flat signal counts over time", "Results"),
            ("figures/figure_10_simple_backtest_cumulative_pnl.png", "Cumulative PnL (simple backtest)", "Results"),
            ("figures/figure_11_simple_backtest_daily_pnl_distribution.png", "Daily PnL distribution (simple backtest)", "Appendix"),
            ("figures/figure_12_realistic_unhedged_cumulative_pnl.png", "Cumulative PnL (realistic strict unhedged)", "Results"),
            ("figures/figure_13_realistic_unhedged_daily_pnl_distribution.png", "Daily PnL distribution (realistic strict)", "Appendix"),
            ("figures/figure_14_portfolio_exposures_timeseries.png", "Delta/Gamma/Vega time series (unhedged)", "Results"),
            ("figures/figure_15_portfolio_exposures_boxplot.png", "Exposure distributions", "Appendix"),
            ("figures/figure_16_hedging_cumulative_pnl_comparison.png", "Cumulative PnL: unhedged vs hedged", "Results"),
            ("figures/figure_17_hedging_daily_pnl_distribution.png", "Daily PnL distribution under hedging", "Appendix"),
            ("figures/figure_18_hedge_effectiveness_before_after.png", "Mean abs delta/gamma before-after hedging", "Results"),
            ("figures/figure_19_delta_gamma_before_after_timeseries.png", "Delta/Gamma before vs after DG hedging", "Appendix"),
        ]
        write_manifest(root, table_files, figure_files)

    except Exception as exc:  # pylint: disable=broad-except
        if xlsx_writer is not None:
            try:
                xlsx_writer.close()
            except Exception:
                pass
        sys.exit(f"Eroare la build_dissertation_outputs: {exc}")

    if xlsx_writer is not None:
        xlsx_writer.close()

    print("\nBuild completed.")
    print(f"Outputs root: {root}")
    print(f"Tables: {tables_dir}")
    print(f"Figures: {figures_dir}")
    print(f"Appendix: {appendix_dir}")
    print(f"Manifest: {manifests_dir / 'README_dissertation_outputs.md'}")


if __name__ == "__main__":
    main()
