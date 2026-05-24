"""Generează versiunile în limba română pentru figurile principale din disertație."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from build_methodological_workflow_figure_ro import build_figure as build_workflow_ro
from dissertation_plot_style import MODEL_COLORS, PALETTE, STRATEGY_COLORS, apply_theme, save_figure


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def fmt_thousands(x: float, _: int) -> str:
    if not np.isfinite(x):
        return ""
    return f"{int(round(x)):,}".replace(",", ".")


def _require(path: pathlib.Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Lipsește fișierul necesar: {path}")


def figure_01_data_cleaning(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = base_dir / "dissertation_outputs/tables/data_cleaning_retention_by_step.csv"
    _require(src)
    df = pd.read_csv(src)

    stage_col = "stage" if "stage" in df.columns else "step"
    value_col = "n_observations" if "n_observations" in df.columns else "n_rows"
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    mapping = {
        "raw_options_rows": "Observații brute",
        "after_cleaning_filters": "După filtrele de curățare",
        "after_iv_pipeline_rows": "După calculul IV",
        "with_valid_implied_vol": "Cu IV validă",
    }
    keep = [k for k in mapping if k in set(df[stage_col].astype(str))]
    plot_df = df[df[stage_col].isin(keep)].copy()
    plot_df["label_ro"] = plot_df[stage_col].map(mapping)
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        raise ValueError("Nu există valori numerice pentru figura 01.")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(
        plot_df["label_ro"],
        plot_df[value_col],
        color=[PALETTE["navy"], PALETTE["teal"], PALETTE["gray"], PALETTE["orange"]][: len(plot_df)],
        alpha=0.92,
    )
    ax.set_title("Curățarea Datelor și Retenția Eșantionului")
    ax.set_ylabel("Număr de observații")
    ax.set_xlabel("Etapa procesului metodologic")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_thousands))
    ax.tick_params(axis="x", rotation=15)
    ax.margins(y=0.10)
    for i, v in enumerate(plot_df[value_col]):
        ax.text(i, v, fmt_thousands(v, 0), ha="center", va="bottom", fontsize=9)

    save_figure(fig, out_dir / "figure_01_retentia_esantionului.png", dpi=300, save_pdf=True)


def figure_02_grid_coverage_heatmap(base_dir: pathlib.Path, out_dir: pathlib.Path, chunksize: int = 300_000) -> None:
    src = base_dir / "iv_grid_long.csv"
    _require(src)

    day_set: set[pd.Timestamp] = set()
    node_set: set[tuple[float, float]] = set()
    valid_counts: pd.Series | None = None

    usecols = ["quote_date", "log_moneyness", "T", "iv_grid"]
    for chunk in pd.read_csv(src, usecols=usecols, chunksize=chunksize):
        qd = parse_dates(chunk["quote_date"]).dt.normalize()
        lm = pd.to_numeric(chunk["log_moneyness"], errors="coerce")
        ttm = pd.to_numeric(chunk["T"], errors="coerce")
        iv = pd.to_numeric(chunk["iv_grid"], errors="coerce")

        qd_valid = qd.dropna().unique()
        for d in qd_valid:
            day_set.add(pd.Timestamp(d))

        nodes_chunk = pd.DataFrame({"log_moneyness": lm, "T": ttm}).dropna().drop_duplicates()
        for row in nodes_chunk.itertuples(index=False):
            node_set.add((float(row.log_moneyness), float(row.T)))

        valid_mask = lm.notna() & ttm.notna() & np.isfinite(iv)
        if valid_mask.any():
            vc = (
                pd.DataFrame({"log_moneyness": lm[valid_mask], "T": ttm[valid_mask]})
                .groupby(["log_moneyness", "T"], observed=True)
                .size()
            )
            valid_counts = vc if valid_counts is None else valid_counts.add(vc, fill_value=0)

    n_days = len(day_set)
    if n_days == 0:
        raise ValueError("Nu au fost identificate zile valide în iv_grid_long.csv.")

    if not node_set:
        raise ValueError("Nu au fost identificate noduri valide în iv_grid_long.csv.")

    nodes_df = pd.DataFrame(sorted(node_set), columns=["log_moneyness", "T"])
    if valid_counts is None:
        cov_df = nodes_df.copy()
        cov_df["coverage"] = 0.0
    else:
        cov_count_df = valid_counts.rename("valid_days").reset_index()
        cov_df = nodes_df.merge(cov_count_df, on=["log_moneyness", "T"], how="left")
        cov_df["valid_days"] = pd.to_numeric(cov_df["valid_days"], errors="coerce").fillna(0.0)
        cov_df["coverage"] = cov_df["valid_days"] / float(n_days)

    pvt = cov_df.pivot_table(index="log_moneyness", columns="T", values="coverage", aggfunc="mean")
    pvt = pvt.sort_index(axis=0).sort_index(axis=1)
    if pvt.empty:
        raise ValueError("Heatmap-ul de acoperire nu poate fi construit (pivot gol).")

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    arr = pvt.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap="viridis", aspect="auto", origin="lower", vmin=0.0, vmax=1.0)

    x_vals = pvt.columns.to_numpy(dtype=float)
    y_vals = pvt.index.to_numpy(dtype=float)
    xt = np.linspace(0, len(x_vals) - 1, min(8, len(x_vals))).round().astype(int)
    yt = np.linspace(0, len(y_vals) - 1, min(8, len(y_vals))).round().astype(int)
    xt = np.unique(xt)
    yt = np.unique(yt)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{x_vals[i]:.2f}" for i in xt])
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y_vals[i]:.2f}" for i in yt])

    ax.set_title("Acoperirea Nodurilor Pe Grila Standardizată A Suprafeței IV")
    ax.set_xlabel("Maturitate")
    ax.set_ylabel("Log-moneyness")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Grad de acoperire")

    save_figure(fig, out_dir / "figure_02_grid_node_coverage_heatmap.png", dpi=300, save_pdf=True)


def _load_grid_day(iv_grid_long_path: pathlib.Path, target_day: pd.Timestamp, chunksize: int = 400_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    target_day = pd.Timestamp(target_day).normalize()
    usecols = ["quote_date", "log_moneyness", "T", "iv_grid"]
    for chunk in pd.read_csv(iv_grid_long_path, usecols=usecols, chunksize=chunksize):
        qd = parse_dates(chunk["quote_date"]).dt.normalize()
        mask = qd.eq(target_day)
        if not mask.any():
            continue
        sub = chunk.loc[mask, usecols].copy()
        sub["quote_date"] = qd.loc[mask]
        sub["log_moneyness"] = pd.to_numeric(sub["log_moneyness"], errors="coerce")
        sub["T"] = pd.to_numeric(sub["T"], errors="coerce")
        sub["iv_grid"] = pd.to_numeric(sub["iv_grid"], errors="coerce")
        parts.append(sub.dropna(subset=["log_moneyness", "T", "iv_grid"]))
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def figure_03_iv_surface(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    day_stats_path = base_dir / "iv_grid_day_stats.csv"
    iv_long_path = base_dir / "iv_grid_long.csv"
    _require(day_stats_path)
    _require(iv_long_path)

    day_stats = pd.read_csv(day_stats_path)
    day_stats["quote_date"] = parse_dates(day_stats["quote_date"]).dt.normalize()
    day_stats["coverage"] = pd.to_numeric(day_stats["coverage"], errors="coerce")
    day_stats = day_stats.dropna(subset=["quote_date", "coverage"])
    if day_stats.empty:
        raise ValueError("Nu există date de acoperire pentru figura 03.")

    high_day = day_stats.sort_values("coverage", ascending=False).iloc[0]["quote_date"]
    day_df = _load_grid_day(iv_long_path, high_day)
    if day_df.empty:
        raise ValueError("Nu am găsit puncte IV pentru ziua cu acoperire ridicată.")

    pvt = day_df.pivot_table(index="T", columns="log_moneyness", values="iv_grid", aggfunc="mean")
    x_vals = pvt.columns.to_numpy(dtype=float)
    t_vals = pvt.index.to_numpy(dtype=float)
    X, Y = np.meshgrid(x_vals, t_vals)
    Z = np.ma.masked_invalid(pvt.to_numpy(dtype=float))

    fig = plt.figure(figsize=(9.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        Y,
        X,
        Z,
        cmap="viridis",
        linewidth=0.15,
        edgecolor=(0, 0, 0, 0.15),
        antialiased=True,
        alpha=0.97,
    )
    ax.set_title("Suprafața Volatilității Implicite – Zi Cu Acoperire Ridicată")
    ax.set_xlabel("Maturitate")
    ax.set_ylabel("Log-moneyness")
    ax.set_zlabel("Volatilitate implicită")
    ax.view_init(elev=24, azim=35)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label("Volatilitate implicită")

    save_figure(fig, out_dir / "figure_03_suprafata_iv_acoperire_ridicata.png", dpi=300, save_pdf=True)


def figure_05_forecast_comparison(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = base_dir / "dissertation_outputs/tables/final_forecast_model_comparison.csv"
    _require(src)
    df = pd.read_csv(src)
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")
    df = df.dropna(subset=["RMSE", "MAE"])
    if df.empty:
        raise ValueError("Nu există valori RMSE/MAE pentru figura 05.")

    order = ["Naive", "ARIMA", "XGBoost"]
    df["order"] = df["model"].map({m: i for i, m in enumerate(order)})
    df = df.sort_values("order")

    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    x = np.arange(len(df))
    w = 0.34
    ax.bar(x - w / 2, df["RMSE"], width=w, label="RMSE", color=PALETTE["navy"])
    ax.bar(x + w / 2, df["MAE"], width=w, label="MAE", color=PALETTE["teal"])
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_title("Compararea Erorilor De Prognoză")
    ax.set_ylabel("Eroare")
    ax.legend()

    save_figure(fig, out_dir / "figure_05_comparatie_erori_forecast.png", dpi=300, save_pdf=True)


def figure_08_signal_strength(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = base_dir / "signals_xgboost.csv"
    _require(src)
    sig = pd.read_csv(src, usecols=["signal_strength", "selected"])
    sig["signal_strength"] = pd.to_numeric(sig["signal_strength"], errors="coerce")
    sig["selected"] = pd.to_numeric(sig["selected"], errors="coerce")

    all_vals = sig["signal_strength"].abs().dropna()
    sel_vals = sig.loc[sig["selected"] == 1, "signal_strength"].abs().dropna()
    if all_vals.empty:
        raise ValueError("Nu există valori de intensitate a semnalului.")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 45)
    ax.hist(all_vals, bins=bins, alpha=0.45, label="Toate semnalele", color=PALETTE["gray"], density=True)
    if not sel_vals.empty:
        ax.hist(sel_vals, bins=bins, alpha=0.55, label="Semnale selectate", color=PALETTE["teal"], density=True)
    ax.set_title("Distribuția Intensității Semnalului")
    ax.set_xlabel("Intensitatea semnalului")
    ax.set_ylabel("Densitate")
    ax.legend()

    save_figure(fig, out_dir / "figure_08_distributia_intensitatii_semnalului.png", dpi=300, save_pdf=True)


def figure_10_simple_cum_pnl(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = base_dir / "portfolio_daily_pnl_simple.csv"
    _require(src)
    df = pd.read_csv(src)
    df["quote_date"] = parse_dates(df["quote_date"])
    df["cumulative_pnl"] = pd.to_numeric(df["cumulative_pnl"], errors="coerce")
    df = df.dropna(subset=["quote_date", "cumulative_pnl"]).sort_values("quote_date")
    if df.empty:
        raise ValueError("Nu există date pentru figura 10.")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(df["quote_date"], df["cumulative_pnl"], color=STRATEGY_COLORS["Simple"], label="Backtest sintetic IV")
    ax.set_title("PnL Cumulat Al Backtestului Sintetic În Spațiul IV")
    ax.set_xlabel("Dată")
    ax.set_ylabel("PnL cumulat")
    ax.legend()
    save_figure(fig, out_dir / "figure_10_pnl_cumulat_backtest_sintetic_iv.png", dpi=300, save_pdf=True)


def figure_11_simple_daily_pnl_distribution(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = base_dir / "portfolio_daily_pnl_simple.csv"
    _require(src)
    df = pd.read_csv(src)
    vals = pd.to_numeric(df["daily_pnl"], errors="coerce").dropna()
    if vals.empty:
        raise ValueError("Nu există date pentru figura 11.")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.hist(vals, bins=40, color=STRATEGY_COLORS["Simple"], alpha=0.75, density=True)
    ax.set_title("Distribuția PnL-ului Zilnic În Backtestul Sintetic")
    ax.set_xlabel("PnL zilnic")
    ax.set_ylabel("Densitate")
    save_figure(fig, out_dir / "figure_11_distributia_pnl_zilnic_backtest_sintetic_iv.png", dpi=300, save_pdf=True)


def figure_12_realistic_cum_pnl(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = base_dir / "portfolio_daily_pnl_realistic_strict.csv"
    _require(src)
    df = pd.read_csv(src)
    df["quote_date"] = parse_dates(df["quote_date"])
    df["cumulative_pnl"] = pd.to_numeric(df["cumulative_pnl"], errors="coerce")
    df = df.dropna(subset=["quote_date", "cumulative_pnl"]).sort_values("quote_date")
    if df.empty:
        raise ValueError("Nu există date pentru figura 12.")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(df["quote_date"], df["cumulative_pnl"], color=STRATEGY_COLORS["Unhedged"])
    ax.set_title("PnL Cumulat Al Portofoliului Realist Neacoperit")
    ax.set_xlabel("Dată")
    ax.set_ylabel("PnL cumulat")

    save_figure(fig, out_dir / "figure_12_pnl_cumulat_portofoliu_realist_neacoperit.png", dpi=300, save_pdf=True)


def figure_14_greeks_timeseries(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = base_dir / "portfolio_daily_greeks_strict.csv"
    _require(src)
    df = pd.read_csv(src)
    df["quote_date"] = parse_dates(df["quote_date"])
    df = df.dropna(subset=["quote_date"]).sort_values("quote_date")

    series = []
    if "total_delta" in df.columns:
        series.append(("total_delta", "Delta", PALETTE["navy"]))
    if "total_gamma" in df.columns:
        series.append(("total_gamma", "Gamma", PALETTE["teal"]))
    if "total_vega" in df.columns:
        series.append(("total_vega", "Vega", PALETTE["burgundy"]))
    if "total_theta" in df.columns:
        series.append(("total_theta", "Theta", PALETTE["gray"]))
    if not series:
        raise ValueError("Nu există coloane Greeks pentru figura 14.")

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    for col, label, color in series:
        vals = pd.to_numeric(df[col], errors="coerce")
        ax.plot(df["quote_date"], vals, label=label, color=color)
    ax.axhline(0.0, color=PALETTE["light_gray"], linewidth=1.0, linestyle="--")
    ax.set_title("Expunerile Agregate Ale Portofoliului")
    ax.set_xlabel("Dată")
    ax.set_ylabel("Expunere agregată")
    ax.legend()

    save_figure(fig, out_dir / "figure_14_expuneri_greeks_portofoliu.png", dpi=300, save_pdf=True)


def figure_16_hedging_cum_pnl(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src = base_dir / "hedged_daily_pnl.csv"
    _require(src)
    d = pd.read_csv(src)
    d["quote_date"] = parse_dates(d["quote_date"])
    d = d.dropna(subset=["quote_date"]).sort_values("quote_date")

    unhedged = pd.to_numeric(d["portfolio_daily_pnl_unhedged"], errors="coerce").fillna(0.0).cumsum()
    delta = pd.to_numeric(d["daily_pnl_delta_hedged"], errors="coerce").fillna(0.0).cumsum()
    dg = pd.to_numeric(d["daily_pnl_delta_gamma_hedged"], errors="coerce").fillna(0.0).cumsum()

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.plot(d["quote_date"], unhedged, label="Fără hedging", color=STRATEGY_COLORS["Unhedged"])
    ax.plot(d["quote_date"], delta, label="Delta-hedged", color=STRATEGY_COLORS["Delta Hedged"])
    ax.plot(d["quote_date"], dg, label="Delta-gamma hedged", color=STRATEGY_COLORS["Delta-Gamma Hedged"])
    ax.set_title("Compararea PnL-ului Cumulat Înainte Și După Hedging")
    ax.set_xlabel("Dată")
    ax.set_ylabel("PnL cumulat")
    ax.legend()

    save_figure(fig, out_dir / "figure_16_comparatie_pnl_cumulat_hedging.png", dpi=300, save_pdf=True)


def figure_18_hedge_effectiveness(base_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    src_candidates = [
        base_dir / "hedge_effectiveness_summary.csv",
        base_dir / "dissertation_outputs/tables/final_hedge_effectiveness.csv",
    ]
    src = next((p for p in src_candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            "Lipsește fișierul de input pentru eficiența hedgingului: "
            "hedge_effectiveness_summary.csv sau dissertation_outputs/tables/final_hedge_effectiveness.csv"
        )
    eff = pd.read_csv(src)
    if not {"metric", "value"}.issubset(eff.columns):
        raise ValueError("Format invalid pentru hedge_effectiveness_summary.csv.")
    m = eff.set_index("metric")["value"]

    delta_before = float(m.get("mean_abs_delta_before_hedge", np.nan))
    delta_after = float(m.get("mean_abs_delta_after_delta_gamma", np.nan))
    gamma_before = float(m.get("mean_abs_gamma_before_hedge", np.nan))
    gamma_after = float(m.get("mean_abs_gamma_after_delta_gamma", np.nan))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    x = np.arange(2)
    labels = ["Înainte de hedging", "După hedging"]
    colors = [PALETTE["navy"], PALETTE["teal"]]

    # Panoul A: Delta (scară proprie)
    delta_vals = np.array([delta_before, delta_after], dtype=float)
    ax_d = axes[0]
    bars_d = ax_d.bar(x, delta_vals, color=colors, width=0.58)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, rotation=10)
    ax_d.set_title("Panoul A: Delta")
    ax_d.set_ylabel("Expunere absolută medie")
    if np.isfinite(delta_vals).any():
        ypad = max(np.nanmax(delta_vals) * 0.06, 1e-8)
        ax_d.set_ylim(0, np.nanmax(delta_vals) + ypad * 2.0)
    for rect, val in zip(bars_d, delta_vals):
        if np.isfinite(val):
            ax_d.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + max(rect.get_height() * 0.02, 1e-8),
                f"{val:.6f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Panoul B: Gamma (scară proprie)
    gamma_vals = np.array([gamma_before, gamma_after], dtype=float)
    ax_g = axes[1]
    bars_g = ax_g.bar(x, gamma_vals, color=colors, width=0.58)
    ax_g.set_xticks(x)
    ax_g.set_xticklabels(labels, rotation=10)
    ax_g.set_title("Panoul B: Gamma")
    ax_g.set_ylabel("Expunere absolută medie")
    if np.isfinite(gamma_vals).any():
        ypad = max(np.nanmax(gamma_vals) * 0.08, 1e-12)
        ax_g.set_ylim(0, np.nanmax(gamma_vals) + ypad * 2.0)
    for rect, val in zip(bars_g, gamma_vals):
        if np.isfinite(val):
            ax_g.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + max(rect.get_height() * 0.02, 1e-12),
                f"{val:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["navy"]),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["teal"]),
    ]
    fig.legend(legend_handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle("Eficiența Hedgingului: Expuneri Înainte Și După Acoperire", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    save_figure(fig, out_dir / "figure_18_eficienta_hedgingului.png", dpi=300, save_pdf=True)


def update_manifest_ro(out_dir: pathlib.Path) -> pathlib.Path:
    manifest_dir = out_dir.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / "README_dissertation_outputs.md"
    lines = [
        "# Manifest Outputuri Disertație (RO)",
        "",
        "Figuri în limba română generate pentru corpul lucrării:",
        "",
        "- `figures/figure_01_retentia_esantionului.png`",
        "- `figures/figure_02_grid_node_coverage_heatmap.png`",
        "- `figures/figure_03_suprafata_iv_acoperire_ridicata.png`",
        "- `figures/figure_05_comparatie_erori_forecast.png`",
        "- `figures/figure_08_distributia_intensitatii_semnalului.png`",
        "- `figures/figure_10_pnl_cumulat_backtest_sintetic_iv.png`",
        "- `figures/figure_11_distributia_pnl_zilnic_backtest_sintetic_iv.png`",
        "- `figures/figure_12_pnl_cumulat_portofoliu_realist_neacoperit.png`",
        "- `figures/figure_14_expuneri_greeks_portofoliu.png`",
        "- `figures/figure_16_comparatie_pnl_cumulat_hedging.png`",
        "- `figures/figure_18_eficienta_hedgingului.png`",
        "- `figures/figure_20_flux_metodologic.png`",
        "",
        "Pentru fiecare figură s-a salvat și versiunea PDF cu același nume de bază.",
        "",
        "Notă: Nu s-au recalculat rezultate; s-au folosit outputurile existente.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generează figurile principale în limba română pentru disertație.")
    parser.add_argument("--base-dir", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("dissertation_outputs_ro/figures"))
    args = parser.parse_args()

    apply_theme()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    figure_01_data_cleaning(args.base_dir, outdir)
    figure_02_grid_coverage_heatmap(args.base_dir, outdir)
    figure_03_iv_surface(args.base_dir, outdir)
    figure_05_forecast_comparison(args.base_dir, outdir)
    figure_08_signal_strength(args.base_dir, outdir)
    figure_10_simple_cum_pnl(args.base_dir, outdir)
    figure_11_simple_daily_pnl_distribution(args.base_dir, outdir)
    figure_12_realistic_cum_pnl(args.base_dir, outdir)
    figure_14_greeks_timeseries(args.base_dir, outdir)
    figure_16_hedging_cum_pnl(args.base_dir, outdir)
    figure_18_hedge_effectiveness(args.base_dir, outdir)
    build_workflow_ro(outdir)
    manifest_path = update_manifest_ro(outdir)

    print(f"Figurile RO au fost generate în: {outdir}")
    print(f"Manifest actualizat: {manifest_path}")


if __name__ == "__main__":
    main()
