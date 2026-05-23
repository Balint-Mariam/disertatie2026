import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np
import pandas as pd


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Valoare booleana invalida: {value}")


def load_selected_signals(signals_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(signals_path)
    required = {
        "quote_date",
        "node",
        "signal",
        "selected",
        "log_moneyness",
        "T",
        "observed_iv",
        "realized_iv",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane in signals_xgboost.csv: {', '.join(sorted(missing))}")

    out = df.copy()
    out["quote_date"] = parse_dates(out["quote_date"]).dt.normalize()
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce")
    out["selected"] = pd.to_numeric(out["selected"], errors="coerce")
    out["log_moneyness"] = pd.to_numeric(out["log_moneyness"], errors="coerce")
    out["T"] = pd.to_numeric(out["T"], errors="coerce")
    out["observed_iv"] = pd.to_numeric(out["observed_iv"], errors="coerce")
    out["realized_iv"] = pd.to_numeric(out["realized_iv"], errors="coerce")

    out = out.dropna(
        subset=[
            "quote_date",
            "node",
            "signal",
            "selected",
            "log_moneyness",
            "T",
            "observed_iv",
            "realized_iv",
        ]
    )
    out = out[(out["selected"] == 1) & (out["signal"].isin([1, -1]))].copy()
    out = out.sort_values(["quote_date", "node"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("Nu exista semnale selectate valide (selected=1, signal in {-1,1}).")
    return out


def add_next_quote_date(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    uniq_dates = sorted(out["quote_date"].dropna().unique())
    next_map = {uniq_dates[i]: uniq_dates[i + 1] for i in range(len(uniq_dates) - 1)}
    out["next_quote_date"] = out["quote_date"].map(next_map)
    out["has_next_day"] = out["next_quote_date"].notna().astype(int)
    return out


def prepare_option_columns(header_cols: Iterable[str]) -> list[str]:
    base = ["quote_date", "expiration", "strike", "option_type", "underlying_symbol", "root"]
    optional = [
        "mid",
        "bid_1545",
        "ask_1545",
        "moneyness",
        "T",
        "spot",
        "underlying_bid_1545",
        "underlying_ask_1545",
        "trade_volume",
        "open_interest",
    ]
    cols = [c for c in base if c in header_cols]
    missing_base = set(base) - set(cols)
    if missing_base:
        raise ValueError(f"Lipsesc coloane obligatorii in options dataset: {', '.join(sorted(missing_base))}")
    cols += [c for c in optional if c in header_cols]
    return cols


def load_options_subset(
    options_path: pathlib.Path,
    needed_dates: set[pd.Timestamp],
    chunksize: int,
) -> pd.DataFrame:
    header = pd.read_csv(options_path, nrows=0)
    usecols = prepare_option_columns(set(header.columns))

    parts: list[pd.DataFrame] = []
    needed_dates = {pd.Timestamp(d).normalize() for d in needed_dates}

    for chunk in pd.read_csv(options_path, usecols=usecols, chunksize=chunksize):
        qd = parse_dates(chunk["quote_date"]).dt.normalize()
        chunk = chunk.assign(quote_date=qd)
        chunk = chunk[chunk["quote_date"].isin(needed_dates)]
        if chunk.empty:
            continue

        chunk["expiration"] = parse_dates(chunk["expiration"]).dt.normalize()
        chunk["strike"] = pd.to_numeric(chunk["strike"], errors="coerce")
        chunk["option_type"] = chunk["option_type"].astype(str).str.upper().str.strip().str[0]
        chunk["underlying_symbol"] = chunk["underlying_symbol"].astype(str).str.strip().str.upper()
        chunk["root"] = chunk["root"].astype(str).str.strip().str.upper()

        if "mid" in chunk.columns:
            chunk["mid_price"] = pd.to_numeric(chunk["mid"], errors="coerce")
        else:
            chunk["mid_price"] = np.nan

        if {"bid_1545", "ask_1545"}.issubset(chunk.columns):
            bid = pd.to_numeric(chunk["bid_1545"], errors="coerce")
            ask = pd.to_numeric(chunk["ask_1545"], errors="coerce")
            mid_alt = (bid + ask) / 2.0
            chunk["mid_price"] = chunk["mid_price"].fillna(mid_alt)

        if "moneyness" in chunk.columns:
            chunk["moneyness"] = pd.to_numeric(chunk["moneyness"], errors="coerce")
        elif "spot" in chunk.columns:
            spot = pd.to_numeric(chunk["spot"], errors="coerce")
            chunk["moneyness"] = chunk["strike"] / spot
        elif {"underlying_bid_1545", "underlying_ask_1545"}.issubset(chunk.columns):
            ub = pd.to_numeric(chunk["underlying_bid_1545"], errors="coerce")
            ua = pd.to_numeric(chunk["underlying_ask_1545"], errors="coerce")
            spot = (ub + ua) / 2.0
            chunk["moneyness"] = chunk["strike"] / spot
        else:
            chunk["moneyness"] = np.nan

        if "T" in chunk.columns:
            chunk["T"] = pd.to_numeric(chunk["T"], errors="coerce")
        else:
            chunk["T"] = (chunk["expiration"] - chunk["quote_date"]).dt.days / 365.0

        chunk["trade_volume"] = pd.to_numeric(chunk.get("trade_volume", 0), errors="coerce").fillna(0.0)
        chunk["open_interest"] = pd.to_numeric(chunk.get("open_interest", 0), errors="coerce").fillna(0.0)

        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(
            subset=["quote_date", "expiration", "strike", "option_type", "mid_price", "moneyness", "T"]
        )
        chunk = chunk[chunk["mid_price"] > 0]
        chunk = chunk[chunk["moneyness"] > 0]
        chunk = chunk[chunk["T"] > 0]
        chunk = chunk[chunk["option_type"].isin(["C", "P"])]
        if chunk.empty:
            continue

        chunk["log_moneyness"] = np.log(chunk["moneyness"])
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_moneyness"])
        if chunk.empty:
            continue

        parts.append(
            chunk[
                [
                    "quote_date",
                    "underlying_symbol",
                    "root",
                    "expiration",
                    "strike",
                    "option_type",
                    "mid_price",
                    "moneyness",
                    "log_moneyness",
                    "T",
                    "trade_volume",
                    "open_interest",
                ]
            ].copy()
        )

    if not parts:
        raise ValueError("Nu am gasit observatii de optiuni pentru datele necesare.")

    options = pd.concat(parts, ignore_index=True)
    options["strike_norm"] = options["strike"].round(6)

    group_cols = [
        "quote_date",
        "underlying_symbol",
        "root",
        "expiration",
        "strike_norm",
        "option_type",
    ]
    options = (
        options.groupby(group_cols, as_index=False)
        .agg(
            strike=("strike", "mean"),
            mid_price=("mid_price", "mean"),
            moneyness=("moneyness", "mean"),
            log_moneyness=("log_moneyness", "mean"),
            T=("T", "mean"),
            trade_volume=("trade_volume", "sum"),
            open_interest=("open_interest", "sum"),
        )
        .sort_values(["quote_date", "underlying_symbol", "expiration", "strike_norm", "option_type"])
        .reset_index(drop=True)
    )

    options["contract_key"] = (
        options["underlying_symbol"].astype(str)
        + "|"
        + options["root"].astype(str)
        + "|"
        + options["expiration"].dt.strftime("%Y-%m-%d")
        + "|"
        + options["strike_norm"].map(lambda x: f"{x:.6f}")
        + "|"
        + options["option_type"].astype(str)
    )
    return options


def map_signal_to_option(
    sig_log: float,
    sig_t: float,
    day_df: pd.DataFrame,
    log_scale: float,
    t_scale: float,
) -> tuple[int, float, float, float]:
    log_arr = day_df["log_moneyness"].to_numpy(dtype=float)
    t_arr = day_df["T"].to_numpy(dtype=float)

    dlog = log_arr - sig_log
    dt = t_arr - sig_t
    dist = np.sqrt((dlog / log_scale) ** 2 + (dt / t_scale) ** 2)
    min_dist = float(np.nanmin(dist))

    idxs = np.where(np.isfinite(dist) & (np.abs(dist - min_dist) <= 1e-12))[0]
    if len(idxs) == 0:
        idx = int(np.nanargmin(dist))
    elif len(idxs) == 1:
        idx = int(idxs[0])
    else:
        # Tie-break: prefer lichiditate mai buna.
        tie = day_df.iloc[idxs].copy()
        tie = tie.sort_values(["open_interest", "trade_volume"], ascending=[False, False])
        idx = int(tie.index[0] - day_df.index[0])

    abs_log_diff = float(abs(dlog[idx]))
    abs_t_diff = float(abs(dt[idx]))
    return idx, min_dist, abs_log_diff, abs_t_diff


def build_positions_mapping(
    signals: pd.DataFrame,
    options: pd.DataFrame,
    log_scale: float,
    t_scale: float,
    max_distance: float,
) -> pd.DataFrame:
    day_options = {d: g.reset_index(drop=True) for d, g in options.groupby("quote_date")}

    rows = []
    for row in signals.itertuples(index=False):
        base = {
            "quote_date": row.quote_date,
            "next_quote_date": row.next_quote_date,
            "node": row.node,
            "signal": int(row.signal),
            "selected": int(row.selected),
            "target_log_moneyness": float(row.log_moneyness),
            "target_T": float(row.T),
            "observed_iv": float(row.observed_iv),
            "realized_iv": float(row.realized_iv),
            "iv_change": float(row.realized_iv - row.observed_iv),
            "mapping_status": "",
            "mapping_reason": "",
            "mapping_distance": np.nan,
            "abs_log_moneyness_diff": np.nan,
            "abs_T_diff": np.nan,
            "contract_key": "",
            "underlying_symbol": "",
            "root": "",
            "strike": np.nan,
            "expiration": pd.NaT,
            "option_type": "",
            "mapped_log_moneyness": np.nan,
            "mapped_T": np.nan,
            "entry_price": np.nan,
            "next_day_price": np.nan,
        }

        if pd.isna(row.next_quote_date):
            base["mapping_status"] = "unmatched"
            base["mapping_reason"] = "no_next_day"
            rows.append(base)
            continue

        if row.quote_date not in day_options:
            base["mapping_status"] = "unmatched"
            base["mapping_reason"] = "no_options_entry_day"
            rows.append(base)
            continue

        if not np.isfinite(row.log_moneyness) or not np.isfinite(row.T):
            base["mapping_status"] = "unmatched"
            base["mapping_reason"] = "missing_signal_coordinates"
            rows.append(base)
            continue

        candidates = day_options[row.quote_date]
        if candidates.empty:
            base["mapping_status"] = "unmatched"
            base["mapping_reason"] = "no_candidates_entry_day"
            rows.append(base)
            continue

        idx, dist, abs_log_diff, abs_t_diff = map_signal_to_option(
            sig_log=float(row.log_moneyness),
            sig_t=float(row.T),
            day_df=candidates,
            log_scale=log_scale,
            t_scale=t_scale,
        )

        if np.isfinite(max_distance) and dist > max_distance:
            base["mapping_status"] = "unmatched"
            base["mapping_reason"] = "distance_above_threshold"
            base["mapping_distance"] = dist
            base["abs_log_moneyness_diff"] = abs_log_diff
            base["abs_T_diff"] = abs_t_diff
            rows.append(base)
            continue

        c = candidates.iloc[idx]
        base.update(
            {
                "mapping_status": "mapped",
                "mapping_reason": "ok",
                "mapping_distance": dist,
                "abs_log_moneyness_diff": abs_log_diff,
                "abs_T_diff": abs_t_diff,
                "contract_key": c["contract_key"],
                "underlying_symbol": c["underlying_symbol"],
                "root": c["root"],
                "strike": float(c["strike"]),
                "expiration": c["expiration"],
                "option_type": c["option_type"],
                "mapped_log_moneyness": float(c["log_moneyness"]),
                "mapped_T": float(c["T"]),
                "entry_price": float(c["mid_price"]),
            }
        )
        rows.append(base)

    mapped = pd.DataFrame(rows)
    return mapped


def attach_next_day_prices(
    mapped: pd.DataFrame,
    options: pd.DataFrame,
    allow_next_day_fallback: bool,
    fallback_same_option_type: bool,
    log_scale: float,
    t_scale: float,
    max_exit_distance: float,
) -> pd.DataFrame:
    out = mapped.copy()

    out["exit_match_type"] = "none"
    out["exit_contract_key"] = ""
    out["exit_strike"] = np.nan
    out["exit_expiration"] = pd.NaT
    out["exit_option_type"] = ""
    out["exit_mapping_distance"] = np.nan
    out["exit_abs_log_moneyness_diff"] = np.nan
    out["exit_abs_T_diff"] = np.nan

    next_px = options[["quote_date", "contract_key", "mid_price"]].rename(
        columns={"quote_date": "next_quote_date", "mid_price": "exact_next_day_price"}
    )
    out = out.merge(next_px, on=["next_quote_date", "contract_key"], how="left")
    out["next_day_price"] = out["exact_next_day_price"]
    out = out.drop(columns=["exact_next_day_price"])

    day_options = {d: g.reset_index(drop=True) for d, g in options.groupby("quote_date")}

    for idx, row in out.iterrows():
        if row["mapping_status"] != "mapped":
            out.at[idx, "trade_status"] = "not_traded"
            if not out.at[idx, "mapping_reason"]:
                out.at[idx, "mapping_reason"] = "entry_not_mapped"
            continue

        if np.isfinite(pd.to_numeric(row["next_day_price"], errors="coerce")):
            out.at[idx, "trade_status"] = "traded"
            out.at[idx, "exit_match_type"] = "exact_contract"
            out.at[idx, "exit_contract_key"] = row["contract_key"]
            out.at[idx, "exit_strike"] = row["strike"]
            out.at[idx, "exit_expiration"] = row["expiration"]
            out.at[idx, "exit_option_type"] = row["option_type"]
            out.at[idx, "exit_mapping_distance"] = 0.0
            out.at[idx, "exit_abs_log_moneyness_diff"] = 0.0
            out.at[idx, "exit_abs_T_diff"] = 0.0
            out.at[idx, "mapping_reason"] = "ok_exact"
            continue

        if not allow_next_day_fallback:
            out.at[idx, "trade_status"] = "not_traded"
            out.at[idx, "mapping_reason"] = "missing_contract_tplus1"
            continue

        next_day = row["next_quote_date"]
        if pd.isna(next_day) or next_day not in day_options:
            out.at[idx, "trade_status"] = "not_traded"
            out.at[idx, "mapping_reason"] = "no_options_tplus1_day"
            continue

        candidates = day_options[next_day]
        if fallback_same_option_type:
            candidates = candidates[candidates["option_type"] == row["option_type"]]
        candidates = candidates.reset_index(drop=True)
        if candidates.empty:
            out.at[idx, "trade_status"] = "not_traded"
            out.at[idx, "mapping_reason"] = "no_exit_candidates_tplus1"
            continue

        idx_c, dist, abs_log_diff, abs_t_diff = map_signal_to_option(
            sig_log=float(row["mapped_log_moneyness"]),
            sig_t=float(row["mapped_T"]),
            day_df=candidates,
            log_scale=log_scale,
            t_scale=t_scale,
        )

        out.at[idx, "exit_mapping_distance"] = dist
        out.at[idx, "exit_abs_log_moneyness_diff"] = abs_log_diff
        out.at[idx, "exit_abs_T_diff"] = abs_t_diff

        if np.isfinite(max_exit_distance) and dist > max_exit_distance:
            out.at[idx, "trade_status"] = "not_traded"
            out.at[idx, "mapping_reason"] = "exit_distance_above_threshold"
            continue

        c = candidates.iloc[idx_c]
        out.at[idx, "next_day_price"] = float(c["mid_price"])
        out.at[idx, "trade_status"] = "traded"
        out.at[idx, "exit_match_type"] = "nearest_fallback"
        out.at[idx, "exit_contract_key"] = c["contract_key"]
        out.at[idx, "exit_strike"] = float(c["strike"])
        out.at[idx, "exit_expiration"] = c["expiration"]
        out.at[idx, "exit_option_type"] = c["option_type"]
        out.at[idx, "mapping_reason"] = "ok_fallback"

    out["trade_status"] = out["trade_status"].fillna("not_traded")
    return out


def apply_daily_weights_and_pnl(positions: pd.DataFrame) -> pd.DataFrame:
    out = positions.copy()
    out["weight"] = 0.0
    out["position_pnl"] = 0.0
    out["price_change"] = np.nan
    out["day_status"] = "unprocessed"

    for day, idx in out.groupby("quote_date").groups.items():
        day_df = out.loc[idx]
        tradable = day_df[day_df["trade_status"] == "traded"].copy()
        long_idx = tradable[tradable["signal"] == 1].index
        short_idx = tradable[tradable["signal"] == -1].index
        n_long = len(long_idx)
        n_short = len(short_idx)

        if n_long > 0 and n_short > 0:
            w_long = 0.5 / n_long
            w_short = -0.5 / n_short
            out.loc[long_idx, "weight"] = w_long
            out.loc[short_idx, "weight"] = w_short
            out.loc[idx, "day_status"] = "tradeable_side_neutral"
        elif n_long > 0:
            out.loc[idx, "day_status"] = "skipped_no_short_side_after_mapping"
        elif n_short > 0:
            out.loc[idx, "day_status"] = "skipped_no_long_side_after_mapping"
        else:
            out.loc[idx, "day_status"] = "skipped_no_tradable_positions"

    out["price_change"] = out["next_day_price"] - out["entry_price"]
    out["position_pnl"] = out["weight"] * out["price_change"]
    return out


def build_daily_pnl(positions: pd.DataFrame) -> pd.DataFrame:
    g = positions.groupby("quote_date", as_index=False)
    daily = g.agg(
        n_long=("weight", lambda x: int((x > 0).sum())),
        n_short=("weight", lambda x: int((x < 0).sum())),
        gross_exposure=("weight", lambda x: float(np.abs(x).sum())),
        net_exposure=("weight", "sum"),
        daily_pnl=("position_pnl", "sum"),
        num_unmatched_positions=("trade_status", lambda x: int((x != "traded").sum())),
    )
    daily = daily.sort_values("quote_date").reset_index(drop=True)
    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()

    day_status = positions.groupby("quote_date")["day_status"].first().rename("day_status").reset_index()
    daily = daily.merge(day_status, on="quote_date", how="left")
    return daily


def max_drawdown_from_cum(cum: pd.Series) -> float:
    if cum.empty:
        return np.nan
    dd = cum - cum.cummax()
    return float(dd.min())


def build_performance(positions: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    daily_pnl = daily["daily_pnl"].to_numpy(dtype=float)
    total_pnl = float(np.nansum(daily_pnl))
    mean_daily = float(np.nanmean(daily_pnl)) if len(daily_pnl) > 0 else np.nan
    vol_daily = float(np.nanstd(daily_pnl, ddof=1)) if len(daily_pnl) > 1 else np.nan
    sharpe = np.nan
    if np.isfinite(vol_daily) and vol_daily > 0:
        sharpe = float(np.sqrt(252.0) * mean_daily / vol_daily)

    perf = pd.DataFrame(
        [
            {
                "strategy": "xgboost_realistic_option_mapping_no_hedge",
                "total_pnl": total_pnl,
                "mean_daily_pnl": mean_daily,
                "daily_volatility": vol_daily,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown_from_cum(daily["cumulative_pnl"]),
                "num_total_positions_selected": int(len(positions)),
                "num_positions_successfully_mapped": int((positions["mapping_status"] == "mapped").sum()),
                "num_positions_traded": int((positions["trade_status"] == "traded").sum()),
                "num_days_total": int(daily["quote_date"].nunique()),
                "num_days_traded": int((daily["gross_exposure"] > 0).sum()),
            }
        ]
    )
    return perf


def build_mapping_quality(positions: pd.DataFrame) -> pd.DataFrame:
    total = len(positions)
    mapped = int((positions["mapping_status"] == "mapped").sum())
    traded = int((positions["trade_status"] == "traded").sum())
    unmatched = total - traded
    avg_dist = float(
        positions.loc[positions["mapping_status"] == "mapped", "mapping_distance"].mean()
    ) if mapped > 0 else np.nan

    exact_count = int((positions["exit_match_type"] == "exact_contract").sum())
    fallback_count = int((positions["exit_match_type"] == "nearest_fallback").sum())
    avg_fallback_dist = float(
        positions.loc[positions["exit_match_type"] == "nearest_fallback", "exit_mapping_distance"].mean()
    ) if fallback_count > 0 else np.nan

    reason_counts = (
        positions.loc[positions["trade_status"] != "traded", "mapping_reason"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("mapping_reason")
        .reset_index(name="count")
    )

    summary_rows = [
        {"metric": "total_selected_signals", "value": float(total)},
        {"metric": "positions_mapped_entry_day", "value": float(mapped)},
        {"metric": "positions_traded_with_tplus1_price", "value": float(traded)},
        {"metric": "positions_unmatched_or_untradable", "value": float(unmatched)},
        {"metric": "mapping_success_rate_vs_selected", "value": float(mapped / total) if total > 0 else np.nan},
        {"metric": "traded_rate_vs_selected", "value": float(traded / total) if total > 0 else np.nan},
        {"metric": "num_exit_exact_contract", "value": float(exact_count)},
        {"metric": "num_exit_nearest_fallback", "value": float(fallback_count)},
        {"metric": "num_still_not_traded", "value": float(total - traded)},
        {"metric": "tradability_rate_after_fallback", "value": float(traded / total) if total > 0 else np.nan},
        {"metric": "avg_mapping_distance_mapped_only", "value": avg_dist},
        {"metric": "avg_fallback_exit_distance", "value": avg_fallback_dist},
    ]
    summary = pd.DataFrame(summary_rows)

    out = pd.concat(
        [
            summary.assign(section="overall"),
            reason_counts.rename(columns={"mapping_reason": "metric", "count": "value"}).assign(section="reason_counts"),
        ],
        ignore_index=True,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Backtest realist pe optiuni mapate din semnalele XGBoost (fara hedging)."
    )
    parser.add_argument("--signals", type=pathlib.Path, default=pathlib.Path("signals_xgboost.csv"))
    parser.add_argument("--options", type=pathlib.Path, default=pathlib.Path("options_eod_all_with_iv.csv"))
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--log-scale", type=float, default=0.02, help="Scala pentru distanta pe log-moneyness.")
    parser.add_argument("--t-scale", type=float, default=0.05, help="Scala pentru distanta pe maturitate T.")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=np.inf,
        help="Prag maxim pentru distanta de mapping (inf = fara prag).",
    )
    parser.add_argument(
        "--allow-next-day-fallback",
        type=parse_bool,
        default=True,
        help="Permite fallback la cea mai apropiata optiune in t+1 cand lipseste contractul exact (default true).",
    )
    parser.add_argument(
        "--fallback-same-option-type",
        type=parse_bool,
        default=True,
        help="Fallbackul in t+1 cauta implicit aceeasi option_type (default true).",
    )
    parser.add_argument(
        "--max-exit-distance",
        type=float,
        default=np.inf,
        help="Prag maxim pentru distanta fallback in t+1 (inf = fara prag).",
    )
    parser.add_argument(
        "--out-positions",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_positions_realistic.csv"),
    )
    parser.add_argument(
        "--out-daily",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_daily_pnl_realistic.csv"),
    )
    parser.add_argument(
        "--out-performance",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_performance_realistic.csv"),
    )
    parser.add_argument(
        "--out-mapping-quality",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_mapping_quality.csv"),
    )
    args = parser.parse_args()

    if not args.signals.exists():
        sys.exit(f"Nu gasesc fisierul de semnale: {args.signals}")
    if not args.options.exists():
        sys.exit(f"Nu gasesc fisierul cu optiuni: {args.options}")
    if args.log_scale <= 0 or args.t_scale <= 0:
        sys.exit("log-scale si t-scale trebuie sa fie > 0.")

    try:
        print("Pas 1/6: citesc semnalele selectate...")
        signals = load_selected_signals(args.signals)
        signals = add_next_quote_date(signals)
        signals = signals.dropna(subset=["next_quote_date"]).copy()
        if signals.empty:
            raise ValueError("Nu exista semnale cu next_quote_date disponibil.")

        print("Pas 2/6: incarc subsetul de optiuni pentru zilele necesare...")
        needed_dates = set(signals["quote_date"].unique()) | set(signals["next_quote_date"].unique())
        options = load_options_subset(args.options, needed_dates=needed_dates, chunksize=args.chunksize)

        print("Pas 3/6: mapez nodurile la contracte reale (nearest moneyness + T)...")
        mapped = build_positions_mapping(
            signals=signals,
            options=options,
            log_scale=args.log_scale,
            t_scale=args.t_scale,
            max_distance=float(args.max_distance),
        )

        print("Pas 4/6: adaug pretul contractului la t+1 si status de tranzactionare...")
        mapped = attach_next_day_prices(
            mapped,
            options,
            allow_next_day_fallback=args.allow_next_day_fallback,
            fallback_same_option_type=args.fallback_same_option_type,
            log_scale=args.log_scale,
            t_scale=args.t_scale,
            max_exit_distance=float(args.max_exit_distance),
        )

        print("Pas 5/6: construiesc portofoliul zilnic side-neutral si calculez PnL...")
        positions = apply_daily_weights_and_pnl(mapped)
        daily = build_daily_pnl(positions)
        perf = build_performance(positions, daily)
        map_quality = build_mapping_quality(positions)

        print("Pas 6/6: salvez outputurile...")
        keep_pos_cols = [
            "quote_date",
            "next_quote_date",
            "node",
            "signal",
            "selected",
            "observed_iv",
            "realized_iv",
            "iv_change",
            "contract_key",
            "underlying_symbol",
            "root",
            "strike",
            "expiration",
            "option_type",
            "target_log_moneyness",
            "target_T",
            "mapped_log_moneyness",
            "mapped_T",
            "entry_price",
            "next_day_price",
            "exit_match_type",
            "exit_contract_key",
            "exit_strike",
            "exit_expiration",
            "exit_option_type",
            "exit_mapping_distance",
            "exit_abs_log_moneyness_diff",
            "exit_abs_T_diff",
            "weight",
            "position_pnl",
            "mapping_distance",
            "abs_log_moneyness_diff",
            "abs_T_diff",
            "mapping_status",
            "mapping_reason",
            "trade_status",
            "day_status",
        ]
        positions_out = positions[keep_pos_cols].copy()

        keep_daily_cols = [
            "quote_date",
            "n_long",
            "n_short",
            "gross_exposure",
            "net_exposure",
            "daily_pnl",
            "cumulative_pnl",
            "num_unmatched_positions",
            "day_status",
        ]
        daily_out = daily[keep_daily_cols].copy()

        for p in [args.out_positions, args.out_daily, args.out_performance, args.out_mapping_quality]:
            p.parent.mkdir(parents=True, exist_ok=True)
        positions_out.to_csv(args.out_positions, index=False)
        daily_out.to_csv(args.out_daily, index=False)
        perf.to_csv(args.out_performance, index=False)
        map_quality.to_csv(args.out_mapping_quality, index=False)

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare: {exc}")

    print("\n=== Summary ===")
    exact_count = int((positions["exit_match_type"] == "exact_contract").sum())
    fallback_count = int((positions["exit_match_type"] == "nearest_fallback").sum())
    print(f"Signals selected: {len(signals)}")
    print(f"Positions mapped entry day: {(positions['mapping_status'] == 'mapped').sum()}")
    print(f"Positions traded (have t+1 price): {(positions['trade_status'] == 'traded').sum()}")
    print(f"Exit exact-contract: {exact_count}")
    print(f"Exit nearest-fallback: {fallback_count}")
    print(f"Days traded: {(daily['gross_exposure'] > 0).sum()} / {len(daily)}")
    print(f"Total PnL: {perf.loc[0, 'total_pnl']:.6f}")
    print(f"Output positions: {args.out_positions}")
    print(f"Output daily: {args.out_daily}")
    print(f"Output performance: {args.out_performance}")
    print(f"Output mapping quality: {args.out_mapping_quality}")


if __name__ == "__main__":
    main()
