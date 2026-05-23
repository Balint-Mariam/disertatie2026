import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def max_drawdown(cum_pnl: pd.Series) -> float:
    if cum_pnl.empty:
        return np.nan
    dd = cum_pnl - cum_pnl.cummax()
    return float(dd.min())


def compute_perf_row(name: str, pnl_series: pd.Series, hedged_days: int) -> dict[str, float | str]:
    pnl = pnl_series.to_numpy(dtype=float)
    total = float(np.nansum(pnl))
    mean = float(np.nanmean(pnl)) if len(pnl) > 0 else np.nan
    vol = float(np.nanstd(pnl, ddof=1)) if len(pnl) > 1 else np.nan
    sharpe = np.nan
    if np.isfinite(vol) and vol > 0:
        sharpe = float(np.sqrt(252.0) * mean / vol)
    cum = pd.Series(pnl_series).fillna(0.0).cumsum()
    return {
        "strategy": name,
        "total_pnl": total,
        "mean_daily_pnl": mean,
        "daily_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(cum),
        "num_days": int(len(pnl_series)),
        "num_days_hedged": int(hedged_days),
    }


def bs_delta_gamma(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt_t = np.sqrt(T)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    disc_q = np.exp(-q * T)
    delta_call = disc_q * norm.cdf(d1)
    delta_put = disc_q * (norm.cdf(d1) - 1.0)
    delta = np.where(is_call, delta_call, delta_put)
    gamma = disc_q * norm.pdf(d1) / (S * vol_sqrt_t)
    return delta, gamma


def build_contract_key(
    underlying_symbol: pd.Series,
    root: pd.Series,
    expiration: pd.Series,
    strike: pd.Series,
    option_type: pd.Series,
) -> pd.Series:
    strike_norm = pd.to_numeric(strike, errors="coerce").round(6)
    return (
        underlying_symbol.astype(str).str.strip().str.upper()
        + "|"
        + root.astype(str).str.strip().str.upper()
        + "|"
        + expiration.dt.strftime("%Y-%m-%d")
        + "|"
        + strike_norm.map(lambda x: f"{x:.6f}" if np.isfinite(x) else "nan")
        + "|"
        + option_type.astype(str).str.upper().str.strip().str[0]
    )


def load_daily_inputs(daily_greeks_path: pathlib.Path, daily_pnl_path: pathlib.Path) -> pd.DataFrame:
    g = pd.read_csv(daily_greeks_path)
    p = pd.read_csv(daily_pnl_path)

    required_g = {"quote_date", "total_delta", "total_gamma", "total_vega", "total_theta"}
    required_p = {"quote_date", "daily_pnl"}
    missing_g = required_g - set(g.columns)
    missing_p = required_p - set(p.columns)
    if missing_g:
        raise ValueError(f"Lipsesc coloane in {daily_greeks_path.name}: {', '.join(sorted(missing_g))}")
    if missing_p:
        raise ValueError(f"Lipsesc coloane in {daily_pnl_path.name}: {', '.join(sorted(missing_p))}")

    g["quote_date"] = parse_dates(g["quote_date"]).dt.normalize()
    p["quote_date"] = parse_dates(p["quote_date"]).dt.normalize()
    for col in ["total_delta", "total_gamma", "total_vega", "total_theta"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    p["daily_pnl"] = pd.to_numeric(p["daily_pnl"], errors="coerce")

    out = g.merge(
        p[["quote_date", "daily_pnl"]].rename(columns={"daily_pnl": "portfolio_daily_pnl_unhedged"}),
        on="quote_date",
        how="left",
    )
    out = out.dropna(subset=["quote_date"]).sort_values("quote_date").reset_index(drop=True)

    uniq = sorted(out["quote_date"].unique())
    next_map = {uniq[i]: uniq[i + 1] for i in range(len(uniq) - 1)}
    out["next_quote_date"] = out["quote_date"].map(next_map)
    return out


def infer_underlying(positions_with_greeks_path: pathlib.Path, user_value: str) -> str:
    if user_value:
        return user_value.strip().upper()

    df = pd.read_csv(positions_with_greeks_path, usecols=["contract_key"])
    if df.empty:
        raise ValueError("Nu pot detecta underlying_symbol: positions_with_greeks este gol.")
    sym = (
        df["contract_key"]
        .astype(str)
        .str.split("|")
        .str[0]
        .value_counts()
    )
    if sym.empty:
        raise ValueError("Nu pot detecta underlying_symbol din contract_key.")
    return str(sym.index[0]).upper()


def prepare_option_columns(header_cols: Iterable[str]) -> list[str]:
    required = [
        "quote_date",
        "underlying_symbol",
        "root",
        "expiration",
        "strike",
        "option_type",
    ]
    optional = [
        "mid",
        "bid_1545",
        "ask_1545",
        "spot",
        "underlying_bid_1545",
        "underlying_ask_1545",
        "T",
        "implied_vol",
        "r_annual",
        "q_annual",
        "open_interest",
        "trade_volume",
    ]
    missing = set(required) - set(header_cols)
    if missing:
        raise ValueError("Lipsesc coloane in options_eod_all_with_iv.csv: " + ", ".join(sorted(missing)))
    cols = [c for c in required if c in header_cols]
    cols += [c for c in optional if c in header_cols]
    return cols


def load_options_subset(
    options_path: pathlib.Path,
    needed_dates: set[pd.Timestamp],
    underlying_symbol: str,
    chunksize: int,
) -> pd.DataFrame:
    header = pd.read_csv(options_path, nrows=0)
    usecols = prepare_option_columns(set(header.columns))

    needed_dates = {pd.Timestamp(d).normalize() for d in needed_dates}
    parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(options_path, usecols=usecols, chunksize=chunksize):
        chunk["quote_date"] = parse_dates(chunk["quote_date"]).dt.normalize()
        chunk["underlying_symbol"] = chunk["underlying_symbol"].astype(str).str.strip().str.upper()
        chunk = chunk[(chunk["quote_date"].isin(needed_dates)) & (chunk["underlying_symbol"] == underlying_symbol)]
        if chunk.empty:
            continue

        chunk["expiration"] = parse_dates(chunk["expiration"]).dt.normalize()
        chunk["root"] = chunk["root"].astype(str).str.strip().str.upper()
        chunk["strike"] = pd.to_numeric(chunk["strike"], errors="coerce")
        chunk["option_type"] = chunk["option_type"].astype(str).str.upper().str.strip().str[0]

        if "mid" in chunk.columns:
            chunk["mid_price"] = pd.to_numeric(chunk["mid"], errors="coerce")
        else:
            chunk["mid_price"] = np.nan

        if {"bid_1545", "ask_1545"}.issubset(chunk.columns):
            bid = pd.to_numeric(chunk["bid_1545"], errors="coerce")
            ask = pd.to_numeric(chunk["ask_1545"], errors="coerce")
            chunk["mid_price"] = chunk["mid_price"].fillna((bid + ask) / 2.0)

        if "spot" in chunk.columns:
            chunk["spot_use"] = pd.to_numeric(chunk["spot"], errors="coerce")
        else:
            chunk["spot_use"] = np.nan
        if {"underlying_bid_1545", "underlying_ask_1545"}.issubset(chunk.columns):
            ub = pd.to_numeric(chunk["underlying_bid_1545"], errors="coerce")
            ua = pd.to_numeric(chunk["underlying_ask_1545"], errors="coerce")
            chunk["spot_use"] = chunk["spot_use"].fillna((ub + ua) / 2.0)

        if "T" in chunk.columns:
            chunk["T"] = pd.to_numeric(chunk["T"], errors="coerce")
        else:
            chunk["T"] = (chunk["expiration"] - chunk["quote_date"]).dt.days / 365.0

        chunk["implied_vol"] = pd.to_numeric(chunk.get("implied_vol", np.nan), errors="coerce")
        chunk["r_annual"] = pd.to_numeric(chunk.get("r_annual", np.nan), errors="coerce")
        chunk["q_annual"] = pd.to_numeric(chunk.get("q_annual", np.nan), errors="coerce")
        chunk["open_interest"] = pd.to_numeric(chunk.get("open_interest", 0), errors="coerce").fillna(0.0)
        chunk["trade_volume"] = pd.to_numeric(chunk.get("trade_volume", 0), errors="coerce").fillna(0.0)

        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=["quote_date", "expiration", "strike", "option_type", "mid_price", "spot_use", "T"])
        chunk = chunk[(chunk["mid_price"] > 0) & (chunk["spot_use"] > 0) & (chunk["T"] > 0)]
        chunk = chunk[chunk["option_type"].isin(["C", "P"])]
        if chunk.empty:
            continue

        chunk["contract_key"] = build_contract_key(
            chunk["underlying_symbol"],
            chunk["root"],
            chunk["expiration"],
            chunk["strike"],
            chunk["option_type"],
        )
        chunk["log_moneyness"] = np.log(chunk["strike"] / chunk["spot_use"])
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_moneyness"])
        if chunk.empty:
            continue

        parts.append(
            chunk[
                [
                    "quote_date",
                    "underlying_symbol",
                    "root",
                    "contract_key",
                    "expiration",
                    "strike",
                    "option_type",
                    "mid_price",
                    "spot_use",
                    "T",
                    "implied_vol",
                    "r_annual",
                    "q_annual",
                    "open_interest",
                    "trade_volume",
                    "log_moneyness",
                ]
            ].copy()
        )

    if not parts:
        raise ValueError("Nu am gasit observatii de optiuni pentru zilele necesare.")

    options = pd.concat(parts, ignore_index=True)
    options = (
        options.groupby(["quote_date", "contract_key"], as_index=False)
        .agg(
            underlying_symbol=("underlying_symbol", "first"),
            root=("root", "first"),
            expiration=("expiration", "first"),
            strike=("strike", "mean"),
            option_type=("option_type", "first"),
            mid_price=("mid_price", "mean"),
            spot_use=("spot_use", "mean"),
            T=("T", "mean"),
            implied_vol=("implied_vol", "mean"),
            r_annual=("r_annual", "mean"),
            q_annual=("q_annual", "mean"),
            open_interest=("open_interest", "sum"),
            trade_volume=("trade_volume", "sum"),
            log_moneyness=("log_moneyness", "mean"),
        )
        .sort_values(["quote_date", "expiration", "strike", "option_type"])
        .reset_index(drop=True)
    )
    return options


def select_hedge_option(
    day_options: pd.DataFrame,
    next_day_options: pd.DataFrame,
    t_min: float,
    t_max: float,
    min_abs_gamma: float,
    default_r: float,
    default_q: float,
    contract_multiplier: float,
) -> dict[str, float | str | pd.Timestamp]:
    if day_options.empty:
        return {"status": "not_hedged_no_options_t"}
    if next_day_options.empty:
        return {"status": "not_hedged_no_options_tplus1"}

    c = day_options.copy()
    c = c[(c["T"] >= t_min) & (c["T"] <= t_max)]
    if c.empty:
        return {"status": "not_hedged_no_options_in_T_window"}

    next_price_map = next_day_options.set_index("contract_key")["mid_price"]
    c["hedge_option_price_tplus1"] = c["contract_key"].map(next_price_map)
    c = c[np.isfinite(c["hedge_option_price_tplus1"]) & (c["hedge_option_price_tplus1"] > 0)]
    if c.empty:
        return {"status": "not_hedged_no_exact_contract_tplus1"}

    c["r_use"] = c["r_annual"].where(np.isfinite(c["r_annual"]), default_r)
    c["q_use"] = c["q_annual"].where(np.isfinite(c["q_annual"]), default_q)

    valid = (
        np.isfinite(c["spot_use"])
        & (c["spot_use"] > 0)
        & np.isfinite(c["strike"])
        & (c["strike"] > 0)
        & np.isfinite(c["T"])
        & (c["T"] > 0)
        & np.isfinite(c["implied_vol"])
        & (c["implied_vol"] > 0)
    )
    c = c[valid].copy()
    if c.empty:
        return {"status": "not_hedged_no_valid_greeks_inputs"}

    is_call = c["option_type"].astype(str).str.upper().str[0].eq("C").to_numpy()
    delta, gamma = bs_delta_gamma(
        S=c["spot_use"].to_numpy(dtype=float),
        K=c["strike"].to_numpy(dtype=float),
        T=c["T"].to_numpy(dtype=float),
        r=c["r_use"].to_numpy(dtype=float),
        q=c["q_use"].to_numpy(dtype=float),
        sigma=c["implied_vol"].to_numpy(dtype=float),
        is_call=is_call,
    )
    c["hedge_option_delta"] = delta
    c["hedge_option_gamma"] = gamma

    c = c[np.isfinite(c["hedge_option_gamma"]) & (np.abs(c["hedge_option_gamma"]) >= min_abs_gamma)]
    if c.empty:
        return {"status": "not_hedged_gamma_below_threshold"}

    # Prefer ATM and lichiditate.
    c["atm_distance"] = np.abs(c["log_moneyness"])
    c = c.sort_values(
        ["atm_distance", "open_interest", "trade_volume"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    best = c.iloc[0]

    return {
        "status": "hedged",
        "hedge_option_contract_key": best["contract_key"],
        "hedge_option_type": best["option_type"],
        "hedge_option_strike": float(best["strike"]),
        "hedge_option_expiration": best["expiration"],
        "hedge_option_price_t": float(best["mid_price"]),
        "hedge_option_price_tplus1": float(best["hedge_option_price_tplus1"]),
        "hedge_option_delta": float(best["hedge_option_delta"]),
        "hedge_option_gamma": float(best["hedge_option_gamma"]),
        "hedge_option_r": float(best["r_use"]),
        "hedge_option_q": float(best["q_use"]),
        "contract_multiplier": float(contract_multiplier),
    }


def build_hedging_outputs(
    daily: pd.DataFrame,
    options: pd.DataFrame,
    default_r: float,
    default_q: float,
    hedge_t_min: float,
    hedge_t_max: float,
    min_abs_hedge_gamma: float,
    contract_multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    options_by_day = {d: g.reset_index(drop=True) for d, g in options.groupby("quote_date")}
    spot_by_day = options.groupby("quote_date")["spot_use"].median()

    rows = []
    for row in daily.itertuples(index=False):
        d = row.quote_date
        nd = row.next_quote_date
        pnl_unhedged = float(row.portfolio_daily_pnl_unhedged) if np.isfinite(row.portfolio_daily_pnl_unhedged) else 0.0
        delta_before = float(row.total_delta)
        gamma_before = float(row.total_gamma)

        spot_t = float(spot_by_day.get(d, np.nan))
        spot_t1 = float(spot_by_day.get(nd, np.nan)) if pd.notna(nd) else np.nan

        # Delta-only
        if pd.notna(nd) and np.isfinite(spot_t) and np.isfinite(spot_t1):
            hedge_underlying_units_do = -delta_before
            delta_only_pnl = hedge_underlying_units_do * (spot_t1 - spot_t)
            status_do = "hedged"
            residual_delta_do = delta_before + hedge_underlying_units_do
        elif pd.isna(nd):
            hedge_underlying_units_do = np.nan
            delta_only_pnl = 0.0
            status_do = "not_hedged_no_next_day"
            residual_delta_do = delta_before
        else:
            hedge_underlying_units_do = np.nan
            delta_only_pnl = 0.0
            status_do = "not_hedged_missing_spot"
            residual_delta_do = delta_before

        pnl_delta_hedged = pnl_unhedged + delta_only_pnl

        # Delta-gamma hedge
        dg = {
            "status": "not_hedged_no_next_day" if pd.isna(nd) else "not_hedged_unknown",
            "hedge_option_contract_key": "",
            "hedge_option_type": "",
            "hedge_option_strike": np.nan,
            "hedge_option_expiration": pd.NaT,
            "hedge_option_price_t": np.nan,
            "hedge_option_price_tplus1": np.nan,
            "hedge_option_delta": np.nan,
            "hedge_option_gamma": np.nan,
            "hedge_option_units": np.nan,
            "hedge_underlying_units_delta_gamma": np.nan,
            "residual_delta_after_hedge": delta_before,
            "residual_gamma_after_hedge": gamma_before,
            "delta_gamma_option_hedge_pnl": 0.0,
            "delta_gamma_underlying_hedge_pnl": 0.0,
        }

        if pd.notna(nd):
            if d in options_by_day and nd in options_by_day and np.isfinite(spot_t) and np.isfinite(spot_t1):
                sel = select_hedge_option(
                    day_options=options_by_day[d],
                    next_day_options=options_by_day[nd],
                    t_min=hedge_t_min,
                    t_max=hedge_t_max,
                    min_abs_gamma=min_abs_hedge_gamma,
                    default_r=default_r,
                    default_q=default_q,
                    contract_multiplier=contract_multiplier,
                )
                dg["status"] = sel["status"]
                if sel["status"] == "hedged":
                    gamma_h = float(sel["hedge_option_gamma"])
                    delta_h = float(sel["hedge_option_delta"])
                    opt_units = -gamma_before / (gamma_h * contract_multiplier)
                    und_units = -(delta_before + opt_units * delta_h * contract_multiplier)

                    opt_pnl = opt_units * contract_multiplier * (
                        float(sel["hedge_option_price_tplus1"]) - float(sel["hedge_option_price_t"])
                    )
                    und_pnl = und_units * (spot_t1 - spot_t)

                    residual_gamma = gamma_before + opt_units * gamma_h * contract_multiplier
                    residual_delta = delta_before + opt_units * delta_h * contract_multiplier + und_units

                    dg.update(
                        {
                            "hedge_option_contract_key": sel["hedge_option_contract_key"],
                            "hedge_option_type": sel["hedge_option_type"],
                            "hedge_option_strike": sel["hedge_option_strike"],
                            "hedge_option_expiration": sel["hedge_option_expiration"],
                            "hedge_option_price_t": sel["hedge_option_price_t"],
                            "hedge_option_price_tplus1": sel["hedge_option_price_tplus1"],
                            "hedge_option_delta": delta_h,
                            "hedge_option_gamma": gamma_h,
                            "hedge_option_units": opt_units,
                            "hedge_underlying_units_delta_gamma": und_units,
                            "residual_delta_after_hedge": residual_delta,
                            "residual_gamma_after_hedge": residual_gamma,
                            "delta_gamma_option_hedge_pnl": opt_pnl,
                            "delta_gamma_underlying_hedge_pnl": und_pnl,
                        }
                    )
            elif not np.isfinite(spot_t) or not np.isfinite(spot_t1):
                dg["status"] = "not_hedged_missing_spot"
            else:
                dg["status"] = "not_hedged_missing_options"

        # If delta-gamma hedge is invalid, keep unhedged pnl for continuity.
        if dg["status"] == "hedged":
            pnl_delta_gamma_hedged = pnl_unhedged + dg["delta_gamma_option_hedge_pnl"] + dg["delta_gamma_underlying_hedge_pnl"]
        else:
            pnl_delta_gamma_hedged = pnl_unhedged

        rows.append(
            {
                "quote_date": d,
                "next_quote_date": nd,
                "portfolio_daily_pnl_unhedged": pnl_unhedged,
                "spot": spot_t,
                "spot_tplus1": spot_t1,
                "total_delta_before": delta_before,
                "total_gamma_before": gamma_before,
                "total_vega_before": float(row.total_vega),
                "total_theta_before": float(row.total_theta),
                "hedge_underlying_units_delta_only": hedge_underlying_units_do,
                "delta_only_hedge_pnl": delta_only_pnl,
                "daily_pnl_delta_hedged": pnl_delta_hedged,
                "hedge_status_delta_only": status_do,
                "hedge_option_contract_key": dg["hedge_option_contract_key"],
                "hedge_option_type": dg["hedge_option_type"],
                "hedge_option_strike": dg["hedge_option_strike"],
                "hedge_option_expiration": dg["hedge_option_expiration"],
                "hedge_option_price_t": dg["hedge_option_price_t"],
                "hedge_option_price_tplus1": dg["hedge_option_price_tplus1"],
                "hedge_option_delta": dg["hedge_option_delta"],
                "hedge_option_gamma": dg["hedge_option_gamma"],
                "hedge_option_units": dg["hedge_option_units"],
                "hedge_underlying_units_delta_gamma": dg["hedge_underlying_units_delta_gamma"],
                "delta_gamma_option_hedge_pnl": dg["delta_gamma_option_hedge_pnl"],
                "delta_gamma_underlying_hedge_pnl": dg["delta_gamma_underlying_hedge_pnl"],
                "daily_pnl_delta_gamma_hedged": pnl_delta_gamma_hedged,
                "residual_delta_after_delta_only": residual_delta_do,
                "residual_delta_after_hedge": dg["residual_delta_after_hedge"],
                "residual_gamma_after_hedge": dg["residual_gamma_after_hedge"],
                "hedge_status_delta_gamma": dg["status"],
            }
        )

    daily_out = pd.DataFrame(rows).sort_values("quote_date").reset_index(drop=True)
    daily_out["cumulative_pnl_unhedged"] = daily_out["portfolio_daily_pnl_unhedged"].fillna(0.0).cumsum()
    daily_out["cumulative_pnl_delta_hedged"] = daily_out["daily_pnl_delta_hedged"].fillna(0.0).cumsum()
    daily_out["cumulative_pnl_delta_gamma_hedged"] = daily_out["daily_pnl_delta_gamma_hedged"].fillna(0.0).cumsum()

    trades_out = daily_out[
        [
            "quote_date",
            "total_delta_before",
            "total_gamma_before",
            "hedge_underlying_units_delta_only",
            "hedge_option_contract_key",
            "hedge_option_type",
            "hedge_option_strike",
            "hedge_option_expiration",
            "hedge_option_price_t",
            "hedge_option_price_tplus1",
            "hedge_option_delta",
            "hedge_option_gamma",
            "hedge_option_units",
            "hedge_underlying_units_delta_gamma",
            "residual_delta_after_hedge",
            "residual_gamma_after_hedge",
            "hedge_status_delta_gamma",
        ]
    ].rename(columns={"hedge_status_delta_gamma": "hedge_status"})

    return daily_out, trades_out


def build_performance_summary(hedged_daily: pd.DataFrame) -> pd.DataFrame:
    unhedged_days = int(len(hedged_daily))
    delta_hedged_days = int((hedged_daily["hedge_status_delta_only"] == "hedged").sum())
    dg_hedged_days = int((hedged_daily["hedge_status_delta_gamma"] == "hedged").sum())

    rows = [
        compute_perf_row(
            "unhedged",
            hedged_daily["portfolio_daily_pnl_unhedged"],
            hedged_days=unhedged_days,
        ),
        compute_perf_row(
            "delta_hedged",
            hedged_daily["daily_pnl_delta_hedged"],
            hedged_days=delta_hedged_days,
        ),
        compute_perf_row(
            "delta_gamma_hedged",
            hedged_daily["daily_pnl_delta_gamma_hedged"],
            hedged_days=dg_hedged_days,
        ),
    ]
    return pd.DataFrame(rows)


def build_hedge_effectiveness_summary(hedged_daily: pd.DataFrame) -> pd.DataFrame:
    abs_delta_before = np.abs(hedged_daily["total_delta_before"])
    abs_delta_after_do = np.abs(hedged_daily["residual_delta_after_delta_only"])
    abs_delta_after_dg = np.abs(hedged_daily["residual_delta_after_hedge"])
    abs_gamma_before = np.abs(hedged_daily["total_gamma_before"])
    abs_gamma_after_dg = np.abs(hedged_daily["residual_gamma_after_hedge"])

    n = len(hedged_daily)
    n_do = int((hedged_daily["hedge_status_delta_only"] == "hedged").sum())
    n_dg = int((hedged_daily["hedge_status_delta_gamma"] == "hedged").sum())
    n_dg_fail = n - n_dg

    rows = [
        {"metric": "mean_abs_delta_before_hedge", "value": float(abs_delta_before.mean())},
        {"metric": "mean_abs_delta_after_delta_only", "value": float(abs_delta_after_do.mean())},
        {"metric": "mean_abs_delta_after_delta_gamma", "value": float(abs_delta_after_dg.mean())},
        {"metric": "mean_abs_gamma_before_hedge", "value": float(abs_gamma_before.mean())},
        {"metric": "mean_abs_gamma_after_delta_gamma", "value": float(abs_gamma_after_dg.mean())},
        {"metric": "pct_days_delta_only_hedged", "value": float(n_do / n) if n > 0 else np.nan},
        {"metric": "pct_days_delta_gamma_hedged", "value": float(n_dg / n) if n > 0 else np.nan},
        {"metric": "num_days_without_valid_delta_gamma_hedge", "value": float(n_dg_fail)},
    ]

    reason_counts = (
        hedged_daily.loc[hedged_daily["hedge_status_delta_gamma"] != "hedged", "hedge_status_delta_gamma"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("metric")
        .reset_index(name="value")
    )
    if not reason_counts.empty:
        reason_counts["metric"] = "reason_" + reason_counts["metric"].astype(str)
        rows.extend(reason_counts.to_dict("records"))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Delta-only si delta-gamma hedge pentru portofoliul realist strict (exact-contract)."
    )
    parser.add_argument(
        "--positions-greeks",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_positions_with_greeks_strict.csv"),
        help="Folosit pentru detectia underlying-ului principal.",
    )
    parser.add_argument(
        "--daily-greeks",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_daily_greeks_strict.csv"),
    )
    parser.add_argument(
        "--daily-pnl",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_daily_pnl_realistic_strict.csv"),
    )
    parser.add_argument(
        "--options",
        type=pathlib.Path,
        default=pathlib.Path("options_eod_all_with_iv.csv"),
    )
    parser.add_argument("--underlying-symbol", type=str, default="", help="Daca gol, se detecteaza automat.")
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--hedge-t-min", type=float, default=0.08, help="Maturitate minima pentru optiunea de hedge.")
    parser.add_argument("--hedge-t-max", type=float, default=0.25, help="Maturitate maxima pentru optiunea de hedge.")
    parser.add_argument(
        "--min-abs-hedge-gamma",
        type=float,
        default=1e-5,
        help="Prag minim absolut gamma al optiunii de hedge pentru a evita pozitii extreme.",
    )
    parser.add_argument(
        "--contract-multiplier",
        type=float,
        default=1.0,
        help="Multiplier pentru contractul de optiune in hedge (default 1).",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Fallback r daca lipseste in options dataset.",
    )
    parser.add_argument(
        "--dividend-yield",
        type=float,
        default=0.0,
        help="Fallback q daca lipseste in options dataset.",
    )
    parser.add_argument("--out-daily", type=pathlib.Path, default=pathlib.Path("hedged_daily_pnl.csv"))
    parser.add_argument("--out-trades", type=pathlib.Path, default=pathlib.Path("hedge_trades_daily.csv"))
    parser.add_argument("--out-performance", type=pathlib.Path, default=pathlib.Path("hedged_performance_summary.csv"))
    parser.add_argument("--out-effectiveness", type=pathlib.Path, default=pathlib.Path("hedge_effectiveness_summary.csv"))
    args = parser.parse_args()

    for p in [args.positions_greeks, args.daily_greeks, args.daily_pnl, args.options]:
        if not p.exists():
            sys.exit(f"Nu gasesc fisierul: {p}")
    if args.hedge_t_min <= 0 or args.hedge_t_max <= args.hedge_t_min:
        sys.exit("Interval invalid pentru maturitatea optiunii de hedge.")
    if args.min_abs_hedge_gamma <= 0:
        sys.exit("--min-abs-hedge-gamma trebuie sa fie > 0.")
    if args.contract_multiplier <= 0:
        sys.exit("--contract-multiplier trebuie sa fie > 0.")

    try:
        print("Pas 1/6: citesc expunerile zilnice si pnl-ul unhedged...")
        daily = load_daily_inputs(args.daily_greeks, args.daily_pnl)
        if daily.empty:
            raise ValueError("Nu exista date zilnice pentru hedging.")

        print("Pas 2/6: detectez underlying-ul principal...")
        underlying = infer_underlying(args.positions_greeks, args.underlying_symbol)

        print("Pas 3/6: incarc subsetul de optiuni pentru zilele necesare...")
        needed_dates = set(daily["quote_date"].unique()) | set(daily["next_quote_date"].dropna().unique())
        options = load_options_subset(
            options_path=args.options,
            needed_dates=needed_dates,
            underlying_symbol=underlying,
            chunksize=args.chunksize,
        )

        print("Pas 4/6: construiesc hedge-ul delta-only si delta-gamma...")
        hedged_daily, hedge_trades = build_hedging_outputs(
            daily=daily,
            options=options,
            default_r=float(args.risk_free_rate),
            default_q=float(args.dividend_yield),
            hedge_t_min=float(args.hedge_t_min),
            hedge_t_max=float(args.hedge_t_max),
            min_abs_hedge_gamma=float(args.min_abs_hedge_gamma),
            contract_multiplier=float(args.contract_multiplier),
        )

        print("Pas 5/6: calculez sumarul de performanta si eficienta hedge...")
        perf = build_performance_summary(hedged_daily)
        eff = build_hedge_effectiveness_summary(hedged_daily)

        print("Pas 6/6: salvez outputurile...")
        out_daily_cols = [
            "quote_date",
            "portfolio_daily_pnl_unhedged",
            "spot",
            "spot_tplus1",
            "delta_only_hedge_pnl",
            "daily_pnl_delta_hedged",
            "cumulative_pnl_delta_hedged",
            "delta_gamma_option_hedge_pnl",
            "delta_gamma_underlying_hedge_pnl",
            "daily_pnl_delta_gamma_hedged",
            "cumulative_pnl_delta_gamma_hedged",
            "hedge_status_delta_only",
            "hedge_status_delta_gamma",
        ]
        hedged_daily[out_daily_cols].to_csv(args.out_daily, index=False)
        hedge_trades.to_csv(args.out_trades, index=False)
        perf.to_csv(args.out_performance, index=False)
        eff.to_csv(args.out_effectiveness, index=False)

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare: {exc}")

    n_do = int((hedged_daily["hedge_status_delta_only"] == "hedged").sum())
    n_dg = int((hedged_daily["hedge_status_delta_gamma"] == "hedged").sum())
    print("\n=== Summary ===")
    print(f"Underlying folosit: {underlying}")
    print(f"Zile totale: {len(hedged_daily)}")
    print(f"Zile hedged delta-only: {n_do}")
    print(f"Zile hedged delta-gamma: {n_dg}")
    print(f"Output daily: {args.out_daily}")
    print(f"Output hedge trades: {args.out_trades}")
    print(f"Output performance: {args.out_performance}")
    print(f"Output effectiveness: {args.out_effectiveness}")


if __name__ == "__main__":
    main()

