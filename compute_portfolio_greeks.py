import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def load_traded_positions(positions_path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(positions_path)
    required = {
        "quote_date",
        "node",
        "signal",
        "contract_key",
        "option_type",
        "strike",
        "expiration",
        "entry_price",
        "weight",
        "trade_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Lipsesc coloane obligatorii in portfolio_positions_realistic_strict.csv: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    out["quote_date"] = parse_dates(out["quote_date"]).dt.normalize()
    out["expiration"] = parse_dates(out["expiration"]).dt.normalize()
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce")
    out["option_type"] = out["option_type"].astype(str).str.upper().str.strip().str[0]

    out = out[out["trade_status"] == "traded"].copy()
    out = out[np.isfinite(out["weight"]) & (out["weight"] != 0)].copy()
    out = out[np.isfinite(out["entry_price"]) & (out["entry_price"] > 0)].copy()
    out = out.dropna(subset=["quote_date", "contract_key", "option_type", "strike", "expiration"])

    if out.empty:
        raise ValueError("Nu exista pozitii tranzactionate valide dupa filtrele cerute.")
    return out


def prepare_option_columns(header_cols: Iterable[str]) -> list[str]:
    required = [
        "quote_date",
        "expiration",
        "strike",
        "option_type",
        "underlying_symbol",
        "root",
        "implied_vol",
    ]
    optional = [
        "spot",
        "underlying_bid_1545",
        "underlying_ask_1545",
        "T",
        "r_annual",
        "q_annual",
    ]

    missing = set(required) - set(header_cols)
    if missing:
        raise ValueError(
            "Lipsesc coloane obligatorii in options_eod_all_with_iv.csv: "
            + ", ".join(sorted(missing))
        )

    cols = [c for c in required if c in header_cols]
    cols += [c for c in optional if c in header_cols]
    return cols


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


def load_market_snapshot(
    options_path: pathlib.Path,
    needed_dates: set[pd.Timestamp],
    needed_contracts: set[str],
    chunksize: int,
) -> pd.DataFrame:
    header = pd.read_csv(options_path, nrows=0)
    usecols = prepare_option_columns(set(header.columns))

    parts: list[pd.DataFrame] = []
    needed_dates = {pd.Timestamp(d).normalize() for d in needed_dates}

    for chunk in pd.read_csv(options_path, usecols=usecols, chunksize=chunksize):
        chunk["quote_date"] = parse_dates(chunk["quote_date"]).dt.normalize()
        chunk = chunk[chunk["quote_date"].isin(needed_dates)]
        if chunk.empty:
            continue

        chunk["expiration"] = parse_dates(chunk["expiration"]).dt.normalize()
        chunk["strike"] = pd.to_numeric(chunk["strike"], errors="coerce")
        chunk["option_type"] = chunk["option_type"].astype(str).str.upper().str.strip().str[0]
        chunk["underlying_symbol"] = chunk["underlying_symbol"].astype(str).str.strip().str.upper()
        chunk["root"] = chunk["root"].astype(str).str.strip().str.upper()
        chunk["contract_key"] = build_contract_key(
            chunk["underlying_symbol"],
            chunk["root"],
            chunk["expiration"],
            chunk["strike"],
            chunk["option_type"],
        )

        chunk = chunk[chunk["contract_key"].isin(needed_contracts)]
        if chunk.empty:
            continue

        if "spot" in chunk.columns:
            chunk["spot"] = pd.to_numeric(chunk["spot"], errors="coerce")
        else:
            chunk["spot"] = np.nan

        if {"underlying_bid_1545", "underlying_ask_1545"}.issubset(chunk.columns):
            ub = pd.to_numeric(chunk["underlying_bid_1545"], errors="coerce")
            ua = pd.to_numeric(chunk["underlying_ask_1545"], errors="coerce")
            spot_alt = (ub + ua) / 2.0
            chunk["spot"] = chunk["spot"].fillna(spot_alt)

        if "T" in chunk.columns:
            chunk["T"] = pd.to_numeric(chunk["T"], errors="coerce")
        else:
            chunk["T"] = (chunk["expiration"] - chunk["quote_date"]).dt.days / 365.0

        chunk["implied_vol"] = pd.to_numeric(chunk["implied_vol"], errors="coerce")
        chunk["r_annual"] = pd.to_numeric(chunk.get("r_annual", np.nan), errors="coerce")
        chunk["q_annual"] = pd.to_numeric(chunk.get("q_annual", np.nan), errors="coerce")

        parts.append(
            chunk[
                [
                    "quote_date",
                    "contract_key",
                    "underlying_symbol",
                    "root",
                    "strike",
                    "expiration",
                    "option_type",
                    "spot",
                    "T",
                    "implied_vol",
                    "r_annual",
                    "q_annual",
                ]
            ].copy()
        )

    if not parts:
        raise ValueError("Nu am gasit date de piata pentru pozitii in options_eod_all_with_iv.csv.")

    market = pd.concat(parts, ignore_index=True)
    market = (
        market.groupby(["quote_date", "contract_key"], as_index=False)
        .agg(
            underlying_symbol=("underlying_symbol", "first"),
            root=("root", "first"),
            strike=("strike", "mean"),
            expiration=("expiration", "first"),
            option_type=("option_type", "first"),
            spot=("spot", "mean"),
            T=("T", "mean"),
            implied_vol=("implied_vol", "mean"),
            r_annual=("r_annual", "mean"),
            q_annual=("q_annual", "mean"),
        )
        .reset_index(drop=True)
    )
    return market


def merge_positions_with_market(positions: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    out = positions.merge(
        market.rename(
            columns={
                "underlying_symbol": "m_underlying_symbol",
                "root": "m_root",
                "strike": "m_strike",
                "expiration": "m_expiration",
                "option_type": "m_option_type",
                "spot": "m_spot",
                "T": "m_T",
                "implied_vol": "m_implied_vol",
                "r_annual": "m_r_annual",
                "q_annual": "m_q_annual",
            }
        ),
        on=["quote_date", "contract_key"],
        how="left",
        indicator=True,
    )
    out["market_match"] = out["_merge"].eq("both").astype(int)
    out = out.drop(columns=["_merge"])

    out["spot"] = pd.to_numeric(out["m_spot"], errors="coerce")
    out["T"] = pd.to_numeric(out["m_T"], errors="coerce")
    out["implied_vol"] = pd.to_numeric(out["m_implied_vol"], errors="coerce")
    out["r_annual_market"] = pd.to_numeric(out["m_r_annual"], errors="coerce")
    out["q_annual_market"] = pd.to_numeric(out["m_q_annual"], errors="coerce")
    return out


def compute_bs_greeks(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sqrt_t = np.sqrt(T)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    delta_call = disc_q * nd1
    delta_put = disc_q * (nd1 - 1.0)
    delta = np.where(is_call, delta_call, delta_put)

    gamma = disc_q * pdf_d1 / (S * vol_sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t

    theta_call = (
        -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
        - r * K * disc_r * nd2
        + q * S * disc_q * nd1
    )
    theta_put = (
        -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
        + r * K * disc_r * norm.cdf(-d2)
        - q * S * disc_q * norm.cdf(-d1)
    )
    theta = np.where(is_call, theta_call, theta_put)
    return delta, gamma, vega, theta


def attach_greeks(
    merged: pd.DataFrame,
    default_r: float,
    default_q: float,
) -> pd.DataFrame:
    out = merged.copy()
    out["r_annual"] = out["r_annual_market"].where(np.isfinite(out["r_annual_market"]), default_r)
    out["q_annual"] = out["q_annual_market"].where(np.isfinite(out["q_annual_market"]), default_q)
    out["used_default_r"] = (~np.isfinite(out["r_annual_market"])).astype(int)
    out["used_default_q"] = (~np.isfinite(out["q_annual_market"])).astype(int)

    out["greeks_status"] = "ok"
    out["greeks_reason"] = ""

    invalid_market = out["market_match"] != 1
    out.loc[invalid_market, ["greeks_status", "greeks_reason"]] = ["failed", "no_market_match"]

    invalid_spot = ~(np.isfinite(out["spot"]) & (out["spot"] > 0))
    out.loc[(out["greeks_status"] == "ok") & invalid_spot, ["greeks_status", "greeks_reason"]] = [
        "failed",
        "invalid_spot",
    ]

    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    invalid_strike = ~(np.isfinite(out["strike"]) & (out["strike"] > 0))
    out.loc[(out["greeks_status"] == "ok") & invalid_strike, ["greeks_status", "greeks_reason"]] = [
        "failed",
        "invalid_strike",
    ]

    invalid_t = ~(np.isfinite(out["T"]) & (out["T"] > 0))
    out.loc[(out["greeks_status"] == "ok") & invalid_t, ["greeks_status", "greeks_reason"]] = [
        "failed",
        "invalid_T",
    ]

    invalid_iv = ~(np.isfinite(out["implied_vol"]) & (out["implied_vol"] > 0))
    out.loc[(out["greeks_status"] == "ok") & invalid_iv, ["greeks_status", "greeks_reason"]] = [
        "failed",
        "invalid_implied_vol",
    ]

    invalid_type = ~out["option_type"].astype(str).str.upper().str[0].isin(["C", "P"])
    out.loc[(out["greeks_status"] == "ok") & invalid_type, ["greeks_status", "greeks_reason"]] = [
        "failed",
        "invalid_option_type",
    ]

    out["option_delta"] = np.nan
    out["option_gamma"] = np.nan
    out["option_vega"] = np.nan
    out["option_theta"] = np.nan

    valid = out["greeks_status"].eq("ok")
    if valid.any():
        is_call = out.loc[valid, "option_type"].astype(str).str.upper().str[0].eq("C").to_numpy()
        delta, gamma, vega, theta = compute_bs_greeks(
            S=out.loc[valid, "spot"].to_numpy(dtype=float),
            K=out.loc[valid, "strike"].to_numpy(dtype=float),
            T=out.loc[valid, "T"].to_numpy(dtype=float),
            r=out.loc[valid, "r_annual"].to_numpy(dtype=float),
            q=out.loc[valid, "q_annual"].to_numpy(dtype=float),
            sigma=out.loc[valid, "implied_vol"].to_numpy(dtype=float),
            is_call=is_call,
        )
        out.loc[valid, "option_delta"] = delta
        out.loc[valid, "option_gamma"] = gamma
        out.loc[valid, "option_vega"] = vega
        out.loc[valid, "option_theta"] = theta

    out["weighted_delta"] = out["weight"] * out["option_delta"]
    out["weighted_gamma"] = out["weight"] * out["option_gamma"]
    out["weighted_vega"] = out["weight"] * out["option_vega"]
    out["weighted_theta"] = out["weight"] * out["option_theta"]
    return out


def build_daily_greeks(positions_with_greeks: pd.DataFrame) -> pd.DataFrame:
    valid = positions_with_greeks[positions_with_greeks["greeks_status"] == "ok"].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "quote_date",
                "num_positions",
                "total_delta",
                "total_gamma",
                "total_vega",
                "total_theta",
                "gross_delta",
                "gross_gamma",
                "gross_vega",
                "gross_theta",
            ]
        )

    daily = (
        valid.groupby("quote_date", as_index=False)
        .agg(
            num_positions=("contract_key", "count"),
            total_delta=("weighted_delta", "sum"),
            total_gamma=("weighted_gamma", "sum"),
            total_vega=("weighted_vega", "sum"),
            total_theta=("weighted_theta", "sum"),
            gross_delta=("weighted_delta", lambda x: float(np.abs(x).sum())),
            gross_gamma=("weighted_gamma", lambda x: float(np.abs(x).sum())),
            gross_vega=("weighted_vega", lambda x: float(np.abs(x).sum())),
            gross_theta=("weighted_theta", lambda x: float(np.abs(x).sum())),
        )
        .sort_values("quote_date")
        .reset_index(drop=True)
    )
    return daily


def build_summary(
    positions_with_greeks: pd.DataFrame,
    daily_greeks: pd.DataFrame,
    default_r: float,
    default_q: float,
) -> pd.DataFrame:
    total_positions = len(positions_with_greeks)
    ok_positions = int((positions_with_greeks["greeks_status"] == "ok").sum())
    failed_positions = total_positions - ok_positions

    rows = [
        {"section": "overall", "metric": "num_days_analyzed", "value": float(daily_greeks["quote_date"].nunique())},
        {"section": "overall", "metric": "num_positions_input", "value": float(total_positions)},
        {"section": "overall", "metric": "num_positions_greeks_ok", "value": float(ok_positions)},
        {"section": "overall", "metric": "num_positions_greeks_failed", "value": float(failed_positions)},
        {"section": "assumptions", "metric": "default_risk_free_rate", "value": float(default_r)},
        {"section": "assumptions", "metric": "default_dividend_yield", "value": float(default_q)},
        {
            "section": "assumptions",
            "metric": "positions_using_default_r",
            "value": float(positions_with_greeks.loc[positions_with_greeks["greeks_status"] == "ok", "used_default_r"].sum()),
        },
        {
            "section": "assumptions",
            "metric": "positions_using_default_q",
            "value": float(positions_with_greeks.loc[positions_with_greeks["greeks_status"] == "ok", "used_default_q"].sum()),
        },
    ]

    if not daily_greeks.empty:
        for col in ["total_delta", "total_gamma", "total_vega", "total_theta"]:
            rows.append({"section": "exposure_stats", "metric": f"{col}_mean", "value": float(daily_greeks[col].mean())})
            rows.append({"section": "exposure_stats", "metric": f"{col}_std", "value": float(daily_greeks[col].std(ddof=1))})
            rows.append({"section": "exposure_stats", "metric": f"{col}_min", "value": float(daily_greeks[col].min())})
            rows.append({"section": "exposure_stats", "metric": f"{col}_max", "value": float(daily_greeks[col].max())})

    reason_counts = (
        positions_with_greeks.loc[positions_with_greeks["greeks_status"] != "ok", "greeks_reason"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("metric")
        .reset_index(name="value")
    )
    if not reason_counts.empty:
        reason_counts["section"] = "failed_reasons"
        rows.extend(reason_counts.to_dict("records"))
    else:
        rows.append({"section": "failed_reasons", "metric": "none", "value": 0.0})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Calculeaza Greeks pentru portofoliul realist strict (fara hedging)."
    )
    parser.add_argument(
        "--positions",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_positions_realistic_strict.csv"),
    )
    parser.add_argument(
        "--options",
        type=pathlib.Path,
        default=pathlib.Path("options_eod_all_with_iv.csv"),
    )
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Fallback annual risk-free rate daca r_annual lipseste din date.",
    )
    parser.add_argument(
        "--dividend-yield",
        type=float,
        default=0.0,
        help="Fallback annual dividend yield daca q_annual lipseste din date.",
    )
    parser.add_argument(
        "--out-positions",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_positions_with_greeks_strict.csv"),
    )
    parser.add_argument(
        "--out-daily",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_daily_greeks_strict.csv"),
    )
    parser.add_argument(
        "--out-summary",
        type=pathlib.Path,
        default=pathlib.Path("portfolio_greeks_summary_strict.csv"),
    )
    args = parser.parse_args()

    if not args.positions.exists():
        sys.exit(f"Nu gasesc fisierul de pozitii: {args.positions}")
    if not args.options.exists():
        sys.exit(f"Nu gasesc fisierul de optiuni: {args.options}")
    if not np.isfinite(args.risk_free_rate) or not np.isfinite(args.dividend_yield):
        sys.exit("risk-free-rate si dividend-yield trebuie sa fie valori finite.")

    try:
        print("Pas 1/6: citesc pozitiile stricte tranzactionate...")
        positions = load_traded_positions(args.positions)

        print("Pas 2/6: incarc snapshot-ul de piata pentru contractele necesare...")
        needed_dates = set(positions["quote_date"].unique())
        needed_contracts = set(positions["contract_key"].astype(str).unique())
        market = load_market_snapshot(
            options_path=args.options,
            needed_dates=needed_dates,
            needed_contracts=needed_contracts,
            chunksize=args.chunksize,
        )

        print("Pas 3/6: fac merge pozitii + date de piata...")
        merged = merge_positions_with_market(positions, market)

        print("Pas 4/6: calculez Greeks Black-Scholes pe pozitii...")
        pos_greeks = attach_greeks(
            merged,
            default_r=float(args.risk_free_rate),
            default_q=float(args.dividend_yield),
        )

        print("Pas 5/6: agreg expunerile zilnice...")
        daily = build_daily_greeks(pos_greeks)
        summary = build_summary(
            positions_with_greeks=pos_greeks,
            daily_greeks=daily,
            default_r=float(args.risk_free_rate),
            default_q=float(args.dividend_yield),
        )

        print("Pas 6/6: salvez outputurile...")
        keep_positions = [
            "quote_date",
            "node",
            "signal",
            "contract_key",
            "option_type",
            "strike",
            "expiration",
            "entry_price",
            "weight",
            "spot",
            "T",
            "implied_vol",
            "r_annual",
            "q_annual",
            "option_delta",
            "option_gamma",
            "option_vega",
            "option_theta",
            "weighted_delta",
            "weighted_gamma",
            "weighted_vega",
            "weighted_theta",
            "greeks_status",
            "greeks_reason",
            "used_default_r",
            "used_default_q",
        ]
        pos_out = pos_greeks[keep_positions].copy()

        keep_daily = [
            "quote_date",
            "num_positions",
            "total_delta",
            "total_gamma",
            "total_vega",
            "total_theta",
            "gross_delta",
            "gross_gamma",
            "gross_vega",
            "gross_theta",
        ]
        daily_out = daily[keep_daily].copy()

        for p in [args.out_positions, args.out_daily, args.out_summary]:
            p.parent.mkdir(parents=True, exist_ok=True)

        pos_out.to_csv(args.out_positions, index=False)
        daily_out.to_csv(args.out_daily, index=False)
        summary.to_csv(args.out_summary, index=False)

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare: {exc}")

    ok = int((pos_out["greeks_status"] == "ok").sum())
    failed = int((pos_out["greeks_status"] != "ok").sum())
    print("\n=== Summary ===")
    print(f"Positions input (traded, weight!=0): {len(pos_out)}")
    print(f"Positions with valid Greeks: {ok}")
    print(f"Positions failed Greeks: {failed}")
    print(f"Days with aggregated Greeks: {daily_out['quote_date'].nunique()}")
    print(f"Output positions: {args.out_positions}")
    print(f"Output daily Greeks: {args.out_daily}")
    print(f"Output summary: {args.out_summary}")


if __name__ == "__main__":
    main()
