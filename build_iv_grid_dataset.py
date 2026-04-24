import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def smooth_surface(z_grid: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return z_grid
    mask = np.isfinite(z_grid).astype(float)
    values = np.where(np.isfinite(z_grid), z_grid, 0.0)
    smooth_vals = gaussian_filter(values, sigma=sigma)
    smooth_mask = gaussian_filter(mask, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smooth_vals / smooth_mask
    out[smooth_mask < 1e-6] = np.nan
    return out


def prepare_points(
    csv_path: pathlib.Path,
    chunksize: int,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    cols = set(header.columns)
    required = {"quote_date", "strike", "implied_vol"}
    missing = required - cols
    if missing:
        raise ValueError(f"Lipsesc coloane obligatorii: {', '.join(sorted(missing))}")

    usecols = ["quote_date", "strike", "implied_vol"]
    for c in ["T", "expiration", "spot", "underlying_bid_1545", "underlying_ask_1545", "moneyness", "k_over_s"]:
        if c in cols:
            usecols.append(c)

    dt_from = pd.to_datetime(date_from, errors="coerce") if date_from else pd.NaT
    dt_to = pd.to_datetime(date_to, errors="coerce") if date_to else pd.NaT
    if date_from and pd.isna(dt_from):
        raise ValueError("`--date-from` invalid. Use YYYY-MM-DD.")
    if date_to and pd.isna(dt_to):
        raise ValueError("`--date-to` invalid. Use YYYY-MM-DD.")

    parts = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["quote_date"] = parse_dates(chunk["quote_date"])
        chunk["implied_vol"] = pd.to_numeric(chunk["implied_vol"], errors="coerce")
        chunk["strike"] = pd.to_numeric(chunk["strike"], errors="coerce")

        if "T" in chunk.columns:
            chunk["T"] = pd.to_numeric(chunk["T"], errors="coerce")
        elif "expiration" in chunk.columns:
            chunk["expiration"] = parse_dates(chunk["expiration"])
            chunk["T"] = (chunk["expiration"] - chunk["quote_date"]).dt.days / 365.0
        else:
            raise ValueError("Lipseste T si nu exista expiration pentru calcul maturitate.")

        if "k_over_s" in chunk.columns:
            chunk["k_over_s"] = pd.to_numeric(chunk["k_over_s"], errors="coerce")
        elif "moneyness" in chunk.columns:
            chunk["k_over_s"] = pd.to_numeric(chunk["moneyness"], errors="coerce")
        elif "spot" in chunk.columns:
            chunk["spot"] = pd.to_numeric(chunk["spot"], errors="coerce")
            chunk["k_over_s"] = chunk["strike"] / chunk["spot"]
        elif {"underlying_bid_1545", "underlying_ask_1545"}.issubset(chunk.columns):
            bid = pd.to_numeric(chunk["underlying_bid_1545"], errors="coerce")
            ask = pd.to_numeric(chunk["underlying_ask_1545"], errors="coerce")
            spot = (bid + ask) / 2.0
            chunk["k_over_s"] = chunk["strike"] / spot
        else:
            raise ValueError("Nu pot calcula K/S (lipsesc moneyness/k_over_s/spot).")

        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=["quote_date", "implied_vol", "k_over_s", "T"])
        chunk = chunk[(chunk["implied_vol"] > 0) & (chunk["k_over_s"] > 0) & (chunk["T"] >= t_min) & (chunk["T"] <= t_max)]
        if chunk.empty:
            continue

        if not pd.isna(dt_from):
            chunk = chunk[chunk["quote_date"] >= dt_from]
        if not pd.isna(dt_to):
            chunk = chunk[chunk["quote_date"] <= dt_to]
        if chunk.empty:
            continue

        chunk["log_moneyness"] = np.log(chunk["k_over_s"])
        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=["log_moneyness"])
        chunk = chunk[(chunk["log_moneyness"] >= x_min) & (chunk["log_moneyness"] <= x_max)]
        if chunk.empty:
            continue

        parts.append(
            chunk[["quote_date", "log_moneyness", "T", "implied_vol"]].copy()
        )

    if not parts:
        return pd.DataFrame(columns=["quote_date", "log_moneyness", "T", "implied_vol"])

    points = pd.concat(parts, ignore_index=True)
    points["quote_date"] = points["quote_date"].dt.strftime("%Y-%m-%d")
    return points


def build_grid_and_features(
    x_min: float,
    x_max: float,
    x_points: int,
    t_min: float,
    t_max: float,
    t_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    x_grid = np.linspace(x_min, x_max, x_points)
    t_grid = np.linspace(t_min, t_max, t_points)
    grid_x, grid_t = np.meshgrid(x_grid, t_grid)

    features = []
    grid_map_rows = []
    for t_idx, t_val in enumerate(t_grid):
        for x_idx, x_val in enumerate(x_grid):
            feat = f"iv_x{x_idx:02d}_t{t_idx:02d}"
            features.append(feat)
            grid_map_rows.append(
                {
                    "feature": feat,
                    "x_idx": x_idx,
                    "t_idx": t_idx,
                    "log_moneyness": float(x_val),
                    "T": float(t_val),
                }
            )
    grid_map = pd.DataFrame(grid_map_rows)
    return x_grid, t_grid, grid_x, grid_t, features, grid_map


def interpolate_day_surface(
    day_df: pd.DataFrame,
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    method: str,
    fill_nearest: bool,
    smooth_sigma: float,
) -> tuple[np.ndarray, str]:
    day_unique = (
        day_df.groupby(["log_moneyness", "T"], as_index=False)
        .agg(implied_vol=("implied_vol", "median"))
    )
    points = day_unique[["log_moneyness", "T"]].to_numpy(dtype=float)
    values = day_unique["implied_vol"].to_numpy(dtype=float)

    z_grid = griddata(points, values, (grid_x, grid_t), method=method)
    used_method = method

    if np.isnan(z_grid).all() and method == "cubic":
        z_grid = griddata(points, values, (grid_x, grid_t), method="linear")
        used_method = "linear"

    if fill_nearest and np.isnan(z_grid).any():
        z_nn = griddata(points, values, (grid_x, grid_t), method="nearest")
        z_grid = np.where(np.isnan(z_grid), z_nn, z_grid)
        used_method = f"{used_method}+nearest_fill"

    z_grid = smooth_surface(z_grid, sigma=smooth_sigma)
    return z_grid, used_method


def write_outputs(
    points: pd.DataFrame,
    out_long: pathlib.Path,
    out_wide: pathlib.Path,
    out_day_stats: pathlib.Path,
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    features: list[str],
    method: str,
    fill_nearest: bool,
    smooth_sigma: float,
    min_points_day: int,
    min_coverage: float,
    max_days: int,
):
    for p in [out_long, out_wide, out_day_stats]:
        if p.exists():
            p.unlink()

    dates = sorted(points["quote_date"].unique())
    if max_days > 0:
        dates = dates[:max_days]

    flat_x = grid_x.ravel()
    flat_t = grid_t.ravel()
    grid_size = grid_x.size

    first_long = True
    first_wide = True
    kept_days = 0
    stats_rows = []

    for i, day in enumerate(dates, start=1):
        day_df = points.loc[points["quote_date"] == day, ["log_moneyness", "T", "implied_vol"]]
        n_obs = len(day_df)
        n_unique = day_df[["log_moneyness", "T"]].drop_duplicates().shape[0]

        if n_unique < min_points_day:
            stats_rows.append(
                {
                    "quote_date": day,
                    "n_obs_raw": n_obs,
                    "n_obs_unique": n_unique,
                    "coverage": 0.0,
                    "status": "skipped_min_points",
                    "interp_used": "",
                }
            )
            continue

        z_grid, used_method = interpolate_day_surface(
            day_df=day_df,
            grid_x=grid_x,
            grid_t=grid_t,
            method=method,
            fill_nearest=fill_nearest,
            smooth_sigma=smooth_sigma,
        )

        coverage = float(np.isfinite(z_grid).sum() / grid_size)
        if coverage < min_coverage:
            stats_rows.append(
                {
                    "quote_date": day,
                    "n_obs_raw": n_obs,
                    "n_obs_unique": n_unique,
                    "coverage": coverage,
                    "status": "skipped_min_coverage",
                    "interp_used": used_method,
                }
            )
            continue

        z_flat = z_grid.ravel()

        long_df = pd.DataFrame(
            {
                "quote_date": day,
                "log_moneyness": flat_x,
                "T": flat_t,
                "iv_grid": z_flat,
            }
        )
        long_df.to_csv(out_long, mode="w" if first_long else "a", index=False, header=first_long)
        first_long = False

        wide_row = {"quote_date": day}
        wide_row.update(dict(zip(features, z_flat)))
        pd.DataFrame([wide_row]).to_csv(
            out_wide,
            mode="w" if first_wide else "a",
            index=False,
            header=first_wide,
        )
        first_wide = False

        stats_rows.append(
            {
                "quote_date": day,
                "n_obs_raw": n_obs,
                "n_obs_unique": n_unique,
                "coverage": coverage,
                "status": "kept",
                "interp_used": used_method,
            }
        )
        kept_days += 1

        if i % 50 == 0:
            print(f"Processed {i}/{len(dates)} days...")

    pd.DataFrame(stats_rows).to_csv(out_day_stats, index=False)
    return kept_days, len(dates)


def main():
    parser = argparse.ArgumentParser(
        description="Construieste dataset IV standardizat pe grila fixa (log-moneyness x T)."
    )
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("options_eod_all_with_iv.csv"))
    parser.add_argument("--out-long", type=pathlib.Path, default=pathlib.Path("iv_grid_long.csv"))
    parser.add_argument("--out-wide", type=pathlib.Path, default=pathlib.Path("iv_grid_wide.csv"))
    parser.add_argument("--out-grid-map", type=pathlib.Path, default=pathlib.Path("iv_grid_map.csv"))
    parser.add_argument("--out-day-stats", type=pathlib.Path, default=pathlib.Path("iv_grid_day_stats.csv"))
    parser.add_argument("--chunksize", type=int, default=200000)

    parser.add_argument("--x-min", type=float, default=-0.22)
    parser.add_argument("--x-max", type=float, default=0.18)
    parser.add_argument("--x-points", type=int, default=25)
    parser.add_argument("--t-min", type=float, default=0.08)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--t-points", type=int, default=20)

    parser.add_argument("--interp", type=str, default="linear", choices=["linear", "cubic"])
    parser.add_argument("--fill-nearest", action="store_true")
    parser.add_argument("--smooth-sigma", type=float, default=0.0, help="0 = fara smoothing")

    parser.add_argument("--min-points-day", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.0, help="0..1")
    parser.add_argument("--date-from", type=str, default="", help="YYYY-MM-DD")
    parser.add_argument("--date-to", type=str, default="", help="YYYY-MM-DD")
    parser.add_argument("--max-days", type=int, default=0, help="0 = toate zilele")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Nu gasesc fisierul: {args.csv}")
    if args.x_points < 2 or args.t_points < 2:
        sys.exit("x-points si t-points trebuie >= 2.")
    if args.min_coverage < 0 or args.min_coverage > 1:
        sys.exit("min-coverage trebuie sa fie in [0, 1].")

    try:
        print("Pas 1/3: incarc si pregatesc punctele IV brute...")
        points = prepare_points(
            csv_path=args.csv,
            chunksize=args.chunksize,
            x_min=args.x_min,
            x_max=args.x_max,
            t_min=args.t_min,
            t_max=args.t_max,
            date_from=args.date_from,
            date_to=args.date_to,
        )
        if points.empty:
            sys.exit("Nu exista puncte valide dupa filtre.")
        print(f"Puncte valide: {len(points)}")
        print(f"Zile unice: {points['quote_date'].nunique()}")

        print("Pas 2/3: construiesc grila fixa...")
        _, _, grid_x, grid_t, features, grid_map = build_grid_and_features(
            x_min=args.x_min,
            x_max=args.x_max,
            x_points=args.x_points,
            t_min=args.t_min,
            t_max=args.t_max,
            t_points=args.t_points,
        )
        grid_map.to_csv(args.out_grid_map, index=False)

        print("Pas 3/3: standardizez zilnic pe grila si scriu output...")
        kept_days, total_days = write_outputs(
            points=points,
            out_long=args.out_long,
            out_wide=args.out_wide,
            out_day_stats=args.out_day_stats,
            grid_x=grid_x,
            grid_t=grid_t,
            features=features,
            method=args.interp,
            fill_nearest=args.fill_nearest,
            smooth_sigma=args.smooth_sigma,
            min_points_day=args.min_points_day,
            min_coverage=args.min_coverage,
            max_days=args.max_days,
        )

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare: {exc}")

    print("\n=== Summary ===")
    print(f"Days processed: {total_days}")
    print(f"Days kept: {kept_days}")
    print(f"Grid size: {grid_x.shape[1]} x {grid_x.shape[0]} ({grid_x.size} puncte/zi)")
    print(f"Long dataset: {args.out_long}")
    print(f"Wide dataset: {args.out_wide}")
    print(f"Grid map: {args.out_grid_map}")
    print(f"Day stats: {args.out_day_stats}")


if __name__ == "__main__":
    main()
