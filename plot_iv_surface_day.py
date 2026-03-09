import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def choose_target_date(csv_path: pathlib.Path, date_str: str, chunksize: int) -> pd.Timestamp:
    if date_str:
        target = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(target):
            raise ValueError("Data invalida. Foloseste format YYYY-MM-DD.")
        return target

    counts: dict[str, int] = {}
    for chunk in pd.read_csv(csv_path, usecols=["quote_date"], chunksize=chunksize):
        qd = parse_dates(chunk["quote_date"])
        vc = qd.dt.strftime("%Y-%m-%d").value_counts(dropna=True)
        for d, c in vc.items():
            counts[d] = counts.get(d, 0) + int(c)
    if not counts:
        raise ValueError("Nu am gasit quote_date valide in fisier.")
    best_day = max(counts, key=lambda d: counts[d])
    return pd.to_datetime(best_day)


def select_consecutive_dates(csv_path: pathlib.Path, start_date_str: str, chunksize: int, n_days: int) -> list[pd.Timestamp]:
    available = set()
    for chunk in pd.read_csv(csv_path, usecols=["quote_date"], chunksize=chunksize):
        qd = parse_dates(chunk["quote_date"]).dt.strftime("%Y-%m-%d")
        available.update(qd.dropna().unique())

    dates = sorted(available)
    if not dates:
        raise ValueError("Nu am gasit zile valide in fisier.")

    if start_date_str:
        start = pd.to_datetime(start_date_str, errors="coerce")
        if pd.isna(start):
            raise ValueError("Data invalida. Foloseste format YYYY-MM-DD.")
        start_s = start.strftime("%Y-%m-%d")
        idx = 0
        while idx < len(dates) and dates[idx] < start_s:
            idx += 1
        if idx >= len(dates):
            raise ValueError("Data de start este dupa ultima zi disponibila in fisier.")
    else:
        idx = 0

    selected = dates[idx : idx + max(n_days, 1)]
    if len(selected) < max(n_days, 1):
        raise ValueError(f"Nu exista suficiente zile disponibile pentru {n_days} zile consecutive.")
    return [pd.to_datetime(d) for d in selected]


def load_day_data(csv_path: pathlib.Path, target: pd.Timestamp, chunksize: int) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    required = {"quote_date", "strike", "implied_vol"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane obligatorii: {', '.join(sorted(missing))}")

    usecols = ["quote_date", "strike", "implied_vol"]
    for c in ["T", "expiration", "spot", "underlying_bid_1545", "underlying_ask_1545", "k_over_s", "moneyness"]:
        if c in header.columns:
            usecols.append(c)

    day_parts = []
    target_str = target.strftime("%Y-%m-%d")
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        qd = parse_dates(chunk["quote_date"])
        mask = qd.dt.strftime("%Y-%m-%d").eq(target_str)
        if not mask.any():
            continue
        sub = chunk.loc[mask].copy()
        sub["quote_date"] = qd.loc[mask]
        day_parts.append(sub)

    if not day_parts:
        raise ValueError(f"Nu exista observatii pentru data {target_str}.")

    day = pd.concat(day_parts, ignore_index=True)
    day["implied_vol"] = pd.to_numeric(day["implied_vol"], errors="coerce")
    day["strike"] = pd.to_numeric(day["strike"], errors="coerce")

    if "T" in day.columns:
        day["T"] = pd.to_numeric(day["T"], errors="coerce")
    elif "expiration" in day.columns:
        day["expiration"] = parse_dates(day["expiration"])
        day["T"] = (day["expiration"] - day["quote_date"]).dt.days / 365.0
    else:
        raise ValueError("Lipseste T si nu exista expiration pentru calcul maturitate.")

    if "k_over_s" in day.columns:
        day["k_over_s"] = pd.to_numeric(day["k_over_s"], errors="coerce")
    elif "moneyness" in day.columns:
        day["k_over_s"] = pd.to_numeric(day["moneyness"], errors="coerce")
    elif "spot" in day.columns:
        day["spot"] = pd.to_numeric(day["spot"], errors="coerce")
        day["k_over_s"] = day["strike"] / day["spot"]
    elif {"underlying_bid_1545", "underlying_ask_1545"}.issubset(day.columns):
        bid = pd.to_numeric(day["underlying_bid_1545"], errors="coerce")
        ask = pd.to_numeric(day["underlying_ask_1545"], errors="coerce")
        spot = (bid + ask) / 2.0
        day["k_over_s"] = day["strike"] / spot
    else:
        raise ValueError("Nu pot calcula K/S (lipsesc spot si underlying bid/ask).")

    day = day.replace([np.inf, -np.inf], np.nan)
    day = day.dropna(subset=["k_over_s", "T", "implied_vol"])
    day = day[(day["k_over_s"] > 0) & (day["T"] >= 0.08) & (day["implied_vol"] > 0)]
    if len(day) < 10:
        raise ValueError("Prea putine observatii valide pentru a construi o suprafata robusta.")
    return day


def build_regular_grid(x: np.ndarray, y: np.ndarray, grid_nx: int, grid_ny: int):
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    gx = np.linspace(x_min, x_max, grid_nx)
    gy = np.linspace(y_min, y_max, grid_ny)
    grid_x, grid_y = np.meshgrid(gx, gy)
    return grid_x, grid_y


def interpolate_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str,
    fill_nearest: bool,
) -> tuple[np.ndarray, str]:
    points = np.column_stack([x, y])
    z_grid = griddata(points, z, (grid_x, grid_y), method=method)
    used_method = method

    if np.isnan(z_grid).all() and method == "cubic":
        z_grid = griddata(points, z, (grid_x, grid_y), method="linear")
        used_method = "linear"

    if fill_nearest and np.isnan(z_grid).any():
        z_nn = griddata(points, z, (grid_x, grid_y), method="nearest")
        z_grid = np.where(np.isnan(z_grid), z_nn, z_grid)
        used_method = f"{used_method}+nearest_fill"

    return z_grid, used_method


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


def plot_surface(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    z_grid: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    z_obs: np.ndarray,
    title: str,
    out_path: pathlib.Path,
    overlay_points: bool,
    show: bool,
):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        grid_y,
        grid_x,
        z_grid,
        cmap="viridis",
        linewidth=0.2,
        edgecolor=(0, 0, 0, 0.15),
        antialiased=True,
        alpha=0.95,
    )

    if overlay_points:
        ax.scatter(y_obs, x_obs, z_obs, s=6, c="black", alpha=0.22)

    ax.set_xlabel("T (years)")
    ax.set_ylabel("Moneyness K/S")
    ax.set_zlabel("Implied Volatility")
    ax.view_init(elev=24, azim=35)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.65, pad=0.1, label="Implied Vol")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    print(f"Wrote {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="IV surface (grid interpolation + optional smoothing) for one day.")
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("options_eod_all_with_iv.csv"))
    parser.add_argument("--date", type=str, default="", help="YYYY-MM-DD pentru ziua de start (optional).")
    parser.add_argument("--n-days", type=int, default=4, help="Cate zile consecutive (din cele disponibile) sa proceseze.")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("plots"))
    parser.add_argument("--grid-nx", type=int, default=50, help="Grid points on moneyness axis.")
    parser.add_argument("--grid-ny", type=int, default=50, help="Grid points on maturity axis.")
    parser.add_argument("--interp", type=str, default="linear", choices=["linear", "cubic"])
    parser.add_argument("--fill-nearest", action="store_true", help="Fill interpolation NaN margins with nearest.")
    parser.add_argument("--smooth-sigma", type=float, default=0.8, help="Gaussian smoothing sigma; 0 disables smoothing.")
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"Nu gasesc fisierul: {args.csv}")
    if args.grid_nx < 20 or args.grid_ny < 20:
        sys.exit("Grila prea rara. Foloseste cel putin 20x20.")

    try:
        selected_dates = select_consecutive_dates(args.csv, args.date, args.chunksize, args.n_days)

        for i, day_ts in enumerate(selected_dates):
            day = load_day_data(args.csv, day_ts, args.chunksize)
            x = day["k_over_s"].to_numpy(dtype=float)
            y = day["T"].to_numpy(dtype=float)
            z = day["implied_vol"].to_numpy(dtype=float)
            grid_x, grid_y = build_regular_grid(x, y, args.grid_nx, args.grid_ny)

            z_interp, used_interp = interpolate_surface(
                x=x,
                y=y,
                z=z,
                grid_x=grid_x,
                grid_y=grid_y,
                method=args.interp,
                fill_nearest=args.fill_nearest,
            )
            z_smooth = smooth_surface(z_interp, sigma=args.smooth_sigma)

            d = day_ts.strftime("%Y-%m-%d")
            out_raw = args.outdir / f"iv_surface_{d}_interp.png"
            out_smooth = args.outdir / f"iv_surface_{d}_smooth.png"
            out_overlay = args.outdir / f"iv_surface_{d}_smooth_overlay.png"

            plot_surface(
                grid_x,
                grid_y,
                z_interp,
                x,
                y,
                z,
                title=f"IV Surface (Interpolated) | {d}",
                out_path=out_raw,
                overlay_points=False,
                show=False,
            )
            plot_surface(
                grid_x,
                grid_y,
                z_smooth,
                x,
                y,
                z,
                title=f"IV Surface (Smoothed) | {d}",
                out_path=out_smooth,
                overlay_points=False,
                show=False,
            )
            plot_surface(
                grid_x,
                grid_y,
                z_smooth,
                x,
                y,
                z,
                title=f"IV Surface (Smoothed + Observed Points) | {d}",
                out_path=out_overlay,
                overlay_points=True,
                show=((not args.no_show) and (i == len(selected_dates) - 1)),
            )

            nan_ratio = 100.0 * np.isnan(z_interp).sum() / z_interp.size
            print("\n=== Run Summary ===")
            print(f"Selected date: {d}")
            print(f"Observed points used: {len(day)}")
            print(f"Grid: {args.grid_nx} x {args.grid_ny}")
            print(f"Interpolation: {used_interp}")
            print(f"Interpolated grid NaN ratio: {nan_ratio:.2f}%")
            print(f"Smoothing sigma: {args.smooth_sigma}")
            print(f"Smoothing applied: {'yes' if args.smooth_sigma > 0 else 'no'}")

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare: {exc}")


if __name__ == "__main__":
    main()
