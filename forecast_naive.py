import argparse
import pathlib
import sys

import numpy as np
import pandas as pd


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def load_wide_dataset(wide_path: pathlib.Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(wide_path)
    if "quote_date" not in df.columns:
        raise ValueError("Lipseste coloana 'quote_date' in iv_grid_wide.csv.")

    node_cols = [c for c in df.columns if c != "quote_date"]
    if not node_cols:
        raise ValueError("Nu exista coloane de noduri IV in iv_grid_wide.csv.")

    df["quote_date"] = parse_dates(df["quote_date"])
    df = df.dropna(subset=["quote_date"]).copy()
    df = df.sort_values("quote_date").reset_index(drop=True)

    # Daca exista duplicate pe zi, agregam prin medie pe noduri.
    if df["quote_date"].duplicated().any():
        print("Atentie: duplicate pe quote_date detectate; agreg prin medie.")
        df = (
            df.groupby("quote_date", as_index=False)[node_cols]
            .mean(numeric_only=True)
            .sort_values("quote_date")
            .reset_index(drop=True)
        )

    return df, node_cols


def apply_quality_filter(
    df: pd.DataFrame,
    day_stats_path: pathlib.Path,
    min_coverage: float,
    use_filter: bool,
) -> pd.DataFrame:
    if not use_filter:
        return df
    if not day_stats_path.exists():
        raise ValueError(f"Nu gasesc fisierul de day stats: {day_stats_path}")

    ds = pd.read_csv(day_stats_path)
    required = {"quote_date", "coverage", "status"}
    missing = required - set(ds.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane in iv_grid_day_stats.csv: {', '.join(sorted(missing))}")

    ds["quote_date"] = parse_dates(ds["quote_date"])
    ds = ds.dropna(subset=["quote_date"]).copy()
    ds["status"] = ds["status"].astype(str).str.lower().str.strip()
    ds["coverage"] = pd.to_numeric(ds["coverage"], errors="coerce")

    good_days = ds[(ds["status"] == "kept") & (ds["coverage"] >= min_coverage)]["quote_date"].drop_duplicates()
    before = len(df)
    out = df[df["quote_date"].isin(good_days)].copy()
    out = out.sort_values("quote_date").reset_index(drop=True)
    print(f"Quality filter: kept {len(out)}/{before} zile (min_coverage={min_coverage:.2f}).")
    return out


def split_by_ratio(
    dates: pd.Series,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.Series:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio trebuie sa fie 1.0.")

    n = len(dates)
    n_train = int(np.floor(n * train_ratio))
    n_val = int(np.floor(n * val_ratio))
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Split invalid: fiecare segment (train/val/test) trebuie sa aiba cel putin 1 zi.")

    split = np.array(["train"] * n, dtype=object)
    split[n_train : n_train + n_val] = "validation"
    split[n_train + n_val :] = "test"
    return pd.Series(split, index=dates.index)


def split_by_dates(
    dates: pd.Series,
    train_end: str,
    val_end: str,
) -> pd.Series:
    train_end_ts = pd.to_datetime(train_end, errors="coerce")
    val_end_ts = pd.to_datetime(val_end, errors="coerce")
    if pd.isna(train_end_ts) or pd.isna(val_end_ts):
        raise ValueError("train_end si val_end trebuie sa fie date valide YYYY-MM-DD.")
    if train_end_ts >= val_end_ts:
        raise ValueError("train_end trebuie sa fie strict mai mic decat val_end.")

    split = pd.Series(index=dates.index, dtype="object")
    split[dates <= train_end_ts] = "train"
    split[(dates > train_end_ts) & (dates <= val_end_ts)] = "validation"
    split[dates > val_end_ts] = "test"

    for label in ["train", "validation", "test"]:
        if (split == label).sum() == 0:
            raise ValueError(f"Segmentul '{label}' este gol. Ajusteaza pragurile de split.")
    return split


def add_temporal_split(
    df: pd.DataFrame,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    train_end: str,
    val_end: str,
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("quote_date").reset_index(drop=True)

    if split_mode == "ratio":
        out["split"] = split_by_ratio(
            dates=out["quote_date"],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        out["split"] = split_by_dates(
            dates=out["quote_date"],
            train_end=train_end,
            val_end=val_end,
        )
    return out


def naive_forecast_previous_day(df: pd.DataFrame, node_cols: list[str]) -> pd.DataFrame:
    pred_values = df[node_cols].shift(1).reset_index(drop=True)
    base = df[["quote_date", "split"]].reset_index(drop=True)
    return pd.concat([base, pred_values], axis=1)


def evaluate_split(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    node_cols: list[str],
    split_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actual = df.loc[df["split"] == split_label, ["quote_date"] + node_cols].copy()
    pred = pred_df.loc[pred_df["split"] == split_label, ["quote_date"] + node_cols].copy()

    actual_long = actual.melt(id_vars="quote_date", var_name="node", value_name="actual")
    pred_long = pred.melt(id_vars="quote_date", var_name="node", value_name="forecast")
    out = actual_long.copy()
    out["forecast"] = pred_long["forecast"].to_numpy()
    out["split"] = split_label

    valid = np.isfinite(out["actual"].to_numpy(dtype=float)) & np.isfinite(out["forecast"].to_numpy(dtype=float))
    out = out.loc[valid].copy()
    if out.empty:
        raise ValueError(f"Nu exista observatii valide pentru evaluare pe split='{split_label}'.")

    out["error"] = out["forecast"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = out["error"] ** 2

    rmse = float(np.sqrt(out["sq_error"].mean()))
    mae = float(out["abs_error"].mean())
    overall = pd.DataFrame(
        [
            {
                "model": "naive_persistence",
                "split": split_label,
                "n_predictions": int(len(out)),
                "n_days": int(out["quote_date"].nunique()),
                "n_nodes": int(out["node"].nunique()),
                "rmse": rmse,
                "mae": mae,
            }
        ]
    )

    by_node = (
        out.groupby("node", as_index=False)
        .agg(
            n_predictions=("actual", "count"),
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
        )
    )
    by_node["rmse"] = np.sqrt(by_node["mse"])
    by_node = by_node.drop(columns=["mse"])
    by_node = by_node[["node", "n_predictions", "rmse", "mae"]]
    by_node.insert(0, "model", "naive_persistence")
    by_node.insert(1, "split", split_label)

    return out, overall, by_node


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark naiv (persistence) pentru IV grid forecasting."
    )
    parser.add_argument("--wide", type=pathlib.Path, default=pathlib.Path("iv_grid_wide.csv"))
    parser.add_argument("--day-stats", type=pathlib.Path, default=pathlib.Path("iv_grid_day_stats.csv"))
    parser.add_argument("--skip-quality-filter", action="store_true", help="Nu aplica filtrul de calitate.")
    parser.add_argument("--min-coverage", type=float, default=0.0, help="Prag minim coverage daca filtrul e activ.")

    parser.add_argument("--split-mode", type=str, default="ratio", choices=["ratio", "date"])
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--train-end", type=str, default="", help="Necesar in split-mode=date (YYYY-MM-DD).")
    parser.add_argument("--val-end", type=str, default="", help="Necesar in split-mode=date (YYYY-MM-DD).")

    parser.add_argument("--out-forecast", type=pathlib.Path, default=pathlib.Path("forecast_naive_test.csv"))
    parser.add_argument("--out-metrics-overall", type=pathlib.Path, default=pathlib.Path("metrics_naive_overall.csv"))
    parser.add_argument("--out-metrics-by-node", type=pathlib.Path, default=pathlib.Path("metrics_naive_by_node.csv"))
    args = parser.parse_args()

    if not args.wide.exists():
        sys.exit(f"Nu gasesc fisierul wide: {args.wide}")
    if args.min_coverage < 0 or args.min_coverage > 1:
        sys.exit("min-coverage trebuie sa fie in [0, 1].")
    if args.split_mode == "date" and (not args.train_end or not args.val_end):
        sys.exit("In split-mode=date trebuie sa specifici --train-end si --val-end.")

    try:
        print("Pas 1/4: citesc datasetul wide...")
        df, node_cols = load_wide_dataset(args.wide)
        print(f"Zile totale initiale: {len(df)}")
        print(f"Noduri grid: {len(node_cols)}")

        print("Pas 2/4: aplic filtrul de calitate (optional)...")
        df = apply_quality_filter(
            df=df,
            day_stats_path=args.day_stats,
            min_coverage=args.min_coverage,
            use_filter=(not args.skip_quality_filter),
        )
        if len(df) < 3:
            raise ValueError("Prea putine zile dupa filtrare pentru split train/validation/test.")

        print("Pas 3/4: creez split temporal fara leakage...")
        df = add_temporal_split(
            df=df,
            split_mode=args.split_mode,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            train_end=args.train_end,
            val_end=args.val_end,
        )
        split_counts = df["split"].value_counts()
        print(
            f"Split counts: train={split_counts.get('train', 0)}, "
            f"validation={split_counts.get('validation', 0)}, "
            f"test={split_counts.get('test', 0)}"
        )

        print("Pas 4/4: benchmark naiv (forecast_t = actual_{t-1}) pe test...")
        pred_df = naive_forecast_previous_day(df=df, node_cols=node_cols)
        forecast_test, metrics_overall, metrics_by_node = evaluate_split(
            df=df,
            pred_df=pred_df,
            node_cols=node_cols,
            split_label="test",
        )

        for p in [args.out_forecast, args.out_metrics_overall, args.out_metrics_by_node]:
            p.parent.mkdir(parents=True, exist_ok=True)
        forecast_test.to_csv(args.out_forecast, index=False)
        metrics_overall.to_csv(args.out_metrics_overall, index=False)
        metrics_by_node.to_csv(args.out_metrics_by_node, index=False)

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare: {exc}")

    print("\n=== Summary ===")
    print(f"Forecast file: {args.out_forecast}")
    print(f"Overall metrics: {args.out_metrics_overall}")
    print(f"By-node metrics: {args.out_metrics_by_node}")


if __name__ == "__main__":
    main()
