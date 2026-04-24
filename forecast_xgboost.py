import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecast_naive import (
    add_temporal_split,
    apply_quality_filter,
    evaluate_split,
    load_wide_dataset,
)


def parse_lags(lags_str: str) -> list[int]:
    try:
        lags = sorted({int(x.strip()) for x in lags_str.split(",") if x.strip()})
    except Exception as exc:
        raise ValueError("Format invalid pentru --lags. Ex: 1,2,3,5,10") from exc
    if not lags or any(l <= 0 for l in lags):
        raise ValueError("Lagurile trebuie sa fie intregi pozitivi.")
    return lags


def build_node_supervised_table(
    y: np.ndarray,
    dates: pd.Series,
    splits: pd.Series,
    lags: list[int],
    add_rolling: bool,
    add_time_features: bool,
) -> tuple[pd.DataFrame, list[str]]:
    s = pd.Series(y, dtype=float)
    df = pd.DataFrame({"target": s, "quote_date": pd.to_datetime(dates), "split": splits})

    feature_cols: list[str] = []
    for lag in lags:
        col = f"lag_{lag}"
        df[col] = s.shift(lag)
        feature_cols.append(col)

    if add_rolling:
        s_prev = s.shift(1)
        df["roll_mean_5"] = s_prev.rolling(window=5, min_periods=2).mean()
        df["roll_std_5"] = s_prev.rolling(window=5, min_periods=2).std()
        feature_cols += ["roll_mean_5", "roll_std_5"]

    if add_time_features:
        qd = pd.to_datetime(df["quote_date"], errors="coerce")
        df["dow"] = qd.dt.weekday.astype(float)
        feature_cols += ["dow"]

    # target trebuie sa existe; features cu NaN sunt permise (XGBoost le trateaza).
    df = df[np.isfinite(df["target"].to_numpy(dtype=float))].copy()
    return df, feature_cols


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return np.nan, np.nan
    err = y_pred[mask] - y_true[mask]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


def candidate_param_grid() -> list[dict]:
    # Set mic, robust, usor de explicat (fara grid search mare).
    return [
        {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 300,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    ]


def fit_predict_node_xgboost(
    node_df: pd.DataFrame,
    feature_cols: list[str],
    lags: list[int],
    min_train_samples: int,
    min_val_samples: int,
    random_state: int,
    xgb_n_jobs: int,
) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()

    out_summary = {
        "status": "ok",
        "n_train_samples": 0,
        "n_val_samples": 0,
        "n_test_samples": 0,
        "n_test_predictions": 0,
        "lags_used": ",".join(str(l) for l in lags),
        "feature_count": len(feature_cols),
        "best_params": "",
        "val_rmse": np.nan,
        "val_mae": np.nan,
        "fit_seconds": np.nan,
        "error_message": "",
    }

    train_df = node_df[node_df["split"] == "train"].copy()
    val_df = node_df[node_df["split"] == "validation"].copy()
    test_df = node_df[node_df["split"] == "test"].copy()
    out_summary["n_train_samples"] = int(len(train_df))
    out_summary["n_val_samples"] = int(len(val_df))
    out_summary["n_test_samples"] = int(len(test_df))

    if len(train_df) < min_train_samples:
        out_summary["status"] = "skipped_too_few_train_samples"
        out_summary["fit_seconds"] = float(time.time() - t0)
        return test_df[["quote_date"]].assign(forecast=np.nan), out_summary

    if len(val_df) < min_val_samples:
        out_summary["status"] = "skipped_too_few_val_samples"
        out_summary["fit_seconds"] = float(time.time() - t0)
        return test_df[["quote_date"]].assign(forecast=np.nan), out_summary

    # Alegere parametri pe validation, train -> validation (fara leakage).
    X_train = train_df[feature_cols]
    y_train = train_df["target"].to_numpy(dtype=float)
    X_val = val_df[feature_cols]
    y_val = val_df["target"].to_numpy(dtype=float)

    best_cfg = None
    best_val_rmse = np.inf
    best_val_mae = np.inf

    common_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": xgb_n_jobs,
        "missing": np.nan,
    }

    for cfg in candidate_param_grid():
        params = {**common_params, **cfg}
        try:
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            val_rmse, val_mae = rmse_mae(y_val, pred_val)
            if np.isnan(val_rmse):
                continue
            if (val_rmse < best_val_rmse) or (
                np.isclose(val_rmse, best_val_rmse, atol=1e-12) and val_mae < best_val_mae
            ):
                best_val_rmse = val_rmse
                best_val_mae = val_mae
                best_cfg = cfg
        except Exception:
            continue

    if best_cfg is None:
        out_summary["status"] = "fit_error_no_valid_candidate"
        out_summary["fit_seconds"] = float(time.time() - t0)
        return test_df[["quote_date"]].assign(forecast=np.nan), out_summary

    out_summary["best_params"] = json.dumps(best_cfg, ensure_ascii=True)
    out_summary["val_rmse"] = float(best_val_rmse)
    out_summary["val_mae"] = float(best_val_mae)

    # Refit pe train+validation, apoi predict one-step pe test
    # (features sunt laguri/rolling bazate doar pe trecutul observat).
    trainval_df = node_df[node_df["split"].isin(["train", "validation"])].copy()
    X_trainval = trainval_df[feature_cols]
    y_trainval = trainval_df["target"].to_numpy(dtype=float)
    X_test = test_df[feature_cols]

    try:
        final_model = XGBRegressor(**{**common_params, **best_cfg})
        final_model.fit(X_trainval, y_trainval)
        test_pred = final_model.predict(X_test)
    except Exception as exc:
        out_summary["status"] = "fit_or_predict_error"
        out_summary["error_message"] = str(exc)
        out_summary["fit_seconds"] = float(time.time() - t0)
        return test_df[["quote_date"]].assign(forecast=np.nan), out_summary

    out_summary["n_test_predictions"] = int(np.isfinite(test_pred).sum())
    out_summary["fit_seconds"] = float(time.time() - t0)
    return test_df[["quote_date"]].assign(forecast=test_pred), out_summary


def run_xgboost_pipeline(
    df: pd.DataFrame,
    node_cols: list[str],
    lags: list[int],
    add_rolling: bool,
    add_time_features: bool,
    min_train_samples: int,
    min_val_samples: int,
    random_state: int,
    xgb_n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    n_nodes = len(node_cols)
    pred_matrix = np.full((n_rows, n_nodes), np.nan, dtype=float)
    fit_rows = []

    for j, node in enumerate(node_cols, start=1):
        y = pd.to_numeric(df[node], errors="coerce").to_numpy(dtype=float)
        node_table, feature_cols = build_node_supervised_table(
            y=y,
            dates=df["quote_date"],
            splits=df["split"],
            lags=lags,
            add_rolling=add_rolling,
            add_time_features=add_time_features,
        )

        summary_base = {"node": node}
        if node_table.empty:
            fit_rows.append(
                {
                    **summary_base,
                    "status": "skipped_no_valid_rows",
                    "n_train_samples": 0,
                    "n_val_samples": 0,
                    "n_test_samples": 0,
                    "n_test_predictions": 0,
                    "lags_used": ",".join(str(l) for l in lags),
                    "feature_count": 0,
                    "best_params": "",
                    "val_rmse": np.nan,
                    "val_mae": np.nan,
                    "fit_seconds": 0.0,
                    "error_message": "",
                }
            )
            continue

        try:
            node_forecast_df, node_summary = fit_predict_node_xgboost(
                node_df=node_table,
                feature_cols=feature_cols,
                lags=lags,
                min_train_samples=min_train_samples,
                min_val_samples=min_val_samples,
                random_state=random_state,
                xgb_n_jobs=xgb_n_jobs,
            )
            # Pun predicțiile pe pozițiile corecte din dataframe-ul full.
            if not node_forecast_df.empty:
                idx_map = df.reset_index().set_index("quote_date")["index"]
                valid_fc = node_forecast_df[np.isfinite(node_forecast_df["forecast"].to_numpy(dtype=float))]
                if not valid_fc.empty:
                    idx = idx_map.reindex(valid_fc["quote_date"]).to_numpy(dtype=float)
                    mask = np.isfinite(idx)
                    if np.any(mask):
                        pred_matrix[idx[mask].astype(int), j - 1] = valid_fc["forecast"].to_numpy(dtype=float)[mask]

            fit_rows.append({**summary_base, **node_summary})
        except Exception as exc:
            fit_rows.append(
                {
                    **summary_base,
                    "status": "node_error",
                    "n_train_samples": 0,
                    "n_val_samples": 0,
                    "n_test_samples": 0,
                    "n_test_predictions": 0,
                    "lags_used": ",".join(str(l) for l in lags),
                    "feature_count": len(feature_cols),
                    "best_params": "",
                    "val_rmse": np.nan,
                    "val_mae": np.nan,
                    "fit_seconds": 0.0,
                    "error_message": str(exc),
                }
            )

        if j % 25 == 0 or j == n_nodes:
            ok_nodes = sum(1 for r in fit_rows if r["status"] == "ok")
            print(f"XGBoost progress: {j}/{n_nodes} nodes processed | ok={ok_nodes}")

    pred_df = pd.concat(
        [
            df[["quote_date", "split"]].reset_index(drop=True),
            pd.DataFrame(pred_matrix, columns=node_cols),
        ],
        axis=1,
    )
    fit_summary_df = pd.DataFrame(fit_rows)
    fit_summary_df = fit_summary_df[
        [
            "node",
            "status",
            "n_train_samples",
            "n_val_samples",
            "n_test_samples",
            "n_test_predictions",
            "lags_used",
            "feature_count",
            "best_params",
            "val_rmse",
            "val_mae",
            "fit_seconds",
            "error_message",
        ]
    ]
    return pred_df, fit_summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Forecast XGBoost pe noduri IV grid, comparabil cu benchmarkul naiv si ARIMA."
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

    parser.add_argument("--lags", type=str, default="1,2,3,5,10")
    parser.add_argument("--no-rolling", action="store_true", help="Dezactiveaza rolling mean/std simple.")
    parser.add_argument("--no-time-features", action="store_true", help="Dezactiveaza feature temporal day-of-week.")
    parser.add_argument("--min-train-samples", type=int, default=80)
    parser.add_argument("--min-val-samples", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--xgb-n-jobs", type=int, default=2)

    parser.add_argument("--out-forecast", type=pathlib.Path, default=pathlib.Path("forecast_xgboost_test.csv"))
    parser.add_argument("--out-metrics-overall", type=pathlib.Path, default=pathlib.Path("metrics_xgboost_overall.csv"))
    parser.add_argument("--out-metrics-by-node", type=pathlib.Path, default=pathlib.Path("metrics_xgboost_by_node.csv"))
    parser.add_argument("--out-fit-summary", type=pathlib.Path, default=pathlib.Path("xgboost_fit_summary.csv"))
    args = parser.parse_args()

    if not args.wide.exists():
        sys.exit(f"Nu gasesc fisierul wide: {args.wide}")
    if args.min_coverage < 0 or args.min_coverage > 1:
        sys.exit("min-coverage trebuie sa fie in [0, 1].")
    if args.split_mode == "date" and (not args.train_end or not args.val_end):
        sys.exit("In split-mode=date trebuie sa specifici --train-end si --val-end.")

    try:
        lags = parse_lags(args.lags)

        print("Pas 1/5: citesc datasetul wide...")
        df, node_cols = load_wide_dataset(args.wide)
        print(f"Zile totale initiale: {len(df)}")
        print(f"Noduri grid: {len(node_cols)}")

        print("Pas 2/5: aplic filtrul de calitate (optional)...")
        df = apply_quality_filter(
            df=df,
            day_stats_path=args.day_stats,
            min_coverage=args.min_coverage,
            use_filter=(not args.skip_quality_filter),
        )
        if len(df) < 3:
            raise ValueError("Prea putine zile dupa filtrare pentru split train/validation/test.")

        print("Pas 3/5: creez split temporal fara leakage...")
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

        print("Pas 4/5: rulez XGBoost pe fiecare nod...")
        pred_df, fit_summary_df = run_xgboost_pipeline(
            df=df,
            node_cols=node_cols,
            lags=lags,
            add_rolling=(not args.no_rolling),
            add_time_features=(not args.no_time_features),
            min_train_samples=args.min_train_samples,
            min_val_samples=args.min_val_samples,
            random_state=args.random_state,
            xgb_n_jobs=args.xgb_n_jobs,
        )

        print("Pas 5/5: evaluez out-of-sample pe test (RMSE/MAE)...")
        forecast_test, metrics_overall, metrics_by_node = evaluate_split(
            df=df,
            pred_df=pred_df,
            node_cols=node_cols,
            split_label="test",
        )
        metrics_overall["model"] = "xgboost"
        metrics_by_node["model"] = "xgboost"

        for p in [
            args.out_forecast,
            args.out_metrics_overall,
            args.out_metrics_by_node,
            args.out_fit_summary,
        ]:
            p.parent.mkdir(parents=True, exist_ok=True)
        forecast_test.to_csv(args.out_forecast, index=False)
        metrics_overall.to_csv(args.out_metrics_overall, index=False)
        metrics_by_node.to_csv(args.out_metrics_by_node, index=False)
        fit_summary_df.to_csv(args.out_fit_summary, index=False)

    except Exception as exc:  # pylint: disable=broad-except
        sys.exit(f"Eroare: {exc}")

    ok_nodes = int((fit_summary_df["status"] == "ok").sum())
    skipped_nodes = int((fit_summary_df["status"] != "ok").sum())
    print("\n=== Summary ===")
    print(f"Lags: {lags}")
    print(f"Nodes ok: {ok_nodes}")
    print(f"Nodes skipped/error: {skipped_nodes}")
    print(f"Forecast file: {args.out_forecast}")
    print(f"Overall metrics: {args.out_metrics_overall}")
    print(f"By-node metrics: {args.out_metrics_by_node}")
    print(f"Fit summary: {args.out_fit_summary}")


if __name__ == "__main__":
    main()
