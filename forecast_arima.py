import argparse
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from forecast_naive import (
    add_temporal_split,
    apply_quality_filter,
    evaluate_split,
    load_wide_dataset,
)


def parse_order(order_str: str) -> tuple[int, int, int]:
    try:
        p_str, d_str, q_str = order_str.split(",")
        p, d, q = int(p_str), int(d_str), int(q_str)
    except Exception as exc:
        raise ValueError("Format invalid pentru --order. Foloseste, de exemplu: 1,0,0") from exc
    if p < 0 or d < 0 or q < 0:
        raise ValueError("Parametrii ARIMA trebuie sa fie nenegativi.")
    return p, d, q


def fit_arima_and_predict_node(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    order: tuple[int, int, int],
    min_train_obs: int,
) -> tuple[np.ndarray, dict]:
    t0 = time.time()
    summary = {
        "status": "ok",
        "order": f"{order[0]},{order[1]},{order[2]}",
        "n_train_non_missing": int(np.isfinite(y_train).sum()),
        "n_val_non_missing": int(np.isfinite(y_val).sum()),
        "n_trainval_non_missing": 0,
        "n_test_non_missing": int(np.isfinite(y_test).sum()),
        "n_test_forecasts": 0,
        "n_test_updates": 0,
        "aic": np.nan,
        "bic": np.nan,
        "fit_seconds": np.nan,
        "error_message": "",
    }
    test_preds = np.full(len(y_test), np.nan, dtype=float)

    # Fit pe train+validation; test ramane strict out-of-sample.
    # Pastram timeline-ul (inclusiv NaN) si lasam Kalman filter sa trateze missing.
    y_trainval = np.concatenate([y_train, y_val]).astype(float)
    summary["n_trainval_non_missing"] = int(np.isfinite(y_trainval).sum())

    if summary["n_trainval_non_missing"] < min_train_obs:
        summary["status"] = "skipped_too_few_trainval_obs"
        summary["fit_seconds"] = float(time.time() - t0)
        return test_preds, summary

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_res = ARIMA(
                endog=y_trainval,
                order=order,
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        summary["aic"] = float(getattr(fit_res, "aic", np.nan))
        summary["bic"] = float(getattr(fit_res, "bic", np.nan))
    except Exception as exc:
        summary["status"] = "fit_error"
        summary["error_message"] = str(exc)
        summary["fit_seconds"] = float(time.time() - t0)
        return test_preds, summary

    # Rolling one-step-ahead pe test, eficient:
    # append cu toata secventa test (fara refit) + predicted_mean pe intervalul test.
    # Acesta produce aceleasi one-step forecasts ca loop-ul forecast(1)+append pe fiecare zi.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res_all = fit_res.append(y_test.astype(float), refit=False)
        start = len(y_trainval)
        end = start + len(y_test) - 1
        pred_obj = res_all.get_prediction(start=start, end=end)
        test_preds = np.asarray(pred_obj.predicted_mean, dtype=float)
    except Exception as exc:
        summary["status"] = "forecast_error"
        summary["error_message"] = str(exc)
        summary["fit_seconds"] = float(time.time() - t0)
        return np.full(len(y_test), np.nan, dtype=float), summary

    summary["n_test_forecasts"] = int(np.isfinite(test_preds).sum())
    summary["n_test_updates"] = int(np.isfinite(y_test).sum())
    summary["fit_seconds"] = float(time.time() - t0)
    return test_preds, summary


def run_arima_pipeline(
    df: pd.DataFrame,
    node_cols: list[str],
    order: tuple[int, int, int],
    min_train_obs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_arr = df["split"].to_numpy()
    train_mask = split_arr == "train"
    val_mask = split_arr == "validation"
    test_mask = split_arr == "test"
    test_indices = np.where(test_mask)[0]

    n_rows = len(df)
    n_nodes = len(node_cols)
    pred_matrix = np.full((n_rows, n_nodes), np.nan, dtype=float)
    fit_rows = []

    for j, node in enumerate(node_cols, start=1):
        y = pd.to_numeric(df[node], errors="coerce").to_numpy(dtype=float)
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]

        node_preds, fit_summary = fit_arima_and_predict_node(
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            order=order,
            min_train_obs=min_train_obs,
        )
        pred_matrix[test_indices, j - 1] = node_preds

        fit_summary["node"] = node
        fit_rows.append(fit_summary)

        if j % 25 == 0 or j == n_nodes:
            ok_nodes = sum(1 for r in fit_rows if r["status"] == "ok")
            print(f"ARIMA progress: {j}/{n_nodes} nodes processed | ok={ok_nodes}")

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
            "order",
            "n_train_non_missing",
            "n_val_non_missing",
            "n_trainval_non_missing",
            "n_test_non_missing",
            "n_test_forecasts",
            "n_test_updates",
            "aic",
            "bic",
            "fit_seconds",
            "error_message",
        ]
    ]
    return pred_df, fit_summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Forecast ARIMA pe noduri IV grid, cu split temporal comparabil cu benchmarkul naiv."
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

    parser.add_argument("--order", type=str, default="1,1,0", help="Ordin ARIMA p,d,q (ex: 1,1,0).")
    parser.add_argument("--min-train-obs", type=int, default=60, help="Minim observatii train finite pe nod.")

    parser.add_argument("--out-forecast", type=pathlib.Path, default=pathlib.Path("forecast_arima_test.csv"))
    parser.add_argument("--out-metrics-overall", type=pathlib.Path, default=pathlib.Path("metrics_arima_overall.csv"))
    parser.add_argument("--out-metrics-by-node", type=pathlib.Path, default=pathlib.Path("metrics_arima_by_node.csv"))
    parser.add_argument("--out-fit-summary", type=pathlib.Path, default=pathlib.Path("arima_fit_summary.csv"))
    args = parser.parse_args()

    if not args.wide.exists():
        sys.exit(f"Nu gasesc fisierul wide: {args.wide}")
    if args.min_coverage < 0 or args.min_coverage > 1:
        sys.exit("min-coverage trebuie sa fie in [0, 1].")
    if args.min_train_obs < 5:
        sys.exit("min-train-obs trebuie sa fie >= 5.")
    if args.split_mode == "date" and (not args.train_end or not args.val_end):
        sys.exit("In split-mode=date trebuie sa specifici --train-end si --val-end.")

    try:
        order = parse_order(args.order)

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

        print(f"Pas 4/5: rulez ARIMA({order[0]},{order[1]},{order[2]}) pe fiecare nod...")
        pred_df, fit_summary_df = run_arima_pipeline(
            df=df,
            node_cols=node_cols,
            order=order,
            min_train_obs=args.min_train_obs,
        )

        print("Pas 5/5: evaluez out-of-sample pe test (RMSE/MAE)...")
        forecast_test, metrics_overall, metrics_by_node = evaluate_split(
            df=df,
            pred_df=pred_df,
            node_cols=node_cols,
            split_label="test",
        )
        metrics_overall["model"] = "arima"
        metrics_by_node["model"] = "arima"

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
    print(f"ARIMA order: {order}")
    print(f"Nodes ok: {ok_nodes}")
    print(f"Nodes skipped/error: {skipped_nodes}")
    print(f"Forecast file: {args.out_forecast}")
    print(f"Overall metrics: {args.out_metrics_overall}")
    print(f"By-node metrics: {args.out_metrics_by_node}")
    print(f"Fit summary: {args.out_fit_summary}")


if __name__ == "__main__":
    main()
