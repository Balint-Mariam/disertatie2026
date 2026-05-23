import argparse
import pathlib

import pandas as pd


def build_methodology_forecasting_model_setup() -> pd.DataFrame:
    rows = [
        {
            "model": "Naive / Persistence",
            "role_in_analysis": "Benchmark model",
            "input_data": "Standardized IV grid time series (iv_grid_wide.csv)",
            "features_or_lags": "No estimated features; forecast_t+1 = IV_t",
            "training_scheme": "No training step",
            "validation_scheme": "Not applicable (deterministic benchmark)",
            "test_scheme": "Out-of-sample one-step-ahead over test window",
            "evaluation_metrics": "RMSE, MAE",
            "reason_for_inclusion": "Transparent baseline for incremental predictive value",
        },
        {
            "model": "ARIMA",
            "role_in_analysis": "Classical time-series benchmark",
            "input_data": "Standardized IV grid time series by node",
            "features_or_lags": "Autoregressive + moving-average dynamics per node",
            "training_scheme": "Fit per node using train/validation history only",
            "validation_scheme": "Temporal validation with no shuffle and no leakage",
            "test_scheme": "Rolling one-step-ahead out-of-sample per node",
            "evaluation_metrics": "RMSE, MAE",
            "reason_for_inclusion": "Econometric benchmark against persistence and ML",
        },
        {
            "model": "XGBoost",
            "role_in_analysis": "Main ML forecasting model",
            "input_data": "Standardized IV grid time series by node",
            "features_or_lags": "Lags (1,2,3,5,10), rolling mean/std, day-of-week features",
            "training_scheme": "Supervised learning on lagged node-level features",
            "validation_scheme": "Temporal validation for light hyperparameter selection",
            "test_scheme": "Rolling one-step-ahead out-of-sample per node",
            "evaluation_metrics": "RMSE, MAE",
            "reason_for_inclusion": "Flexible nonlinear dynamics; best OOS model in current pipeline",
        },
    ]
    return pd.DataFrame(rows)


def build_methodology_trading_signal_rules() -> pd.DataFrame:
    rows = [
        {
            "component": "forecast_iv",
            "definition": "Model-implied next-step volatility estimate at node/date",
            "economic_interpretation": "Expected implied volatility level",
            "implementation_details": "Taken from forecast_xgboost_test.csv for selected evaluation dates",
        },
        {
            "component": "observed_iv",
            "definition": "Current market implied volatility at node/date",
            "economic_interpretation": "Reference market valuation",
            "implementation_details": "Taken from standardized grid observed IV at signal date",
        },
        {
            "component": "forecast_minus_observed",
            "definition": "forecast_iv - observed_iv",
            "economic_interpretation": "Relative mispricing signal in IV space",
            "implementation_details": "Computed cross-sectionally each day before ranking",
        },
        {
            "component": "cross-sectional z-score",
            "definition": "Standardized deviation of forecast_minus_observed within day",
            "economic_interpretation": "Signal intensity relative to daily opportunity set",
            "implementation_details": "Daily normalization used to make thresholds comparable in time",
        },
        {
            "component": "LONG_VOL",
            "definition": "Positive high-strength signal",
            "economic_interpretation": "Implied volatility expected to increase / be underpriced",
            "implementation_details": "signal = +1 on top positive tail of cross-sectional ranking",
        },
        {
            "component": "SHORT_VOL",
            "definition": "Negative high-strength signal",
            "economic_interpretation": "Implied volatility expected to decrease / be overpriced",
            "implementation_details": "signal = -1 on bottom negative tail of cross-sectional ranking",
        },
        {
            "component": "FLAT",
            "definition": "Signal near zero or outside selected tails",
            "economic_interpretation": "No strong directional volatility view",
            "implementation_details": "signal = 0 and/or selected = 0",
        },
        {
            "component": "top_k_per_side",
            "definition": "Symmetric selection of strongest long and short signals",
            "economic_interpretation": "Controls concentration and improves comparability",
            "implementation_details": "Implemented in signal selection layer with per-day cross-section",
        },
        {
            "component": "daily rebalancing",
            "definition": "Portfolio refreshed each trading day based on new signals",
            "economic_interpretation": "Dynamic adaptation to changing IV surface expectations",
            "implementation_details": "One-day holding period with next-day realization",
        },
    ]
    return pd.DataFrame(rows)


def build_methodology_backtesting_design() -> pd.DataFrame:
    rows = [
        {
            "design_element": "portfolio universe",
            "chosen_specification": "Selected XGBoost signals mapped to real options",
            "rationale": "Connects statistical forecast layer to tradable option contracts",
        },
        {
            "design_element": "holding period",
            "chosen_specification": "One day (t to t+1)",
            "rationale": "Matches one-step forecast horizon and reduces path-dependence",
        },
        {
            "design_element": "rebalancing",
            "chosen_specification": "Daily",
            "rationale": "Keeps exposure aligned with fresh cross-sectional signals",
        },
        {
            "design_element": "weighting",
            "chosen_specification": "Side-neutral: +0.5 long side, -0.5 short side",
            "rationale": "Separates relative signal value from gross directional bias",
        },
        {
            "design_element": "simple IV-space backtest",
            "chosen_specification": "Signal × IV change synthetic diagnostic",
            "rationale": "Preliminary test of economic content before contract-level frictions",
        },
        {
            "design_element": "realistic strict backtest",
            "chosen_specification": "Real option prices with exact-contract t+1 requirement",
            "rationale": "Main economic test with conservative tradability constraints",
        },
        {
            "design_element": "mapping rule",
            "chosen_specification": "Nearest option at entry by maturity/log-moneyness",
            "rationale": "Operationally links grid-node signals to observable contracts",
        },
        {
            "design_element": "fallback rule",
            "chosen_specification": "Not used in main strict results",
            "rationale": "Avoids inter-contract mismatch and keeps interpretation clean",
        },
        {
            "design_element": "transaction costs",
            "chosen_specification": "Excluded in baseline",
            "rationale": "Keeps focus on signal informational value before cost modeling",
        },
        {
            "design_element": "PnL definition",
            "chosen_specification": "(price_t+1 - price_t) × portfolio weight",
            "rationale": "Direct contract-level payoff proxy under one-day horizon",
        },
    ]
    return pd.DataFrame(rows)


def build_methodology_hedging_framework() -> pd.DataFrame:
    rows = [
        {
            "hedge_type": "No hedge",
            "instruments_used": "None (baseline realistic option portfolio)",
            "exposure_target": "Unconstrained delta/gamma",
            "implementation_rule": "Use strict realistic portfolio returns as baseline",
            "performance_output": "PnL, Sharpe, max drawdown benchmark",
        },
        {
            "hedge_type": "Delta-only hedge",
            "instruments_used": "Underlying index proxy",
            "exposure_target": "Total portfolio delta",
            "implementation_rule": "Underlying units = -total_delta each day",
            "performance_output": "Delta-hedged daily/cumulative PnL and risk metrics",
        },
        {
            "hedge_type": "Delta-gamma hedge",
            "instruments_used": "Hedge option + underlying",
            "exposure_target": "Gamma neutral first, then residual delta neutral",
            "implementation_rule": "Option units = -total_gamma/option_gamma; underlying adjusts residual delta",
            "performance_output": "Delta-gamma hedged PnL, Sharpe, drawdown, residual exposures",
        },
        {
            "hedge_type": "Hedge option selection",
            "instruments_used": "ATM-like option with valid t and t+1 price",
            "exposure_target": "Stable gamma instrument",
            "implementation_rule": "Maturity window and liquidity filters from hedge script",
            "performance_output": "Hedge feasibility and effectiveness diagnostics",
        },
        {
            "hedge_type": "Evaluation layer",
            "instruments_used": "Portfolio-level analytics",
            "exposure_target": "Risk-return and exposure reduction",
            "implementation_rule": "Compare unhedged, delta, and delta-gamma variants",
            "performance_output": "PnL, Sharpe, max drawdown, mean abs delta/gamma before-after",
        },
    ]
    return pd.DataFrame(rows)


def build_hypotheses_empirical_findings_summary() -> pd.DataFrame:
    rows = [
        {
            "hypothesis": "H1 - IV surface has predictable structure",
            "empirical_test": "Out-of-sample RMSE/MAE comparison across Naive, ARIMA, XGBoost",
            "main_evidence": "ARIMA and especially XGBoost improve on naive in key forecasting metrics",
            "conclusion": "Supported / partially supported (model-dependent gain)",
        },
        {
            "hypothesis": "H2 - Forecast errors contain economic information",
            "empirical_test": "Signal-based backtests (simple IV-space and realistic strict portfolio)",
            "main_evidence": "Positive economic value appears when signals are transformed into portfolios",
            "conclusion": "Supported",
        },
        {
            "hypothesis": "H3 - Delta-gamma hedging isolates risk exposures",
            "empirical_test": "Before/after exposure diagnostics and hedged PnL variants",
            "main_evidence": "Mean absolute delta/gamma exposures are materially reduced post-hedging",
            "conclusion": "Supported",
        },
        {
            "hypothesis": "H4 - Economic value must be assessed beyond statistical accuracy",
            "empirical_test": "Compare unhedged vs hedged backtest outcomes and risk metrics",
            "main_evidence": "Risk-adjusted performance differs across hedging designs despite same forecasts",
            "conclusion": "Supported",
        },
    ]
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def update_workbook(
    workbook_path: pathlib.Path,
    tables: dict[str, pd.DataFrame],
) -> bool:
    if not workbook_path.exists():
        return False
    with pd.ExcelWriter(workbook_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Create methodological dissertation tables without rerunning heavy pipeline.")
    parser.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("dissertation_outputs"))
    parser.add_argument("--skip-xlsx", action="store_true", help="Skip updating dissertation_tables.xlsx")
    args = parser.parse_args()

    tables_dir = args.outdir / "tables"
    workbook_path = tables_dir / "dissertation_tables.xlsx"

    tables = {
        "methodology_forecasting_model_setup": build_methodology_forecasting_model_setup(),
        "methodology_trading_signal_rules": build_methodology_trading_signal_rules(),
        "methodology_backtesting_design": build_methodology_backtesting_design(),
        "methodology_hedging_framework": build_methodology_hedging_framework(),
        "hypotheses_empirical_findings_summary": build_hypotheses_empirical_findings_summary(),
    }

    for name, df in tables.items():
        write_csv(df, tables_dir / f"{name}.csv")

    xlsx_updated = False
    if not args.skip_xlsx:
        xlsx_updated = update_workbook(workbook_path=workbook_path, tables=tables)

    print("Methodology tables created:")
    for name in tables:
        print(f"- {tables_dir / f'{name}.csv'}")
    if args.skip_xlsx:
        print("Skipped workbook update (--skip-xlsx).")
    else:
        if xlsx_updated:
            print(f"Workbook updated: {workbook_path}")
        else:
            print(f"Workbook not found, skipped update: {workbook_path}")


if __name__ == "__main__":
    main()
