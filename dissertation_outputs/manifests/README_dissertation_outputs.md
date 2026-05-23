# Dissertation Outputs Manifest

This folder contains final tables and figures prepared for dissertation reporting.

## Tables

- `tables/data_cleaning_summary.csv`: Sample sizes along data cleaning and IV steps (**Methodology**)
- `tables/data_cleaning_retention_by_step.csv`: Retention by major pipeline stage (**Methodology**)
- `tables/grid_dataset_summary.csv`: Grid design and final dataset structure (**Methodology**)
- `tables/daily_grid_coverage_summary.csv`: Daily coverage statistics on grid (**Methodology**)
- `tables/final_forecast_model_comparison.csv`: Naive vs ARIMA vs XGBoost forecast metrics (**Empirical results**)
- `tables/forecast_model_ranking.csv`: Model ranking by RMSE/MAE (**Empirical results**)
- `tables/signals_summary.csv`: Economic signal generation overview (**Methodology / Empirical results**)
- `tables/positions_selection_summary.csv`: Selected positions statistics (**Empirical results**)
- `tables/backtest_simple_summary.csv`: Simple IV-space backtest performance (**Empirical results**)
- `tables/final_mapping_quality.csv`: Strict realistic mapping quality (**Empirical results**)
- `tables/realistic_backtest_summary.csv`: Realistic strict unhedged performance (**Empirical results**)
- `tables/greeks_summary.csv`: Greeks computation summary (**Methodology**)
- `tables/portfolio_exposures_summary.csv`: Portfolio exposure descriptive statistics (**Empirical results**)
- `tables/final_backtest_performance_comparison.csv`: Unhedged vs delta vs delta-gamma performance (**Empirical results**)
- `tables/final_hedge_effectiveness.csv`: Hedge effectiveness metrics (**Empirical results**)
- `tables/thesis_main_results_summary.csv`: Master summary table for thesis main text (**Empirical results**)

### Added Methodology Tables

- `tables/methodology_forecasting_model_setup.csv`: Forecast models setup and rationale (**Methodology**)
- `tables/methodology_trading_signal_rules.csv`: Trading signal construction rules (**Methodology**)
- `tables/methodology_backtesting_design.csv`: Backtesting design choices and rationale (**Methodology**)
- `tables/methodology_hedging_framework.csv`: Hedging framework definitions (**Methodology**)
- `tables/hypotheses_empirical_findings_summary.csv`: Hypotheses and empirical findings mapping (**Empirical results**)

## Figures

- `figures/figure_01_data_cleaning_retention.png`: Data cleaning and sample retention funnel/bar (**Methodology**)
- `figures/figure_02_grid_node_coverage_heatmap.png`: Grid node coverage heatmap (**Methodology**)
- `figures/figure_03_iv_surface_high_coverage.png`: Representative IV surface (high coverage day) (**Empirical results**)
- `figures/figure_04_iv_surface_median_coverage.png`: Representative IV surface (median coverage day) (**Empirical results**)
- `figures/figure_05_forecast_rmse_mae_comparison.png`: RMSE/MAE comparison across forecast models (**Empirical results**)
- `figures/figure_06_forecast_actual_vs_pred_*.png`: Actual vs forecast for representative node(s) (**Appendix**)
- `figures/figure_08_signal_strength_distribution.png`: Signal strength distribution (**Methodology**)
- `figures/figure_09_signal_direction_stacked_counts.png`: Long/Short/Flat signal counts over time (**Empirical results**)
- `figures/figure_10_simple_backtest_cumulative_pnl.png`: Cumulative PnL (simple backtest) (**Empirical results**)
- `figures/figure_11_simple_backtest_daily_pnl_distribution.png`: Daily PnL distribution (simple backtest) (**Appendix**)
- `figures/figure_12_realistic_unhedged_cumulative_pnl.png`: Cumulative PnL (realistic strict unhedged) (**Empirical results**)
- `figures/figure_13_realistic_unhedged_daily_pnl_distribution.png`: Daily PnL distribution (realistic strict) (**Appendix**)
- `figures/figure_14_portfolio_exposures_timeseries.png`: Delta/Gamma/Vega time series (unhedged) (**Empirical results**)
- `figures/figure_15_portfolio_exposures_boxplot.png`: Exposure distributions (**Appendix**)
- `figures/figure_16_hedging_cumulative_pnl_comparison.png`: Cumulative PnL: unhedged vs hedged (**Empirical results**)
- `figures/figure_17_hedging_daily_pnl_distribution.png`: Daily PnL distribution under hedging (**Appendix**)
- `figures/figure_18_hedge_effectiveness_before_after.png`: Mean abs delta/gamma before-after hedging (**Empirical results**)
- `figures/figure_19_delta_gamma_before_after_timeseries.png`: Delta/Gamma before vs after DG hedging (**Appendix**)

- `figures/figure_20_methodological_workflow.png`: End-to-end empirical workflow diagram (**Methodology chapter / Presentation slides**)

### Presentation-only Media

- `figures/iv_surface_animation.mp4`: Time animation of IV surface evolution (**Presentation only**)
- `figures/iv_surface_animation.gif`: Optional lightweight animation preview (**Presentation only**)

## Appendix Files

- `appendix/daily_grid_coverage_detail.csv`
- `appendix/forecast_metrics_by_node_naive.csv`
- `appendix/forecast_metrics_by_node_arima.csv`
- `appendix/forecast_metrics_by_node_xgboost.csv`
- `appendix/portfolio_daily_greeks_detail.csv`
- `appendix/hedged_daily_pnl_detail.csv`
- `appendix/hedge_trades_daily_detail.csv`

## Folder Guide

- `tables/`: final CSV tables for main text.
- `figures/`: final PNG/PDF figures for main text + presentation media.
- `appendix/`: detailed supporting tables for appendix.
- `manifests/`: manifest and documentation files.

## Notes

- Methodology and model outputs are not recomputed in this manifest step.
- Files are consolidated from existing pipeline outputs.
- Figure style is standardized via `dissertation_plot_style.py`.
- Excel workbook with consolidated tables: `tables/dissertation_tables.xlsx`.
