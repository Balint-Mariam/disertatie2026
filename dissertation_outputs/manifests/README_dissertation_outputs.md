# Dissertation Outputs Manifest

This folder contains final tables and figures prepared for dissertation reporting.

## Tables

- `tables/data_cleaning_summary.csv`: Sample sizes along data cleaning and IV steps (Suggested section: **Methodology**)
- `tables/data_cleaning_retention_by_step.csv`: Retention by major pipeline stage (Suggested section: **Methodology**)
- `tables/grid_dataset_summary.csv`: Grid design and final dataset structure (Suggested section: **Methodology**)
- `tables/daily_grid_coverage_summary.csv`: Daily coverage statistics on grid (Suggested section: **Methodology**)
- `tables/final_forecast_model_comparison.csv`: Naive vs ARIMA vs XGBoost forecast metrics (Suggested section: **Results**)
- `tables/forecast_model_ranking.csv`: Model ranking by RMSE/MAE (Suggested section: **Results**)
- `tables/signals_summary.csv`: Economic signal generation overview (Suggested section: **Methodology**)
- `tables/positions_selection_summary.csv`: Selected positions statistics (Suggested section: **Methodology**)
- `tables/backtest_simple_summary.csv`: Simple IV-space backtest performance (Suggested section: **Results**)
- `tables/final_mapping_quality.csv`: Strict realistic mapping quality (Suggested section: **Results**)
- `tables/realistic_backtest_summary.csv`: Realistic strict unhedged performance (Suggested section: **Results**)
- `tables/greeks_summary.csv`: Greeks computation summary (Suggested section: **Methodology**)
- `tables/portfolio_exposures_summary.csv`: Portfolio exposure descriptive statistics (Suggested section: **Results**)
- `tables/final_backtest_performance_comparison.csv`: Unhedged vs delta vs delta-gamma performance (Suggested section: **Results**)
- `tables/final_hedge_effectiveness.csv`: Hedge effectiveness metrics (Suggested section: **Results**)
- `tables/thesis_main_results_summary.csv`: Master summary table for thesis main text (Suggested section: **Results**)

## Figures

- `figures/figure_01_data_cleaning_retention.png`: Data cleaning and sample retention funnel/bar (Suggested section: **Methodology**)
- `figures/figure_02_grid_node_coverage_heatmap.png`: Grid node coverage heatmap (Suggested section: **Methodology**)
- `figures/figure_03_iv_surface_high_coverage.png`: Representative IV surface (high coverage day) (Suggested section: **Results**)
- `figures/figure_04_iv_surface_median_coverage.png`: Representative IV surface (median coverage day) (Suggested section: **Results**)
- `figures/figure_05_forecast_rmse_mae_comparison.png`: RMSE/MAE comparison across forecast models (Suggested section: **Results**)
- `figures/figure_06_forecast_actual_vs_pred_*.png`: Actual vs forecast for representative node(s) (Suggested section: **Appendix**)
- `figures/figure_08_signal_strength_distribution.png`: Signal strength distribution (Suggested section: **Methodology**)
- `figures/figure_09_signal_direction_stacked_counts.png`: Long/Short/Flat signal counts over time (Suggested section: **Results**)
- `figures/figure_10_simple_backtest_cumulative_pnl.png`: Cumulative PnL (simple backtest) (Suggested section: **Results**)
- `figures/figure_11_simple_backtest_daily_pnl_distribution.png`: Daily PnL distribution (simple backtest) (Suggested section: **Appendix**)
- `figures/figure_12_realistic_unhedged_cumulative_pnl.png`: Cumulative PnL (realistic strict unhedged) (Suggested section: **Results**)
- `figures/figure_13_realistic_unhedged_daily_pnl_distribution.png`: Daily PnL distribution (realistic strict) (Suggested section: **Appendix**)
- `figures/figure_14_portfolio_exposures_timeseries.png`: Delta/Gamma/Vega time series (unhedged) (Suggested section: **Results**)
- `figures/figure_15_portfolio_exposures_boxplot.png`: Exposure distributions (Suggested section: **Appendix**)
- `figures/figure_16_hedging_cumulative_pnl_comparison.png`: Cumulative PnL: unhedged vs hedged (Suggested section: **Results**)
- `figures/figure_17_hedging_daily_pnl_distribution.png`: Daily PnL distribution under hedging (Suggested section: **Appendix**)
- `figures/figure_18_hedge_effectiveness_before_after.png`: Mean abs delta/gamma before-after hedging (Suggested section: **Results**)
- `figures/figure_19_delta_gamma_before_after_timeseries.png`: Delta/Gamma before vs after DG hedging (Suggested section: **Appendix**)

## Folder Guide

- `tables/`: final CSV tables for main text.
- `figures/`: final PNG/PDF figures for main text.
- `appendix/`: detailed supporting tables for appendix.
- `manifests/`: manifest and documentation files.

## Notes

- Methodology and model outputs are not recomputed in this manifest step.
- Files are consolidated from existing pipeline outputs.
- Figure style is standardized via `dissertation_plot_style.py`.