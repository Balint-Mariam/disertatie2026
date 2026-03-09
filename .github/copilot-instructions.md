# Copilot instructions for this repo

Purpose
- This repo is a small data pipeline for option implied-volatility (IV) processing and visualization.

Quick architecture (what to know)
- Files of interest:
  - `import pandas as pd.py` — cleans raw options (`options_eod_all.csv`) and writes `options_eod_all_clean.csv` (filters, put-call parity, spread/practice filters).
  - `rfr.py` — builds `div yield and rfr.csv` from T-bill and dividend Excel files used as rate/dividend inputs.
  - `iv.py` — merges option rows with rates and computes implied vol per row (Brent root finding) and writes `options_eod_all_with_iv.csv`.
  - `bound_check.py` — removes rows that violate the no-arbitrage lower bound and writes `options_eod_all_with_iv_lb.csv`.
  - `plot_iv_surface.py` — plots 3D IV surface for a single quote date.

Data flow / CLI pipeline examples
- Typical sequence (run from repo root):
  1) Clean raw options:
     python "import pandas as pd.py" --csv options_eod_all.csv --out options_eod_all_clean.csv
  2) Prepare rates (run `rfr.py` which writes `div yield and rfr.csv`)
     python rfr.py
  3) Compute implied vols:
     python iv.py --csv options_eod_all_clean.csv --rates "div yield and rfr.csv" --out options_eod_all_with_iv.csv
  4) Drop lower-bound violators:
     python bound_check.py --csv options_eod_all_with_iv.csv --out options_eod_all_with_iv_lb.csv
  5) Plot IV surface:
     python plot_iv_surface.py --csv options_eod_all_with_iv.csv --date 2020-03-18

Important, repo-specific conventions & patterns
- Date handling: pandas parsing uses dayfirst=True throughout (CSV dates are day-first). Keep that convention when adding parsing code or tests.
- Chunked processing: Large CSVs are read with pandas `chunksize` and written using `mode='w'` for the first chunk and `mode='a'` afterwards. Re-running will overwrite unless you remove the output; be careful when testing.
- Column names: several scripts validate specific column sets (see `REQUIRED_COLUMNS` in `iv.py` and `bound_check.py`). Keep exact column names when modifying code or adding new columns.
- Implied volatility:
  - IV is computed by `iv.implied_vol` (Brent root finder between 1e-4 and 5.0); it returns NaN if not solvable — downstream code expects and handles NaNs.
  - T is computed as days / 365.0 (not trading days).
- Rates: `rfr.py` converts annual percent yields to daily rates using 252 trading days and stores columns `1M,2M,3M,4M,6M,12M`.
- Error messages and user-facing text are in Romanian; keep this consistent unless making a deliberate change across the repo.

Developer workflow tips
- Run small-scale tests by setting `--chunksize` small (e.g. 1000) or by piping a CSV slice before running to iterate quickly.
- To debug IV solver on a single row:
  - Use interactive Python: from `iv import implied_vol`; call implied_vol(mid, spot, strike, T, r, q, call=True)
- No automated tests discovered; add small unit tests around `implied_vol`, `lower_bound_price`, and the key `apply_filters` logic if you add CI.

Dependencies & environment
- Key Python libs: numpy, pandas, scipy, matplotlib. No requirements file present — use e.g.:
  pip install numpy pandas scipy matplotlib
- Python 3.8+ recommended (uses f-strings and modern pandas API).

Notes for AI agents
- Preserve dayfirst=True behavior when altering date parsing.
- Keep chunked read/write pattern (first chunk writes header; subsequent chunks append) to avoid unexpected memory growth.
- Avoid changing user-facing strings or column names without updating CLI help and validations.
- Use the explicit column sets (`REQUIRED_COLUMNS`) as sanity checks when creating new transforms.

If anything here is unclear or you'd like more detail (examples, refactoring suggestions, or a small test scaffold), tell me what to expand.  

