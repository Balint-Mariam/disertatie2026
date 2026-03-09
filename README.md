# IV Project

Pipeline pentru:
- curatarea datelor de optiuni (`clean date.py`)
- calculul implied volatility (`iv.py`)
- statistici descriptive (`iv_descriptives.py`)
- ploturi pentru IV surface (`plot_iv_surface_day.py`, `plot_iv_surface_day_v2.py`)

## Cerinte

- Python 3.10+
- pachete:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`

Instalare rapida:

```bash
pip install numpy pandas scipy matplotlib
```

## 1) Curatare date

```bash
python "clean date.py" --csv options_eod_all.csv --rates "div yield and rfr.csv" --out options_eod_all_clean.csv
```

## 2) Calcul IV

```bash
python iv.py --csv options_eod_all_clean.csv --rates "div yield and rfr.csv" --out options_eod_all_with_iv.csv
```

`iv.py` foloseste `r_annual`/`q_annual` din input daca exista deja; altfel le calculeaza din fisierul de rate.

## 3) Statistici descriptive IV

```bash
python iv_descriptives.py --csv options_eod_all_with_iv.csv
```

## 4) IV Surface (v1)

```bash
python plot_iv_surface_day.py --csv options_eod_all_with_iv.csv --n-days 4 --no-show
```

## 5) IV Surface (v2, log-moneyness, smooth-only)

```bash
python plot_iv_surface_day_v2.py --csv options_eod_all_with_iv.csv --n-days 0 --no-show
```

Note:
- `--n-days 0` = toate zilele disponibile.
- imaginile sunt salvate implicit in `plots_v2/`.

## GitHub: publicare repo

```bash
git init
git add .
git commit -m "Initial IV cleaning and surface pipeline"
git branch -M main
git remote add origin <URL_REPO_GITHUB>
git push -u origin main
```

`README.md` si `.gitignore` nu afecteaza rularea scripturilor.
