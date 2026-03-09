import pandas as pd

tbill_path = r"us_tbill_yields_2010_2025-09-30 (1).xlsx"
div_path = r"S&P500Index-Dividend-Yield.xlsx"
out_csv = "div yield and rfr.csv"
start_date = pd.Timestamp("2012-01-03")
end_date = pd.Timestamp("2025-09-30")

tbill = pd.read_excel(tbill_path)
tbill["Date"] = pd.to_datetime(tbill["Date"], dayfirst=True)
tbill = tbill[(tbill["Date"] >= start_date) & (tbill["Date"] <= end_date)]

# complete 2M with linear interpolation between 1M and 3M (midpoint)
mask_2m = tbill["2M"].isna()
tbill.loc[mask_2m, "2M"] = (tbill.loc[mask_2m, "1M"] + tbill.loc[mask_2m, "3M"]) / 2

# complete 4M with linear interpolation between 3M and 6M (1 month into a 3-month gap)
mask_4m = tbill["4M"].isna()
tbill.loc[mask_4m, "4M"] = (2 / 3) * tbill.loc[mask_4m, "3M"] + (1 / 3) * tbill.loc[mask_4m, "6M"]

# keep the maturities we need
tbill = tbill[["Date", "1M", "2M", "3M", "4M", "6M", "12M"]]

# convert annualized percent yields to daily rates (252 trading days)
for col in ["1M", "2M", "3M", "4M", "6M", "12M"]:
    tbill[col] = (1 + tbill[col] / 100.0) ** (1 / 252.0) - 1

div = pd.read_excel(div_path)
div["Calendar Date"] = pd.to_datetime(div["Calendar Date"], dayfirst=True)
div = div[(div["Calendar Date"] >= start_date) & (div["Calendar Date"] <= end_date)]

merged = pd.merge(
    div,
    tbill,
    left_on="Calendar Date",
    right_on="Date",
    how="inner",
)

merged = merged.drop(columns=["Date"])
merged = merged[(merged["Calendar Date"] >= start_date) & (merged["Calendar Date"] <= end_date)]

merged.to_csv(out_csv, index=False)
