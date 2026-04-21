"""
============================================================
  BUILD_DATASET.PY  –  Smart Retail Master Dataset Builder
  Merges sales_data.csv + retail_store_inventory.csv and
  engineers all features needed by the ML pipeline.
============================================================
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

# ── 1. Load raw data ──────────────────────────────────────
sales = pd.read_csv("/mnt/user-data/uploads/sales_data.csv", parse_dates=["Sale_Date"])
inv   = pd.read_csv("/mnt/user-data/uploads/retail_store_inventory.csv", parse_dates=["Date"])

# Normalise category names so they match across both files
cat_map = {"Groceries": "Food"}
inv["Category"] = inv["Category"].replace(cat_map)

# ── 2. Aggregate inventory to (Date, Region, Category) ───
inv_agg = (
    inv.groupby([inv["Date"].dt.to_period("M").dt.to_timestamp(), "Region", "Category"])
    .agg(
        Inventory_Level     = ("Inventory Level",    "mean"),
        Units_Sold_Online   = ("Units Sold",         "sum"),
        Units_Ordered       = ("Units Ordered",      "sum"),
        Demand_Forecast     = ("Demand Forecast",    "mean"),
        Avg_Price           = ("Price",              "mean"),
        Avg_Discount_Inv    = ("Discount",           "mean"),
        Weather_Mode        = ("Weather Condition",  lambda x: x.mode().iloc[0]),
        Holiday_Days        = ("Holiday/Promotion",  "sum"),
        Competitor_Pricing  = ("Competitor Pricing", "mean"),
        Seasonality         = ("Seasonality",        lambda x: x.mode().iloc[0]),
    )
    .reset_index()
    .rename(columns={"Date": "Month"})
)

# ── 3. Prepare sales data ─────────────────────────────────
sales["Month"] = sales["Sale_Date"].dt.to_period("M").dt.to_timestamp()
sales_agg = (
    sales.groupby(["Month", "Region", "Product_Category"])
    .agg(
        Total_Sales_Amount  = ("Sales_Amount",   "sum"),
        Total_Qty_Sold      = ("Quantity_Sold",  "sum"),
        Avg_Unit_Cost       = ("Unit_Cost",      "mean"),
        Avg_Unit_Price      = ("Unit_Price",     "mean"),
        Avg_Discount_Sales  = ("Discount",       "mean"),
        Num_Transactions    = ("Product_ID",     "count"),
        New_Customers       = ("Customer_Type",  lambda x: (x == "New").sum()),
        Returning_Customers = ("Customer_Type",  lambda x: (x == "Returning").sum()),
        Online_Txn          = ("Sales_Channel",  lambda x: (x == "Online").sum()),
        Retail_Txn          = ("Sales_Channel",  lambda x: (x == "Retail").sum()),
    )
    .reset_index()
    .rename(columns={"Product_Category": "Category"})
)

# ── 4. Merge ──────────────────────────────────────────────
df = sales_agg.merge(inv_agg, on=["Month", "Region", "Category"], how="left")

# ── 5. Feature Engineering ────────────────────────────────
# NOTE: Unit_Cost and Unit_Price in sales_data are per-TRANSACTION totals (not per-unit).
# Gross_Profit is computed at transaction level then aggregated.

# --- Core financials ---
df["Gross_Profit"]       = df["Total_Sales_Amount"] - (df["Avg_Unit_Cost"] * df["Num_Transactions"])
df["Profit_Margin_Pct"]  = (df["Gross_Profit"] / df["Total_Sales_Amount"].replace(0, np.nan)) * 100
df["Avg_Revenue_Per_Txn"]= df["Total_Sales_Amount"] / df["Num_Transactions"].replace(0, np.nan)

# --- Transportation cost estimate (2% of total cost of goods) ---
df["Transport_Cost_Est"] = df["Avg_Unit_Cost"] * df["Num_Transactions"] * 0.02

# --- Net Profit after transport ---
df["Net_Profit"]         = df["Gross_Profit"] - df["Transport_Cost_Est"]
df["Net_Profit_Margin"]  = (df["Net_Profit"] / df["Total_Sales_Amount"].replace(0, np.nan)) * 100

# --- Spot (retail) vs online split ---
df["Spot_Sales_Ratio"]   = df["Retail_Txn"] / df["Num_Transactions"].replace(0, np.nan)
df["Online_Sales_Ratio"] = df["Online_Txn"] / df["Num_Transactions"].replace(0, np.nan)

# --- Price competitiveness ---
df["Price_vs_Competitor"]= df["Avg_Unit_Price"] - df["Competitor_Pricing"].fillna(df["Avg_Unit_Price"])

# --- Demand fulfilment rate ---
df["Fulfillment_Rate"]   = (df["Units_Sold_Online"] / df["Demand_Forecast"].replace(0, np.nan)).clip(0, 1)

# --- Inventory adequacy ---
df["Inventory_Coverage"] = df["Inventory_Level"] / (df["Total_Qty_Sold"] / 30).replace(0, np.nan)  # days of stock

# --- Month / Year time features ---
df["Year"]  = df["Month"].dt.year
df["MonthNum"] = df["Month"].dt.month
df["Quarter"]  = df["Month"].dt.quarter

# --- Recommended selling price (cost × (1 + target 40% margin)) ---
df["Recommended_Price"]  = df["Avg_Unit_Cost"] * 1.40

# --- Qty to order to meet forecast + 15% safety stock ---
df["Recommended_Order_Qty"] = (df["Demand_Forecast"] * 1.15).round().fillna(df["Units_Ordered"])

# --- MoM growth (per region+category) ---
df = df.sort_values(["Region", "Category", "Month"])
df["Sales_Growth_MoM"]  = df.groupby(["Region","Category"])["Total_Sales_Amount"].pct_change() * 100
df["Profit_Growth_MoM"] = df.groupby(["Region","Category"])["Net_Profit"].pct_change() * 100

# ── 6. Save master dataset ───────────────────────────────
os.makedirs("/home/claude/smart_retail_ml", exist_ok=True)
out = "/home/claude/smart_retail_ml/master_dataset.csv"
df.to_csv(out, index=False)
print(f"✅  Master dataset saved → {out}")
print(f"   Shape : {df.shape}")
print(f"   Columns ({len(df.columns)}):\n   " + "\n   ".join(df.columns.tolist()))
