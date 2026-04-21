"""
============================================================
  ML_PIPELINE.PY  –  Smart Retail Intelligence Engine
  Models trained:
    1. Sales Amount Predictor        (RandomForest Regressor)
    2. Profit Margin Predictor       (GradientBoosting Regressor)
    3. Recommended Sell Qty Predictor(RandomForest Regressor)
    4. High / Low Profit Classifier  (RandomForest Classifier)
  Outputs:
    - predictions.csv
    - improvement_report.csv
    - model_scores.csv
    - feature_importance.csv
============================================================
"""

import pandas as pd
import numpy as np
import json, warnings, os
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, r2_score,
                             classification_report, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

OUT = "/home/claude/smart_retail_ml"
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════
# 0. Load master dataset
# ══════════════════════════════════════════════════════════
df = pd.read_csv(f"{OUT}/master_dataset.csv", parse_dates=["Month"])
print(f"Loaded master dataset: {df.shape}")

# ══════════════════════════════════════════════════════════
# 1. Encode categoricals
# ══════════════════════════════════════════════════════════
le = {}
for col in ["Region", "Category", "Weather_Mode", "Seasonality"]:
    df[col + "_enc"] = LabelEncoder().fit_transform(df[col].astype(str))

# ══════════════════════════════════════════════════════════
# 2. Feature set
# ══════════════════════════════════════════════════════════
FEATURE_COLS = [
    "Region_enc", "Category_enc", "Year", "MonthNum", "Quarter",
    "Avg_Unit_Cost", "Avg_Unit_Price", "Avg_Discount_Sales",
    "Num_Transactions", "New_Customers", "Returning_Customers",
    "Online_Txn", "Retail_Txn",
    "Inventory_Level", "Units_Ordered", "Demand_Forecast",
    "Avg_Discount_Inv", "Holiday_Days", "Competitor_Pricing",
    "Seasonality_enc", "Weather_Mode_enc",
    "Price_vs_Competitor", "Fulfillment_Rate",
    "Recommended_Order_Qty",
]

# Drop rows where all features are NaN
df_model = df.dropna(subset=FEATURE_COLS + ["Total_Sales_Amount", "Net_Profit_Margin"]).copy()
X = df_model[FEATURE_COLS].fillna(df_model[FEATURE_COLS].median())

scores_log = {}

# ══════════════════════════════════════════════════════════
# 3. MODEL A – Sales Amount Predictor
# ══════════════════════════════════════════════════════════
print("\n── Model A: Sales Amount Predictor ──")
y_sales = df_model["Total_Sales_Amount"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y_sales, test_size=0.2, random_state=42)

rf_sales = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
rf_sales.fit(X_tr, y_tr)
y_pred_sales = rf_sales.predict(X_te)

mae_s  = mean_absolute_error(y_te, y_pred_sales)
r2_s   = r2_score(y_te, y_pred_sales)
cv_s   = cross_val_score(rf_sales, X, y_sales, cv=5, scoring="r2").mean()
print(f"  MAE: {mae_s:,.2f}  |  R²: {r2_s:.4f}  |  CV R²: {cv_s:.4f}")
scores_log["Sales_Amount"] = {"MAE": round(mae_s,2), "R2": round(r2_s,4), "CV_R2": round(cv_s,4)}

# ══════════════════════════════════════════════════════════
# 4. MODEL B – Profit Margin Predictor
# ══════════════════════════════════════════════════════════
print("\n── Model B: Net Profit Margin Predictor ──")
y_margin = df_model["Net_Profit_Margin"]
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_margin, test_size=0.2, random_state=42)

gb_margin = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                       max_depth=4, random_state=42)
gb_margin.fit(X_tr2, y_tr2)
y_pred_margin = gb_margin.predict(X_te2)

mae_m  = mean_absolute_error(y_te2, y_pred_margin)
r2_m   = r2_score(y_te2, y_pred_margin)
cv_m   = cross_val_score(gb_margin, X, y_margin, cv=5, scoring="r2").mean()
print(f"  MAE: {mae_m:.2f}%  |  R²: {r2_m:.4f}  |  CV R²: {cv_m:.4f}")
scores_log["Profit_Margin"] = {"MAE": round(mae_m,4), "R2": round(r2_m,4), "CV_R2": round(cv_m,4)}

# ══════════════════════════════════════════════════════════
# 5. MODEL C – Recommended Sell Qty Predictor
# ══════════════════════════════════════════════════════════
print("\n── Model C: Recommended Sell Qty Predictor ──")
y_qty = df_model["Total_Qty_Sold"]
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X, y_qty, test_size=0.2, random_state=42)

rf_qty = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
rf_qty.fit(X_tr3, y_tr3)
y_pred_qty = rf_qty.predict(X_te3)

mae_q  = mean_absolute_error(y_te3, y_pred_qty)
r2_q   = r2_score(y_te3, y_pred_qty)
cv_q   = cross_val_score(rf_qty, X, y_qty, cv=5, scoring="r2").mean()
print(f"  MAE: {mae_q:.2f} units  |  R²: {r2_q:.4f}  |  CV R²: {cv_q:.4f}")
scores_log["Sell_Quantity"] = {"MAE": round(mae_q,2), "R2": round(r2_q,4), "CV_R2": round(cv_q,4)}

# ══════════════════════════════════════════════════════════
# 6. MODEL D – High / Low Profit Classifier
# ══════════════════════════════════════════════════════════
print("\n── Model D: Profit Class Classifier ──")
threshold = df_model["Net_Profit_Margin"].median()
df_model["Profit_Class"] = (df_model["Net_Profit_Margin"] >= threshold).astype(int)
y_cls = df_model["Profit_Class"]
X_tr4, X_te4, y_tr4, y_te4 = train_test_split(X, y_cls, test_size=0.2, random_state=42)

rf_cls = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
rf_cls.fit(X_tr4, y_tr4)
y_pred_cls = rf_cls.predict(X_te4)
acc = accuracy_score(y_te4, y_pred_cls)
print(f"  Accuracy: {acc:.4f}")
print(classification_report(y_te4, y_pred_cls, target_names=["Low Profit", "High Profit"]))
scores_log["Profit_Classifier"] = {"Accuracy": round(acc,4)}

# ══════════════════════════════════════════════════════════
# 7. Predictions on full dataset
# ══════════════════════════════════════════════════════════
X_full = df_model[FEATURE_COLS].fillna(df_model[FEATURE_COLS].median())

df_model["Predicted_Sales"]        = rf_sales.predict(X_full)
df_model["Predicted_Margin_Pct"]   = gb_margin.predict(X_full)
df_model["Predicted_Sell_Qty"]     = rf_qty.predict(X_full).round().astype(int)
df_model["Predicted_Profit_Class"] = pd.Series(rf_cls.predict(X_full), index=df_model.index).map({1: "High Profit", 0: "Low Profit"})

# Derived from predictions
df_model["Predicted_Gross_Profit"]  = df_model["Predicted_Sales"] * (df_model["Predicted_Margin_Pct"]/100)
df_model["Predicted_Transport_Cost"]= df_model["Avg_Unit_Cost"] * df_model["Predicted_Sell_Qty"] * 0.02
df_model["Predicted_Net_Profit"]    = df_model["Predicted_Gross_Profit"] - df_model["Predicted_Transport_Cost"]

# ══════════════════════════════════════════════════════════
# 8. Monthly & Yearly summaries
# ══════════════════════════════════════════════════════════
monthly = (
    df_model.groupby(["Year", "MonthNum", "Region", "Category"])
    .agg(
        Monthly_Sales     = ("Total_Sales_Amount",   "sum"),
        Monthly_Profit    = ("Net_Profit",           "sum"),
        Monthly_Qty       = ("Total_Qty_Sold",       "sum"),
        Avg_Margin_Pct    = ("Net_Profit_Margin",    "mean"),
        Spot_Txn          = ("Retail_Txn",           "sum"),
        Online_Txn_Count  = ("Online_Txn",           "sum"),
        Transport_Cost    = ("Transport_Cost_Est",   "sum"),
        Pred_Sales        = ("Predicted_Sales",      "sum"),
        Pred_Profit       = ("Predicted_Net_Profit", "sum"),
    )
    .reset_index()
)

yearly = (
    df_model.groupby(["Year", "Region", "Category"])
    .agg(
        Yearly_Sales      = ("Total_Sales_Amount",  "sum"),
        Yearly_Profit     = ("Net_Profit",          "sum"),
        Yearly_Qty        = ("Total_Qty_Sold",      "sum"),
        Avg_Margin_Pct    = ("Net_Profit_Margin",   "mean"),
        Total_Transport   = ("Transport_Cost_Est",  "sum"),
        Spot_Txn          = ("Retail_Txn",          "sum"),
        Online_Txn_Count  = ("Online_Txn",          "sum"),
    )
    .reset_index()
)

# ══════════════════════════════════════════════════════════
# 9. Improvement Opportunities Report
# ══════════════════════════════════════════════════════════
insights = []

for (region, cat), grp in df_model.groupby(["Region", "Category"]):
    row = {}
    row["Region"]   = region
    row["Category"] = cat

    # Margin gap
    avg_margin = grp["Net_Profit_Margin"].mean()
    best_margin = df_model["Net_Profit_Margin"].quantile(0.75)
    row["Avg_Margin_Pct"]     = round(avg_margin, 2)
    row["Margin_Gap_vs_Top_Quartile"] = round(best_margin - avg_margin, 2)

    # Discount drag
    row["Avg_Discount_Pct"]   = round(grp["Avg_Discount_Sales"].mean() * 100, 2)

    # Competitor pricing gap
    row["Avg_Price_vs_Competitor"] = round(grp["Price_vs_Competitor"].mean(), 2)

    # Inventory waste
    row["Avg_Inventory_Coverage_Days"] = round(grp["Inventory_Coverage"].mean(), 1)

    # Channel mix
    row["Spot_Sales_Ratio_Avg"]   = round(grp["Spot_Sales_Ratio"].mean() * 100, 2)
    row["Online_Sales_Ratio_Avg"] = round(grp["Online_Sales_Ratio"].mean() * 100, 2)

    # Demand fulfilment
    row["Fulfillment_Rate_Avg"] = round(grp["Fulfillment_Rate"].mean() * 100, 2)

    # Recommendations
    tips = []
    if row["Margin_Gap_vs_Top_Quartile"] > 5:
        tips.append(f"Margin is {row['Margin_Gap_vs_Top_Quartile']:.1f}pp below top performers – review pricing or reduce costs.")
    if row["Avg_Discount_Pct"] > 20:
        tips.append(f"High avg discount ({row['Avg_Discount_Pct']:.1f}%) – consider loyalty rewards instead of blanket discounts.")
    if row["Avg_Price_vs_Competitor"] < -5:
        tips.append("Priced below competitors – possible room to raise prices without volume loss.")
    if row["Avg_Inventory_Coverage_Days"] > 60:
        tips.append("Over-stocked (>60 days coverage) – reduce order quantities to free up working capital.")
    if row["Avg_Inventory_Coverage_Days"] < 10:
        tips.append("Under-stocked (<10 days coverage) – risk of stockouts; increase re-order frequency.")
    if row["Fulfillment_Rate_Avg"] < 70:
        tips.append("Low demand fulfilment – consider stocking more to capture latent demand.")
    if row["Online_Sales_Ratio_Avg"] < 20:
        tips.append("Low online sales share – invest in e-commerce / marketplace presence.")
    row["Recommendations"] = " | ".join(tips) if tips else "Performance is on-track."

    insights.append(row)

improvement_df = pd.DataFrame(insights)

# ══════════════════════════════════════════════════════════
# 10. Feature Importance
# ══════════════════════════════════════════════════════════
fi_df = pd.DataFrame({
    "Feature"    : FEATURE_COLS,
    "Importance_Sales"  : rf_sales.feature_importances_,
    "Importance_Margin" : gb_margin.feature_importances_,
    "Importance_Qty"    : rf_qty.feature_importances_,
    "Importance_Cls"    : rf_cls.feature_importances_,
}).sort_values("Importance_Sales", ascending=False)

# ══════════════════════════════════════════════════════════
# 11. Save all outputs
# ══════════════════════════════════════════════════════════
PRED_COLS = [
    "Month", "Region", "Category",
    "Total_Sales_Amount", "Total_Qty_Sold",
    "Avg_Unit_Cost", "Avg_Unit_Price",
    "Gross_Profit", "Transport_Cost_Est", "Net_Profit",
    "Net_Profit_Margin", "Profit_Margin_Pct",
    "Spot_Sales_Ratio", "Online_Sales_Ratio",
    "Inventory_Level", "Inventory_Coverage",
    "Demand_Forecast", "Recommended_Price", "Recommended_Order_Qty",
    "Predicted_Sales", "Predicted_Margin_Pct",
    "Predicted_Sell_Qty", "Predicted_Net_Profit",
    "Predicted_Profit_Class",
    "Sales_Growth_MoM", "Profit_Growth_MoM",
    "Fulfillment_Rate",
]

df_model[PRED_COLS].to_csv(f"{OUT}/predictions.csv",       index=False)
monthly.to_csv(f"{OUT}/monthly_summary.csv",               index=False)
yearly.to_csv(f"{OUT}/yearly_summary.csv",                 index=False)
improvement_df.to_csv(f"{OUT}/improvement_report.csv",     index=False)
fi_df.to_csv(f"{OUT}/feature_importance.csv",              index=False)
pd.DataFrame(scores_log).T.to_csv(f"{OUT}/model_scores.csv")

print("\n✅  All outputs saved:")
for f in ["predictions.csv","monthly_summary.csv","yearly_summary.csv",
          "improvement_report.csv","feature_importance.csv","model_scores.csv"]:
    print(f"   {OUT}/{f}")
