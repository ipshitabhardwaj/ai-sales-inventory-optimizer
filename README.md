# Smart Retail Intelligence — ML Project

A complete machine learning pipeline for retail shop analytics.
Built from two real datasets: transaction-level sales data + store inventory data.

---

## What This Project Does

Given your shop's monthly sales and inventory records, the system:

- **Predicts** future sales, profit margins, and how many units to sell
- **Classifies** each product-region segment as High Profit or Low Profit
- **Calculates** gross profit, net profit, transport costs, and margin %
- **Splits** sales into Spot (counter/retail) vs Online channels
- **Produces** daily/monthly/yearly summaries
- **Generates** an improvement report — where to price better, discount less, stock smarter
- **Visualises** everything in an interactive HTML dashboard

---

## Project Structure

```
smart_retail_ml/
├── build_dataset.py       ← Step 1: merge + feature engineering
├── ml_pipeline.py         ← Step 2: train models + generate predictions
├── dashboard.html         ← Interactive visual dashboard
├── master_dataset.csv     ← Merged engineered dataset (41 features)
├── predictions.csv        ← Full predictions for all segments
├── monthly_summary.csv    ← Monthly aggregated KPIs
├── yearly_summary.csv     ← Yearly aggregated KPIs
├── improvement_report.csv ← Per-segment improvement recommendations
├── feature_importance.csv ← What drives sales most
└── model_scores.csv       ← R², MAE, CV scores per model
```

---

## The 4 ML Models

| # | Model | Algorithm | Predicts | Score |
|---|-------|-----------|----------|-------|
| A | Sales Amount Predictor | Random Forest Regressor | Total monthly revenue per segment | R² = 0.70 (CV) |
| B | Profit Margin Predictor | Gradient Boosting Regressor | Net profit margin % | R² = 0.12 |
| C | Sell Quantity Predictor | Random Forest Regressor | Units to sell per month | R² = 0.72 (CV) |
| D | Profit Classifier | Random Forest Classifier | High / Low Profit label | Accuracy = 71.8% |

Model B (margin) has a low R² because margin depends heavily on cost data that varies per transaction, not captured in monthly aggregates. 
**To improve it: add transaction-level cost/price variance as features.**

---

## What the Dashboard Shows

### Tab 1 — Overview
- Total Revenue, Net Profit, Avg Margin, Units Sold (KPI cards)
- Monthly Sales vs Profit trend chart (2023)
- Sales breakdown by Category (donut chart)
- Revenue by Region (horizontal bar)
- Spot vs Online sales split by category

### Tab 2 — Predictions
- Filterable table: actual vs predicted sales, profit, margin, sell qty, transport cost, profit class
- Filter by Region and Category

### Tab 3 — Improvements
- Margin gap vs top quartile (where you're losing money)
- Avg discount % heatmap (who is over-discounting)
- 8 actionable recommendations with specific segments named

### Tab 4 — ML Models
- Model cards with scores and confidence bars
- Feature importance chart — what actually drives sales

---

## Key Findings

1. **Transaction count** is by far the #1 driver of sales (0.75 importance) — more customers > everything else
2. **West · Furniture** has the worst margin at 30.9% — 30pp below top performers
3. **West · Food** is the star performer at 60.2% margin — study and replicate
4. **All segments are over-stocked** (>60 days inventory) — capital is being wasted on excess stock
5. **Blanket discounts** (avg 13–17%) are dragging profit — switching to loyalty rewards would improve margins
6. **Electronics in West** is heavily online (68.8%) — signals a market ready for digital-first strategy
7. **North is the strongest region** overall by both revenue and profit

---

## Features in the Master Dataset (41 columns)

**Sales Features**
- Total_Sales_Amount, Total_Qty_Sold, Num_Transactions
- Avg_Unit_Cost, Avg_Unit_Price, Avg_Discount_Sales
- New_Customers, Returning_Customers
- Online_Txn, Retail_Txn (spot vs online count)

**Inventory Features**
- Inventory_Level, Units_Sold_Online, Units_Ordered
- Demand_Forecast, Competitor_Pricing
- Weather_Mode, Holiday_Days, Seasonality

**Engineered Features**
- Gross_Profit = Sales − Cost of Goods
- Transport_Cost_Est = 2% of COGS (adjustable)
- Net_Profit = Gross Profit − Transport Cost
- Net_Profit_Margin, Profit_Margin_Pct
- Spot_Sales_Ratio, Online_Sales_Ratio
- Fulfillment_Rate = Units Sold / Demand Forecast
- Inventory_Coverage = days of stock on hand
- Price_vs_Competitor = your price − competitor price
- Recommended_Price = Cost × 1.40 (40% target margin)
- Recommended_Order_Qty = Demand × 1.15 (15% safety stock)
- Sales_Growth_MoM, Profit_Growth_MoM

---

## What You Can Add Next

| Feature | Benefit |
|---------|---------|
| Supplier name / lead time | Predict stockout risk more accurately |
| Actual transportation bills | Replace the 2% estimate with real data |
| Customer ID / loyalty tier | Enable customer-level CLV prediction |
| Footfall / walk-in count | Separate spot discovery from repeat purchases |
| Product weight / volume | Better transport cost estimation per SKU |
| Seasonal promotions flag | Isolate promotional uplift vs organic growth |
| Return / damage rate | True net-net profit calculation |
| Weather API integration | Automated demand forecasting by weather |
| WhatsApp / social orders | Track digital-word-of-mouth channel separately |
| Daily sales log | Enable day-of-week pattern analysis |

---

## Does Anything Like This Already Exist?

Yes — but they are expensive enterprise tools:

| Product | What it does | Cost |
|---------|-------------|------|
| **SAP Retail** | Full retail ERP + analytics | Enterprise pricing |
| **Microsoft Dynamics 365** | Sales + inventory AI | ₹5,000–₹25,000/month |
| **Zoho Analytics** | BI + basic ML | ₹1,000–₹5,000/month |
| **Google Looker** | Dashboards | Usage-based billing |
| **Vyapar / Marg ERP** | Indian SME accounting + reports | ₹2,000–₹8,000/year |

**This project is different**: it is fully local, free, customisable, and built on your actual data. 
No subscription. No data leaving your machine. You own the model.

---

## How to Run

```bash
# Step 1: Build master dataset
python3 build_dataset.py

# Step 2: Train models + generate all outputs
python3 ml_pipeline.py

# Step 3: Open dashboard
open dashboard.html    # macOS
xdg-open dashboard.html  # Linux
```

**Dependencies**: pandas, numpy, scikit-learn (all standard — no external install needed if Anaconda/standard Python is present)

---

## To Retrain on Your Own Data

Replace the two CSV paths at the top of `build_dataset.py`:
```python
sales = pd.read_csv("YOUR_sales_file.csv", ...)
inv   = pd.read_csv("YOUR_inventory_file.csv", ...)
```

Make sure your files have columns matching the names used, or update the column references accordingly.
The pipeline is modular — each section is clearly labelled and can be extended independently.