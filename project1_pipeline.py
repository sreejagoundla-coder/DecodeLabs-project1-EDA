"""
DecodeLabs - Data Science Project 1: Advanced EDA & Feature Engineering
Architecture: Input -> Process -> Output (IPO)

Run:  python project1_pipeline.py
Needs: pandas, numpy, scikit-learn, matplotlib, pandera, openpyxl
"""
import numpy as np
import pandas as pd
import pandera.pandas as pa
from sklearn.impute import KNNImputer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "Dataset_for_Data_Analytics.xlsx"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
TARGET = "TotalPrice"
ID_COLS = ["OrderID", "CustomerID", "TrackingNumber", "ShippingAddress"]  # unique identifiers, no predictive value
rng = np.random.default_rng(42)

# =====================================================================
# MODULE 1 - INPUT: securing fidelity
# =====================================================================
df = pd.read_excel(SRC)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")

# ---- 1a. Missingness audit + decision matrix (<5% drop | 5-20% impute | >20% multi-dim)
miss = (df.isna().mean() * 100).round(2)
miss = miss[miss > 0]
print("\nMissingness (%):\n", miss.to_string())

df["CouponCode_was_missing"] = df["CouponCode"].isna().astype(int)
# CouponCode is categorical and 25.75% missing (> 20%). KNN needs numeric distances, and
# here a blank means "no coupon was applied" (structural missingness), so we encode it as
# its own category instead of inventing a coupon.
df["CouponCode"] = df["CouponCode"].fillna("NONE")

# ---- 1b. Imputation benchmark (course "Conclusion" suggestion: Mean vs KNN)
# No numeric column is missing, so we MASK 10% of UnitPrice artificially, impute it
# three ways, and compare against the known truth.
num_cols = ["Quantity", "UnitPrice", "ItemsInCart"]
true = df["UnitPrice"].copy()
mask = rng.random(len(df)) < 0.10
masked = df[num_cols].copy()
masked.loc[mask, "UnitPrice"] = np.nan
cands = {
    "Mean": masked["UnitPrice"].fillna(masked["UnitPrice"].mean()),
    "Median": masked["UnitPrice"].fillna(masked["UnitPrice"].median()),
}
knn = KNNImputer(n_neighbors=5).fit_transform(masked)
cands["KNN (k=5)"] = pd.Series(knn[:, 1], index=df.index)
bench = pd.DataFrame({
    name: {
        "RMSE": float(np.sqrt(((s[mask] - true[mask]) ** 2).mean())),
        "Std after imputation": float(s.std()),
    } for name, s in cands.items()
}).T
bench.loc["Original (truth)"] = [np.nan, float(true.std())]
print("\nImputation benchmark on masked UnitPrice:\n", bench.round(2).to_string())

# ---- 1c. Outlier detection (IQR and Z-score) then winsorisation with clip (no row loss)
rows = []
for c in num_cols + [TARGET]:
    q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (df[c] - df[c].mean()) / df[c].std()
    rows.append({"feature": c, "Q1": q1, "Q3": q3, "lower": lo, "upper": hi,
                 "IQR_outliers": int(((df[c] < lo) | (df[c] > hi)).sum()),
                 "Z>3_outliers": int((z.abs() > 3).sum()), "skew": df[c].skew()})
outlier_report = pd.DataFrame(rows).set_index("feature")
print("\nOutlier report:\n", outlier_report.round(2).to_string())

df["TotalPrice_raw"] = df[TARGET]
q1, q3 = df[TARGET].quantile([0.25, 0.75]); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["TotalPrice_outlier_flag"] = ((df[TARGET] < lo) | (df[TARGET] > hi)).astype(int)
df[TARGET] = df[TARGET].clip(lower=max(lo, 0), upper=hi)  # vectorised cap, row count preserved
print(f"\nTotalPrice capped to [{max(lo,0):.2f}, {hi:.2f}] -> {df['TotalPrice_outlier_flag'].sum()} rows capped")

# =====================================================================
# MODULE 2 - PROCESS: vectorised engine
# =====================================================================
# ---- 2a. Feature engineering (all vectorised, none use the target -> no leakage)
d = df["Date"]
df["OrderMonth"] = d.dt.month
df["OrderQuarter"] = d.dt.quarter
df["OrderDayOfWeek"] = d.dt.dayofweek
df["IsWeekend"] = (d.dt.dayofweek >= 5).astype(int)
df["DaysSinceStart"] = (d - d.min()).dt.days                 # trend / recency
df["UnitsPerCartItem"] = df["Quantity"] / df["ItemsInCart"]  # purchase concentration
df["HasCoupon"] = (df["CouponCode"] != "NONE").astype(int)
df["ProductGroup"] = np.where(df["Product"].isin(["Chair", "Desk"]), "Furniture", "Electronics")
df["IsRepeatCustomer"] = df["CustomerID"].duplicated(keep=False).astype(int)
df["PriceTier"] = pd.qcut(df["UnitPrice"], 3, labels=["Low", "Mid", "High"]).astype(str)
new_feats = ["OrderMonth", "OrderQuarter", "OrderDayOfWeek", "IsWeekend", "DaysSinceStart",
             "UnitsPerCartItem", "HasCoupon", "ProductGroup", "IsRepeatCustomer", "PriceTier"]

# ---- 2b. One-hot encoding (drop_first avoids the dummy-variable trap = perfect multicollinearity)
cat_cols = ["Product", "PaymentMethod", "OrderStatus", "CouponCode", "ReferralSource", "ProductGroup", "PriceTier"]
model = df.drop(columns=ID_COLS + ["Date", "TotalPrice_raw", "TotalPrice_outlier_flag", "CouponCode_was_missing"])
model = pd.get_dummies(model, columns=cat_cols, drop_first=True, dtype=int)

# ---- 2c. Collinearity eradication (|r| > 0.80, drop the member weaker vs target)
X = model.drop(columns=[TARGET])
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
pairs = upper.stack().loc[lambda s: s > 0.80]
tcorr = X.corrwith(model[TARGET]).abs()
to_drop = set()
for (a, b), r in pairs.items():  # iterates over a handful of flagged pairs, not over data rows
    to_drop.add(a if tcorr[a] < tcorr[b] else b)
print(f"\nCollinear pairs > 0.80: {len(pairs)} -> dropping {sorted(to_drop) or 'nothing'}")
model = model.drop(columns=list(to_drop))
print("Top |r| with target:\n", model.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(6).round(3).to_string())

# =====================================================================
# MODULE 3 - OUTPUT: contracts
# =====================================================================
schema = pa.DataFrameSchema({
    "Quantity": pa.Column(int, pa.Check.in_range(1, 5)),
    "UnitPrice": pa.Column(float, pa.Check.gt(0)),
    "ItemsInCart": pa.Column(int, pa.Check.in_range(1, 10)),
    "TotalPrice": pa.Column(float, pa.Check.in_range(0, hi + 1e-6)),
    "UnitsPerCartItem": pa.Column(float, pa.Check.in_range(0, 5)),
    "HasCoupon": pa.Column(int, pa.Check.isin([0, 1])),
    "IsWeekend": pa.Column(int, pa.Check.isin([0, 1])),
    "OrderMonth": pa.Column(int, pa.Check.in_range(1, 12)),
}, strict=False, coerce=True)
try:
    schema.validate(model, lazy=True)
    print("\nPandera contract: PASSED (lazy validation, 0 failures)")
except pa.errors.SchemaErrors as e:
    print("\nPandera contract FAILED:\n", e.failure_cases)
assert model.isna().sum().sum() == 0, "NaNs remain in model matrix"

# ---- Save
clean_full = df.drop(columns=[])
with pd.ExcelWriter(OUT / "project1_cleaned_dataset.xlsx") as w:
    clean_full.to_excel(w, sheet_name="Cleaned_Full", index=False)
    model.to_excel(w, sheet_name="Model_Ready", index=False)
    outlier_report.round(3).to_excel(w, sheet_name="Outlier_Report")
    bench.round(3).to_excel(w, sheet_name="Imputation_Benchmark")
model.to_csv(OUT / "project1_model_ready.csv", index=False)

# ---- Figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].boxplot([df["TotalPrice_raw"], df[TARGET]], tick_labels=["Raw", "Capped (IQR)"])
ax[0].set_title("TotalPrice: before vs after winsorisation")
ax[1].bar(bench.index[:3], bench["RMSE"][:3], color=["#7a9", "#79c", "#c97"])
ax[1].set_title("Imputation RMSE on masked UnitPrice (lower = better)")
tc = model.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).head(8)
ax[2].barh(tc.index[::-1], tc.values[::-1], color="#58a")
ax[2].set_title("Top correlations with TotalPrice")
plt.tight_layout(); plt.savefig(OUT / "project1_eda_plots.png", dpi=130)

print(f"\nDone. Cleaned shape {clean_full.shape}, model-ready shape {model.shape}")
