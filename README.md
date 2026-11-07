# Cleaning and Feature Engineering on 1,200 E-commerce Orders

I did this for Project 1 of the DecodeLabs Data Science internship (Batch 2026). The brief was simple: take a raw orders dataset, deal with missing values and outliers, build some new features, and end up with something a model could actually train on. I wanted the cleaning decisions to be defensible, not just "it ran without errors", so most of this README is about why I did things the way I did.

## The data

1,200 orders, 15 columns: product, quantity, unit price, total price, payment method, order status, coupon code, referral source, dates, and a few ID-style columns (order ID, customer ID, tracking number, address).

It turned out to be a pretty clean dataset. Only one column had missing values, and only one had outliers. So a lot of the work was figuring out what the right call was in each case, rather than applying a standard fix everywhere.

## What I found and what I did

**Missing values.** `CouponCode` was blank in about 26% of rows. By the course's decision matrix that is over 20%, which points to KNN. But KNN works on distances between numbers, and a blank coupon here doesn't mean data was lost. It means the customer didn't use one. I filled it with a `NONE` category instead of guessing a coupon that never existed.

**Mean vs median vs KNN.** Since no numeric column had gaps, I couldn't compare imputation methods on real missing data. So I hid 10% of `UnitPrice` at random and imputed it three ways, then checked the error against the true values:

| Method | RMSE |
|---|---|
| Mean | 199 |
| Median | 199 |
| KNN (k=5) | 228 |

KNN did worst. `UnitPrice` barely correlates with anything else in this data, so the nearest neighbours carry no useful signal. I expected KNN to win, so this surprised me a bit. It's a good reminder that the fancier method isn't automatically better.

**Outliers.** I checked every numeric column with both IQR and Z-scores. Only `TotalPrice` had any, 8 rows by IQR and none by Z-score. I capped them at the upper IQR bound (3330.41) using `clip()` so no rows were lost. The original values stay in `TotalPrice_raw` and there's a flag column marking which rows were changed.

I'm not fully comfortable with this one. Those 8 rows are real orders (like 5 units of a ~690 item), not errors. Capping them breaks the rule that `TotalPrice = Quantity x UnitPrice` for those rows. I did it because the project asked for it, but in a real setting I'd probably check with whoever owns the data first.

**Feature engineering.** Ten new features, none of them built from `TotalPrice`:
- `OrderMonth`, `OrderQuarter`, `OrderDayOfWeek`, `IsWeekend`, `DaysSinceStart` (from the order date)
- `UnitsPerCartItem` (quantity divided by items in cart)
- `HasCoupon`
- `ProductGroup` (Electronics vs Furniture)
- `IsRepeatCustomer`
- `PriceTier` (low / mid / high, split into thirds)

**Encoding and collinearity.** I one-hot encoded the categorical columns with `drop_first=True` so the dummy columns don't perfectly predict each other. Then I looked for feature pairs with correlation above 0.80 and dropped whichever one was less correlated with `TotalPrice`. That removed `CouponCode_NONE` (it's just the opposite of `HasCoupon`), `OrderQuarter` (redundant with month) and `PriceTier_Low`. Two of those three were my own features overlapping, which was a useful catch.

**Validation.** The final dataset is checked against a pandera schema (types, value ranges) and there are no NaNs left. It passes.

## Things to keep in mind if you use this data for modelling

- `TotalPrice` is exactly `Quantity x UnitPrice`. If you predict `TotalPrice` using both of those columns, the model will look amazing for no real reason.
- The data is small and fairly uniform, so don't read much into model scores from it.
- I didn't implement the Feast feature store that the course slides mention. It needs infrastructure that didn't make sense for a 1,200-row file.

## Running it

```bash
pip install -r requirements.txt
python project1_pipeline.py
```

Results go to `output/`:
- `project1_cleaned_dataset.xlsx` has the full cleaned data, the model-ready version, the outlier report and the imputation comparison
- `project1_model_ready.csv` is the encoded matrix (1,200 rows, 33 columns)
- `project1_eda_plots.png` shows the before/after outliers, imputation errors and top correlations

## Files

```
data/       original dataset
output/     everything the script produces
project1_pipeline.py
requirements.txt
```

Built with pandas, NumPy, scikit-learn, matplotlib and pandera.
