# DecodeLabs Data Science – Project 1: Advanced EDA & Feature Engineering

Dataset: 1,200 e-commerce orders (`data/Dataset_for_Data_Analytics.xlsx`).
Pipeline follows the Input → Process → Output architecture from the course.

## Run
```bash
pip install -r requirements.txt
python project1_pipeline.py
```
Outputs are written to `output/`.

## What the pipeline does
| Stage | Step | Result |
|---|---|---|
| Input | Missing data | Only `CouponCode` (25.75%) was missing. A blank means "no coupon", so it is filled with a `NONE` category. |
| Input | Imputation benchmark | Hid 10% of `UnitPrice` and compared Mean (RMSE 199), Median (199) and KNN (228). KNN was worst because `UnitPrice` is nearly uncorrelated with other columns. |
| Input | Outliers | IQR found 8 outliers in `TotalPrice` (none by Z-score). They are capped at 3330.41 with `clip()`. The raw value and a flag column are kept. |
| Process | Feature engineering | 10 new features: OrderMonth, OrderQuarter, OrderDayOfWeek, IsWeekend, DaysSinceStart, UnitsPerCartItem, HasCoupon, ProductGroup, IsRepeatCustomer, PriceTier |
| Process | Encoding | One-hot, `drop_first=True` |
| Process | Collinearity | Pairs with abs(r) > 0.80 resolved by keeping the feature more correlated with `TotalPrice` (dropped `CouponCode_NONE`, `OrderQuarter`, `PriceTier_Low`) |
| Output | Validation | Pandera schema, lazy validation, passes with 0 failures |

## Notes and limitations
- Capping legitimate high orders means `TotalPrice` no longer equals `Quantity * UnitPrice` for those 8 rows. Originals are in `TotalPrice_raw`.
- `TotalPrice` is exactly `Quantity * UnitPrice`, so using it as a model target with both columns present leaks the answer.
- Feast feature store (optional slide topic) is not implemented.
