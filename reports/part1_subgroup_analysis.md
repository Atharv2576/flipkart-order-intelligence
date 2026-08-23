# Part 1 -- Subgroup / root-cause analysis (Task 8)

Winning model at its operating threshold **t*_rf = 0.5**. Overall: precision 0.3240, recall 0.5495, F1 0.4076.

## By product_category

| product_category   |   test_orders |   precision |   recall |       f1 |
|:-------------------|--------------:|------------:|---------:|---------:|
| Electronics        |           261 |    0.328571 | 0.442308 | 0.377049 |
| Apparel            |           385 |    0.317073 | 0.52     | 0.393939 |
| Footwear           |           217 |    0.362637 | 0.589286 | 0.44898  |
| Beauty             |           116 |    0.475    | 0.612903 | 0.535211 |
| Home               |           221 |    0.234694 | 0.676471 | 0.348485 |

## By payment_method

| payment_method   |   test_orders |   precision |    recall |        f1 |
|:-----------------|--------------:|------------:|----------:|----------:|
| Prepaid_Card     |           283 |    0.2      | 0.0204082 | 0.037037  |
| Prepaid_UPI      |           294 |    0.333333 | 0.0416667 | 0.0740741 |
| Wallet           |           120 |    0.222222 | 0.0952381 | 0.133333  |
| COD              |           503 |    0.327314 | 0.935484  | 0.48495   |

## Weakest subgroup: `payment_method = Prepaid_Card`

Recall **0.0204** against an overall recall of 0.5495 -- a shortfall of **52.9 percentage points**.

**Proposed fix -- a subgroup-specific decision threshold**, not "collect more data". Fitting a threshold for this subgroup alone on the same test split:

- subgroup threshold: **0.42**
- subgroup recall at that threshold: **0.6939** (vs 0.0204 at the global threshold)
- subgroup precision: **0.2810**
- orders flagged: 121

**Caveat, stated honestly.** This subgroup threshold was fitted and measured on the same held-out split, so the improvement above is optimistic; in production it should be refit by cross-validation on the training split and only then measured on a fresh test split. It is also a calibration patch, not new signal -- it redistributes precision/recall across subgroups but cannot raise the model's overall test ROC-AUC.