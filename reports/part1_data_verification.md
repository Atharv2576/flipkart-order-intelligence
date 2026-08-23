# Part 1 -- Data verification (Task 2)

- rows: **6000**
- columns: **13**
- overall return rate: **22.75%**
- missing `rating_given`: **13.05%**

## Return rate by product_category

| product_category   |   orders |   return_rate |
|:-------------------|---------:|--------------:|
| Apparel            |     1979 |         26.43 |
| Footwear           |     1071 |         25.96 |
| Beauty             |      579 |         20.03 |
| Home               |     1055 |         19.15 |
| Electronics        |     1316 |         18.69 |

## Return rate by payment_method

| payment_method   |   orders |   return_rate |
|:-----------------|---------:|--------------:|
| COD              |     2501 |         30.75 |
| Wallet           |      594 |         17.85 |
| Prepaid_UPI      |     1448 |         16.92 |
| Prepaid_Card     |     1457 |         16.82 |

## Missingness mechanism for rating_given

- COD orders missing rating: **22.83%**
- non-COD orders missing rating: **6.06%**
- gap: **16.77 percentage points**

**Classification: MAR (missing at random), conditional on `payment_method`.**

- Not MCAR: the missing rate is not uniform across the two groups above -- there is a real, measured dependency on payment method.
- MAR: that dependency runs entirely through `payment_method`, a column observed on every single row. Conditioning on it removes any further information the missingness pattern carries.
- Not MNAR: nothing in how the mask is generated depends on the value of `rating_given` itself -- the mask is drawn from `payment_method` alone, so the *unobserved* rating value has no bearing on whether it is missing.