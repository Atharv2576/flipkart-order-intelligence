# Part 1 -- Feature importance (Task 7)

## Top 5 by impurity-based `.feature_importances_`

| feature                   |   importance |
|:--------------------------|-------------:|
| cat__payment_method_COD   |    0.178806  |
| num__price_inr            |    0.132267  |
| num__delivery_distance_km |    0.0956745 |
| num__customer_tenure_days |    0.0900129 |
| num__delivery_days        |    0.0884491 |

## Permutation importance on the held-out test split (n_repeats=10, scoring=roc_auc)

| feature              |   importance_mean |
|:---------------------|------------------:|
| payment_method       |       0.0980164   |
| price_inr            |       0.0102035   |
| num_previous_returns |       0.00846166  |
| product_category     |       0.00603309  |
| delivery_days        |       0.00257122  |
| is_weekend_order     |       0.00119887  |
| delivery_distance_km |      -0.000214959 |
| discount_pct         |      -0.000229975 |
| rating_given         |      -0.00188248  |
| num_previous_orders  |      -0.00239893  |
| customer_tenure_days |      -0.00549095  |

## Why impurity-based importance can overrate a noisy continuous column

`.feature_importances_` totals how much every split on a column reduced Gini impurity on the *training* data. A continuous column with many distinct values offers many candidate split points, so the forest gets more chances to find a cut that separates a node's training rows by chance -- and that lucky split still banks impurity-reduction credit. A one-hot flag has exactly one possible split and no such advantage. Permutation importance instead measures how much *test-set* ROC-AUC actually drops when a column is shuffled, which is immune to that bias because it never looks at how the tree was built, only at what the column contributes to unseen-data performance.