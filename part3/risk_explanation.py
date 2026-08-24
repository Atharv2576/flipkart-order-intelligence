"""Grounds "why is this order risky" explanations in Part 1's own,
already-computed permutation importance (reports/part1_feature_importance.md,
n_repeats=10, scoring=roc_auc on the held-out test split) -- transcribed
here, not recomputed or invented. Only features with meaningfully positive
importance are ever cited, and only when the order being explained actually
has a value for that feature -- no per-instance SHAP-style causal claim is
made, because none was computed.
"""

PERMUTATION_IMPORTANCE = {
    "payment_method": 0.0980164,
    "price_inr": 0.0102035,
    "num_previous_returns": 0.00846166,
    "product_category": 0.00603309,
    "delivery_days": 0.00257122,
    "is_weekend_order": 0.00119887,
    "delivery_distance_km": -0.000214959,
    "discount_pct": -0.000229975,
    "rating_given": -0.00188248,
    "num_previous_orders": -0.00239893,
    "customer_tenure_days": -0.00549095,
}

# The report's own conclusion: delivery_distance_km/customer_tenure_days
# rank #3/#4 by impurity but carry near-zero or negative held-out signal --
# so "meaningful" here is a floor well above the noise band, not just >0.
MEANINGFUL_IMPORTANCE_FLOOR = 0.005

_FEATURE_PHRASES = {
    "payment_method": "it was paid for using {value}, which this model's permutation-importance analysis found to be by far the strongest single predictor of return risk",
    "price_inr": "its price of ₹{value:,.0f}",
    "num_previous_returns": "the customer's history of {value:.0f} previous return(s)",
    "product_category": "its product category ({value})",
    "delivery_days": "its {value:.0f}-day delivery time",
}

_RANKED_FEATURES = sorted(_FEATURE_PHRASES, key=lambda f: PERMUTATION_IMPORTANCE[f], reverse=True)


def explain_risk(order_features: dict) -> list[str]:
    """Human-readable clauses for the (at most 3) features that are both
    genuinely important (per the real permutation-importance report) and
    actually present on this order.
    """
    clauses = []
    for feature in _RANKED_FEATURES:
        if PERMUTATION_IMPORTANCE[feature] < MEANINGFUL_IMPORTANCE_FLOOR:
            continue
        value = order_features.get(feature)
        if value is None:
            continue
        clauses.append(_FEATURE_PHRASES[feature].format(value=value))
        if len(clauses) == 3:
            break
    return clauses
