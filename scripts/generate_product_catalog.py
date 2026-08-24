"""Generates a synthetic, structured product catalog -- an additional
knowledge layer for the support agent's RAG/catalog search, separate from
the graded Part 1 6,000-row orders_dataset.csv (which this script never
touches). Deterministic via np.random.default_rng(SEED), same pattern as
generate_orders.py, so re-running this script reproduces the committed
data/product_catalog.json byte-for-byte.

Categories match Part 1's exact five (generate_orders.py's `categories`
list) and price ranges match Part 1's own `base_price` bands, so the
catalog and the ML model agree on what these categories mean. Per-category
return_window/exchange_available values are pinned to the actual policy KB
documents (POL01-POL05, POL15) rather than invented separately, so the
catalog cannot contradict the policy the agent already retrieves.

Run as: python3 scripts/generate_product_catalog.py
"""
import json
from pathlib import Path

import numpy as np

SEED = 7
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "product_catalog.json"

# Return window (days) and exchange eligibility straight from the policy KB:
# POL01 (15), POL02 (12), POL03/POL04 (10), POL05 (7); POL15 restricts
# exchange to Apparel and Footwear only.
CATEGORY_RULES = {
    "Apparel": {"return_window": 15, "exchange_available": True, "base_price": (400, 2200)},
    "Footwear": {"return_window": 12, "exchange_available": True, "base_price": (500, 4500)},
    "Electronics": {"return_window": 10, "exchange_available": False, "base_price": (1200, 45000)},
    "Home": {"return_window": 10, "exchange_available": False, "base_price": (300, 8000)},
    "Beauty": {"return_window": 7, "exchange_available": False, "base_price": (150, 2500)},
}

SUBCATEGORIES = {
    "Apparel": ["T-Shirt", "Jeans", "Jacket", "Dress", "Shirt", "Hoodie", "Ethnic Wear"],
    "Footwear": ["Running Shoe", "Sneaker", "Formal Shoe", "Sandal", "Slipper", "Boot"],
    "Electronics": [
        "Smartphone", "Headphone", "Earbuds", "Keyboard", "Mouse",
        "Monitor", "Smartwatch", "Charger", "Power Bank",
    ],
    "Home": ["Kitchen Appliance", "Lamp", "Bedsheet", "Furniture", "Storage Unit", "Home Decor"],
    "Beauty": ["Skincare", "Haircare", "Makeup", "Fragrance"],
}

# How many SKUs per category, roughly proportional to Part 1's cat_probs
# (0.32/0.22/0.18/0.18/0.10) scaled to a ~55-item catalog.
CATEGORY_COUNTS = {"Apparel": 18, "Electronics": 12, "Home": 10, "Footwear": 10, "Beauty": 5}

BRAND_PREFIXES = [
    "Veloce", "Northline", "Cirrus", "Baseline", "Kestrel", "Marlow", "Orbient",
    "Solace", "Tresna", "Ferro", "Hollow Peak", "Ambient", "Rivet & Co", "Palisade",
    "Meridian", "Solstice", "Thicket", "Voxel", "Lantern Bay", "Coalmark",
]

DESCRIPTORS = [
    "everyday", "premium", "lightweight", "durable", "compact", "classic",
    "modern", "essential", "all-weather", "travel-friendly",
]

# Non-returnable subcategories, matching POL16's own examples (innerwear,
# opened beauty products, furniture handled outside the standard window per
# POL04) rather than an invented list.
NON_RETURNABLE_SUBCATEGORIES = {"Furniture", "Fragrance"}

WARRANTY_BY_SUBCATEGORY = {
    "Smartphone": "1 year", "Laptop": "1 year", "Headphone": "6 months",
    "Earbuds": "6 months", "Keyboard": "1 year", "Mouse": "1 year",
    "Monitor": "2 years", "Smartwatch": "1 year", "Charger": "6 months",
    "Power Bank": "6 months", "Kitchen Appliance": "1 year",
}

DELIVERY_SLA = "2-4 business days (metro), 5-8 business days (non-metro)"


def _product_name(rng: np.random.Generator, brand: str, descriptor: str, subcat: str, model_no: int) -> str:
    return f"{brand} {descriptor.capitalize()} {subcat} {model_no}"


def _description(brand: str, descriptor: str, subcat: str, category: str) -> str:
    return (
        f"A {descriptor} {subcat.lower()} from {brand}, part of the synthetic "
        f"{category} catalog generated for this project."
    )


def generate_catalog() -> list[dict]:
    rng = np.random.default_rng(SEED)
    products = []
    product_id = 1

    for category, count in CATEGORY_COUNTS.items():
        rules = CATEGORY_RULES[category]
        subcats = SUBCATEGORIES[category]
        low, high = rules["base_price"]

        for i in range(count):
            subcat = subcats[i % len(subcats)]
            brand = rng.choice(BRAND_PREFIXES)
            descriptor = rng.choice(DESCRIPTORS)
            model_no = int(rng.integers(100, 999))
            price = round(float(rng.uniform(low, high)), 0)

            non_returnable = subcat in NON_RETURNABLE_SUBCATEGORIES
            cod_available = bool(rng.random() < 0.85)
            warranty = WARRANTY_BY_SUBCATEGORY.get(subcat, "No warranty")

            products.append(
                {
                    "product_id": f"SKU{product_id:04d}",
                    "product_name": _product_name(rng, brand, descriptor, subcat, model_no),
                    "category": category,
                    "subcategory": subcat,
                    "price_inr": price,
                    "return_window": None if non_returnable else rules["return_window"],
                    "exchange_available": bool(rules["exchange_available"] and not non_returnable),
                    "cod_available": cod_available,
                    "delivery_sla": DELIVERY_SLA,
                    "non_returnable": non_returnable,
                    "warranty": warranty,
                    "description": _description(brand, descriptor, subcat, category),
                }
            )
            product_id += 1

    return products


def main():
    products = generate_catalog()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(products, indent=2), encoding="utf-8")
    print(f"products: {len(products)}")
    print(f"written to {OUTPUT_PATH}")
    return products


if __name__ == "__main__":
    main()
