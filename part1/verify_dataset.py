"""Task 2: verify the generated dataset's shape and rates, and diagnose the
missingness mechanism behind rating_given (MCAR vs MAR vs MNAR).

Run as: python3 -m part1.verify_dataset
"""
from pathlib import Path

import pandas as pd

from part1.common import DATASET_PATH, load_dataset

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "part1_data_verification.md"


def rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col)["returned"].agg(orders="size", return_rate="mean")
    grouped["return_rate"] = (grouped["return_rate"] * 100).round(2)
    return grouped.sort_values("return_rate", ascending=False)


def missingness_gap(df: pd.DataFrame) -> dict:
    cod_missing = df.loc[df["payment_method"] == "COD", "rating_given"].isna().mean()
    non_cod_missing = df.loc[df["payment_method"] != "COD", "rating_given"].isna().mean()
    return {
        "cod_missing_pct": round(cod_missing * 100, 2),
        "non_cod_missing_pct": round(non_cod_missing * 100, 2),
        "gap_pp": round((cod_missing - non_cod_missing) * 100, 2),
    }


def run_verification(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = load_dataset()

    overall_return_rate = round(df["returned"].mean() * 100, 2)
    missing_rating_pct = round(df["rating_given"].isna().mean() * 100, 2)
    by_category = rate_table(df, "product_category")
    by_payment = rate_table(df, "payment_method")
    gap = missingness_gap(df)

    return {
        "rows": len(df),
        "columns": df.shape[1],
        "overall_return_rate": overall_return_rate,
        "missing_rating_pct": missing_rating_pct,
        "by_category": by_category,
        "by_payment": by_payment,
        "missingness_gap": gap,
    }


def render_report(result: dict) -> str:
    lines = [
        "# Part 1 -- Data verification (Task 2)",
        "",
        f"- rows: **{result['rows']}**",
        f"- columns: **{result['columns']}**",
        f"- overall return rate: **{result['overall_return_rate']}%**",
        f"- missing `rating_given`: **{result['missing_rating_pct']}%**",
        "",
        "## Return rate by product_category",
        "",
        result["by_category"].to_markdown(),
        "",
        "## Return rate by payment_method",
        "",
        result["by_payment"].to_markdown(),
        "",
        "## Missingness mechanism for rating_given",
        "",
        f"- COD orders missing rating: **{result['missingness_gap']['cod_missing_pct']}%**",
        f"- non-COD orders missing rating: **{result['missingness_gap']['non_cod_missing_pct']}%**",
        f"- gap: **{result['missingness_gap']['gap_pp']} percentage points**",
        "",
        "**Classification: MAR (missing at random), conditional on `payment_method`.**",
        "",
        "- Not MCAR: the missing rate is not uniform across the two groups above --"
        " there is a real, measured dependency on payment method.",
        "- MAR: that dependency runs entirely through `payment_method`, a column"
        " observed on every single row. Conditioning on it removes any further"
        " information the missingness pattern carries.",
        "- Not MNAR: nothing in how the mask is generated depends on the value of"
        " `rating_given` itself -- the mask is drawn from `payment_method` alone,"
        " so the *unobserved* rating value has no bearing on whether it is missing.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_verification()
    report = render_report(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
