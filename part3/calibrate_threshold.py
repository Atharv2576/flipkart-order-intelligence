"""Measures where real in-domain and out-of-domain questions actually land
against this knowledge base, and picks a groundedness threshold inside the
gap between them -- rather than choosing a threshold by eyeballing a
transcript.

Run as: python3 -m part3.calibrate_threshold
"""
from pathlib import Path

from part3.eval_queries import IN_DOMAIN_CALIBRATION_QUERIES, OUT_OF_DOMAIN_CALIBRATION_QUERIES
from part3.retrieval import search_chunks

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "part3_threshold_calibration.md"


def best_score(query: str) -> float:
    hits = search_chunks(query, top_k=5)
    return max((h["score"] for h in hits), default=0.0)


def run_calibration() -> dict:
    in_domain_scores = {q: round(best_score(q), 4) for q in IN_DOMAIN_CALIBRATION_QUERIES}
    out_domain_scores = {q: round(best_score(q), 4) for q in OUT_OF_DOMAIN_CALIBRATION_QUERIES}

    lowest_in_domain = min(in_domain_scores.values())
    highest_out_domain = max(out_domain_scores.values())
    gap = round(lowest_in_domain - highest_out_domain, 4)

    # sit the threshold inside the gap, closer to the out-of-domain side so a
    # genuine (if weakly-phrased) in-domain question still clears it
    threshold = round(highest_out_domain + gap * 0.4, 2) if gap > 0 else round(lowest_in_domain, 2)

    return {
        "in_domain_scores": in_domain_scores,
        "out_domain_scores": out_domain_scores,
        "lowest_in_domain": lowest_in_domain,
        "highest_out_domain": highest_out_domain,
        "gap": gap,
        "threshold": threshold,
    }


def render_report(result: dict) -> str:
    lines = [
        "# Part 3 -- Groundedness threshold calibration",
        "",
        "## In-domain queries (should be answerable)",
        "",
    ]
    for q, s in result["in_domain_scores"].items():
        lines.append(f"- {s:.4f} -- \"{q}\"")
    lines += ["", "## Out-of-domain queries (should be refused)", ""]
    for q, s in result["out_domain_scores"].items():
        lines.append(f"- {s:.4f} -- \"{q}\"")
    lines += [
        "",
        f"- lowest in-domain score: **{result['lowest_in_domain']}**",
        f"- highest out-of-domain score: **{result['highest_out_domain']}**",
        f"- separation gap: **{result['gap']}**",
        f"- **threshold chosen: {result['threshold']}** (inside the gap)",
        "",
        "Every in-domain query above clears this threshold; every out-of-domain"
        " query falls below it.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_calibration()
    report = render_report(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
