"""Task 10: Precision@3 / Recall@3 over the query/relevant-document answer
key in eval_queries.py, computed at the document level -- each retrieved
chunk is mapped back to its parent document and deduplicated before scoring.

Run as: python3 -m part3.evaluate_retrieval
"""
from pathlib import Path

from part3.eval_queries import RETRIEVAL_EVAL_QUERIES
from part3.retrieval import search_documents

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "part3_retrieval_evaluation.md"
TOP_K = 3


def evaluate_query(query: str, relevant_ids: set[str]) -> dict:
    documents = search_documents(query, top_k=TOP_K)
    retrieved_ids = {d["document_id"] for d in documents}

    hits = retrieved_ids & relevant_ids
    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0
    recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0

    return {
        "query": query,
        "relevant_ids": sorted(relevant_ids),
        "retrieved_ids": sorted(retrieved_ids),
        "hits": sorted(hits),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def run_evaluation() -> dict:
    per_query = [evaluate_query(q["query"], q["relevant_document_ids"]) for q in RETRIEVAL_EVAL_QUERIES]
    avg_precision = round(sum(r["precision"] for r in per_query) / len(per_query), 4)
    avg_recall = round(sum(r["recall"] for r in per_query) / len(per_query), 4)
    return {"per_query": per_query, "avg_precision_at_3": avg_precision, "avg_recall_at_3": avg_recall}


def render_report(result: dict) -> str:
    lines = ["# Part 3 -- Retrieval evaluation (Task 10)", "", "Document-level Precision@3 / Recall@3.", ""]
    for r in result["per_query"]:
        lines.append(f"## \"{r['query']}\"")
        lines.append("")
        lines.append(f"- relevant: {r['relevant_ids']}")
        lines.append(f"- retrieved (top-3 chunks, deduped to documents): {r['retrieved_ids']}")
        lines.append(f"- hits: {r['hits']}")
        lines.append(
            f"- precision = {len(r['hits'])}/{len(r['retrieved_ids'])} = **{r['precision']}**"
            if r["retrieved_ids"] else "- precision = 0/0 = **0.0**"
        )
        lines.append(f"- recall = {len(r['hits'])}/{len(r['relevant_ids'])} = **{r['recall']}**")
        lines.append("")
    lines.append(f"**Average Precision@3: {result['avg_precision_at_3']}**")
    lines.append("")
    lines.append(f"**Average Recall@3: {result['avg_recall_at_3']}**")
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_evaluation()
    report = render_report(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
