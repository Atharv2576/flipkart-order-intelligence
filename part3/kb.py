"""Loads the policy knowledge base: one Flipkart-style policy document per
.md file in part3/knowledge_base/, each with an id derived from its
filename and a title taken from its first markdown heading.
"""
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"


def load_policy_documents() -> list[dict]:
    documents = []
    for path in sorted(KB_DIR.glob("POL*.md")):
        text = path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        title = lines[0].lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()
        doc_id = path.stem  # e.g. "POL01_apparel_return_window"
        documents.append({"id": doc_id, "title": title, "text": body})
    return documents
