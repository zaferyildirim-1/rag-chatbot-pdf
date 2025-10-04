"""Evaluation utilities for RAG QA.

Contains a small set of golden questions and a function to run them and record
whether the expected pages are present in results.
"""
from typing import List, Dict, Any
import csv
from pathlib import Path

from .rag_pipeline import get_answers, load_index

# Small golden set: question -> expected pages (example)
GOLDEN_QS = [
    {"q": "What are the two pre-training tasks used in BERT, and what does each one train the model to do?", "pages": [1, 2]},
    {"q": "Describe BERT model architecture.", "pages": [3, 4]},
]


def evaluate_answers(output_csv: str = "evaluation_results.csv", k: int = 5) -> str:
    """Run golden questions through get_answers and write simple pass/fail rows.

    Returns path to CSV file written.
    """
    out_path = Path(output_csv)
    retriever = load_index()(k)

    rows = []
    for item in GOLDEN_QS:
        res = get_answers(item["q"], k=k, retriever=retriever)
        found_pages = set(res.get("pages", []))
        expected = set(item["pages"])
        pass_pages = expected.issubset(found_pages)
        rows.append({
            "question": item["q"],
            "expected_pages": ",".join(map(str, sorted(expected))),
            "found_pages": ",".join(map(str, sorted(found_pages))),
            "pass": pass_pages,
            "confidence": res.get("confidence", "unknown"),
        })

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "expected_pages", "found_pages", "pass", "confidence"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return str(out_path)


__all__ = ["evaluate_answers", "GOLDEN_QS"]
