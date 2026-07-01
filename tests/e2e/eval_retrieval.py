"""Direct retrieval evaluation — tests ContextEngine embedding quality.

Bypasses openclaw gateway entirely, calls ContextEngine /assemble directly.
For each QA question:
1. Calls /api/v1/assemble with the question
2. Checks if retrieved memories are relevant to the expected answer
3. Scores keyword overlap between working set and expected answer

Usage:
    PYTHONPATH=. python3 tests/e2e/eval_retrieval.py \
        tests/e2e/data/locomo10.json --output tests/e2e/results_retrieval.csv

    # Specific samples only
    PYTHONPATH=. python3 tests/e2e/eval_retrieval.py \
        tests/e2e/data/locomo10.json --sample 0 --sample 1

    # With count limit per sample
    PYTHONPATH=. python3 tests/e2e/eval_retrieval.py \
        tests/e2e/data/locomo10.json --count 20
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import requests


CONTEXTENGINE_URL = os.environ.get("CONTEXTENGINE_URL", "http://localhost:8090")


def load_locomo_data(path: str, sample_indices: list[int] | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if sample_indices:
        result = []
        for idx in sample_indices:
            if 0 <= idx < len(data):
                result.append(data[idx])
        return result
    return data


def call_assemble(question: str, session_id: str = "eval-retrieval") -> dict:
    """Call ContextEngine assemble endpoint directly."""
    try:
        resp = requests.post(
            f"{CONTEXTENGINE_URL}/api/v1/assemble",
            json={
                "prompt": question,
                "messages": [{"role": "user", "content": question}],
                "accountId": "acct-demo",
                "userId": "u-alice",
                "agentId": "main",
                "sessionId": session_id,
                "tokenBudget": 128000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def keyword_overlap(text: str, keywords: list[str]) -> float:
    """Fraction of keywords found in text (case-insensitive)."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def extract_answer_keywords(answer: str) -> list[str]:
    """Extract meaningful keywords from expected answer."""
    # Simple keyword extraction: split, remove stop words, keep meaningful tokens
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "than", "too", "very", "just", "about", "up", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "what", "which", "who", "whom", "this", "that",
        "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
        "him", "his", "she", "her", "it", "its", "they", "them", "their",
        "week", "weeks", "month", "year", "day", "time",
    }
    # Split on whitespace and punctuation
    words = answer.replace(",", " ").replace(".", " ").replace(";", " ").split()
    keywords = []
    for w in words:
        w = w.strip().strip("()[]{}\"'")
        if w and w.lower() not in stop_words and len(w) > 1:
            keywords.append(w)
    return keywords


CATEGORY_NAMES = {
    "1": "single_session",
    "2": "cross_session",
    "3": "inference",
    "4": "forgetting",
    "5": "persistence",
}


def main():
    parser = argparse.ArgumentParser(description="Direct retrieval evaluation")
    parser.add_argument("input", help="Path to locomo10.json")
    parser.add_argument("--output", default="tests/e2e/results_retrieval", help="Output CSV prefix")
    parser.add_argument("--sample", type=int, action="append", help="Sample index(s) to evaluate")
    parser.add_argument("--count", type=int, help="Max questions per sample")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests (seconds)")
    args = parser.parse_args()

    samples = load_locomo_data(args.input, args.sample)
    csv_path = f"{args.output}.csv"

    # CSV headers
    fieldnames = [
        "sample_id", "qi", "question", "expected", "category", "category_name",
        "evidence", "working_set_count", "working_set_text",
        "top_score", "top_abstract",
        "answer_in_working_set", "keyword_overlap", "has_relevant_hit",
    ]

    total_qa = 0
    total_with_hits = 0
    total_with_answer = 0
    stats_by_category = defaultdict(lambda: {"total": 0, "with_hits": 0, "with_answer": 0})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(samples):
            sample_id = item["sample_id"]
            qas = [q for q in item.get("qa", []) if str(q.get("category", "")) != "5"]
            if args.count:
                qas = qas[:args.count]

            print(f"\n=== Sample {sample_id} [{idx}] ({len(qas)} QA) ===", file=sys.stderr)

            for qi, qa in enumerate(qas, start=1):
                question = qa["question"]
                expected = str(qa["answer"])
                category = str(qa.get("category", ""))
                category_name = CATEGORY_NAMES.get(category, f"cat_{category}")
                evidence = qa.get("evidence", [])

                # Call assemble
                session_id = f"eval-ret-{sample_id}-{qi}"
                result = call_assemble(question, session_id)

                if "error" in result:
                    print(f"  Q{qi}: ERROR - {result['error'][:60]}", file=sys.stderr)
                    writer.writerow({
                        "sample_id": sample_id, "qi": qi, "question": question,
                        "expected": expected, "category": category,
                        "category_name": category_name,
                        "evidence": json.dumps(evidence),
                        "working_set_count": 0, "working_set_text": "",
                        "top_score": 0, "top_abstract": "",
                        "answer_in_working_set": False, "keyword_overlap": 0,
                        "has_relevant_hit": False,
                    })
                    total_qa += 1
                    stats_by_category[category]["total"] += 1
                    continue

                # Extract working set
                ws_text = result.get("memoryUserMessage", "")
                ws_count = ws_text.count("相关度:") if ws_text else 0

                # Check if expected answer keywords appear in working set
                keywords = extract_answer_keywords(expected)
                overlap = keyword_overlap(ws_text, keywords) if keywords else 0.0
                answer_found = overlap > 0.3  # At least 30% of keywords found

                # Extract top hit info
                top_score = 0
                top_abstract = ""
                if ws_text:
                    # Parse scores from working set text
                    import re
                    scores = re.findall(r"相关度:\s*(\d+)%", ws_text)
                    if scores:
                        top_score = int(scores[0]) / 100.0
                    # Extract first abstract
                    abstracts = re.findall(r"\[.*?\]\s*(.+?)\s*\(相关度", ws_text)
                    if abstracts:
                        top_abstract = abstracts[0][:200]

                has_relevant = ws_count > 0 and top_score >= 0.3

                print(f"  Q{qi}: hits={ws_count} top={top_score:.0%} "
                      f"answer_found={answer_found} overlap={overlap:.0%} "
                      f"[{category_name}]", file=sys.stderr)

                writer.writerow({
                    "sample_id": sample_id, "qi": qi, "question": question,
                    "expected": expected, "category": category,
                    "category_name": category_name,
                    "evidence": json.dumps(evidence),
                    "working_set_count": ws_count,
                    "working_set_text": ws_text[:500],
                    "top_score": f"{top_score:.3f}",
                    "top_abstract": top_abstract,
                    "answer_in_working_set": answer_found,
                    "keyword_overlap": f"{overlap:.3f}",
                    "has_relevant_hit": has_relevant,
                })

                total_qa += 1
                if ws_count > 0:
                    total_with_hits += 1
                if answer_found:
                    total_with_answer += 1
                stats_by_category[category]["total"] += 1
                if ws_count > 0:
                    stats_by_category[category]["with_hits"] += 1
                if answer_found:
                    stats_by_category[category]["with_answer"] += 1

                if args.delay:
                    time.sleep(args.delay)

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Retrieval Evaluation Summary (BGE-large-en-v1.5, 1024d)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Total questions: {total_qa}", file=sys.stderr)
    print(f"With memory hits (score >= 30%): {total_with_hits} ({total_with_hits/total_qa*100:.1f}%)", file=sys.stderr)
    print(f"With answer keywords in hits: {total_with_answer} ({total_with_answer/total_qa*100:.1f}%)", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"By category:", file=sys.stderr)
    for cat in sorted(stats_by_category.keys()):
        s = stats_by_category[cat]
        name = CATEGORY_NAMES.get(cat, f"cat_{cat}")
        hit_rate = s["with_hits"] / s["total"] * 100 if s["total"] else 0
        answer_rate = s["with_answer"] / s["total"] * 100 if s["total"] else 0
        print(f"  {name} (cat={cat}): {s['total']} questions, "
              f"hit_rate={hit_rate:.1f}%, answer_rate={answer_rate:.1f}%", file=sys.stderr)
    print(f"\nResults saved to: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
