"""
Cross-lingual retrieval experiment.

Runs the same 28 queries twice - once in Chinese, once in English - against the
same corpus and the same ground truth. The only variable is query language, so
any difference isolates the cross-lingual factor of the embedding model.

Usage (see rag-eval/README.md for the full setup):
    AAE_REPO=../../agentic-ai-engineering uv run python rag-eval/compare_lang.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import K_VALUES, load_eval_set, retrieve  # noqa: E402
from run_eval import first_hit_rank  # noqa: E402
from indexer.embedder import Embedder  # noqa: E402
from store.vector import VectorStore  # noqa: E402

EN_PATH = Path(__file__).parent / "queries_en.json"


def run_all(store, embedder, eval_set, collection, use_english, en_queries):
    """Run every query once and return its first-hit rank."""
    rows = []
    for item in eval_set["queries"]:
        query = en_queries[item["id"]] if use_english else item["query"]
        results = retrieve(store, embedder, query, collection)
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "rank": first_hit_rank(results, item["expect_contains"]),
                "query": query,
            }
        )
    return rows


def recall_at(rows, k):
    """Fraction of queries whose first correct chunk ranks at or above k."""
    hits = 0
    for row in rows:
        if row["rank"] is not None and row["rank"] <= k:
            hits += 1
    return hits / len(rows) if rows else 0.0


def by_category(rows, category):
    subset = []
    for row in rows:
        if row["category"] == category:
            subset.append(row)
    return subset


def main() -> None:
    eval_set = load_eval_set()
    with open(EN_PATH, encoding="utf-8") as f:
        en_queries = json.load(f)["queries"]

    store = VectorStore()
    embedder = Embedder()
    collection = eval_set["collection"]

    zh = run_all(store, embedder, eval_set, collection, False, en_queries)
    en = run_all(store, embedder, eval_set, collection, True, en_queries)

    print("\n" + "=" * 62)
    print("整体 recall@k        中文查询   英文查询   差值")
    print("=" * 62)
    for k in K_VALUES:
        a, b = recall_at(zh, k), recall_at(en, k)
        print(f"  recall@{k:<3}          {a:>6.0%}     {b:>6.0%}   {b - a:>+6.0%}")

    print("\n" + "=" * 62)
    print("按类别 (recall@5)    中文查询   英文查询   差值")
    print("=" * 62)
    for category in eval_set["categories"]:
        zs, es = by_category(zh, category), by_category(en, category)
        if not zs:
            continue
        a, b = recall_at(zs, 5), recall_at(es, 5)
        print(f"  {category:<18} {a:>6.0%}     {b:>6.0%}   {b - a:>+6.0%}")

    print("\n" + "=" * 62)
    print("逐条排名变化 (中文 → 英文)")
    print("=" * 62)
    zh_by_id = {row["id"]: row for row in zh}
    for row_en in en:
        row_zh = zh_by_id[row_en["id"]]
        a = row_zh["rank"] or "MISS"
        b = row_en["rank"] or "MISS"
        if a == b:
            continue
        arrow = "改善" if (row_en["rank"] or 99) < (row_zh["rank"] or 99) else "变差"
        print(f"  {row_en['id']}  {str(a):>4} -> {str(b):<4}  {arrow}   {row_en['query']}")


if __name__ == "__main__":
    main()
