"""
Retrieval eval harness.

Measures recall@k of the vector store alone - no LLM call, no agent loop.
Deterministic and free, so it can be re-run after every retrieval change.

This harness deliberately lives OUTSIDE the indexed corpus. When it sat in
06-codebase-navigator/eval/ it got indexed itself, and its Chinese print
strings then won 4 of the semantic queries it was supposed to measure.
A measurement tool must not be part of what it measures.

Usage (see rag-eval/README.md for the full setup):
    AAE_REPO=../../agentic-ai-engineering uv run python rag-eval/run_eval.py
"""

import json
import os
import sys
from pathlib import Path

# The corpus and the retrieval code both live in the upstream tutorial repo
# (agenticloops-ai/agentic-ai-engineering). Default to a sibling checkout;
# override with AAE_REPO when it sits somewhere else.
DEFAULT_UPSTREAM = Path(__file__).resolve().parent.parent.parent / "agentic-ai-engineering"
UPSTREAM = Path(os.environ.get("AAE_REPO", DEFAULT_UPSTREAM)).expanduser().resolve()
NAVIGATOR = UPSTREAM / "01-foundations" / "06-codebase-navigator"

if not NAVIGATOR.is_dir():
    sys.exit(
        f"找不到上游教程仓库: {NAVIGATOR}\n"
        f"请 clone https://github.com/agenticloops-ai/agentic-ai-engineering\n"
        f"然后用 AAE_REPO=<path> 指向它。"
    )

sys.path.insert(0, str(NAVIGATOR))

from indexer.chunker import collect_files  # noqa: E402
from indexer.embedder import Embedder  # noqa: E402
from store.vector import VectorStore  # noqa: E402

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)


def load_eval_set() -> dict:
    """Load the eval set from disk."""
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_ground_truth(eval_set: dict) -> list[str]:
    """
    Check that every expected string actually exists in the corpus.

    An eval set whose ground truth is wrong produces meaningless numbers,
    so this runs before any retrieval.
    """
    corpus_path = UPSTREAM / eval_set["corpus"]
    files = collect_files(corpus_path)

    all_content = []
    for path in files:
        try:
            all_content.append(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    corpus_text = "\n".join(all_content)

    problems = []
    for item in eval_set["queries"]:
        for expected in item["expect_contains"]:
            if expected not in corpus_text:
                problems.append(f"{item['id']}: 语料里找不到 {expected!r}")

    return problems


def retrieve(store: VectorStore, embedder: Embedder, query: str, collection: str) -> list[dict]:
    """Run one retrieval and return the top MAX_K chunks."""
    query_embedding = embedder.embed_query(query)
    return store.search(
        query_embedding=query_embedding,
        collection_name=collection,
        n_results=MAX_K,
    )


def first_hit_rank(results: list[dict], expected: list[str]) -> int | None:
    """Return the 1-based rank of the first chunk containing any expected string."""
    for rank, result in enumerate(results, start=1):
        for needle in expected:
            if needle in result["content"]:
                return rank
    return None


def print_report(rows: list[dict], categories: dict) -> None:
    """Print recall@k overall and broken down by category."""
    print("\n" + "=" * 72)
    print("整体 recall@k")
    print("=" * 72)
    for k in K_VALUES:
        hits = 0
        for row in rows:
            if row["rank"] is not None and row["rank"] <= k:
                hits += 1
        print(f"  recall@{k:<3} {hits:>2}/{len(rows)}   {hits / len(rows):.0%}")

    print("\n" + "=" * 72)
    print("按类别拆分 (这才是能指导行动的数字)")
    print("=" * 72)
    for category in categories:
        subset = []
        for row in rows:
            if row["category"] == category:
                subset.append(row)
        if not subset:
            continue

        parts = []
        for k in K_VALUES:
            hits = 0
            for row in subset:
                if row["rank"] is not None and row["rank"] <= k:
                    hits += 1
            parts.append(f"@{k}={hits / len(subset):>4.0%}")
        print(f"  {category:<18} n={len(subset):<3} " + "  ".join(parts))

    print("\n" + "=" * 72)
    print("未命中的查询 (top-10 里一个正确 chunk 都没有)")
    print("=" * 72)
    misses = []
    for row in rows:
        if row["rank"] is None:
            misses.append(row)
    if not misses:
        print("  (无)")
    for row in misses:
        print(f"  [{row['category']}] {row['id']}  {row['query']}")
        print(f"      top-1 实际捞回: {row['top1']}")


def main() -> None:
    eval_set = load_eval_set()

    print("校验 ground truth ...")
    problems = validate_ground_truth(eval_set)
    if problems:
        print(f"\n评测集有 {len(problems)} 处问题, 先修好再跑:")
        for problem in problems:
            print("  -", problem)
        sys.exit(1)
    print(f"  {len(eval_set['queries'])} 条查询的 ground truth 全部在语料中找到\n")

    store = VectorStore()
    embedder = Embedder()
    collection = eval_set["collection"]

    rows = []
    for item in eval_set["queries"]:
        results = retrieve(store, embedder, item["query"], collection)
        rank = first_hit_rank(results, item["expect_contains"])

        if results:
            meta = results[0]["metadata"]
            top1 = f"{meta.get('filepath')}:{meta.get('start_line')}-{meta.get('end_line')}"
        else:
            top1 = "(空)"

        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "category": item["category"],
                "rank": rank,
                "top1": top1,
            }
        )

        mark = f"@{rank}" if rank else "MISS"
        print(f"  {item['id']}  {mark:<5} {item['query']}")

    print_report(rows, eval_set["categories"])


if __name__ == "__main__":
    main()
