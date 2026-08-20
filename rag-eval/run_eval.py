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

from hybrid import build_bm25, fuse  # noqa: E402

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)

# RETRIEVER=vector (default) | hybrid. Kept as an env switch so the two
# retrievers run against the same eval set, same index and same metric -
# the retriever is the only variable.
RETRIEVER = os.environ.get("RETRIEVER", "vector")

# How deep each arm goes before fusion. Fusing two top-10 lists would leave
# almost nothing for RRF to reorder.
CANDIDATE_K = 50

W_VECTOR = float(os.environ.get("W_VECTOR", "1.0"))
W_BM25 = float(os.environ.get("W_BM25", "1.0"))

_bm25_cache: dict[str, tuple] = {}


def load_eval_set() -> dict:
    """Load the eval set from disk."""
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def expected_strings(item: dict) -> list[str]:
    """Every ground-truth string for a query, whichever field it declares."""
    if "expect_all" in item:
        return item["expect_all"]
    return item["expect_any"]


def metric_of(item: dict) -> str:
    """Which metric this query is scored with."""
    if "expect_all" in item:
        return "coverage"
    return "hit"


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
        if ("expect_any" in item) == ("expect_all" in item):
            problems.append(f"{item['id']}: 必须且只能有 expect_any 或 expect_all 其中一个")
            continue
        for expected in expected_strings(item):
            if expected not in corpus_text:
                problems.append(f"{item['id']}: 语料里找不到 {expected!r}")

    return problems


def get_bm25(store: VectorStore, collection: str) -> tuple:
    """Build the BM25 index once per collection, then reuse it."""
    if collection not in _bm25_cache:
        _bm25_cache[collection] = build_bm25(store, collection)
    return _bm25_cache[collection]


def retrieve(store: VectorStore, embedder: Embedder, query: str, collection: str) -> list[dict]:
    """
    Run one retrieval and return the top MAX_K chunks.

    vector: the vector store alone.
    hybrid: vector and BM25 rankings fused with RRF.
    """
    query_embedding = embedder.embed_query(query)

    if RETRIEVER == "vector":
        return store.search(
            query_embedding=query_embedding,
            collection_name=collection,
            n_results=MAX_K,
        )

    bm25, by_id = get_bm25(store, collection)

    vector_hits = store.search(
        query_embedding=query_embedding,
        collection_name=collection,
        n_results=CANDIDATE_K,
    )
    vector_ids = []
    for hit in vector_hits:
        vector_ids.append(hit["id"])
        by_id[hit["id"]] = hit

    bm25_ids = []
    for doc_id, _score in bm25.search(query, CANDIDATE_K):
        bm25_ids.append(doc_id)

    results = []
    for doc_id in fuse(vector_ids, bm25_ids, W_VECTOR, W_BM25)[:MAX_K]:
        results.append(by_id[doc_id])
    return results


def first_hit_rank(results: list[dict], expected: list[str]) -> int | None:
    """Return the 1-based rank of the first chunk containing any expected string."""
    for rank, result in enumerate(results, start=1):
        for needle in expected:
            if needle in result["content"]:
                return rank
    return None


def found_count(results: list[dict], expected: list[str], k: int) -> int:
    """How many of the expected strings appear anywhere in the top-k chunks."""
    found = 0
    for needle in expected:
        for result in results[:k]:
            if needle in result["content"]:
                found += 1
                break
    return found


def score_at(item: dict, results: list[dict], k: int) -> float:
    """
    Score one query at k, using the metric its schema asks for.

    expect_any  -> hit@k       备选写法, 命中任一就算答上
    expect_all  -> coverage@k  答案分散在多处, 按命中比例给分

    Why the split: hit@k asks only whether top-k touched the topic. For a query
    whose answer is spread across three files, finding 1 of 3 scored the same as
    finding all 3 - a third of an answer counted as a full one.

    coverage <= hit for every query, so this can only lower a score, never raise
    one. That is the point: the old cross_file number was inflated.
    """
    expected = expected_strings(item)
    if metric_of(item) == "coverage":
        return found_count(results, expected, k) / len(expected)

    rank = first_hit_rank(results, expected)
    if rank is not None and rank <= k:
        return 1.0
    return 0.0


def mean_score(rows: list[dict], k: int) -> float:
    """Average score at k over the given rows."""
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        total += row["scores"][k]
    return total / len(rows)


def print_metric_block(title: str, note: str, rows: list[dict]) -> None:
    """Print one metric's @k table, or say so when no query uses it."""
    print("\n" + "=" * 72)
    print(f"{title}   n={len(rows)}")
    print(note)
    print("=" * 72)
    if not rows:
        print("  (没有查询使用这个指标)")
        return
    for k in K_VALUES:
        print(f"  @{k:<3} {mean_score(rows, k):>6.0%}")


def print_report(rows: list[dict], categories: dict) -> None:
    """Print both metrics separately, then the per-category breakdown."""
    hit_rows = []
    coverage_rows = []
    for row in rows:
        if row["metric"] == "coverage":
            coverage_rows.append(row)
        else:
            hit_rows.append(row)

    print_metric_block(
        "hit@k        (expect_any)",
        "  备选写法, top-k 里命中任一即算答上",
        hit_rows,
    )
    print_metric_block(
        "coverage@k   (expect_all)",
        "  答案分散在多处, 分数 = 命中数 / 应命中数",
        coverage_rows,
    )

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

        metrics = set()
        for row in subset:
            metrics.add(row["metric"])
        label = "+".join(sorted(metrics))

        parts = []
        for k in K_VALUES:
            parts.append(f"@{k}={mean_score(subset, k):>4.0%}")
        print(f"  {category:<18} [{label:<8}] n={len(subset):<3} " + "  ".join(parts))

    print("\n" + "=" * 72)
    print("没答全的查询 (按 top-10 判定)")
    print("=" * 72)
    problems = []
    for row in rows:
        if row["scores"][MAX_K] < 1.0:
            problems.append(row)
    if not problems:
        print("  (无)")
    for row in problems:
        if row["metric"] == "coverage":
            state = f"coverage@{MAX_K}={row['scores'][MAX_K]:.0%} ({row['found']}/{row['total']})"
        elif row["rank"] is None:
            state = "MISS (top-10 里一个正确 chunk 都没有)"
        else:
            state = f"命中在 rank {row['rank']}"
        print(f"  [{row['category']}] {row['id']}  {row['query']}")
        print(f"      {state}")
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

    if RETRIEVER == "hybrid":
        print(f"检索器: hybrid (RRF, w_vector={W_VECTOR}, w_bm25={W_BM25})\n")
    else:
        print("检索器: vector\n")

    rows = []
    for item in eval_set["queries"]:
        results = retrieve(store, embedder, item["query"], collection)
        expected = expected_strings(item)
        metric = metric_of(item)

        scores = {}
        for k in K_VALUES:
            scores[k] = score_at(item, results, k)

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
                "metric": metric,
                "scores": scores,
                "rank": first_hit_rank(results, expected),
                "found": found_count(results, expected, MAX_K),
                "total": len(expected),
                "top1": top1,
            }
        )

        if metric == "coverage":
            mark = f"{found_count(results, expected, 5)}/{len(expected)}@5"
        elif rows[-1]["rank"]:
            mark = f"@{rows[-1]['rank']}"
        else:
            mark = "MISS"
        print(f"  {item['id']}  {mark:<7} {item['query']}")

    print_report(rows, eval_set["categories"])


if __name__ == "__main__":
    main()
