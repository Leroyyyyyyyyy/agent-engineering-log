"""
Cross-lingual retrieval experiment.

Runs the same 28 queries twice - once in Chinese, once in English - against the
same corpus and the same ground truth. The only variable is query language, so
any difference isolates the cross-lingual factor of the embedding model.

Scoring comes from run_eval, so both scripts always report the same metric per
query: hit@k for expect_any, coverage@k for expect_all.

Usage (see rag-eval/README.md for the full setup):
    AAE_REPO=../../agentic-ai-engineering uv run python rag-eval/compare_lang.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_eval import K_VALUES, load_eval_set, retrieve  # noqa: E402
from run_eval import expected_strings, first_hit_rank, mean_score  # noqa: E402
from run_eval import found_count, metric_of, score_at  # noqa: E402

from indexer.embedder import Embedder  # noqa: E402
from store.vector import VectorStore  # noqa: E402

EN_PATH = Path(__file__).parent / "queries_en.json"


def run_all(store, embedder, eval_set, collection, use_english, en_queries):
    """Score every query once, in one language."""
    rows = []
    for item in eval_set["queries"]:
        query = en_queries[item["id"]] if use_english else item["query"]
        results = retrieve(store, embedder, query, collection)
        expected = expected_strings(item)

        scores = {}
        for k in K_VALUES:
            scores[k] = score_at(item, results, k)

        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "metric": metric_of(item),
                "scores": scores,
                "rank": first_hit_rank(results, expected),
                "found": found_count(results, expected, 5),
                "total": len(expected),
                "query": query,
            }
        )
    return rows


def by_category(rows, category):
    subset = []
    for row in rows:
        if row["category"] == category:
            subset.append(row)
    return subset


def describe(row):
    """One-cell summary of how a query did, in its own metric's terms."""
    if row["metric"] == "coverage":
        return f"{row['found']}/{row['total']}"
    if row["rank"] is None:
        return "MISS"
    return str(row["rank"])


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
    print("整体得分            中文查询   英文查询   差值")
    print("  (hit 和 coverage 混合平均, 只用于看总体趋势)")
    print("=" * 62)
    for k in K_VALUES:
        a, b = mean_score(zh, k), mean_score(en, k)
        print(f"  @{k:<3}                {a:>6.0%}     {b:>6.0%}   {b - a:>+6.0%}")

    print("\n" + "=" * 62)
    print("按类别 (@5)          中文查询   英文查询   差值")
    print("=" * 62)
    for category in eval_set["categories"]:
        zs, es = by_category(zh, category), by_category(en, category)
        if not zs:
            continue
        metric = zs[0]["metric"]
        a, b = mean_score(zs, 5), mean_score(es, 5)
        print(f"  {category:<18} {a:>6.0%}     {b:>6.0%}   {b - a:>+6.0%}   [{metric}]")

    print("\n" + "=" * 62)
    print("逐条变化 (中文 → 英文;  hit 类看排名, coverage 类看命中数/总数)")
    print("=" * 62)
    zh_by_id = {row["id"]: row for row in zh}
    for row_en in en:
        row_zh = zh_by_id[row_en["id"]]
        if row_zh["scores"][5] == row_en["scores"][5] and describe(row_zh) == describe(row_en):
            continue
        if row_en["scores"][5] > row_zh["scores"][5]:
            arrow = "改善"
        elif row_en["scores"][5] < row_zh["scores"][5]:
            arrow = "变差"
        else:
            arrow = "持平"
        left, right = describe(row_zh), describe(row_en)
        print(f"  {row_en['id']}  {left:>5} -> {right:<5}  {arrow}   {row_en['query']}")


if __name__ == "__main__":
    main()
