"""
Run Ragas' non-LLM retrieval metrics over the SAME 60 queries, same index,
same retriever as run_eval.py - so the metric is the only variable.

Why only the non-LLM metrics: Ragas' headline metrics (faithfulness,
answer_relevancy) score GENERATION. They need a generated answer, a reference
answer and an LLM judge. This eval set has none of those - its ground truth is
code substrings like "def check_command(". Only NonLLMContextRecall and
NonLLMContextPrecisionWithReference measure the same thing run_eval.py does,
and they need no API key.

Usage:
    uv run --directory <navigator> python <path-to>/run_ragas.py
"""

import asyncio
import json
import os
from pathlib import Path

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextRecall

from run_eval import (  # noqa: E402
    MAX_K,
    expected_strings,
    load_eval_set,
    metric_of,
    retrieve,
    score_at,
)

from indexer.embedder import Embedder  # noqa: E402
from store.vector import VectorStore  # noqa: E402

# The k our own harness is compared at. Ragas' recall has no k - it scores
# whatever list you hand it - so we hand it the same top-5 we score ourselves on.
COMPARE_K = 5

# LANG=en (default) uses queries_en.json - the 76% baseline in HANDOFF is the
# English number. LANG=zh uses the Chinese queries inside eval_set.json.
LANG = os.environ.get("LANG_SET", "en")
EN_PATH = Path(__file__).parent / "queries_en.json"


def load_queries() -> dict:
    """id -> query text, in whichever language LANG_SET asks for."""
    if LANG != "en":
        return {}
    with open(EN_PATH, encoding="utf-8") as f:
        return json.load(f)["queries"]


def max_similarity(needle: str, chunks: list[str], measure) -> float:
    """
    The single number that explains everything below.

    Ragas takes, for each reference string, the best similarity against any
    retrieved chunk. We surface that raw value instead of only the 0/1 verdict,
    because "0%" alone does not show WHY it is zero.
    """
    best = 0.0
    for chunk in chunks:
        sample = SingleTurnSample(reference=chunk, response=needle)
        score = asyncio.run(measure.single_turn_ascore(sample, None))
        if score > best:
            best = score
    return best


def main() -> None:
    eval_set = load_eval_set()
    store = VectorStore()
    embedder = Embedder()
    collection = eval_set["collection"]

    en_queries = load_queries()
    metric = NonLLMContextRecall()
    print(f"查询语言:   {LANG}")
    print(f"Ragas 指标: {metric.name}")
    print(f"  距离函数: {metric.distance_measure.distance_measure}")
    print(f"  阈值:     {metric.threshold}")
    print(f"对比 k:     {COMPARE_K}\n")

    rows = []
    for item in eval_set["queries"]:
        query_text = en_queries.get(item["id"], item["query"])
        results = retrieve(store, embedder, query_text, collection)
        chunks = []
        for result in results[:COMPARE_K]:
            chunks.append(result["content"])

        expected = expected_strings(item)

        sample = SingleTurnSample(
            user_input=item["query"],
            retrieved_contexts=chunks,
            reference_contexts=expected,
        )
        ragas_score = asyncio.run(metric.single_turn_ascore(sample, None))
        mine = score_at(item, results, COMPARE_K)

        best_sim = 0.0
        for needle in expected:
            sim = max_similarity(needle, chunks, metric.distance_measure)
            if sim > best_sim:
                best_sim = sim

        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "metric": metric_of(item),
                "mine": mine,
                "ragas": ragas_score,
                "best_sim": best_sim,
                "needle_len": len(expected[0]),
                "chunk_len": len(chunks[0]) if chunks else 0,
                "query": query_text,
            }
        )
        print(f"  {item['id']}  我的={mine:>4.0%}  ragas={ragas_score:>4.0%}  最大相似度={best_sim:.3f}")

    print("\n" + "=" * 78)
    print(f"整体对比 @{COMPARE_K}")
    print("=" * 78)

    mine_total = 0.0
    ragas_total = 0.0
    for row in rows:
        mine_total += row["mine"]
        ragas_total += row["ragas"]
    n = len(rows)
    print(f"  我的指标 (hit/coverage)     {mine_total / n:>6.0%}")
    print(f"  Ragas {metric.name}   {ragas_total / n:>6.0%}")

    print("\n" + "=" * 78)
    print("按类别")
    print("=" * 78)
    for category in eval_set["categories"]:
        subset = []
        for row in rows:
            if row["category"] == category:
                subset.append(row)
        if not subset:
            continue
        m = 0.0
        r = 0.0
        for row in subset:
            m += row["mine"]
            r += row["ragas"]
        c = len(subset)
        print(f"  {category:<18} n={c:<3} 我的={m / c:>4.0%}   ragas={r / c:>4.0%}")

    print("\n" + "=" * 78)
    print("两边判定不一致的查询")
    print("=" * 78)
    disagree = []
    for row in rows:
        if abs(row["mine"] - row["ragas"]) >= 0.5:
            disagree.append(row)
    print(f"  共 {len(disagree)} / {n} 条\n")
    for row in disagree[:10]:
        print(f"  [{row['category']}] {row['id']}  {row['query']}")
        print(
            f"      我的={row['mine']:.0%}  ragas={row['ragas']:.0%}  "
            f"最大相似度={row['best_sim']:.3f}  "
            f"(needle {row['needle_len']} 字符 vs top1 chunk {row['chunk_len']} 字符)"
        )

    print("\n" + "=" * 78)
    print("最大相似度分布 (这是解释,不是结果)")
    print("=" * 78)
    sims = []
    for row in rows:
        sims.append(row["best_sim"])
    sims.sort()
    print(f"  min={sims[0]:.3f}  中位={sims[len(sims) // 2]:.3f}  max={sims[-1]:.3f}")
    over = 0
    for s in sims:
        if s > metric.threshold:
            over += 1
    print(f"  超过阈值 {metric.threshold} 的: {over} / {len(sims)}")


if __name__ == "__main__":
    main()
