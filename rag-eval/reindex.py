"""
Rebuild the ChromaDB index that the eval harness measures.

Must be re-run after every chunker change. Skip it and you measure the old
chunks, see no movement, and conclude the change did nothing.

Corpus and collection name both come from eval_set.json, so the index and the
eval set can never drift apart.

Usage (see README.md for the full setup):
    uv run --directory <navigator> python <path-to>/reindex.py
"""

import sys
from collections import Counter
from typing import Any

from run_eval import UPSTREAM, load_eval_set  # noqa: E402

from indexer.chunker import chunk_repository  # noqa: E402
from indexer.embedder import Embedder, index_chunks  # noqa: E402
from store.vector import VectorStore  # noqa: E402


def print_source_distribution(chunks: list[dict[str, Any]], top_n: int = 12) -> None:
    """
    Show which top-level directory each chunk came from.

    This is the check that caught the corpus pollution: 391 of 539 chunks (73%)
    came from a third-party repo that had been cloned into the corpus, and the
    retrieval numbers were meaningless until it was excluded. Eyeball this
    before trusting any recall figure.
    """
    counts: Counter[str] = Counter()
    for chunk in chunks:
        top_dir = chunk["filepath"].split("/")[0]
        counts[top_dir] += 1

    total = len(chunks)
    print(f"\nchunk 来源分布 (共 {total} 个):")
    for name, count in counts.most_common(top_n):
        print(f"  {count:>5}  {count / total:>5.0%}  {name}")


def main() -> None:
    eval_set = load_eval_set()
    collection_name = eval_set["collection"]
    corpus = UPSTREAM / eval_set["corpus"]

    if not corpus.is_dir():
        sys.exit(f"语料目录不存在: {corpus}")

    print(f"语料:       {corpus}")
    print(f"collection: {collection_name}")

    chunks = chunk_repository(corpus, corpus.name)
    if not chunks:
        sys.exit("没有切出任何 chunk, 检查语料路径和 SKIP_DIRS")

    print_source_distribution(chunks)

    store = VectorStore()

    # 必须先删旧 collection, 不能直接覆盖写。
    # chunk id 是 "<collection>:<filepath>:<start_line>", 改了切块边界之后大部分
    # id 都会变: 直接 add 只是把新块追加进去, 旧块原地不动, 语料变成新旧混合。
    # 这个坑不报错, 只是让每次改动的效果都测不准。
    if store.collection_exists(collection_name):
        store.client.delete_collection(collection_name)
        print(f"\n已删除旧 collection '{collection_name}'")

    embedder = Embedder()
    indexed = index_chunks(embedder, store, collection_name, chunks)

    print(f"\n索引完成: {indexed} 个 chunk 写入 '{collection_name}'")
    print("现在可以跑 run_eval.py 了")


if __name__ == "__main__":
    main()
