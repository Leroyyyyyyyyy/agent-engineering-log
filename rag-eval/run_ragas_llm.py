"""
Ragas' LLM-judged generation metrics, run as an EXPERIMENT rather than a score.

The question this answers is not "how good is my RAG" - it is:

    Can generation-side metrics detect a retrieval failure?

faithfulness and answer_relevancy never look at the ground truth. They ask
"is the answer grounded in the retrieved chunks" and "does the answer address
the question". Neither knows whether the RIGHT chunks were retrieved. So we
split the eval set by something we DO know - our own score@5 - and check
whether the metrics separate the two groups.

  HIT  group: score@5 == 1.0   (retrieval found the ground truth)
  MISS group: score@5 == 0.0   (retrieval found nothing correct)

If the gap is small, these metrics cannot substitute for a retrieval metric,
and that is the finding.

Costs money: 1 generation + several judge calls per query, 26 queries.
Set RAGAS_MODEL to pick a cheaper model.

Usage:
    uv run --directory <navigator> python <path-to>/run_ragas_llm.py
"""

import asyncio
import json
import os
from pathlib import Path

import anthropic
from dotenv import dotenv_values
from openai import AsyncOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, ResponseRelevancy

from run_eval import load_eval_set, retrieve, score_at  # noqa: E402

from indexer.embedder import Embedder  # noqa: E402
from store.vector import VectorStore  # noqa: E402

# The upstream tutorial repo holds the credentials. Default to a sibling
# checkout; override with AAE_ENV when it sits somewhere else. Never hard-code
# an absolute path in a public repo - it leaks the layout of someone's disk and
# breaks for everyone else.
ENV_PATH = Path(
    os.environ.get(
        "AAE_ENV",
        Path(__file__).resolve().parent.parent.parent / "agentic-ai-engineering" / ".env",
    )
).expanduser()
EN_PATH = Path(__file__).parent / "queries_en.json"

TOP_K = 5

# Ragas is provider-agnostic: llm_factory takes a pre-built client, so any
# OpenAI-compatible endpoint works. DeepSeek is the default because it is the
# cheapest judge that supports the tool-calling instructor needs.
#   RAGAS_PROVIDER=deepseek   -> OPENAI_API_KEY  + OPENAI_BASE_URL
#   RAGAS_PROVIDER=anthropic  -> ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL
PROVIDER = os.environ.get("RAGAS_PROVIDER", "deepseek")
DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "anthropic": "claude-haiku-4-5-20251001",
}
MODEL = os.environ.get("RAGAS_MODEL", DEFAULT_MODELS[PROVIDER])
# Same local model the index was built with - keeps answer_relevancy free.
EMBED_MODEL = "all-MiniLM-L6-v2"
# How many HIT queries to sample. There are only 13 MISS queries, so matching
# that count keeps the two groups the same size and the comparison honest.
N_PER_GROUP = 13


class LocalEmbeddings:
    """
    Adapter: ResponseRelevancy wants the legacy embedding interface
    (embed_query / embed_documents). ragas ships two embedding base classes and
    its own HuggingFaceEmbeddings implements the NEW one (embed_text), which
    this metric never calls. Rather than add a langchain dependency, wrap the
    same local Embedder the index was built with - so the questions and the
    corpus are compared in one embedding space.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)


def build_client():
    """
    Build the LLM client from .env, NOT from the process environment.

    Claude Code's process presets ANTHROPIC_BASE_URL=https://api.anthropic.com,
    and load_dotenv() does not override an existing variable - so reading the
    env would silently send these calls to the official API and 401.
    dotenv_values reads the file and ignores the process entirely.

    Returns (client, ragas_provider_name). ragas calls every OpenAI-compatible
    endpoint "openai", so DeepSeek reports as openai here.
    """
    env = dotenv_values(ENV_PATH)

    if PROVIDER == "anthropic":
        token = env.get("ANTHROPIC_AUTH_TOKEN")
        base_url = env.get("ANTHROPIC_BASE_URL")
        if not token or not base_url:
            raise SystemExit(f"{ENV_PATH} 缺 ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL")
        client = anthropic.AsyncAnthropic(
            auth_token=token, base_url=base_url, api_key=None, timeout=60
        )
        return client, "anthropic"

    key = env.get("OPENAI_API_KEY")
    base_url = env.get("OPENAI_BASE_URL")
    if not key or not base_url:
        raise SystemExit(f"{ENV_PATH} 缺 OPENAI_API_KEY / OPENAI_BASE_URL")
    if not key.isascii():
        raise SystemExit(
            f"{ENV_PATH} 里的 OPENAI_API_KEY 还是占位符(含非 ASCII 字符), 先填真 key"
        )
    client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=60)
    return client, "openai"


async def generate_answer(client, question: str, contexts: list[str]) -> str:
    """Answer the question using only the retrieved chunks - the generator half of RAG."""
    joined = "\n\n---\n\n".join(contexts)
    prompt = (
        f"Answer the question using ONLY the code context below. "
        f"If the context does not contain the answer, say so.\n\n"
        f"CONTEXT:\n{joined}\n\nQUESTION: {question}\n\nANSWER:"
    )
    messages = [{"role": "user", "content": prompt}]

    if isinstance(client, anthropic.AsyncAnthropic):
        response = await client.messages.create(
            model=MODEL, max_tokens=400, messages=messages
        )
        return response.content[0].text.strip()

    response = await client.chat.completions.create(
        model=MODEL, max_tokens=400, messages=messages
    )
    return response.choices[0].message.content.strip()


def pick_subset(eval_set, store, embedder, en_queries) -> list[dict]:
    """Split by our own score@5, then take an equal number from each side."""
    hits = []
    misses = []
    for item in eval_set["queries"]:
        question = en_queries.get(item["id"], item["query"])
        results = retrieve(store, embedder, question, eval_set["collection"])
        score = score_at(item, results, TOP_K)

        contexts = []
        for result in results[:TOP_K]:
            contexts.append(result["content"])

        row = {
            "id": item["id"],
            "category": item["category"],
            "question": question,
            "contexts": contexts,
            "mine": score,
        }
        if score == 1.0:
            hits.append(row)
        elif score == 0.0:
            misses.append(row)

    for row in hits:
        row["group"] = "HIT"
    for row in misses:
        row["group"] = "MISS"

    return hits[:N_PER_GROUP] + misses[:N_PER_GROUP]


async def score_rows(rows, client, faithfulness, relevancy) -> None:
    """Generate an answer per row, then judge it. Mutates rows in place."""
    for row in rows:
        row["answer"] = await generate_answer(client, row["question"], row["contexts"])
        sample = SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
        )
        row["faithfulness"] = await faithfulness.single_turn_ascore(sample, None)
        row["relevancy"] = await relevancy.single_turn_ascore(sample, None)
        print(
            f"  [{row['group']:<4}] {row['id']}  "
            f"faithfulness={row['faithfulness']:>5.2f}  relevancy={row['relevancy']:>5.2f}  "
            f"{row['question'][:44]}"
        )


def group_mean(rows, group: str, key: str) -> float:
    """Mean of one metric over one group, skipping NaN verdicts."""
    total = 0.0
    count = 0
    for row in rows:
        if row["group"] != group:
            continue
        value = row.get(key)
        if value is None or value != value:  # NaN
            continue
        total += value
        count += 1
    if count == 0:
        return float("nan")
    return total / count


def report(rows) -> None:
    """The only table that matters: does the metric separate HIT from MISS?"""
    print("\n" + "=" * 78)
    print("生成侧指标能不能识别出检索失败?")
    print("=" * 78)
    print(f"{'指标':<18}{'HIT 组':>10}{'MISS 组':>10}{'差距':>10}")
    for key, label in [("faithfulness", "faithfulness"), ("relevancy", "answer_relevancy")]:
        hit = group_mean(rows, "HIT", key)
        miss = group_mean(rows, "MISS", key)
        print(f"{label:<18}{hit:>10.2f}{miss:>10.2f}{hit - miss:>10.2f}")

    print("\n  对照: 我的 score@5   HIT 组 = 1.00   MISS 组 = 0.00   差距 = 1.00")
    print("  生成侧指标的差距越接近 0, 越说明它测的不是检索。")


def main() -> None:
    with open(EN_PATH, encoding="utf-8") as f:
        en_queries = json.load(f)["queries"]

    eval_set = load_eval_set()
    store = VectorStore()
    embedder = Embedder()

    rows = pick_subset(eval_set, store, embedder, en_queries)
    hit_n = 0
    for row in rows:
        if row["group"] == "HIT":
            hit_n += 1
    print(f"子集: HIT {hit_n} 条 + MISS {len(rows) - hit_n} 条")
    print(f"生成与评判模型: {MODEL}  (provider={PROVIDER})")
    print(f"answer_relevancy 的 embedding: {EMBED_MODEL} (本地, 不花钱)\n")

    client, ragas_provider = build_client()
    ragas_llm = llm_factory(MODEL, provider=ragas_provider, client=client)
    ragas_embeddings = LocalEmbeddings(embedder)

    faithfulness = Faithfulness(llm=ragas_llm)
    relevancy = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    asyncio.run(score_rows(rows, client, faithfulness, relevancy))
    report(rows)

    out = Path(__file__).parent / "ragas_llm_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n逐条结果写入 {out}")


if __name__ == "__main__":
    main()
