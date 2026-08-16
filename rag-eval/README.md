# rag-eval

检索层评测。只测向量库捞回来的 chunk 对不对,**不调 LLM**——所以确定性、免费,
每次改检索都能立刻复跑。

## 为什么脚本放在语料外面

这套脚本最早建在 `06-codebase-navigator/eval/`,结果 **它把自己也索引进去了**:
`.json` 在可索引扩展名里,36 条 ground truth 里 33 条原样写在 `eval_set.json` 中,
检索直接命中答案键,recall@5 从 43% 虚涨到 64%。脚本里的中文 print 字符串
还在一堆英文代码里成了新的污染源,赢走了 4 条本该由代码 chunk 命中的语义查询。

> **答案键必须和语料物理隔离——放到语料目录外面,不是加 `SKIP_DIRS`。**
> 配置会被改回去,目录结构是约束。

## 文件

| 文件 | 作用 |
|--|--|
| `eval_set.json` | 28 条中文查询 + ground truth,分 4 类 |
| `queries_en.json` | 同样 28 条的英文版,用于跨语言对照 |
| `run_eval.py` | 主评测,算 recall@k;跑前先自检 ground truth |
| `compare_lang.py` | 中英文对照实验 |
| `reindex.py` | 重建索引,并打印 chunk 来源分布 |
| `patches/chunker.patch` | 对上游 `indexer/chunker.py` 的两处改动 |

四个类别:

- `exact_identifier` — 问确切的函数名/类名/常量名,关键词检索的活
- `semantic` — 概念性问题,用自然语言描述行为,向量检索的主场
- `value_lookup` — 问某个常量的具体取值,必须精确命中定义处
- `cross_file` — 答案分散在多个文件,需要多个 chunk 才能拼齐

## 前置:上游仓库

评测跑在上游教程仓库的 `01-foundations` 上,检索代码(`indexer/`、`store/`)也在那边,
所以需要先 clone 它:

```bash
git clone https://github.com/agenticloops-ai/agentic-ai-engineering.git
```

默认假设它和本仓库**并列**放置:

```
父目录/
├── agent-engineering-log/     ← 本仓库
└── agentic-ai-engineering/    ← 上游
```

放在别处就用 `AAE_REPO` 环境变量指过去。

## 应用切块改动

`patches/chunker.patch` 是让 recall@5 从 54% 涨到 57%(英文 86% → 93%)的那次改动:

```bash
cd ../agentic-ai-engineering && git apply ../agent-engineering-log/rag-eval/patches/chunker.patch
```

## 建索引

上游的 `06-codebase-navigator` 需要 **Python 3.13**——`chromadb → onnxruntime`
没有 cp314 的轮子,用 3.14 装不上:

```bash
cd ../agentic-ai-engineering/01-foundations/06-codebase-navigator && uv sync --python 3.13
```

然后建索引。语料和 collection 名都从 `eval_set.json` 读,所以索引和评测集不会跑偏:

```bash
uv run --directory ../agentic-ai-engineering/01-foundations/06-codebase-navigator python "$PWD/reindex.py"
```

> **改了 `chunker.py` 就必须重建索引**,否则测的还是旧块,会看到「改动没有任何效果」。

`reindex.py` 做了两件容易被忽略的事:

- **先删旧 collection 再写**。chunk id 是 `<collection>:<filepath>:<start_line>`,
  改了切块边界之后大部分 id 都会变——直接 `add` 只是把新块追加进去,**旧块原地不动**,
  语料变成新旧混合。这个坑不报错,只是让每次改动的效果都测不准
- **打印 chunk 来源分布**。就是这个检查抓出了语料污染(539 个 chunk 里 391 个来自
  一个 clone 进来的第三方仓库)。在相信任何 recall 数字之前,先扫一眼这张表

## 跑评测

```bash
uv run --directory ../agentic-ai-engineering/01-foundations/06-codebase-navigator python "$PWD/run_eval.py"
```

对照实验换成 `compare_lang.py` 即可。三个脚本都**不需要 API key**——
embedding 是本地 MiniLM,检索是本地 ChromaDB。

输出三段:整体 recall@k、**按类别拆分**(这段才是能指导行动的)、以及未命中查询的 top-1 实际捞回内容。
单一总分是用来汇报的,分类拆分才是用来干活的。
