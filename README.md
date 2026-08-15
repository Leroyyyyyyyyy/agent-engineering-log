# Agent Engineering Log

学 Agent 开发过程中,**自己动手做出来的东西和踩到的坑**。不是教程代码的搬运——跟练的部分在
[agenticloops-ai/agentic-ai-engineering](https://github.com/agenticloops-ai/agentic-ai-engineering),
这里只放我自己写的评测工具、重写的实现,以及从中提炼的规则。

---

## 核心内容:给一个 RAG 检索系统建评测闭环

顶石项目是一个 codebase navigator(索引本地代码库 → 向量检索 → LLM 回答)。跑通之后我发现
**它答错了却看不出来**——问 `BLOCKED_COMMANDS 的值`,它编了一个 6 元素的列表(真实是 10 个),
格式漂亮、语气笃定。

所以先不优化,**先做能测准的那一层**:28 条查询 + ground truth 的检索层评测集,只测 recall@k,
不调 LLM,确定性且免费。然后每次只改一个变量,改一次测一次。

### 可归因的改进记录

| 改动 | chunk 数 | recall@5 (中文查询) | recall@5 (英文查询) |
|--|--:|--:|--:|
| 基线 | 539 | 43% | *未测* |
| ① `SKIP_DIRS` 排除 `repos/`、`data/` | 149 | 54% | 86% |
| ② 顶层常量单独成块 | 213 | **57%** | **93%** |

改动 ① 是一行配置:索引时把 clone 进来的第三方仓库和向量库自身的持久化目录扫了进去,
**539 个 chunk 里 391 个(73%)是垃圾**。

改动 ② 见 [`rag-eval/patches/chunker.patch`](rag-eval/patches/chunker.patch):原来只按 `class`/`def`
切块,模块级常量全被塞进同一个 chunk。一个 384 维向量要同时代表 docstring + import + 4 个常量 +
一份 tool schema,离哪件都不够近。受控实测(同一查询,只改被比较的文本长度):

| 查询 | vs 整个 chunk(1094 字符) | vs 只有那几行常量 |
|--|--:|--:|
| `SHELL_OPERATORS 包含哪些操作符` | 0.2670 | **0.5724** |
| `ALLOWED_COMMANDS 白名单里有哪些命令` | 0.2169 | **0.3969** |

### 评测集三次证明了自己的价值

1. **抓出语料污染**——73% 的 chunk 来自一个不该被索引的目录
2. **抓出数据泄漏**——把答案键建在语料目录内,分数从 43% 跳到 64%。看着很合理,所以差点就信了。
   实测发现 36 条 ground truth 里 **33 条(92%)原样写在被索引的 `eval_set.json` 里**,
   检索命中的是答案键本身。移出语料后真实值是 54%
3. **拦下一个方向错误的优化**——我已经规划好要改切块策略,评测集显示 `semantic` 类的瓶颈
   根本不是切块,是**查询语言**

### 一个我没修但测清楚了的问题

`all-MiniLM-L6-v2` 是英文模型,语料是英文代码。同一语料、同一 ground truth,
**只把 28 条查询从中文换成英文**:

| 类别 (recall@5) | 中文查询 | 英文查询 | Δ |
|--|--:|--:|--:|
| `exact_identifier` | 100% | 100% | **±0** |
| `semantic` | 25% | **100%** | **+75** |
| `value_lookup` | 33% | 67% | +33 |
| `cross_file` | 0% | 33% | +33 |
| **整体** | 54% | **86%** | **+32** |

`exact_identifier` 纹丝不动是**对照组**:`chunk_python`、`VectorStore` 这类查询的关键 token
中英文本来就一样,没有跨语言损失。这条对照证实了机制不是别的东西。

修法是换多语言 embedding 或加查询改写,**我选择不修**——这条线的价值在于已经拿到的数字。

---

## 目录

| 路径 | 是什么 |
|--|--|
| [`rag-eval/`](rag-eval/) | 检索评测集与脚本(28 条查询,分 4 类) |
| [`rag-eval/patches/chunker.patch`](rag-eval/patches/chunker.patch) | 对上游切块器的两处改动 |
| [`agent-loop/my_agent.py`](agent-loop/my_agent.py) | 关掉源文件凭理解重写的 agent loop |
| [`NOTES.md`](NOTES.md) | 38 条学习笔记,格式是「现象 → 规则」成对 |

### `my_agent.py` 修掉的两个洞

跟练的实现里:

1. **`stop_reason` 只判断了 `end_turn`**。遇到 `max_tokens` 时会:不满足退出条件 →
   找不到 `tool_use` → `tool_results` 是空列表 → 追加 `{"role":"user","content":[]}` →
   下次调用 **400 崩溃**。重写版每个枚举值都有分支,外加 `else` 兜底
2. **子串黑名单不是护栏**。实测:`echo format` 被拦(`"rm"` 是 `"fo`**`rm`**`at"` 的子串,误伤),
   `mv a.txt /dev/null` 放行(漏网),`python3 -c "..."` 一句话绕过全部黑名单。
   换成三层独立防御:拒 shell 操作符 → 可执行文件白名单 → `shell=False`

第 1 层为什么必须在第 2 层之前:白名单回答的是「哪个程序允许运行」,
它隐含前提是**字符串里只有一个程序**。`echo $(whoami)` 里 `$(...)` 是命令替换、
**在主命令之前执行**,你检查第一个词是 `echo` 就放行了,而 `whoami` 早跑完了。

---

## 已知局限

诚实记录,没修:

- **`cross_file` 只有 33%**,是四类里唯一没解决的
- **这个 33% 还是虚高的**。现在 recall 的定义是「top-k 里**任一**期望块命中就算成功」,
  但 `cross_file` 类问题的答案分散在多个文件,需要**多个块同时进 top-k** 才算真答上。
  命中 1/3 现在被记为满分。正确指标应该是 coverage@k,改完这个数字大概率会**下降**——
  这是对的,先把测量做对再谈优化
- **中文查询 57% vs 英文 93%**,见上文,已定位未修复
- 评测集 n=28,**边界附近单条排名的抖动是噪声**,只看类别级别的系统性移动

---

## 怎么跑

见 [`rag-eval/README.md`](rag-eval/README.md)。检索层评测不需要任何 API key。
