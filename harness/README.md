# harness

一个自己写的 agent runtime。不用框架,每一层都是为了能回答一个具体的面试追问而做的。

```bash
python3 test_loop.py                                  # 16 条,系统 python3 直跑,不需要 venv
.venv/bin/uvicorn server:app --port 8000              # SSE 服务(需要 .venv)
curl -N -s -X POST localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"goal":"where am I?","fake":true}'
```

| 文件 | 行数 | 是什么 |
|--|--:|--|
| `loop.py` | 289 | 循环、工具、护栏、取消令牌 |
| `server.py` | 157 | SSE 接口 + `/stop` |
| `provider.py` | 84 | 模型访问接口 + 不联网的 FakeProvider |
| `tracing.py` | 199 | OTel GenAI 埋点(透传生成器) |
| `cost.py` | 102 | 四费率成本核算 |
| `evaluate.py` | 216 | 轨迹评测 |
| `tasks.json` + `fixtures/` | 6 条 | 冻结的任务集 |
| `test_loop.py` | 375 | 24 条测试,不需要 API key |

```bash
python3 evaluate.py                                   # 轨迹评测(scripted,免费)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
  .venv/bin/uvicorn server:app --port 8000            # 带埋点跑服务
```

---

# 分层决策日志

> 记「为什么这么写」和「被否决的方案」。代码注释里只留一行「为什么不是显而易见的那种写法」。

## 2026-08-23 · 骨架:provider 注入 + 审批策略参数化

把模块级的 client 和写死的 `input()` 改成参数。

**这一步买到的唯一东西:一个能跑 100 次、不花钱、不用按 y 的循环。** 除此之外没有直接价值。

**被否决:包一层自己的 Response 类型。** `AnthropicProvider` 原样交回 SDK 对象,
`FakeProvider` 模仿它的形状。代价是循环耦合在 Anthropic 的 block 布局上;
理由是第二家真 vendor 还没出现,现在抽象就是猜。**等它出现再付这笔钱。**

## 2026-09-02 · 事件流:`print` → `yield`,以及 SSE 接口

### 决策 1:循环改成生成器

`agent()` 原来一边 `print` 一边跑、最后 `return` 字符串。`print` 送不过 HTTP,
测试它还得捕获 stdout。改成 `run()` 生成器,每件事 `yield` 一个带 `type` 的 dict。

**同一份数据不加转换层就能进三个地方:SSE 帧、JSONL 日志、测试断言。**
观测埋点、轨迹评测、成本核算之后都挂在这一条流上,不用各起一套。

`agent()` 保留成薄包装(排干事件流、只返回最终字符串),**13 条老测试一行没改。**

**被否决:直接把 `agent()` 改成返回事件列表。** 那样调用方全要改,而且失去流式——
列表要等跑完才有,生成器边跑边出。

### 决策 2:选 SSE 不选 WebSocket

这条流是单向的,控制指令走独立的 `POST /stop/{run_id}`。
SSE 是纯 `text/event-stream` 的 HTTP 响应,`curl -N` 一条命令能验。

**被否决:WebSocket。** 它买的是双向和低延迟帧,这里两样都不需要,
代价是独立握手、独立心跳重连、独立代理配置,而且 curl 测不了。

**已知的两个 SSE 代价,没有回避:**

- HTTP/1.1 下浏览器对同域名只有 6 个并发连接,每个 SSE 流占一个 → 生产要求 HTTP/2
- 浏览器原生 `EventSource` **只能发 GET**,而 `/run` 是 POST 带 JSON
  → **浏览器端用不了 `EventSource`,得 `fetch` + `ReadableStream` 自己解帧**。
  想用 `EventSource` 就得改成「POST 建 run 拿 id → GET `/stream/{run_id}` 订阅」两段式。**这笔欠着。**

### 决策 3:取消属于 harness,不属于 HTTP 层

`/stop` **完全不碰连接**,只翻一个 `CancellationToken`;循环自己在**每步开头**和
**每次执行工具前**检查。

**关连接停的是传输,不是计算。** socket 断了,已经起来的 subprocess 照跑、
模型请求照发、token 照烧——客户端看着停了,后端还在花钱。

**实测**:`max_steps=50` 的 run,第 2 个 `tool_result` 之后调 `/stop`,**实际跑 3 步就停**,
以 `tool_result → cancelled → run_finished` 收尾,`/runs` 立刻变空。

检查点必须是这两处:每步开头(还没发请求就退,省一次模型调用)、
每个 `tool_use` 执行前(工具是不可逆的那一半)。放在「每步结束」就晚了。

**取消时必须合成一条 `tool_result`** —— 取消发生在 `tool_use` 和 `tool_result` 之间会留下
未配对的 `tool_use`,下次请求直接 400。有测试守着
(`test_cancel_mid_run_still_pairs_every_tool_use`)。这是「配对不变量」,顺手做掉了。

**已知边界:多 worker 下取消不了。** `RUNS` 是进程内 dict,`--workers 4` 时 `/stop` 打到
worker B、run 在 worker A 就停不掉。要修得把取消状态放进 Redis 或数据库让循环轮询 ——
**那是持久化状态的问题,归下一层。**

### 决策 4:SSE 的两个协议细节

- **帧尾的空行不能少。** 它是帧分隔符,少了客户端会一直往同一帧拼,
  **症状是页面什么都不显示——看起来像卡死,不像出错,没有任何报错。**
- **`X-Accel-Buffering: no`。** nginx 默认 `proxy_buffering`,会攒满 buffer 才转发,
  流式白做。用响应头声明的好处是**不用改 nginx 配置**。
- `data` 里不放裸换行(会被当成新字段行),整个事件走 `json.dumps`,天然转义。

### 决策 5:token 统计只做了一半,是已知欠账

每步 `yield` 一个 `usage` 事件,`run_finished` 汇总。

**但 Anthropic 的 `Usage` 实际有 7 个字段**(introspect SDK 得到):
`input_tokens` / `output_tokens` / `cache_creation_input_tokens` /
`cache_read_input_tokens` / `cache_creation` / `server_tool_use` / `service_tier`。

**算钱必须把缓存拆开**——写缓存比普通输入贵,读缓存便宜一个量级。
全按 `input_tokens` 算会同时高估和低估,**误差方向随缓存命中率漂移**。
现在只累加了 input/output 两个。

另外:**agent 的 token 消耗随步数平方增长**,因为每步都要重发完整历史。
这是 compaction 和 prompt caching 不是优化项而是必需项的原因。

### 还没做,且知道为什么

**断线重连**(`id:` + `Last-Event-ID`)。要能「从第 N 个事件之后接着推」,
事件必须先被持久化;现在事件即时生成即时丢弃,重连只能从头跑。
**断线重连和崩溃恢复是同一个机制的两个用法**,一起做。


## 2026-09-02 · 观测、成本、轨迹评测

### 决策 6:按 OTel GenAI 语义约定埋点,不自造格式

三层 span,对齐实际发生的三件事:`invoke_agent` → `chat` → `execute_tool`。
**属性名一个都不是自己起的**,全部取自 `opentelemetry.semconv` 的 `gen_ai_attributes` 常量。

**这是「我按 GenAI 语义约定埋点」和「我写了个 trace 格式」的全部区别,工作量一样。**
agentevals 的文档明写:它靠看见 `gen_ai.request.model` / `gen_ai.input.messages`
**自动识别**格式——所以接进去**一行适配代码都没写**。换 Jaeger / ARMS / Langfuse 同理。

**被否决:自定义 JSONL trace。** 写起来一样快,但答案从「能直接接现成后端」
降级成「我自己定义了一套」,而且要为每个消费端写一次适配。

### 决策 7:三个 bug,全部由真实消费端暴露

发出去的 JSON 自己看毫无问题。喂进 agentevals 才现形:

| bug | 我干了什么 | 消费端怎么读 |
|--|--|--|
| tokens 5700 报成 11400 | `gen_ai.usage.*` 同时写在每个 `chat` span **和** `invoke_agent` 上(想放个总数) | 按属性名跨整棵树求和,总数成了又一个加数 |
| 3 次工具调用报成 6 次 | 同一次调用既写进 chat 的 `output.messages`,又开 `execute_tool` span | 两处都收 |
| `agentText` 为空 | 最终答案只在 `done` 事件里,循环不为它 yield `text` 事件 | 它读 `output.messages`,那儿没有 |

**规则:遥测属性是给聚合器加总的,不是给人读的。父层级上放「小计」,
聚合器只理解成又一个加数。** 层级汇总改用 `harness.run.input_tokens`,退出它的求和范围。

**用别人的命名空间就得接受别人的聚合规则**——`gen_ai.*` 是契约不是前缀。

**已知欠账**:工具调用的去重是**为 agentevals 单独做的取舍**,不是纯粹的修复。
按规范 chat span 的 `output.messages` 本该包含模型请求的工具调用
(那是「模型说它要调」,`execute_tool` 是「真的跑了」,两件事)。
换个会去重的后端,这条信息就少了。

### 决策 8:埋点写成透传生成器

```python
for e in run(...)              # 之前
for e in instrument(run(...))  # 之后 —— 调用方只改了这一行
```

收什么吐什么,`agent()` 和 `server.py` 其余代码一句没动。

**理由不是背压**(实测同步 Python 里回调也阻塞,两种写法时间线一致 —— 这条我一开始说错了)。
真正的三条:①消费者 `break` 能让循环**立刻停住**,回调拦不住
——对 agent 这是钱,每步都是一次模型调用;②中间层可以一层套一层,各层互不知道;
③`StreamingResponse` 要的就是可迭代对象,回调式要额外的线程+队列把「推」翻译成「拉」。

### 决策 9:成本按四个费率算,不是两个

```
input 1.00 / output 5.00 / cache_write 1.25 / cache_read 0.10   (USD per 1M)
```

**写缓存比普通输入贵,读缓存便宜一个量级。** 折成一个 input 费率会同时高估和低估,
**误差随缓存命中率漂移,没法用系数校正**。有测试守着这个不变量。

未知模型返回 `None` 不返回 `0.0` —— **静默的 0.00 会被读成「免费」而不是「没定价」。**

**价格表会过期,引用前自己核。形状是稳的,数字不是。**

### 决策 10:轨迹分和答案分不合成一个

`trajectory`(工具/参数/顺序对不对)和 `answer`(最终文本对不对)**分开报**。

**因为它们独立失败,而有意思的正是不一致的时候**:
轨迹错但答对 = 蒙的(答案来自模型先验,不是工具输出);
轨迹对但答错 = 读错了,不是规划错。合成一个分数两边都看不见。
报告里单列这两组。

打分用**加权 LCS** 而非精确匹配,因为**顺序有意义**——先找文件再读它是个计划,
反过来不是。实测区分度:

```
完全正确 100%  多一步探索 100%  参数错 75%  顺序颠倒 50%  少一步 50%
换了个命令 0%  该拒绝却调了工具 0%
```

**⚠️ 只有一个工具,所以 ground truth 必须带 `input_parameters`。**
只比工具名的话每条任务都是 100%,量不出任何东西。

### 已知欠账

- **任务集只有 6 条。** 路线的合格线是 50–100 条起,**现在这个数字不能写进简历**。
- **scripted provider 跑出的 100% 没有意义**(它重放的就是期望轨迹),只验管道。
  真数字要接真模型跑 —— **「自评比独立评乐观多少」这个目标还没达成**。
- **`server.py` 的埋点没有单测**,只做过端到端 curl 验证。
