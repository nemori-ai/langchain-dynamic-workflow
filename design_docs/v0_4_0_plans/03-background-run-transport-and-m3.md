# v0.4.0 · 后台 run transport 底座 + M3（下钻 + 结果回传）— 设计 spec

> **For agentic workers:** 本文件是 v0.4.0 **切片 1** 的设计定稿（承重的 context-independent
> transport 底座 + M3①下钻 + 独立的 M3②结果回传）。实现阶段用 `superpowers:writing-plans` 将其
> 拆成逐任务的 TDD plan。M2（持久侧栏多-workflow 实时流程图）作为**同一底座**的后续消费面，详细
> 设计留到 `04-*.md` 在切片 2 启动时展开——本文件只在 §9 钉死"底座对 M2 够用"这一约束。

**目标（一句话）：** 让一条 **detached 后台 run**（`run board` 启动的那些）的 leaf 运行事件，在
"启动它的 host turn 早已返回、那一轮的 stream writer 已死"之后，仍能**定向**流回一条活着的 UI
surface（M3①下钻）；并让每条后台 run 完成后把**精炼结果**喂回发起它的 host agent，使 host 能向
用户**汇报实质结论**（M3②）。

**架构（一句话）：** 在引擎侧 `BgRunSlot` 上挂一个**有界 raw-event buffer**，后台 run 的引擎
sink 只往 buffer **append**（不需 writer）；一条活着的轮询/下钻 turn **每轮全量重放** buffer 里的
事件，经一个**绑当前活轮 emit 的 fresh `UiAdapter`** 渲染成与 inline run 一模一样的 Gen-UI 卡。
结果回传复用现成 `RunResult` + `ResultStore`（摘要 inline、大结果按需 fetch）。

**Tech：** Python 3.12 async-first、LangGraph durable execution、deepagents leaf、LangChain
callback 运行树、agent-chat-ui Generative-UI。

---

## 0. 范围

| 纳入本切片 | 不纳入（留后续） |
|---|---|
| context-independent transport 底座（引擎） | M2 持久侧栏多-workflow 实时流程图（→ `04`，消费本底座） |
| M3① board 下钻（demo 消费，选择式/对话式） | 真·实时 push（带外 SSE 侧信道；A2，已否决） |
| M3② 后台 run 结果回传 host（独立，不依赖底座） | 事件持久化/回放日志（A3，已否决；遥测瞬态即可） |

---

## 1. 问题：后台 run 为何 UI-dark（现状机理）

inline run **能**实时可观测，靠的是 `ui_bridge.make_host_ui_emit`：`push_ui_message` 从 contextvar
`var_child_runnable_config` 解析"当前图的 stream writer"，`make_host_ui_emit` 在 **host node 上下文里**
`get_config()` 抓拍 host 的 `RunnableConfig`，然后每个引擎事件临时把该 contextvar rebind 回这份抓拍
——事件就打到 host 图的 writer + `ui` 通道。其前置条件是：**host node 仍在栈上、那一轮的 writer 仍活**
（`run_workflow` 在 host turn 内被直接 `await`）。`ui_bridge.py` 模块注释亲口声明这套**只对 inline 成立**：
"Deferring it to a background task would both lose the host context and require an awaitable sink the engine
does not offer."

后台 run 两条前置全断：

1. `BgRunManager.start()` 用 `asyncio.ensure_future(self._run_wrapped(...))` 把协程 **detach** 上事件循环
   （`src/_background.py`）。等后台 run 真正执行 leaf 时，**启动它的那一轮 host turn 早已返回**（这正是
   "不阻塞 host turn"的设计本意），那一轮绑定的 stream writer / SSE 通道已关闭。
2. 即便后台 run 产出事件，也无活 writer 可推。

**冒烟枪（现状代码）：** `host_graph.py::launch_background_run`（L1074）的
`run_workflow(_orchestrate, roster=make_roster(), workflows=workflows)` **一个 sink 都没传**——
后台 run 压根不产出可观测事件。`run_runs_board_live`（L1220）只轮询 `BgRunManager.list_runs` 拿
聚合状态。`RunBoard.tsx`（L14-17）与 `run_runs_board`（L1295）的注释亲口钉死："background runs are
UI-dark … A row carries no drill-in affordance, because there is no interior to drill into from a
detached run."

**结论：** run-tree 的 `run_id`/`parent_run_id` 给了**身份**（知道哪条 host run 生了哪条后台 run），
唯一缺的是**传输通道**——writer 死后事件去哪。这正是本底座要补的。

---

## 2. 架构总览（A1：缓冲 + 活轮 pump）

被分层强制的关键判断：**引擎不能依赖 demo 的 `UiAdapter`**，所以引擎 buffer 只能存 **raw 引擎事件**，
coalesce（边→卡的原地折叠）留在消费者 `UiAdapter` 的 pump 时刻做。引擎 buffer = 哑传输，消费者 coalesce
——引擎保持 demo-agnostic、框架原生（任何 tracer/adapter 都能复用，与 M1 定位一致）。

```
后台 run（detached task）
  引擎 5 个 sink（on_span_begin/on_span/on_leaf_event/on_progress/on_command）
        │  纯进程内 append，不需 writer / adapter
        ▼
  ┌─────────────────────────────────────────────┐
  │ BgRunSlot.events: 有界 raw-event buffer       │   ← 引擎侧，挂在 slot 上
  │ + dropped 计数（溢出不静默）                   │
  └─────────────────────────────────────────────┘
        │
        │  稍后，在一条"活着"的轮询/下钻 turn 上（drill_run 的 poll loop）
        ▼
  每 tick：buffered_events(run_id) 全量取出
        → 重放进一个【绑当前活轮 emit 的 fresh UiAdapter】
        → make_host_ui_emit → push_ui_message
        ▼
  前端：agent_span / fanout_graph / phase_timeline 卡（与 inline run 同款）
         稳定 span_id 作 event_id → SDK reducer 同 id 原地 merge，不出重复卡
```

三条已拍板的设计取舍（对话记录）：
- **transport = A1**（缓冲 + 活轮 pump），否决 A2（带外 SSE，面太大、背离 Gen-UI-on-turn）、A3（持久日志，
  过度工程、与 delivered-not-recorded 先例相悖）。
- **buffer 存 raw 引擎事件**（分层强制）。
- **每轮全量重放 + fresh adapter**（复用 `UiAdapter` 100%、零新状态管理；稳定 id 保幂等），否决游标增量
  drain（要管每-run 持久 adapter 生命周期 + emit 逐轮换绑，面更大）。

---

## 3. 引擎切片 — context-independent transport 底座

### 3.1 `BufferedEvent` — raw 引擎事件的标签联合

后台 run 的 5 个 sink 各自的入参类型不同，buffer 需以统一形态存下、pump 时再分派回 `adapter.on_*`。

```python
# src/langchain_dynamic_workflow/_background.py（或新 _run_events.py，见 §3.7 归属）
from dataclasses import dataclass

# 引擎已有：SpanBegin / Span（_observability）、LeafEvent（_leaf_events）、
# ProgressEntry（_progress）、CommandEvent（_observability，M5）。

type BufferedPayload = SpanBegin | Span | LeafEvent | ProgressEntry | CommandEvent

@dataclass(frozen=True, slots=True)
class BufferedEvent:
    """One raw engine event captured for later replay to a live UI surface.

    Attributes:
        kind: Which engine sink produced it ("span_begin" | "span" | "leaf_event" |
            "progress" | "command"), so the pump dispatches to the matching
            ``UiAdapter`` method without isinstance-sniffing.
        payload: The verbatim engine event object.
    """
    kind: str
    payload: BufferedPayload
```

> `kind` 显式记下而非 pump 时 `isinstance` 判别——五种里 `Span`/`SpanBegin` 同源易混，显式标签让分派
> 一目了然且不依赖类型擦除。

### 3.2 `BgRunSlot` 事件 buffer（有界、dropped/truncated）

在 `BgRunSlot`（`_background.py`）上增量加三个字段（向后兼容、皆有默认）：

```python
@dataclass
class BgRunSlot:
    # ...既有字段不动...
    events: list[BufferedEvent] = field(default_factory=list)
    dropped: int = 0                       # 溢出丢弃计数（不静默）
    _events_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
```

**有界策略（资源耗尽守卫）：** 封顶 `max_buffered_events`（默认建议 `2000`，host 可调）。达到上限后
**丢弃新到事件**（drop-newest）并 `dropped += 1`——保留 run 的早期/结构性事件（begin 边、fan-out 边），
与 `UiAdapter._MAX_SUBTREE_NODES=200` drop-new + `truncated` 的**现成先例一致**。`buffered_events`
返回时一并带出 `dropped`，pump 经卡上的 `truncated` 标志**诚实告知**前端，绝不静默截断。

> 常见量级（数十~低百叶）永不触顶；数千叶的 `batch_map` 后台 run 是压力个例，而那种规模前端本就渲染
> 聚合而非真下钻到每一叶，触顶即 `truncated=True` 显示"已截断 N 条"。

### 3.3 `BgRunManager.event_sinks(run_id)` — sink 工厂

```python
@dataclass(frozen=True, slots=True)
class RunEventSinks:
    """The five engine sinks that append a background run's events to its slot buffer."""
    on_span_begin: Callable[[SpanBegin], None]
    on_span: Callable[[Span], None]
    on_leaf_event: Callable[[LeafEvent], None]
    on_progress: Callable[[ProgressEntry], None]
    on_command: Callable[[CommandEvent], None]

class BgRunManager:
    def event_sinks(self, run_id: str, *, thread_id: str) -> RunEventSinks:
        """Build append-only sinks bound to ``run_id``'s slot buffer.

        Each sink wraps a ``BufferedEvent`` and appends under the slot's lock,
        dropping past the cap (incrementing ``dropped``). Synchronous, never raises,
        never blocks (a pure bounded append) — satisfying the engine's inline-sink
        contract so a sink can never unwind the orchestration.
        """
```

每个 sink 形如（以 `on_leaf_event` 为例）：

```python
def _append(slot, kind, payload, *, cap) -> None:
    with slot._events_lock:
        if len(slot.events) >= cap:
            slot.dropped += 1
            return
        slot.events.append(BufferedEvent(kind=kind, payload=payload))
```

**契约对齐：** 引擎 sink 是同步 `Callable`，且引擎**直接调用**（raising sink 会拆掉编排）。本 sink 只做
有界 append——不 await、不 I/O、不 raise——天然满足契约。`leaf_event_include_payloads` /
`command_include_payloads` 默认 `False`（shape-only），与 inline 一致、隔离照旧。

### 3.4 `buffered_events` — 读 API

```python
class BgRunManager:
    def buffered_events(self, run_id: str, *, thread_id: str | None = None,
                        ) -> tuple[list[BufferedEvent], int]:
        """Return a snapshot copy of the run's buffered events and its dropped count.

        Returns ``([], 0)`` for an unknown run. The list is a shallow copy taken under
        the slot lock, so a concurrent worker-thread ``on_command`` append cannot tear
        the read.
        """
```

### 3.5 线程安全（`on_command` 跨线程 append）

`UiAdapter` 注释已记载：`on_command` 边由 deepagents 的 execute 路径经 `asyncio.to_thread` worker 触发，
**跑在非事件循环线程**，与 loop 线程的其余 sink 并发。故后台 run 的 buffer append **必须加锁**
（`slot._events_lock`），否则 worker 线程的 command append 会与 loop 线程的 span/leaf append、以及
`buffered_events` 的读拷贝相互撕裂。这是必须落实的正确性细节，复用 `UiAdapter` 的 `threading` 锁范式。

### 3.6 生命周期 + 资源界 + sweep

- buffer 随 slot 而生（`start` 创建 slot 时初始化），随 slot 而灭（既有 `sweep` 回收 settled/过期 slot 时，
  `events` 一并随 slot 释放——无需额外清理路径）。
- 双重资源界：单 run 被 `max_buffered_events` 封顶；全局被既有 `max_concurrent_runs × max_buffered_events`
  绑定。两者皆 host 可调，皆有默认，是 fast-fail / bounded-queue 守卫。

### 3.7 瞬态遥测：不入 journal / 确定性 / replay；隔离照旧

- buffer 是 **transient 遥测**——**不入** journal、**不进**确定性 guard、**不参与** replay
  （沿用 E 的 BATCH 进度 delivered-not-recorded 先例）。resume 时事件由实时执行重新产生，buffer 自然重建；
  cached/replayed leaf 本就不 fire 内部事件（与 inline 一致），故下钻只对真执行路径有内容、绝不伪造。
- **隔离红线照旧：** 事件进 buffer（带外遥测），**永不**进 host LLM 的消息上下文。host LLM 只在 M3②
  收到精炼 report（§5），二者是分开的两条路。
- **归属：** `BufferedEvent` / `RunEventSinks` 与 buffer 字段紧贴 `BgRunManager`，就近放 `_background.py`；
  若该文件因此过大，可拆出 `_run_events.py` 承载 `BufferedEvent` + sink 工厂，由 `_background.py` 复用。
  二者皆可，落地时按文件体量定，保持单一职责。

---

## 4. M3① 消费切片 — board 下钻（demo）

### 4.1 `launch_background_run` 接 event_sinks

`host_graph.py::launch_background_run`（今 L1074 一个 sink 没传）改为把 `manager.event_sinks(run_id,
thread_id=...)` 的五个 sink 传进 `run_workflow`：

```python
sinks = manager.event_sinks(run_id, thread_id=thread_id)
result = await run_workflow(
    _orchestrate, roster=make_roster(), workflows=workflows,
    on_span_begin=sinks.on_span_begin, on_span=sinks.on_span,
    on_leaf_event=sinks.on_leaf_event, on_progress=sinks.on_progress,
    on_command=sinks.on_command,
)
```

> 注意 `run_id` 的产生时机：`BgRunManager.start` 可生成 `run_id`。需让 `launch_background_run` 在构造
> `_run()` 协程前先确定 `run_id`（显式传给 `start(run_id=...)`），sink 才能绑对 slot。落地时把 `run_id`
> 的解析上移到 `start` 之前（或令 `event_sinks` 惰性按 key 查 slot），writing-plans 阶段定细节。

### 4.2 选择式 drill + 每轮全量重放 fresh UiAdapter

下钻交互 = **选择式/对话式**：host 暴露一个**专用 `drill_run(run_id | label)` 工具**（定稿选择——比给
`run_runs_board_live` 加 `focus` 参数更干净：不重载 board-live 的聚合语义，下钻是一个语义独立的动作）。
用户说"看看 X 这条 run"，host 调 `drill_run(X)` 进入一个**单 turn 内的 poll loop**（与 `run_runs_board_live`
同构）：

每 tick：
1. `events, dropped = manager.buffered_events(run_id)`；
2. 起一个**绑当前活轮 `make_host_ui_emit(anchor=...)` 的 fresh `UiAdapter`**；
3. 顺序把 `events` 按 `kind` 分派重放：`adapter.on_span_begin/on_span/on_leaf_event/on_progress/on_command`；
4. `dropped > 0` 时令卡带 `truncated`。

稳定 `span_id` 作 event_id → 每 tick 全量重放在 SDK `ui_message_reducer` 处**幂等**（同 id 原地 merge，
不堆重复卡）。loop 在该 run 进入终态（done/failed/cancelled）或用户转移注意后收束。

> **anchor 一致性：** drill 是**单 turn 内的循环**（非跨用户 turn），所有 tick 锚定本轮 host 消息、复用
> 稳定 id——与 `run_runs_board_live` 在一轮内反复 upsert `run-board-1` 完全同构，无跨轮锚点漂移问题。
>
> **选择式 vs 全量：** 只 pump 被 drill 的那一条 run（有界：一条 run 的全量重放，非全部在飞 run）。
> "每轮把所有在飞 run 的内部都 pump" 是 M2 侧栏的活（§9），在 M3① 做会模糊切片边界。

### 4.3 RunBoard 下钻 affordance + 诚实注释改写

- `RunBoard.tsx`：行加一个下钻提示（点击/展开 → 触发 host 的 drill，或在已 drill 的 run 下内联其卡）；
  改写 L14-17 那段"no interior to drill into"的诚实注释——底座落地后该限制不再成立。
- 渲染**整套复用现有卡**：`agent_span`（AgentSpan.tsx 已支持 `on_leaf_event` 子树折叠下钻）、
  `fanout_graph`、`phase_timeline`、`journal_badge`、`execution_command`。前端组件词汇**零新增**
  （除 RunBoard 的下钻 affordance）。

---

## 5. M3② — 后台 run 结果回传 host（独立，不依赖底座）

### 5.1 `run_runs_board` 收尾返回 per-run `RunResult`

现状缺口：`run_runs_board_live` 只返回一句聚合 "Ran 3 of 3: 3 finished"，**没把每个 run 的实际 result
喂回 host 上下文**，host 因此无内容可汇报。

底座已现成：`BgRunManager.get_result(run_id)` 返回 `RunResult`——小结果 inline 全文 `value`、大结果
（> `ResultStore.inline_max_chars`，默认 2000）`summary` + `handle`（经 `ResultStore` 卸载）。

M3② = 把 `run_runs_board` 收尾的 tool 返回值，从一句聚合改为**每个被启动 run 的 `RunResult`**
（每 run 一行：label + status + summary，大结果附 handle）。host 据此能对每条 run 汇报，而非只数个数。

### 5.2 `fetch_run_result` 工具

新增一个 host 工具 `fetch_run_result(run_id | handle)`：对 host 想**细说**的某条 run，按需拉全文
（`ResultStore.fetch(handle)`）。

### 5.3 守控制流反转

- **默认精炼**：host 上下文收到的是每 run 的 capped summary（+ handle），**非** N 份全文——不灌爆上下文。
- **按需拉全**：host 仅对要向用户详述结论的那一两条 run 调 `fetch_run_result` 取全文，再综合汇报。
- 贴合 M3② 目标："不只是'3 个完成了'，而是'RAG vs long-context 的结论是 X……'"。
- 复用 `RunResult` + `ResultStore`，引擎**几乎零新增**（M3② 主要是 demo 侧 tool 接线 + 一个 fetch 工具）。

---

## 6. 数据流（端到端）

```
host turn A: run_runs_board 启动 N 条后台 run
   → 每条 launch_background_run 把 event_sinks(run_id) 传进 run_workflow（§4.1）
   → BgRunManager.start detach；turn A 立即返回（不阻塞）
[turn A 已结束，其 writer 已死]

后台 run 执行中：引擎 sink 把 SpanBegin/Span/LeafEvent/Progress/Command append 进 slot.events（§3.2，加锁）

host turn B: 用户"看看 X 这条" → drill_run(X) poll loop（§4.2）
   每 tick: buffered_events(X) → fresh UiAdapter（绑 turn B emit）重放 → push_ui_message
   → 前端在 turn B 下渲染 X 的实时流程图 + leaf 状态（稳定 id 原地 merge）

后台 run 完成：BgRunManager._settle 落 RunResult（小 inline / 大 offload 到 ResultStore）

host turn C（收尾/汇报）: run_runs_board 返回 per-run RunResult（§5.1）
   → host 对要详述的 run 调 fetch_run_result（§5.2）→ 向用户汇报实质结论
```

隔离：左侧（事件 → buffer → UI）与右侧（结果 → ResultStore → host LLM）是**两条独立的路**；leaf 事件
永不进 host LLM 上下文，host LLM 只见 M3② 的精炼 report。

---

## 7. 错误处理 + 资源界 + 隔离（红线）

- **sink 不拆编排**：buffer append 同步、不 raise、不 block（§3.3）；pump 侧的 emit 经 `UiAdapter` +
  `make_host_ui_emit` 已吞异常（既有红线）。
- **资源界**：单 run `max_buffered_events` 封顶 + drop-newest + `dropped`/`truncated` 诚实告知；全局
  `max_concurrent_runs × cap` 绑定；buffer 随 slot `sweep` 回收（§3.2/3.6）。
- **线程安全**：`on_command` 跨 to_thread worker → buffer append/读拷贝全程加锁（§3.5）。
- **隔离**：事件只进 buffer（带外），永不进 host LLM 上下文；下钻只对真执行路径有内容，cached leaf 不伪造（§3.7）。
- **确定性/resume**：buffer 不入 journal/确定性/replay；resume 重建（§3.7）。

---

## 8. 测试策略

| 层 | 覆盖 |
|---|---|
| 引擎单元 | `event_sinks` 五路 append；`max_buffered_events` 触顶 drop-newest + `dropped` 递增；`buffered_events` 快照拷贝；slot `sweep` 连带释放 buffer；跨线程 append（模拟 worker 线程 `on_command` 与 loop 线程并发）锁正确性 |
| 引擎集成 | `launch_background_run` 传 sinks 后，后台 `run_workflow` 真产出 SpanBegin/Span/LeafEvent 入 buffer；resume 不重放 buffer（瞬态）；隔离（事件不进 host 上下文） |
| demo 集成 | drill_run poll loop：buffered_events → fresh UiAdapter 重放 → 断言 push_ui_message 收到 agent_span/fanout_graph 卡 + 稳定 id 幂等（多 tick 不出重复卡）；`truncated` 透传；M3② `run_runs_board` 返回 per-run RunResult + `fetch_run_result` 拉全文 |
| 真模型 E2E（gated） | 起多条真后台 run → drill 一条 → board 行展开见**真扇出 + 真 leaf 状态流转**（headline path，非 fallback）；run 完成后 host 经 M3② 汇报某条的**实质结论**。须 `uv sync --group example`、tracing 开、实际跑（见 memory：gated real-E2E must actually run） |

---

## 9. 与 M2 的关系（同一底座；M2 设计留 `04`）

M2（持久侧栏多-workflow 实时流程图）**消费同一 transport 底座**：M2 = "每轮把**所有**在飞 run 的事件
都 drain+pump 到一个**常驻侧栏** surface"，而 M3① = "选择式 pump **一条** run 到 chat inline"。二者
共用 §3 的引擎底座（buffer + event_sinks + buffered_events），差异全在消费侧（渲染 surface + pump 范围）。

**本切片对底座的硬约束（保证 M2 够用）：**
- `buffered_events(run_id)` 可对**任意** run_id 调用（M2 遍历所有在飞 run）——已满足（按 key 查 slot）。
- buffer 含 fan-out（parallel/pipeline/dag）span 事件，足以渲染流程图拓扑——已满足（`on_span`/`on_span_begin`
  覆盖全 SpanKind）。
- 全量重放幂等（M2 侧栏每 tick 重渲不堆卡）——稳定 span_id 保证。

M2 的渲染技术（侧栏布局、多-workflow 并列、真·拓扑 DAG 图 vs 现 `FanoutGraph.tsx` 的宽度条）、pump 全量
范围的性能权衡等，留 `04-*.md` 在切片 2 启动时设计。

---

## 10. 公共 API 增量面（全 additive，keyword-only / 新符号）

引擎（`langchain_dynamic_workflow`）：
- `BufferedEvent`（包根导出）— raw 事件标签联合记录。
- `RunEventSinks`（包根导出）— 五路 sink 束。
- `BgRunManager.event_sinks(run_id, *, thread_id) -> RunEventSinks` — 新方法。
- `BgRunManager.buffered_events(run_id, *, thread_id=None) -> tuple[list[BufferedEvent], int]` — 新方法。
- `BgRunManager.__init__` 增 `max_buffered_events: int = 2000`（keyword-only，有默认）。
- `BgRunSlot` 增 `events` / `dropped` 字段（有默认，向后兼容）。

无对既有签名的破坏性改动；`run_workflow` 的 sink 形参早已存在（§见签名核实），M3① 只是开始**使用**它们。

demo（host）：`drill_run(run_id | label)`、`fetch_run_result(run_id | handle)` 两个 host 工具；
`launch_background_run` 接 sinks；`run_runs_board` 收尾返回 per-run RunResult；`RunBoard.tsx` 下钻 affordance。

---

## 11. 非目标 / YAGNI

- **不**做带外 SSE/WS 侧信道（A2）——真·跨 turn push 留作 M2 若性能逼迫时的备选，本切片不引入新服务面。
- **不**持久化事件日志（A3）——遥测瞬态即可，resume 重建。
- **不**在 M3① 做"全量在飞 run 实时"——那是 M2。
- **不**回传 N 份全文给 host——守控制流反转，默认精炼 + 按需 fetch。
- **不**改 `parallel`/`pipeline`/`dag` 等编排原语的公共契约——底座只加 BgRunManager 表面 + demo 接线。
