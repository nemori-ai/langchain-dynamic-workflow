# Claude Code Dynamic Workflows 逆向工程参考报告

**面向在 LangChain deepagents 上重实现该机制的工程师**
**日期**：2026-06-01
**范围**：本报告系统梳理 Anthropic Claude Code「Dynamic Workflows」(2026-05-28 随 Claude Opus 4.8 进入 research preview，要求 Claude Code v2.1.154+) 的范式、原语语义、journal/resume/determinism 机制、meta/codegen/校验层、执行底座与硬上限，以及编排范式。目标读者是 `langchain-dynamic-workflow` 的实现者——一个将该机制移植到 LangChain deepagents 生态的社区 Python 端口。

**可信度免责声明**：本报告是**逆向工程**产物，不是 Anthropic 官方规范。Anthropic 官方仅文档化了**行为契约**(behavior)，从未发布过**原语级 API 参考**(mechanism)。读者必须严格区分两层证据：(1) **official-confirmed** —— 已逐字核对 `code.claude.com/docs/en/workflows`（权威主源）或 Anthropic 博客/HN 上经核实的 Anthropic 员工发言；(2) **community-inferred** —— 来自社区逆向，其中部分（barrier 行为、determinism 禁用清单、`meta` 纯字面量规则）有**两个独立社区源**交叉印证（ray-amjad + alexop.dev），可信度较高；另一部分（content-hash 缓存键、`workflow()` 一级嵌套、`budget` 对象表面、512KB/180s/30s 等数值常量）**仅单一社区源**(ray-amjad 探测 pre-GA 二进制)，应视为合理但未经证实；(3) **speculative/contradicted** —— 例如「V8 isolate 底座」与「LangGraph 底座」之说均不被任何官方源支持。**关键提醒**：官方编排语言是 **JavaScript，不是 Python**；本 Python/deepagents 端口的原语命名镜像的是被逆向出来的 JS 表面，而非任何官方 Python API。port 的设计文档（`docs/plans/`）才是本端口预期行为的权威。

---

## 1. 范式：控制流反转 (Control-Flow Inversion)

Dynamic Workflows 的核心是**把"下一步做什么"的决策权从 LLM 转移到确定性脚本**。这是官方明确定义并 official-confirmed 的(`code.claude.com/docs/en/workflows`)。官方原文：「A dynamic workflow is a JavaScript script that orchestrates subagents at scale. Claude writes the script for the task you describe, and a runtime executes it in the background while your session stays responsive.」

官方对比表把差异讲得很直白：

| 维度 | 普通 subagent/skill | Dynamic Workflow |
|---|---|---|
| 谁决定下一步运行什么 | Claude，逐轮(turn by turn) | **脚本**(the script) |
| 中间结果存放在哪 | Claude 的 context window | **脚本变量**(script variables) |
| 到达调用方 context 的内容 | 整个 trajectory | 只有最终答案 |
| 中断后 | 重启该轮(restart the turn) | 同会话内可 resume |
| 规模 | 每轮几个委派任务 | 单次运行数十到数百 agent |

官方一句话概括：「**A workflow moves the plan into code.**」在普通模式下 Claude 是 orchestrator，每个中间结果都落进它的 context；workflow 模式下脚本持有 loop / branching / fan-out，Claude 的 context 只收到最终答案(official-confirmed)。

**为什么这能解决普通 agent 的问题**(official-confirmed)：单个对话无法协调足够多的 agent，且会被中间 context 淹没。Workflow 把规模扩展到「数十到数百 agent / 次运行」，把中间产物挡在 context 之外，做到「同会话内可 resume」，并能施加一种可复用的质量模式——让独立 agent **对抗式交叉审查**(adversarial cross-review) 彼此的发现，或从多个角度起草方案再加权比较——从而得到比单次更可信的结果(official-confirmed，`claude.com/blog/introducing-dynamic-workflows-in-claude-code`)。博客原话：「Work you'd normally plan in quarters now finishes in days.」「Agents address the problem from independent angles, other agents try to refute what they found, and the run keeps iterating until the answers converge.」

**对实现者的含义**：在 deepagents 端口里，每个 leaf `agent()` 调用应运行在一个隔离、用后即弃的 context（复用 deepagents 的 context quarantine），中间结果只活在 Python 编排脚本的局部变量里。这是把官方"行为契约"忠实落地的第一原则。

---

## 2. 七个原语的语义

下表是七个原语的概览。**除"控制流反转"这一范式外，所有原语级签名与失败语义均为 community-inferred**——官方文档从未命名或文档化 `agent/parallel/pipeline/phase/log/budget/workflow` 中任何一个。证据链注明在每行后。

| 原语 | 签名 (社区逆向) | barrier? | 失败语义 | 可信度 |
|---|---|---|---|---|
| `agent(prompt, opts?)` | `→ Promise<string \| object>` | n/a | 被用户 skip → 返回 `null` | community-inferred (双源) |
| `parallel(thunks)` | `→ Promise<any[]>` | **是**(barrier) | thunk 抛错 → 该槽位 `null`，调用本身**永不 reject** | community-inferred (双源) |
| `pipeline(items, ...stages)` | `→ Promise<any[]>` | **否**(streaming) | stage 抛错 → 该 item 落为 `null` 并跳过其剩余 stage | community-inferred (双源) |
| `phase(title)` | `→ void` | n/a | — | community-inferred |
| `log(message)` | `→ void` | n/a | — | community-inferred |
| `budget` | `{ total, spent(), remaining() }` 注入对象 | n/a | `spent() >= total` 后 `agent()` 抛错 | community-inferred (单源) |
| `workflow(nameOrRef, args?)` | `→ Promise<any>` | n/a | 未知名/不可读路径/子语法错/二级嵌套 → 抛错 | community-inferred (单源) |

### 2.1 `agent(prompt, opts?)` —— leaf subagent 派生

派生一个全新 context 的 subagent。无 `schema` 时逐字返回 subagent 最终文本(string)；传入 `schema` 时返回 AJV 校验过的 object。若用户在 `/workflows` 里跳过该 agent，返回 `null`（因此到处可见 `.filter(Boolean)` 习语）。

`opts` 字段(community-inferred, ray-amjad)：
- `label` (string)：显示名，默认取 prompt 前 60 字符；**不进 resume 缓存键**。
- `phase` (string)：把该 agent 归入命名进度组，供并发 stage 内使用；**不进缓存键**。
- `schema` (JSON Schema)：强制结构化输出。
- `model` (`'haiku'|'sonnet'|'opus'|'inherit'|完整 model ID`)：**不做校验**，打错的字会透传到 API 调用层才失败。
- `isolation` (`'worktree'`)：在新 git worktree 中运行该 agent（`'remote'` 在二进制里存在但本 build 禁用）。
- `agentType`：从活动 registry 取注册的 subagent 类型；**会校验**，未知值抛错并列出可用 agent（与 `model` 不同）。
- `stallMs` (number)：覆盖每 agent 180000ms (3 min) 的 stall 超时。

**缓存键四要素**：`schema/model/isolation/agentType`——改其中任一都会重跑该调用；`label/phase` 纯装饰，永不失效缓存。

### 2.2 `parallel(thunks)` —— barrier 式 fan-out

接收**thunk 数组** `[() => agent(...), () => agent(...)]`，**不是** promise 数组（裸调用会立即启动，绕过并发限制器）。它是 **barrier**：等待每个 thunk 才返回。**失败语义**(双源印证 ray-amjad + alexop.dev)：抛错的 thunk 在结果数组对应槽位变成 `null`，**调用本身永不 reject**——所以使用前必须 `.filter(Boolean)`。结果数组按设计带"空洞"(null = 抛错/跳过/预算丢弃)。**仅当下游 stage 确实需要一次拿到整组前序结果时才用 `parallel`**（跨全集去重/合并、基于计数的提前退出、把一项与全体对比）；否则优先 `pipeline`。

### 2.3 `pipeline(items, ...stages)` —— 无 barrier 流式

每个 item 独立流过**所有 stage，stage 间无 barrier**——item A 可在 stage 3 而 item B 还在 stage 1。墙钟时间≈最慢单 item 整条链，而非各步最慢 stage 之和。这是多 stage 工作的**默认**。每个 stage 回调收到 `(prevResult, originalItem, index)`。**失败语义**(双源)：抛错的 stage 把该 item 落为 `null` 并跳过其剩余 stage。`pipeline` 与 `parallel` 可组合——pipeline stage 常返回一个对前序发现的 `parallel(...)`。

### 2.4 `phase(title)` / `log(message)` —— 进度叙事

`phase(title)` 开启进度组，其后派生的 agent 加入该组（`/workflows` 实时树）；`meta.phases[].title` 与 `phase()` 调用**精确匹配**。`log(message)` 在进度树上方发一行叙事。在并发 pipeline/parallel stage 内，优先用每-agent 的 `phase` 选项而非全局 `phase()`，以避免组归属竞态。官方仅确认用户可见概念：「`/workflows` shows each phase with its agent count, token total, and elapsed time」(official-confirmed)，但从未文档化函数 API。

### 2.5 `budget` —— 共享 token 池 (community-inferred, **单源**)

注入对象 `{ total, spent(), remaining() }`，反映用户经 `'+500k'` 式指令设的 token 目标。`budget.total` = 目标或 `null`；`budget.spent()` = 本轮**输出 token**，跨主循环与所有 workflow **共享**(非 per-workflow)；`budget.remaining()` = `max(0, total − spent())`，无目标时为 `Infinity`。目标是**硬上限**：`spent()` 达 `total` 后新 `agent()` 调用抛 `WorkflowBudgetExceededError`，在飞 agent 完成并保留结果。预算循环**必须**用 `budget.total` 守卫（如 `while (budget.total && budget.remaining() > 50_000)`），否则无目标时 `remaining()` 为 `Infinity`，无守卫循环会冲进 1000-agent 上限。

### 2.6 `workflow(nameOrRef, args?)` —— 一级内联嵌套 (community-inferred, **单源**)

内联运行另一个 workflow 并返回其返回值；传保存的 workflow 名或 `{ scriptPath }`。子 workflow **共享**本次运行的并发上限、agent 计数器、abort signal、token 预算；其 agent 显示为嵌套组。**仅一级嵌套**——在子 workflow 里调 `workflow()` 抛错。未知名/不可读路径/子语法错也抛错（catch 以优雅降级）。alexop.dev 提到 `workflow()` 存在但**未**印证"一级嵌套/共享计数器"细节，故此条仅单源。

---

## 3. journal / resume / determinism 机制

### 3.1 resume 行为 (official-confirmed)

官方逐字：「If you stop a run, you can resume it: agents that already completed return their cached results, and the rest run live.」「The runtime tracks each agent's result as the run progresses, which is what makes a run resumable within the same session.」**resume 仅限创建该 run 的同一 Claude Code 会话**——「If you exit Claude Code while a workflow is running, the next session starts the workflow fresh.」这强烈暗示已完成 agent 的结果是会话/内存作用域，不跨进程重启持久化（持久化细节本身是推断，但 same-session-only 是显式的）。

### 3.2 content-hash journal (community-inferred, **单源关键机制**)

官方只确认**行为**(per-agent 结果追踪)，从未公布**缓存键如何推导**。ray-amjad 的逆向称：每个 `agent()` 调用按其 `(prompt, opts)` 的**哈希**为键、连同结果一起 journal（subagent transcript 存为 `agent-<id>.jsonl`）；resume 经 `Workflow({ scriptPath, resumeFromRunId })`，哈希命中已完成条目即瞬时返回缓存（不调模型），第一个新/改调用及其后全部 live 运行。"相同脚本 + 相同 args = 100% 缓存命中"。**重要核验结论**：alexop.dev **并未**说键是 `(prompt, opts)` 的 content hash，只说「Workflows journal every agent() call」。因此 content-hash-of-`(prompt,opts)` 的键推导**只靠单一逆向源**；Anthropic 是 content-hash 还是 index/position-based **官方完全未文档化**。

**对 port 的直接含义**：本端口的设计假设 LangGraph 原生缓存是 index-based，故额外加一层 content-hash journal——这是**port 自己的设计取舍**，不是对 Anthropic runtime 的镜像。实现者应把 content-hash 当作"为得到稳健 resume 而采用的工程选择"，而非"复刻官方"。

### 3.3 determinism guard (community-inferred, **双源**)

因为 run 被 journal 以支持 resume，任何非确定性都会使缓存失效，所以三个经典 JS 非确定源在 workflow 脚本内**抛错**(fail-loud)：`Date.now()`、`Math.random()`、以及**无参** `new Date()`/`Date()`（`new Date(specificValue)` 仍可用）。这是 alexop.dev 的亲手观察（非引用 Anthropic）与 ray-amjad 的二进制探测**双源印证**，可信度因此提升到 partially-confirmed。**官方仅在粗粒度上印证**："No direct filesystem or shell access from the workflow itself"与"no non-deterministic mid-run input"，从未给出被禁标识符清单或"抛错"行为。

**仍未确证**：guard 是**预执行 AST gate** 还是**运行时 throw**？官方产品未定论。推荐 workaround：时间戳经 `args` 传入；要让 agent 各异，按 loop index 或 per-index label 变 prompt，而非随机化。

---

## 4. meta 层 / codegen / 校验

### 4.1 触发与 codegen 入口 (official-confirmed)

用户描述任务并在 prompt 任意处含 `workflow` 一词(Claude 高亮，alt+w 取消)，**或** `/effort ultracode`(把 xhigh reasoning 与每个实质任务的自动 workflow 编排结合)。Claude 随即为该任务写一个自包含 JS 编排脚本，交给 Workflow 工具(输入字段 `script | name | scriptPath`，外加可选 `args` 与 `resumeFromRunId`)。捆绑的 `/deep-research` 随产品发货。

### 4.2 `export const meta = {…}` 纯字面量块 (community-inferred, **双源 + 官方旁证**)

每个脚本**第一条语句**必须是 `export const meta = {…}`。字段：`name`(必填，非空 string，这才是 workflow 名而非文件名)、`description`(必填，单行，显示在权限对话框)、`whenToUse`(可选)、`phases`(可选 `{title, detail?, model?}` 数组——`title` 与 `phase()` 精确匹配；**`phases[].model` 是 display-only**，无任何代码读它来选模型，模型仅由 `agent()` 的 `model` 选项决定)。

`meta` **必须是纯字面量**：无变量、无函数调用、无 spread (`...`)、无模板插值；保留键 `__proto__`/`constructor`/`prototype` 被拒。**理由(static-extraction)**：parser 走语法树、在 body 执行**前**静态读取 `meta` 来填充审批对话框。官方确认 per-run prompt「shows the planned phases」并提供「View raw script」(Ctrl+G)——这要求元数据在不执行不可信 body 的前提下可提取。但 `meta` 字面量语法、保留键拒绝、AST 强制均**不在官方文档**。

### 4.3 预执行校验 (community-inferred)

parser 强制：`meta` 存在、为首语句、纯字面量(保留键拒)；`name`/`description` 必填。捆绑的社区 linter `validate-workflow.mjs` 镜像 parser 规则：剥注释/字符串后，对缺失/非首/非字面 `meta`、缺 `name`/`description`、被禁非确定调用(`Date.now`/`Math.random`/无参 `new Date`)、host-API 误用、超尺寸脚本(>512KB/524288 字节，解析前拒)报错；对 `require()`/`import…from`/`process.*` 与 `parallel([agent(...)])` 裸 promise 误用告警。**这是社区对 parser 规则的近似重实现(comment/string 剥离 + regex)，不是 Anthropic 的真实 validator 代码**。

**对 port 的含义**：在 Layer 2 meta 层，对 LLM 生成的 Python 编排脚本应做 **AST gate**(禁 imports / dunders / banned names) 校验后再执行。这与 Anthropic 的 AST 校验在精神上一致，但官方校验器的真实实现(真 ESTree walk vs tokenizer)未公开。

### 4.4 args 序列化 (community-inferred，经验观测，可能漂移)

`args` 全局：工具 input-schema 类型标为 `unknown`，但**经验探测(2026-05-29)**显示 runtime 在脚本运行前把 `args` **序列化为字符串**——JSON 对象/数组到达时是 JSON 文本，纯字符串保持字符串，未传则 `undefined`；脚本须归一化(`typeof === 'string'` 守卫 + try/parse)。**作者明确指出此行为与工具自身 inline schema 描述矛盾，可能是 pre-release 怪癖，GA 后可能改为 live-object passthrough**——需在当前 build 重新核验。

### 4.5 codegen 模式：一次成型 + 审批门 → 迭代式 edit-and-resume (official-confirmed)

初次 codegen 实质是**一次成型**：Claude 写出完整脚本，runtime 解析/校验/持久化到会话目录文件，再走审批门(CLI: Yes run / Yes don't ask again / View raw script Ctrl+G / No；Desktop 卡片显示名+phase 列表+token 警示，Once/Always/Deny)。**没有证据**表明首跑前存在自动 self-correcting codegen 循环。**精化**走 persist-and-edit-resume：跑一次 → 用 Write/Edit 改保存的 `.js` → 以 `{scriptPath, resumeFromRunId}` 重调；首改之前的 `agent()` 调用从 journal 缓存瞬时重放，仅被改调用及其后 live 重跑。

---

## 5. 执行底座 / sandbox / 硬上限

### 5.1 底座：JS 脚本在隔离环境中运行 (官方确认范围有限)

官方仅说「a runtime executes the script ... in an isolated environment, separate from your conversation」，且 Boris Cherny(HN `bcherny`，经核实 Anthropic 员工)确认 runtime 是「JavaScript, running locally or in the cloud」，构建于共享的 Claude Agent SDK 之上，并称「more docs + technical details coming soon」。

**「V8 isolate」框定被核验为 contradicted/speculative**：没有任何源用「V8 isolate」描述 workflow 编排 runtime。显式的「V8 isolate vs microVM」语言属于**另一个产品**——Cloudflare「Claude Managed Agents」的 **agent 代码沙箱后端**，不是 workflow 编排 runtime；混淆二者是已知错误。社区"VM synchronous timeout"标签反而指向 **vm-module 式进程内沙箱**，而非单独 provision 的 V8 isolate。**同样地，"LangGraph @entrypoint/@task/checkpointer 底座"是本 Python port 的设计选择，不是 Anthropic runtime 的确认镜像**——Anthropic 从未披露内部实现。实现者不应把任一具体底座当作"复刻官方"。

### 5.2 script-vs-agent I/O 切分 (official-confirmed)

官方"Behavior and limits"表逐字：约束「No direct filesystem or shell access from the workflow itself」，原因「Agents read, write, and run commands. The script coordinates the agents.」这是控制流反转/I-O-切分的核心：脚本是纯确定性协调器，所有副作用工作委派给用后即弃 context 的 subagent，仅最终结果返回。社区源补充：编排器内禁 `require()`/`fs`/`process`/网络。

### 5.3 硬上限

| 上限 | 值 | 可信度 |
|---|---|---|
| 并发 agent | **"up to 16, fewer on limited cores"** | **official-confirmed**；精确公式 `min(16, max(2, cores−2))` 与 floor-of-2 为 community-inferred **单源** |
| 单次运行总 agent | **1,000**(防失控循环) | **official-confirmed**；超限抛 `WorkflowAgentCapError` 为社区**单源**，错误类名未经证实 |
| 超并发的额外调用 | 排队，slot 空出再跑(非报错) | official(粗) + community |
| 脚本尺寸 | 524288 字节(512KB)，解析前拒 | community **单源**，lower-confidence |
| 每-agent stall | 180000ms(3min)，重试至多 5×，再放弃 | community **单源** |
| VM 同步超时 | 30000ms，仅捕获无限同步循环(非墙钟上限) | community **单源** |
| token 预算 | 用户经 `'+500k'` 式指令设；超限抛 `WorkflowBudgetExceededError` | community **单源** |

**注意**：只有 16-并发与 1000-总数两个数字是官方确认；字节/stall/VM/重试常量全是 ray-amjad 单源探测 **pre-GA 二进制**所得，无第二源印证。鉴于 enablement gate 在 GA 时已漂移（pre-GA 的 `CLAUDE_CODE_WORKFLOWS=1` opt-IN → GA 的 `CLAUDE_CODE_DISABLE_WORKFLOWS=1` opt-OUT），**其余单源常量同样可能在 GA 漂移，硬编码前应对当前 build 重新核验**。

### 5.4 隔离、权限、后台与通知 (官方多处确认)

- **worktree 隔离**(official-confirmed 为 subagent 能力)：frontmatter `isolation: worktree` 或 Agent-tool `isolation: "worktree"`，在临时 git worktree(默认从 default branch 分出)中运行；无改动则自动清理。成本 `~200–500ms + disk/agent` 为 community **单源**。
- **subagent 权限继承**(official-confirmed)：workflow 派生的 subagent **总是 acceptEdits 模式**并继承用户 tool allowlist，**无论会话权限模式**；文件编辑自动批准；allowlist 外的 shell/web/MCP 工具仍可能 mid-run 弹窗。
- **无 mid-run 用户输入**(official-confirmed)：仅 agent permission prompt 能暂停 run；要在 stage 间签字，把每个 stage 做成独立 workflow。
- **后台执行 + 完成通知**(official-confirmed)：run 立即返回 run ID(社区称 `wf_…`)，进度流到 `/workflows`；完成时 `<task-notification>` 注入对话(GitHub issue #18544 印证后台任务完成通知的注入机制)。
- **MCP/session 工具**(official-confirmed)：subagent 默认继承主对话的内部 + MCP 工具，受 session allowlist 过滤，经 `tools`/`disallowedTools`/`mcpServers` frontmatter 作 per-agent scoping。

---

## 6. 编排范式与实战范例

以下范式中，对抗式验证为 official-confirmed（概念），其余命名范式为 community-inferred（来自 ray-amjad 范式目录 + alexop.dev）。

- **fan-out + synthesize**：把综合任务拆成可并行的每部分（"review every file in this diff"、"audit all 40 dependencies"），再综合。官方头条用例（codebase-wide bug sweep、500-file migration、auth audit）。
- **pipeline-by-default**：默认 `pipeline()`，因为 item 一就绪即推进（无队头阻塞）；只在下游 stage 需整组前序结果时才用 `parallel()` barrier。反模式：`parallel → 无跨项依赖的 transform → parallel` 本该是 pipeline。
- **adversarial verification**(official-confirmed 概念)：对每个发现，派生 N 个独立 skeptic agent 去**反驳**；多数存活才保留(如 3 票/claim，<2 反驳则存活)。官方原话：「other agents try to refute what they found, and the run keeps iterating until the answers converge.」N-skeptic/多数表决的实现细节是社区化的。**diverse-lens 变体**：给每个 verifier 不同视角(correctness/security/performance/reproducibility)。
- **judge panel**：从不同角度生成 N 个独立 attempt → 并行 judge 评分 → 从赢家综合、可嫁接亚军最佳部分。官方"draft a plan from several angles and weigh them"的结构化形式。
- **loop-until-budget / loop-until-dry / loop-until-target**：计数循环 `while(bugs.length < 10)`；预算循环 `while(budget.total && budget.remaining() > 50_000)`；dry 循环——连续 K 轮无新发现才停。**关键陷阱**：dedupe 要对**所有已见**(not just confirmed)去重，否则永不终止。所有循环都须带硬停(counter/budget)。
- **structured-output 作可靠数据路径**：你之后会读取的每个结果字段都该有 `schema` 撑腰；schema/model/isolation/agentType 改动会失效 journal 缓存(强制重跑)，label/phase 只是装饰。

**结构化输出机制**(community-inferred)：传 `opts.schema` 时，runtime 用 **AJV** 编译 schema，合成一个隐藏 `StructuredOutput` 工具(其 input 即该 schema)，指示 subagent 恰好调用一次；调用经 AJV 校验，**不匹配则把校验错误交回 agent 重试**；若 agent 结束时未调该工具，runtime 至多再 nudge 两次才失败。`agent()` 返回校验过的 tool input（无需 `JSON.parse`）。

**实战锚点**：Jarred Sumner(Bun)用 dynamic workflows + 对抗式 review 把 Bun 从 Zig 移植到 Rust——一个 workflow 映射每个 struct field 的正确 Rust lifetime，下一个把每个 `.rs` 写成行为等价端口；数百 agent 并行、每文件两个 reviewer；per-unit 模式 = do-work → adversarial-review → apply。约 750K 行 Rust、99.8% 测试通过——但 canary-only，未入生产(community-inferred；时间线在各源不一致："six days" vs "eleven days to merge"，未由 Sumner 原始声明调和)。**主导批评**(HN, official-confirmed 存在该批评)是 **token 成本**("tokenmaxxing"，有人 62 个 Opus agent 18 分钟烧掉 5 小时 cap)、"slop debt"、弱 mid-run 人工控制。

---

## 7. 机制可信度表

| Claim(声明) | Verdict(裁决) | 最佳信源 |
|---|---|---|
| workflow 是 **JS**(非 Python)脚本，runtime 后台执行；脚本持有 loop/branch/fan-out，中间结果在脚本变量，仅最终答案到 context | **confirmed** | `code.claude.com/docs/en/workflows` |
| 并发上限 "up to 16, fewer on limited cores" | **confirmed**（公式 `min(16,max(2,cores−2))` 仅单源、未证实） | 同上 |
| 单次运行硬上限 **1,000 agent**(防失控) | **confirmed**（`WorkflowAgentCapError` 类名单源、未证实） | 同上 |
| 脚本**无**直接 fs/shell 访问，仅 leaf agent 做 I/O | **confirmed** | 同上 |
| resume 仅 same-session；完成 agent 返回缓存，其余 live；退出 CC 则重头 | **confirmed** | 同上 |
| resume 缓存键 = `(prompt,opts)` 的 **content hash**(schema/model/isolation/agentType 入键) | **partially-confirmed**（行为官方；键推导**单源** ray-amjad，alexop.dev 未印证） | `raw.githubusercontent.com/ray-amjad/.../api-reference.md` |
| determinism guard：`Date.now()`/`Math.random()`/无参 `new Date()` **抛错** | **partially-confirmed**（**双源**，但 Anthropic 未确认；AST-gate vs runtime-throw 未定） | `alexop.dev/posts/claude-code-workflows-deterministic-orchestration/` |
| `parallel(thunks)` 是 barrier，抛错→`null`，**永不 reject**，收 thunk 非 promise | **partially-confirmed**（**双源**，官方未命名） | ray-amjad api-reference |
| `pipeline` 无 barrier，stage 收 `(prevResult,originalItem,index)`，抛错→item 落 `null` 跳过余 stage | **partially-confirmed**（**双源**，官方未命名） | ray-amjad api-reference |
| `export const meta` 首语句、纯字面量、保留键拒、静态读取填审批框 | **partially-confirmed**（纯字面量**双源** + 官方旁证 View raw script） | ray-amjad api-reference |
| 无 mid-run 用户输入，仅 agent permission prompt 能暂停 | **confirmed** | `code.claude.com/docs/en/workflows` |
| subagent 总是 acceptEdits + 继承 allowlist，无论会话模式 | **confirmed** | 同上 |
| `workflow(nameOrRef, args?)` 一级嵌套；子共享并发/计数/预算 | **unconfirmed**（**单源** ray-amjad，alexop.dev 未印证细节） | ray-amjad api-reference |
| `budget` 对象 `{total,spent(),remaining()}`，超限抛 `WorkflowBudgetExceededError` | **unconfirmed**（**单源**） | ray-amjad api-reference |
| 512KB 脚本上限 / 180s stall(×5) / 30s VM 同步超时 | **unconfirmed**（**单源**，无第二源） | ray-amjad api-reference |
| 编排 runtime 是 **V8 isolate** / **LangGraph** 底座 | **contradicted/speculative**（V8-isolate 语言属 Cloudflare 另一产品；LangGraph 是本 port 设计选择） | `blog.cloudflare.com/claude-managed-agents/` |
| 要求 v2.1.154+，2026-05-28 research preview，paid+API+Bedrock/Vertex/Foundry，disable via `CLAUDE_CODE_DISABLE_WORKFLOWS=1` (opt-OUT) | **confirmed** | `code.claude.com/docs/en/workflows` |

---

## 8. 仍未确证的 open questions / 空白点

1. **官方 Python 表面是否存在**：官方语言全程是 JS；未找到任何官方 Python API。本端口纯属社区重实现，其原语命名镜像逆向出的 JS 表面。
2. **原语签名是否稳定**：Anthropic 从不发布原语级 API 参考。`agent/parallel/pipeline/phase/log/budget/workflow` 这些精确名与签名是否就是 Anthropic 内部所用，无从证实——社区可能标准化了 Anthropic 实际不用的名字。
3. **真实内部 runtime 未知**：官方只说"isolated runtime"追踪 per-agent 结果。是 Node `vm` 模块、QuickJS 内嵌，还是真 `isolated-vm` V8 isolate？未证实。content-hash vs index-based 缓存键官方未文档化。
4. **determinism guard 形态**：是预执行 AST gate 还是运行时 fail-loud throw？官方未定论；错误信息/实现(shadow globals vs AST transform vs restricted runtime)未公开。
5. **缓存键如何索引**：按 call index 还是 prompt/args 的 content hash？官方未文档化。本 port 假设 LangGraph 原生为 index-based 并补 content-hash journal——这是 port 自身假设。
6. **云端并发上限**：本地 CLI 是 "16, fewer on low-core"；Bedrock/Vertex/Foundry/Agent SDK 上是否不同？未知。
7. **`isolation:'worktree'`/`agentType` 选项真实性**：community-inferred；`'remote'` isolation 与 `workflow-remote-agent` 在探测 build 中"存在但禁用"，GA 行为未知。
8. **错误类名稳定性**：`WorkflowAgentCapError`/`WorkflowBudgetExceededError`/`wf_…` run-ID 方案均社区报告，跨版本稳定性未知。
9. **单源探测的 GA 漂移**：ray-amjad 探测 pre-GA 二进制且 enablement gate 已漂移；其余单源常量可能已在 GA 改变，硬编码前须重核。
10. **`budget.spent()` 计什么**：仅输出 token 还是总 token？`'+500k'` 指令精确语法？社区推断，官方文档无。
11. **workflow 派生 agent 能否再派生 subagent**：官方 sub-agents 文档说"Subagents cannot spawn other subagents"，但 workflow 派生 agent 是否为特例，未核实。
12. **质量无独立基准**：未找到 dynamic-workflow 输出**质量**(vs 单 agent 或 vs LangGraph/CrewAI)的独立量化基准；99.8% Bun 通过率是自报且 canary-only。

---

## 9. 带标注的信源清单

**Official(官方主源)**
- `https://code.claude.com/docs/en/workflows` — **PRIMARY/权威**。确认 JS+runtime、控制流反转、16-并发与 1000-总数上限、无-fs/shell、same-session resume、无 mid-run 输入、acceptEdits+allowlist 继承、per-stage model routing、`/deep-research`、v2.1.154+、`/config`+`disableWorkflows`。**不**命名/文档化原语 API。
- `https://claude.com/blog/introducing-dynamic-workflows-in-claude-code` — **PRIMARY**。高层：数百并行 subagent、folding 前验证、对抗式交叉检查、ultracode 触发。无 API 签名。
- `https://www.anthropic.com/news/claude-opus-4-8` — 发布上下文(2026-05-28)，仅一句 workflow 提及。
- `https://code.claude.com/docs/en/worktrees` — worktree 隔离、`isolation: worktree`、清理规则。
- `https://code.claude.com/docs/en/sub-agents` — frontmatter 字段、tools/MCP 继承、isolation、background。
- `https://code.claude.com/docs/en/agents` — subagent vs agent view vs teams vs workflows。
- `https://blog.cloudflare.com/claude-managed-agents/` — **注意：另一产品**。"V8 isolate vs microVM"语言的真实出处(agent 沙箱后端，**非** workflow 编排 runtime)。

**Community-reverse(社区逆向，最高保真但未经 Anthropic 证实)**
- `https://alexop.dev/posts/claude-code-workflows-deterministic-orchestration/` — 最佳社区深挖；determinism throw 声明、journaling/cache 框定、parallel-barrier、pipeline-no-barrier、meta 纯字面量的**第二独立源**。
- `https://raw.githubusercontent.com/ray-amjad/claude-code-workflow-creator/main/references/api-reference.md` — **全部 API 级语义的主源**：完整签名、`label/phase/schema/model/isolation/agentType/stallMs` 选项、barrier 行为、AJV StructuredOutput+retry、budget 共享池、`workflow()` 一级嵌套、上限、determinism sandbox、content-hash journal & resume。**多数单源细节出处**。
- `https://github.com/ray-amjad/claude-code-workflow-creator/blob/main/SKILL.md` — 撰写流程；印证 pipeline-vs-parallel 默认、args 字符串序列化(2026-05-29 探测)、`.filter(Boolean)` 陷阱。
- `https://kenhuangus.substack.com/p/claude-code-orchestration-dynamic` — 16-并发/1000-per-run/后台 的社区印证。
- `https://yage.ai/share/claude-code-workflow-determinism-en-20260528.html` — process vs result determinism 分析，引官方 docs/blog。
- `https://news.ycombinator.com/item?id=48311705` — HN 线程，含经核实的 Boris Cherny(`bcherny`)发言("JavaScript, locally or in the cloud"、Claude Agent SDK、"more docs coming soon")；token-cost/slop/control 批评、Bun 重写辩论。

**Repo(社区代码工件)**
- `https://github.com/ray-amjad/claude-code-workflow-creator/blob/main/scripts/validate-workflow.mjs` — 社区 linter，编码 parser 硬规则(512KB、meta-first-pure-literal、禁 Date.now/Math.random/无参 new Date、no require/import/process、parallel-thunks-not-promises)。**启发式重实现，非 Anthropic 校验器代码**。
- `https://github.com/ray-amjad/claude-code-workflow-creator/blob/main/references/patterns.md` — 可运行 fan-out/pipeline/loop-until-budget/adversarial-verify/judge-panel/nested-workflow 范式。
- `https://github.com/anthropics/claude-code/issues/18544` — 后台任务完成通知注入机制。
- `https://github.com/anthropics/claude-code/issues/58933` — determinism gap 背景(社区用户，无 maintainer 确认)。

**Blog(二手报道/印证)**
- `https://www.marktechpost.com/2026/05/28/anthropic-ships-claude-opus-4-8-...-capped-at-1000-subagents/` — 1000-cap 与 2026-05-28 发布的二手印证。
- `https://claudefa.st/blog/guide/development/dynamic-workflows` — when-not-to-use、failure modes、Bun/Klarna 示例。
- `https://developertoolkit.ai/en/claude-code/advanced-techniques/dynamic-workflows/` — 原语表、约束、disable 设置(转述官方)。
- `https://www.mindstudio.ai/blog/claude-code-agentic-workflow-patterns` — "completeness-critic"/"multi-modal sweep"等**较松散**社区框定(不在 canonical 目录)。
- `https://www.lowcode.agency/blog/claude-code-vs-langgraph`、`https://www.developersdigest.tech/guides/ai-agent-frameworks-compared` — 框架对比(workflows = 模型撰写、dev-time、runtime-决定 fan-out；LangGraph = 人撰写、显式类型 DAG、跨会话持久状态、硬 HITL checkpoint)。
- `https://digg.com/ai/rb5xj3bt`、`https://www.testingcatalog.com/anthropic-launches-dynamic-workflows-for-claude-code/`、`https://pasqualepillitteri.it/en/news/3663/...`、`https://www.techtimes.com/articles/317363/...`、`https://www.reworked.co/digital-workplace/anthropic-announces-dynamic-workflows-in-claude-code/`、`https://agentpedia.codes/blog/claude-opus-4-8-claude-code-workflows`、`https://azukiazusa.dev/en/blog/claude-code-dynamic-workflow/`、`https://www.agentupdate.ai/blog/claude-code-dynamic-workflows-best-practices/` — 发布报道与 hands-on，多为转述/印证官方。
