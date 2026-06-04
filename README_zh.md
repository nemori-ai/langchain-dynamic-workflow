# langchain-dynamic-workflow

[![CI](https://github.com/nemori-ai/langchain-dynamic-workflow/actions/workflows/ci.yml/badge.svg)](https://github.com/nemori-ai/langchain-dynamic-workflow/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Types: pyright strict](https://img.shields.io/badge/types-pyright%20strict-blue.svg)
![Status: alpha 0.2.0](https://img.shields.io/badge/status-alpha%200.2.0-orange.svg)

[English](README.md) | **中文**

> 面向 LangChain [`deepagents`](https://github.com/langchain-ai/deepagents) 的确定性、脚本化、可断点续跑的多 agent 编排引擎 —— Claude Code **Dynamic Workflows** 的社区复刻。

普通 agent **逐回合**决定自己的控制流：每一次循环、分支、扇出都活在模型的上下文窗口里，既烧 token 又不断堆积中间状态。`langchain-dynamic-workflow` 把这件事**反转**过来 —— 由一段确定性的编排**脚本**掌控控制流，只有叶子 `agent()` 调用才委派给 deepagent，每个 deepagent 跑在隔离、用完即弃的上下文里，因此**只有最终结果回到调用方的上下文**。

|  | 普通 agent | Dynamic workflow |
|---|---|---|
| 谁决定下一步 | LLM，逐回合 | **脚本**（确定性代码） |
| 中间结果存哪 | 模型的上下文窗口 | **脚本变量里** |
| 什么进入调用方上下文 | 整条执行轨迹 | **只有最终结果** |

## 目录

- [为什么](#为什么)
- [特性速览](#特性速览)
- [架构](#架构)
- [安装](#安装)
- [快速上手](#快速上手)
- [host agent 与 meta 层](#host-agent-与-meta-层)
- [断点续跑、预算与可观测性](#断点续跑预算与可观测性)
- [示例](#示例)
- [公开 API](#公开-api)
- [开发](#开发)
- [状态](#状态)
- [许可证](#许可证)

## 为什么

逐回合的控制流有三笔随任务增长而复利的成本：上下文窗口被中间推理填满、执行轨迹不确定、一旦中断便无法在不重放模型的前提下续跑。控制流反转一次性解掉这三件事 —— 脚本掌管循环与分支，中间结果留在普通变量里，内容哈希 journal 让续跑时已完成的工作以零模型成本重放。

适用场景：任务**扇出密集**（研究 N 个角度、评审 M 个候选）、**长链多步**（轨迹否则会撑爆上下文）、或需要**确定性续跑与跨多个 sub-agent 的共享 token 预算**。

## 特性速览

- **确定性控制流** —— 循环、分支、扇出写在代码里，而非模型脑中。
- **上下文隔离** —— 每个叶子跑在全新、用完即弃的 deepagents 上下文里，只回吐折叠后的结果。
- **parallel、pipeline 与 race 扇出** —— `parallel()`（阻塞 barrier）、`pipeline()`（无 barrier 流式）与 `race()`（best-of-N 早退：首个令 `win` 为真者胜、在飞 loser 全数 cancel、决策内容哈希 journal 故 resume 复现胜者），共享一道并发闸门。
- **按内容哈希续跑** —— success-only 的 journal 在续跑时以零模型成本重放已完成的叶子。
- **fail-loud 确定性 guard** —— 重放时 `agent()` 调用序列一旦分叉即抛错，绝不喂回错位的缓存。
- **共享 token 预算** —— 一道上限管住所有叶子，配套 `loop-until-budget` 范式。
- **默认可观测** —— 每个 `agent` / `parallel` / `pipeline` / `race` 调用都向可选 sink 发一个 span（不接 sink 时零成本）。
- **per-leaf sandbox 隔离** —— 执行类叶子各租一个隔离 backend，`/shared/` 路由支持显式的 producer→consumer 交接。
- **meta 层** —— host agent 运行时当场写编排脚本，经 AST gate 校验后才进入单点受限 `exec`。
- **工程从严** —— Python 3.12、async-first、pyright `strict`，Layer 0/1/2 边界由 import-linter 机械守护。

## 架构

三层，依赖单向（Layer 2 → Layer 1 → Layer 0），由 import-linter 机械守护：

- **Layer 0 —— 底座**：LangGraph 持久化执行（`@entrypoint` + `@task` + checkpointer），自带续跑、重放、命中缓存跳过。
- **Layer 1 —— 编排运行时**：核心原语 —— `agent()`、`parallel()`（barrier）、`pipeline()`（无 barrier）、`race()`（best-of-N 早退）、`phase()`、`log()`、`budget`、`workflow()` —— 外加两个 LangGraph 缺失的补丁：**内容哈希 journal**（原生缓存是 index-based）与 **fail-loud 确定性 guard**（原生把确定性当约定而非不变量）。
- **Layer 2 —— meta 层**：由 LLM 编写 Python 编排脚本；进入受限 builtins 的 `exec` 之前，先经 **AST gate** 校验（禁 import、dunder、禁用名）。

叶子 `agent()` 调用从一个**命名注册表（roster）**中解析出对应的 deepagent，并作为 `@task` 调起，复用 deepagents 的上下文隔离与 sandbox backend。

## 安装

```bash
uv sync   # 安装依赖 + 创建 .venv
```

Python 3.12+，依赖管理用 [uv](https://docs.astral.sh/uv/)。尚未发布到 PyPI；请从源码克隆或作为 git 依赖安装。

## 快速上手

写一段编排脚本（`ctx` 暴露核心原语），把叶子 agent 注册进 roster，然后跑起来。叶子可以是任何状态里带 `messages` 键的 runnable —— 通常是一个 `deepagents.create_deep_agent(...)`：

```python
import asyncio

from deepagents import create_deep_agent
from langchain_dynamic_workflow import Ctx, Roster, run_workflow


async def main() -> None:
    # 1. 按名字注册叶子 agent（build-time 接线；agent 自己不做这件事）。
    roster = Roster().register(
        "researcher",
        create_deep_agent(model="anthropic:claude-haiku-4-5"),
        description="研究单个主题",
    )

    # 2. 编排脚本掌控控制流；只有叶子 agent() 调用才委派给 deepagent。
    #    parallel() 是阻塞 barrier；失败的叶子落为 None（自行过滤），永不中断 barrier。
    async def orchestrate(ctx: Ctx) -> str:
        ctx.phase("research")
        findings = await ctx.parallel(
            [
                lambda t=topic: ctx.agent(f"Research {t}", agent_type="researcher")
                for topic in ("batteries", "solar", "wind")
            ]
        )
        surviving = [f for f in findings if f is not None]
        return f"synthesized {len(surviving)} findings: " + " | ".join(surviving)

    # 3. 跑起来。只有最终结果回到你手里 —— 而不是整条执行轨迹。
    result = await run_workflow(orchestrate, roster=roster)
    print(result)


asyncio.run(main())
```

## host agent 与 meta 层

想让 **host agent** 在后台驱动 workflow，把 `create_workflow_middleware(roster, workflows=...)` 挂到一个 host `create_deep_agent` 上。agent 便通过单个 `workflow` 工具调度所有 run：

| 命令 | 作用 |
|---|---|
| `run` | 按名启动**已注册**的 workflow，立即返回 `run_id`。 |
| `run_script` | 启动 **agent 当场手写的临时脚本**（即 meta 层）。 |
| `status` | 轮询某个 run 并取回结果（过大的结果会 offload 到 handle 后面）。 |
| `resume` | 针对同一 journal 重跑，已完成的叶子以零成本重放。 |
| `cancel` | 中止一个在跑的 run。 |

run 在后台执行，完成通知会在 host 下一回合前注入，因此启动从不阻塞对话。

**meta 层（`run_script`）。** host 写一段 `async def orchestrate(ctx, args)` 并提交源码。源码先过一道 **AST 安全 gate**（禁 import、dunder 访问、禁用 builtins、`str.format` 注入），再在受限 builtins 命名空间下经一次收敛的 `exec` 执行。被拒的脚本会带回**具体违规**，host 据此改正重提。build-time 代码可用 `run_workflow_from_source(source, roster=...)` 编程式做同一件事。

> **安全边界。** gate 加受限命名空间只挡"好心模型手滑"——它**不是安全沙箱**，对抗性脚本仍可能逃逸。只提交 host 自己写的脚本；对抗输入请把引擎跑在进程外隔离 backend 后面。

## 断点续跑、预算与可观测性

`run_workflow` 的几个 keyword-only 旋钮可自由组合：

- **`journal=`** —— 跨多次调用传入**同一个** journal 即可命中缓存续跑：已完成的叶子以零模型成本重放，确定性 guard 同时校验调用序列未分叉。
- **`budget=`** —— 所有叶子共享的 token 上限；耗尽后下一次 `agent()` 抛 `WorkflowBudgetExceededError`，驱动 `while ctx.budget.remaining() > T` 范式。
- **`on_span=`** —— 接收每个 `agent` / `parallel` / `pipeline` 调用 span 的 sink；续跑时会重发被标记 `cached=True` 的 span。
- **`sandbox_manager=`** —— 为需要执行环境的叶子各租一个隔离 backend；纯推理叶子不分配。

## 示例

[`examples/`](examples/) 下每个示例都**离线、无需 API key**（确定性 fake model）即可运行。若要通过 OpenRouter 驱动真实叶子并用 LangSmith 看 trace，先 `uv sync --group example` 装上示例附加依赖，在本地 `.env` 里配好 `OPENROUTER_API_KEY` 与 `LANGSMITH_*`，再设 `LDW_DEMO_REAL_MODEL`（默认 `anthropic/claude-opus-4.8`，设成任意 OpenRouter slug 即可覆盖）。

| 示例 | 演示内容 |
|---|---|
| [`01_single_agent`](examples/01_single_agent.py) | 单个叶子 `agent()` 调用，端到端。 |
| [`02_fanout`](examples/02_fanout.py) | `parallel()` barrier 扇出与失败叶子过滤。 |
| [`03_loop_until_budget`](examples/03_loop_until_budget.py) | 由 `ctx.budget.remaining()` 驱动的精化循环。 |
| [`04_sandbox_artifacts`](examples/04_sandbox_artifacts.py) | per-leaf sandbox 隔离 + `/shared/` 产物交接。 |
| [`05_host_agent_workflow`](examples/05_host_agent_workflow.py) | host agent 经 `workflow` 工具驱动一条 workflow。 |
| [`06_capstone`](examples/06_capstone.py) | 旗舰：`parallel` 研究 → `pipeline` 精化 → 对抗式验证 → 综述。 |
| [`07_deep_research_real_e2e`](examples/07_deep_research_real_e2e.py) | 真实 OpenRouter host 启动已注册的 `deep_research` workflow。 |
| [`08_meta_layer_run_script`](examples/08_meta_layer_run_script.py) | meta 层：host **当场手写**脚本并经 `run_script` 提交。 |

```bash
uv run python examples/06_capstone.py

# 全真端到端（真实 OpenRouter host + leaves）：
LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8 uv run python examples/07_deep_research_real_e2e.py
```

## 公开 API

稳定的公开面从包根导出，自 `0.1.0` 起遵循语义化版本：

- **库 core** —— `run_workflow`：开发者 / build-time 入口。
- **meta 层** —— `compile_workflow_source` / `run_workflow_from_source` / `extract_meta`：把 LLM 当场写的源码经 AST gate 编译并运行。
- **注册表** —— `Roster` / `RosterEntry`、`WorkflowRegistry`。
- **host 面** —— `create_workflow_tool`、`create_workflow_middleware`、`skills_path` / `skill_files`。
- **原语** —— 挂在传给脚本的 `Ctx` 上：`agent` / `parallel` / `pipeline` / `race` / `phase` / `log` / `budget` / `workflow`。`ctx.race(candidates, *, win, win_tag="")` 把若干 `RaceCandidate` 并发跑起，返回第一个令 `win` 为真者的 `RaceResult`，其余 cancel；决策内容哈希 journal（`win_tag` 折进 key），故 resume 复现胜者且零派发。
- **跨叶归约** —— 折叠 `parallel` / `pipeline` 返回的结果列表的纯函数：`survives`（refute-by-default 投票）、`dedup`、`reconcile`（双盲复核调解）、`corroborate`（跨叶相互印证），外加 `ReviewItem` / `Reconciled` / `Consensus` 结果类型。同时注入 `run_script` 命名空间——host 当场写的脚本无需 import 即可按名调用。
- **race 值类型** —— `RaceCandidate`（一份可内容哈希的 agent 调用规格，镜像 `agent()` 调用）与 `RaceResult`（胜者、其下标、`.won`）。两者同样注入 `run_script` 命名空间——host 当场写的脚本无需 import 即可按名构造和读取。
- **类型与异常** —— `Budget`、`JournalStore` / `InMemoryJournalStore` / `JournalRecord` / `race_key`、`SandboxManager`、`Span` / `SpanKind` / `SpanSink`、`BgRunManager` 家族，以及 `Workflow*Error` 系列异常（含 `WorkflowScriptError`）。

公开签名稳定；新增参数一律 keyword-only 带默认值。以 `_` 开头的模块和成员属于内部实现，可能随时变动。

## 开发

```bash
uv sync                 # 安装依赖 + 创建 .venv
uv run pytest           # 跑测试（覆盖率门，行覆盖 >= 85%）
uv run ruff check .     # lint
uv run ruff format .    # 格式化
uv run pyright          # 类型检查（strict）
uv run lint-imports     # 校验 Layer 0/1/2 架构边界
```

## 状态

**0.1.0 —— 架构已锁定、公开 API 已稳定、尚未发布到 PyPI。** 三层全部落地，含 Layer 2 meta 层。发布日志见 [`CHANGELOG.md`](CHANGELOG.md)。

## 许可证

[MIT](LICENSE)
