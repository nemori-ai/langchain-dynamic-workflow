# langchain-dynamic-workflow

[English](README.md) | **中文**

> 面向 LangChain [`deepagents`](https://github.com/langchain-ai/deepagents) 的确定性、脚本化、可断点续跑的多 agent 编排引擎 —— Claude Code **Dynamic Workflows** 的社区复刻。

## 是什么

普通 agent **逐回合**决定自己的控制流 —— 每一次循环、分支、扇出都活在模型的上下文窗口里，既烧 token 又不断堆积中间状态。`langchain-dynamic-workflow` 把这件事**反转**过来：由一段确定性的编排**脚本**掌控控制流，只有叶子 `agent()` 调用才委派给 deepagent —— 每个 deepagent 跑在一个隔离的、用完即弃的上下文里，因此**只有最终结果回到调用方的上下文**。

| | 普通 agent | Dynamic workflow |
|---|---|---|
| 谁决定下一步 | LLM，逐回合 | **脚本**（确定性代码） |
| 中间结果存哪 | 模型的上下文窗口 | **脚本变量里** |
| 什么进入调用方上下文 | 整条执行轨迹 | **只有最终结果** |

## 为什么

引擎建在 LangGraph 的持久化执行（`@entrypoint` + `@task`）之上 —— 后者已自带断点续跑 / 重放 / 命中缓存跳过。在此之上，本库补齐了把 deepagents 技术栈变成一个**可脚本化、可扇出、可续跑**的编排运行时所需的部件；并可选地提供一个 meta 层：由 LLM 为你描述的任务当场写出编排脚本。

## 架构（三层）

- **Layer 0 —— 底座**：LangGraph 持久化执行（`@entrypoint` + `@task` + checkpointer）。
- **Layer 1 —— 编排运行时**：核心原语 —— `agent()`、`parallel()`（barrier）、`pipeline()`（无 barrier）、`phase()`、`log()`、`budget`、`workflow()` —— 外加一个内容哈希 journal 和一个 fail-loud 确定性 guard。
- **Layer 2 —— meta 层**：由 LLM 编写 Python 编排脚本；执行前先经 AST gate 校验。

叶子 `agent()` 调用从一个**命名注册表（roster）**中解析出对应的 deepagent，并将其作为 `@task` 调起，复用 deepagents 的上下文隔离（context quarantine）与 sandbox backend。

完整设计依据见 [`docs/plans/`](docs/plans/)（设计基线；gitignored，不进版本控制）。

## 快速上手

```bash
uv sync   # 安装依赖 + 创建 .venv
```

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

跨多次调用传入**同一个** `journal=` 即可获得命中缓存的断点续跑（已完成的叶子以零模型成本重放）；`budget=` 设共享 token 上限；`on_span=` 接一个可观测性 trace。想让 **host agent** 在后台驱动 workflow，就把 `create_workflow_middleware(roster, workflows=...)` 挂到一个 host `create_deep_agent` 上 —— agent 通过单个 `workflow` 工具按名启动已注册 workflow（`run`）或**当场手写一段临时脚本提交**（`run_script` —— meta 层），再 `status` 轮询、`resume`、`cancel`，并在 run 完成时收到通知。

`run_script` 即 meta 层：agent 写一段 `async def orchestrate(ctx, args)` 并提交源码，源码先过一道 **AST 安全 gate**、在受限 builtins 命名空间下执行。这道 gate 只挡"好心手滑"，**不是安全沙箱**——对抗性脚本仍可能逃逸，所以**只提交 agent 自己写的脚本**（对抗输入请把引擎跑在进程外隔离 backend 后面）。build-time 代码可用 `run_workflow_from_source(source, roster=...)` 编程式做同一件事。

[`examples/`](examples/) 下的每个示例都**离线、无需 API key**（用 fake model）即可运行；若要通过 OpenRouter 驱动真实叶子并用 LangSmith 看 trace，先用 `uv sync --group example` 装上示例附加依赖，在本地 `.env` 里配好 `OPENROUTER_API_KEY` 与 `LANGSMITH_*`，再设 `LDW_DEMO_REAL_MODEL`（模型默认 `anthropic/claude-opus-4.8`，设成任意 OpenRouter slug 即可覆盖）。旗舰示例是 [`examples/06_capstone.py`](examples/06_capstone.py)：host agent 在后台驱动一条 `parallel` 研究 → `pipeline` 提炼 → 对抗式验证 → 综述的 workflow。若要全真运行，[`examples/07_deep_research_real_e2e.py`](examples/07_deep_research_real_e2e.py) 让一个**真实的 OpenRouter host agent**自行决定启动已注册的 `deep_research` workflow（search → extract → 对抗式验证 → 综述），端到端跑通。

```bash
uv run python examples/06_capstone.py

# 全真端到端（真实 OpenRouter host + leaves）：
LDW_DEMO_REAL_MODEL=anthropic/claude-opus-4.8 uv run python examples/07_deep_research_real_e2e.py
```

## 公开 API

稳定的公开面从包根导出，自 `0.1.0` 起遵循语义化版本：

- **库 core**：`run_workflow` —— 开发者 / build-time 入口。
- **meta 层**：`compile_workflow_source` / `run_workflow_from_source` / `extract_meta` —— 把 LLM 当场写的源码经 AST gate 编译并运行。
- **注册表**：`Roster` / `RosterEntry`、`WorkflowRegistry`。
- **host 面**：`create_workflow_tool`、`create_workflow_middleware`、`skills_path` / `skill_files`。
- **原语**：挂在传给脚本的 `Ctx` 上 —— `agent` / `parallel` / `pipeline` / `phase` / `log` / `budget` / `workflow`。
- **类型与异常**：`Budget`、`JournalStore` / `InMemoryJournalStore` / `JournalRecord`、`SandboxManager`、`Span` / `SpanKind` / `SpanSink`、`BgRunManager` 家族，以及 `Workflow*Error` 系列异常（含 `WorkflowScriptError`）。

公开签名稳定；新增参数一律 keyword-only 带默认值。以 `_` 开头的模块和成员属于内部实现，可能随时变动。

## 状态

**v0.1.0 —— 架构已锁定，公开 API 已稳定。** 尚未发布到 PyPI。详见 [`CHANGELOG.md`](CHANGELOG.md)。

## 开发

```bash
uv sync                 # 安装依赖 + 创建 .venv
uv run pytest           # 跑测试（带覆盖率门，行覆盖 >= 85%）
uv run ruff check .     # lint
uv run ruff format .    # 格式化
uv run pyright          # 类型检查（strict）
uv run lint-imports     # 校验 Layer 0/1/2 架构边界
```

Python 3.12+。依赖管理用 [uv](https://docs.astral.sh/uv/)。

## 许可证

[MIT](LICENSE)
