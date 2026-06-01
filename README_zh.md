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

## 状态

**早期阶段 —— 架构已锁定，公开 API 正在构建中。** 尚未发布到 PyPI。

## 开发

```bash
uv sync                 # 安装依赖 + 创建 .venv
uv run pytest           # 跑测试
uv run ruff check .     # lint
uv run ruff format .    # 格式化
uv run pyright          # 类型检查（strict）
```

Python 3.12+。依赖管理用 [uv](https://docs.astral.sh/uv/)。

## 许可证

[MIT](LICENSE)
