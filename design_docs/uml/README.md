# UML

`langchain-dynamic-workflow` 的 UML 视图。以"**消费者是 AI agent、唯一动作是 tool call**"为中心。

| 文档 | 视图 | 看点 |
|---|---|---|
| [01-component.md](01-component.md) | 组件图 | agent 运行时边界、middleware 交付载体、引擎对 agent 不可见的三层 |
| [02-class.md](02-class.md) | 类图 | 公共面 / tool / middleware+后台机制 / 引擎核心 / 底座 的类与关系 |
| [03-sequence.md](03-sequence.md) | 时序图 | async run→notify 闭环、resume 重放 |

机制详解见 [../01-engine-mechanism.md](../01-engine-mechanism.md);架构与对外形态见 [../02-architecture.md](../02-architecture.md)。

> 图用 Mermaid 表达,GitHub / mkdocs / 多数 Markdown 查看器可直接渲染。
