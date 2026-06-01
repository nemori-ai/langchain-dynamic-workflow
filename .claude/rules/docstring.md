## Docstring 约定

### 作用域

编写或修改 Python 代码中的 docstring、inline comment、`Field(description=...)` 时适用。

### 规则

- **语言**：英文，Google 风格
- **禁止出现研发进度说明**：docstring 任何位置不得包含 TODO、roadmap、"will be implemented"、当前开发状态等进度描述
- **禁止引用设计文档和 ADR**：docstring 不得引用或链接任何设计文档、ADR 文档或内部规划文件
- **字段文档不重复**：每个类选择且仅选择一种字段文档策略（class `Attributes:` 或 `Field(description=...)`），不得同时在两处描述同一字段
- **枚举值同步**：增删改枚举值时必须同步更新 docstring
- **跨引用**：若使用文档站交叉引用，采用 mkdocstrings Markdown 链接语法，禁止 Sphinx roles
