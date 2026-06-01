## 测试规范

### 作用域

每当对源码进行修改时，都需要关注和思考对已有测试用例的维护——包括新增、更新或删除测试，确保测试与源码保持一致。

### 规则

- 每次写完代码后，涉及的模块都要执行 ruff 和 pyright 确保格式和静态检查都没问题
- 写测试时总是使用 pytest 和 pytest-asyncio 构建测试用例和实现相关测试逻辑，可适当搭配其他库（如 `unittest.mock`）
- 设计的测试用例**必须**是有实际意义的、能够指导和帮助进行架构反腐和回归验证的；**禁止写 filler 凑数**
- **复用优先**：优先复用项目内的公共测试模块和脚手架（放在 `tests/` 下的共享 fixtures / mocks / factories）
- **测试基础设施归属**：新增 mock、patch、fixtures、factories 时，思考其通用性并放置到合适层级——通用的放入 `tests/fixtures/` 或 `tests/mocks/`，仅单文件使用的就近定义
- **测试层级与源码性质对应**：
  - 纯逻辑 / 数据模型层 → 单元测试即可
  - 编排运行时 / 跨组件流程（agent / parallel / pipeline / resume 等）→ 单元测试 + 集成测试
- **运行测试**：使用 `uv run pytest`（`asyncio_mode = "auto"` 已在 `pyproject.toml` 配置）

### 测试输出重定向到本地日志文件

任何测试运行（`uv run pytest` 等）应将完整 stdout+stderr 写入 `/tmp/` 下的日志文件，以便查看完整 summary、失败详情和 stack trace 而无需重跑。不要 live pipe 到 `head` / `grep`（会截断 summary 行、吞掉 traceback）。范式：

```bash
uv run pytest -q > /tmp/ldw-test.log 2>&1; echo "EXIT=$?"
tail -30 /tmp/ldw-test.log
grep -E "FAILED|ERROR" /tmp/ldw-test.log
```

### 测试覆盖率标准

- 以**有实际意义**的测试达成合理覆盖（架构反腐 + 回归验证），禁止 filler 凑数
- 核心编排流程（`agent` / `parallel` / `pipeline` / `resume`）必须至少有集成测试覆盖，不能只靠单元测试
