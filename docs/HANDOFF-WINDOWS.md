# Windows 适配交接（给 Windows 机器上的 AI 会话）

## 目标（唯一目标）

让 zgrok 在 Windows 上**构建成功并运行正常**，行为与 Linux 构建一致。
这是一次移植验证，**不是优化任务**。禁止重写任何已对齐的协议逻辑。

## 开工前必读（按顺序）

1. `docs/ZGROK_WORKFLOW.md` —— 内核协议知识、v4 命令面、全部实证结论。
   修复时遇到"为什么这么写"的疑问，答案几乎都在里面。
2. 本文件余下部分。

## 架构事实（不要推翻，只需围绕它修）

- pager（TUI）与 zcode agent **同进程**：`crates/codegen/xai-grok-pager/src/acp/spawn.rs`
  的 `ZcodeAgent::new(gateway, kernel_bin)` 直接构造，无 IPC 边界。
- 内核是独立子进程：`zcode app-server`（stdin/stdout JSON-RPC），
  `crates/codegen/xai-zcode-agent/src/kernel.rs` 的 `Kernel::spawn`。
- 后端选择：环境变量 `GROK_BACKEND`（默认 zcode 后端；`grok`/`shell` 切回原生）；
  内核二进制 `ZCODE_BIN`（默认 PATH 上的 `zcode`）。
- 认证/内核数据：`%USERPROFILE%\.zcode`（与官方 Windows 客户端同布局）。
  **凭据只读，禁止打印任何凭据内容。**
- 平台适配已做一轮（commit d40ab9b）：`zcode_home()`（HOME→USERPROFILE 回退，
  agent.rs 18 处调用点）、日志走 `std::env::temp_dir()`。

## 构建步骤

```
1. 安装 Rust（MSVC 工具链）+ git
2. git clone -b zgrok https://github.com/ZouDongj/grok-build.git
3. 安装 Windows 版 zcode CLI，登录一次 coding plan（zcode 在 PATH，或设 ZCODE_BIN）
4. cd grok-build && cargo build --release
5. 在 Windows Terminal 里进入任意项目目录运行 target\release\grok.exe
```

## 预期错误区域与修法

| 区域 | 修法 |
|---|---|
| 我们 crate 里残留的 Unix 假设（路径拼接、信号、权限位） | 用 `cfg(windows)` 分支修，**不要改 Linux 行为** |
| 上游 cfg(windows) 缺口（本 fork 从未在 Windows 编过） | 最小修复，cfg 隔离 |
| `grok wrap` PTY 功能 | 允许直接 cfg 掉（非核心） |
| 字体/终端问题 | 建议用户用 Windows Terminal，不要改渲染逻辑 |

## 红线

1. **禁止**重写/重构 agent.rs、kernel.rs 的协议逻辑（v4 命令、事件映射、
   replay、CAS 流程）——它们在 Linux 上有 48 项回归实证（`examples/verify_0169.rs`）。
2. 平台修复必须 cfg 隔离，保证 Linux 构建不回归（这边会复验）。
3. 每轮修改后跑：`cargo build --release`，能跑则再跑
   `cargo run -p xai-zcode-agent --example verify_0169`（真实内核全链路 48 项，
   会消耗额度，产出测试会话后按 ZGROK_WORKFLOW.md 的清理模式删）。
4. 修不动的问题：记录现象（编译错误全文/行为差异）推到 zgrok 分支，
   由 Linux 侧（源头环境）继续排查。

## 完成标准

- `cargo build --release` 零错误；
- grok.exe 在 Windows Terminal 可正常对话（含中文输入、模型切换、图片粘贴）；
- verify_0169 在 Windows 上跑通（或至少明确列出未过项及原因）。
