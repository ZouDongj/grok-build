# zgrok 开发工作流

zgrok = xAI grok-build TUI（前端 100% 原生）+ ZCode 内核（0.16.9+，app-server 协议）。
本文件是"在 zgrok 里开发 zgrok"的守则——你脚下的地板就是被改造的对象。

## 仓库与关键路径

- 仓库：`/data/zdj/ZCode/grok-build`（上游 f2c7c39 + 鲲鹏移植 + 全部 zgrok 提交）
- agent（我们写的桥接层）：`crates/codegen/xai-zcode-agent/`（agent.rs 是核心）
- TUI（上游对齐，原则上不改）：`crates/codegen/xai-grok-pager/`
- bin 组合根：`crates/codegen/xai-grok-pager-bin/`（`-p xai-grok-pager` 只编译 lib，装 bin 必须用它）
- 内核：`~/.local/opt/zcode/<ver>/`（launcher 按 sort -V 选最高）；3.12.1 保留可回退

## 黄金法则

1. **重装一律用 `grok-install`**（原子 rename；`cat > ~/.local/bin/grok` 原地截断会让运行中实例 SIGBUS）。
   回滚：`mv ~/.local/bin/grok.bak ~/.local/bin/grok`。运行中的实例持有旧 inode，重启才换新。
2. **重启 zgrok = 内核死 + 会话断，但 transcript 在内核库里永生** → 重启后 resume 即接续。
   重启前三件事：git commit、`verify_0169` 绿、grok-install 完成。
3. **迭代闭环（不打断当前会话）**：改码 → `cargo build -p xai-zcode-agent --example verify_0169 && ./target/debug/examples/verify_0169`
   （16 项回归：turn/plan 审批/模式切换/ack/插话/mcp/plugins/skills/图片/resume/删除）→ grok-install →
   另一个 tmux 窗口起测试实例（cwd 用 /tmp 草稿）→ 全绿才重启主实例。
4. **禁止**：会话里 pkill grok / 杀内核；跑任何登录或凭据命令（凭据状态是验证过的稳定态）；
   未经确认删内核库数据。测试会话固定用 /tmp 草稿目录，测完删。
5. release 链接慢（增量 ~2min），日常回归用 debug example（~1min）。多会话同时 cargo 会等锁，别并行构建。

## 已知特性

- 大上下文轮次首字延迟 1-3 分钟是 GLM prefill 的真实耗时（进度指示一直在，不是挂死）。
- Ctrl+. 在多数 SSH 终端不可达（非 ASCII 控制字符），用 Ctrl+X；命令面板 Ctrl+P；会话列表 Ctrl+\。
- GLM-5.3 主模型不支持图像输入（官方能力表），贴图请切 Flash。
- 每内核启动会实时拉 V4 签名门控；凭据解密失败会以"身份验证失败"形式出现在握手层
  （agent 的 decrypt_credential 负责 enc:v1 AES-GCM 解封）。

## 排查手册

- agent 调试日志：`/tmp/zcode-agent-debug.log`（prompt/interject/ext 调用全记录）
- 内核统一日志：`~/.zcode/cli/log/zcode-YYYY-MM-DD.jsonl`（UTC 时间戳；client_signing / turn.failed /
  mcp.server.* / session/send accepted{attachmentCount} 都在这里）
- 协议线：ZCODE_WIRE_LOG=1 启动 grok（reader 线程 tee）
- 会话存储：`~/.zcode/cli/db/db.sqlite`（删除走 zgrok picker 的 `d` 键或 rusqlite 事务）

## 生态桥接现状（Extensions 浮层）

| 标签 | 状态 |
|---|---|
| MCP Servers | ✅ x.ai/mcp/list → mcp/list(mode:status) |
| Plugins | ✅ 列表 + enable/disable → plugins/setEnabled |
| Skills | ✅ x.ai/skills/list → skills/referenceCatalog |
| Marketplace | ⬜ 可桥 plugins/marketplace/add + plugins/overview |
| Hooks | ⬜ grok-shell 专属体系，需评估内核 workspace/hooks 对应关系 |
| Workflows | ⬜ 内核有 workflows/* 全套方法 |

模型侧（技能调用、斜杠直呼、MCP 工具、图片输入）全部原生可用——管理面只是 UI。
