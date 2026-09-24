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

## /compact 的真实语义（2026-09-23 修复）

内核 `session/compact` 是**异步受理**：~300ms 返回 `{state:"accepted"}`，真正的压缩
作为后台 prompt 回合（`turn.started` 带 `input:"/compact"`、model-only 可见性）跑几秒
到几分钟；期间 `session/send` 被拒 `-32010 "A prompt is already running"`。zgrok 侧：
ext 等待压缩回合真实完成才回 Ok；压缩中的 prompt 排队为 continuation，压缩结束自动
发出；send 的早期拒绝显式报错（不再吞掉让压缩回合冒充回答）。识别依据 turnId
（pump 已把 params.turnId 提升进事件 payload）。

## v4 命令面（开源客户端揭示，zgrok 已接入 steer）

开源客户端（/data/zdj/ZCode/zcode-oss，3.14.0，clone 需走本机 7890 代理）揭示了
与旧 session/* 并存的 v4 协议：`v4/command` 统一入口，30 个命令。**发行版 3.14.1
内核已实现**（probe_v4 实测 guide 转向落地）。已接入：
- `sendText(requestedDelivery:"guide")` → x.ai/interject 回合中真转向（不打断注入）
- `sendText(requestedDelivery:"startNow")` → meta.sendNow 原子抢占
两者均带回退（老内核 → 回合边界队列 / stop+retry）。新增 x.ai/v4/command 透传。
已接入（v4 订阅模型落地后）：
- `editUserQuery` 深回滚已实现——rewind = fork 在目标前一回合的 assistant 行（fork 实测
  只含之前历史）+ kernel_id 换轨 + pager 自截断视图；目标 0 = 新建空会话
- 队列五操作（x.ai/queue/{edit,remove,reorder,clear,interject} 通知）→ v4 队列命令，
  队列广播携带内核 queueItemId
- `setFollowupMode(queue|guide)`（x.ai/session/set_followup_mode）
- 内核自发压缩的 auto_compact_started/completed 横幅
- v4 订阅模型：帧解包 + snapshot/delta 折叠（含 row.removed 截断）+ kernel_id 事件重映射
- 内核怪癖：**压缩回合运行中 v4 sendText(queue) 会卡死队列排水**——压缩路径保持本地
  continuation 槽（已实证并记录）
待接入（v4 均已可达）：
- `editUserQuery`（截断式 rewind；CAS 需 baseRevision+baseLogEpoch+rowTarget，
  需先接 v4/conversation/subscribe 的 rows/revision 模型）
- `setFollowupMode(queue|guide)`（会话级转向默认）
- 队列项操作（edit/reorder/delete/sendQueuedNow/setAutoDrain）
MCP 配置写：v4 无独立命令（仅 createSession 的 mcpServers 启动期配置）——仍走
settings 文件。v4 订阅模型（conversation/subscribe rows）是后续深度整合的主线。

## 插话 / 强插 / 队列的真实语义（011e44c）

- **强插（send now）**：pager 发普通 PromptRequest + `meta.sendNow=true`，期望 agent
  先取消运行中的回合再发。agent 现在走 session/stop → 重试 send 直到内核放行
  （-32010 拒绝期间 250ms 间隔重试，25s 上限）→ 旧回合按 Cancelled 结算。
  正常路径遇到 "already running" 拒绝也有同款停旧重发兜底（drain 竞态时可能撞上）。
- **队列广播**：pager 的本地队列排水被 `server_queue_owns_next_turn` 门闩卡住——
  服务端队列里有非运行行就永不排水。官方 shell 靠 `x.ai/queue/changed` 全量快照对账，
  agent 现在在每个队列转换点广播（排队 continuation 为一行、接受/投递为空快照+running）。
- **插话投递广播**：`x.ai/session/interjection`（带 interjectionId）让 pager 认领
  乐观回显块。空闲插话改 fire-and-forget（send RPC 的 Ok 在回合结束才回来，
  call() 的 30s 超时会误报）。
- **真·回合中转向（steer）是内核运行时层私有能力**（steerTurn delivery guide/queue），
  app-server RPC 无条件拒绝并发 send——RPC 面上只能做到回合边界投递。
- turn_epoch 守卫：陈旧的 turn.failed 宽限期不再误杀新回合。

## 夜班对齐记录（resume 子代理修复 + 三项对齐）

- **resume 后首条消息误报 "Ran 1 subagent 1 failed"**：session/subagents 返回历史子代理，
  轮询器误当新增播报。load_session 现预置历史 id 为已播报状态。
- **TodoWrite 清单**：db todo 表 → ACP 原生 SessionUpdate::Plan（回合/压缩完成后推送）。
  注意 x.ai/session_notification 的 XaiSessionUpdate 枚举没有 plan 变体，必须走 ACP 原生通道。
- **会话删除**：优先 v4 deleteSession（内核侧清理），db 手术兜底。
- **计量对表**：v4/conversation/usage 与 SQL 逐字段一致（x.ai/v4/usage ext 可直查）；
  唯一口径差：cacheRead（v4 查询返 0，db 行有值）。
- **斜杠命令**：纯文本 /goal 经 session/send 被内核解析（goal 语义生效）。
- **视频**：ACP 通道只有 Text/Image 块，TUI 结构性不支持（内核侧就绪）。
- **v4 createSession**：开源注释明说其 binder 调旧 create op——无迁移紧迫性。

## goal 模式 / 后台任务 / 重放思考时长（最新）

- **重放 "Thought for 0.0s" 已修**：重放思考块带 meta.agentTimestampMs（账本
  created→completed，字符安全中点拆分），pager 计算真实时长。教训：拆分中文文本
  必须 is_char_boundary；/goal 之类斜杠命令 session/send 不解析（会当纯文本喂给
  模型），已由 agent 拦截路由到 sessionGoal RPC。
- **goal 面板**：v4 投影 goal 字段 → goal_updated 通知（active/user_paused/
  complete/blocked 映射）；/goal 文本命令 + x.ai/session/goal ext 均可控制。
- **后台任务**：v4 投影 backgroundWorks（bash 类）→ background_tasks 通知；
  x.ai/task/kill → v4 cancelBackgroundWork。

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
| Marketplace | ✅ 列表 + 安装/卸载/更新动作（plugins/marketplace/add 等） |
| Hooks | ✅ 空列表（grok-shell 专属体系，内核无对应；标签正常打开） |
| Workflows | ✅ x.ai/workflows/list → workflows/list（project scope） |

模型侧（技能调用、斜杠直呼、MCP 工具、图片输入）全部原生可用——管理面只是 UI。

其他已接线：auth/check_subscription、billing（订阅轮询静默化）、session/info、
prompt_history（上箭头召回，读内核 input_history 表）、session/usage（turn_usage 聚合）、
compact_conversation（/compact → session/compact）、session/fork、interject、
plugins enable/disable/install/uninstall/update。

## 子代理可视化的真实通道（edca112）

ZCode 客户端看子代理**不走 RPC**——子会话的 session/subscribe、session/events、
session/messages 一律被内核拒绝（`Session is not active`，verify_0169 有固化断言）。
官方通道有两条，zgrok 现已都用：

1. **原生 `subagent.lifecycle` 事件**：跟着父会话流下发（phase spawned/stopped，
   带 agentId/childSessionId/agentType/status），handle_event 直接消费；
   session/subagents 轮询降级为兜底（只负责 progress 心跳）。
2. **转录文件**：`~/.zcode/cli/agents/<父内核会话>/agent_<id>/`（客户端的
   subagentTranscripts 存储类就是这个前缀）。metadata.json 有状态/令牌/时长，
   output.txt 按字节游标增量 tail 出内容；SQL 读消息账本仅作无文件内核的 fallback。

教训：下"内核没暴露"结论前，先反查官方客户端（asar 的 scheduler/host 包）用的是什么。

## 内核不开门清单（剩两项）

| 功能 | 现状 | 出路 |
|---|---|---|
| rewind 执行 | fork 保留全账本，无截断式回滚 | 等内核方法或做账本手术 |
| MCP 配置写 | 只认 settings 文件 | 写文件 + 重启提示 |
