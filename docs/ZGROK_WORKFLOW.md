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
- **0.0s 二次修复（bc866bd，第一次修得不彻底）**：pager 冻结时长的真公式是
  `末块 agentTimestampMs - streamStartMs`，且 **meta 必须带 isReplay:true** 才会
  禁用本地计时块——只盖 agentTimestampMs 时本地计时器（started_at=now）仍抢跑，
  两块微秒级到达 → finish() 冻结 ~0ms 盖掉服务端值。正确做法：按 part 级
  `time.start/end`（纯思考跨度，session/resume 的 parts 里就有）逐 part 重放，
  每块盖齐三个 meta 字段；多段思考（思考→工具→思考）按 streamStart 分块。
  教训：**回归检查必须复刻消费方的公式**，只验证"字段存在"不等于"渲染正确"。
- **goal 面板**：v4 投影 goal 字段 → goal_updated 通知（active/user_paused/
  complete/blocked 映射）；/goal 文本命令 + x.ai/session/goal ext 均可控制。
  入口可见性（5cee23d）：zcode agent 必须**广告** /goal —— initialize 的
  `meta.availableCommands` 引导 + new/load_session 后 400ms 延迟推
  AvailableCommandsUpdate（早于会话面板注册的会被 pager 丢弃）；AcpSlashCommand
  走 PassThrough，即 `/goal <args>` 原样进 prompt 被拦截。设置 goal 后按 `g`
  键打开 goal 详情面板。
- **后台任务**：v4 投影 backgroundWorks（bash 类）→ background_tasks 通知；
  x.ai/task/kill → v4 cancelBackgroundWork。
- **goal 状态深度映射（d0516b2）**：内核 goal schema 有独立 `verifying` 状态、
  `verifications[]`（每轮 outcome/reason/nextAction）、`iterations[]`。映射：
  verifying→verifying_completion 覆盖层；verifications.len→verify 轮数；
  iterations.len→worker 轮数（面板 Rounds 行）；末轮 outcome→verdict 徽章
  （achieved/not_achieved）；notSatisfied/failed→blocked+pause_message（验证
  原因，Reason 块）。端到端实证（probe_goal_e2e）：设定目标→内核自主创建
  文件并自验→active→verifying→complete(verify=1, achieved)。
- **goal token 统计（a0f30cf）**：内核 v4 不按 goal 分账，但 pager 的 goal
  token 行本就是 `当前上下文 - token_baseline`（grok 原生 shell 同款机制）——
  设 goal 时快照基线（新会话无观测则 0），终态冻结
  `tokens_used = 最终usage - 基线`。实证：complete 冻结 19128。注意
  GoalUpdated 的 `token_baseline` 是裸 i64，发 null 会整条解析失败被丢弃。
  值为 0 时 pager 隐藏 token 段（状态行 + 弹窗），不显示假 0。

## 上下文仪表实时刷新（compact 后立即回落，官方一致）

- **官方机制**（OSS `zcode-protocol-v4/snapshot.ts`）：v4 投影的
  `usage.contextWindow {usedTokens, maxTokens, autoCompactThresholdTokens}`，
  注释明言 "conflation：值未变不下发" —— 内核在值变化时（含 compact 完成的骤降）
  主动推 `state.updated` 增量，桌面右上角实时刷新。
- **zgrok 旧缺陷**：上下文条取 `model_usage` 表最近一次 `main_turn` 的
  input+output —— compact 后没有新 main_turn，条停在压缩前的大数，直到下一条
  用户消息才回落；内核自发 auto-compact 的 banner 还带 `tokens_used: 0`/
  `tokens_after: 0`，会把条刷成 0%。
- **修复**（agent.rs）：`v4_usage` 投影消费（snapshot + state.updated patch）；
  帧分发点 diff 变化即推 ACP `UsageUpdate(used, max)`；`push_context_usage`
  优先投影值（db 账本降级兜底）；banner 带真实 token 数；fork 换 id 后 v4 帧
  按 kernel_id 回查路由（此前 queue/goal/works 对换 id 会话全丢）。
- **实测**（probe_usage）：GLM-5.3 maxTokens=1,000,000；compact 完成 115ms 后
  USAGE 通知 18365→6969 实时到达，无需再发消息。

## 对齐批次：8 项收尾（827bcfd + 6a3fb2c）

- **行定位铁律**：v4 行定位命令（setAssistantFeedback/retryTurn/applyFileRewind）
  的 target 是 **turnHeader 行**（canRewindFiles/canRetry/canFork 只在它身上），
  打 assistant/user 行会得到 `guard.actionUnavailable`。turnHeader 已入投影
  （v4_turn_headers）+ 诊断转储。
- **/rate like|dislike|clear**：setAssistantFeedback（CAS），作用于最后一条
  非中断回复；ack.status 必须判 accepted（rejected 也走 Ok 通道！）。
- **/drain on|off|show**：setAutoDrain（**也是 CAS 命令**，无 token 会被
  proto.invalidPayload 拒）；queue.autoDrain 已入投影。实证：3.14.1 内核 v4 stop
  被 accepted 但**不**置 autoDrain=false（OSS 注释是桌面端模型），stop 后队列
  照常排水。
- **held-queue choice**：队列 held（有项+autoDrain 关）时 v4 sendText 自动带
  `heldQueueDisposition=keepQueueAndSend + expectedHeldQueueItemIds`（TUI 安全
  默认不清队列）；cancel 走 v4 stop（legacy 兜底）。
- **/filerewind [apply]**：文件级回退（不截历史）。预览走
  `v4/conversation/fileRewindPreview` RPC，执行走 applyFileRewind CAS。
  /rewind 点位的 hasFileChanges 徽章从账本真实计算（Write/Edit part 落在轮次
  序列区间内）。
- **内核标题同步**：turn 结束/resume 时把 session.title（generated）写进
  pager 的 summary.json（session_summary + updated_at），resume 列表显示真标题。
- **/retry**：v4 retryTurn 官方通道已接；3.14.1 guard 对 completed 和
  interrupted 轮都不放行（actions=null），失败轮大概率才可用——如实上报。
- **workflows**：v4 workflowRuns.runs 入投影 → workflow_updated 通知
  （pending/running→active 等）；本机 workflows=0，映射按 OSS schema，未经
  实机 e2e。
- 广告的命令：goal/rate/drain/filerewind/retry（initialize meta + 每会话 ACU）。

## Windows 构建与使用（d40ab9b）

- 上游 grok-build 本就跨平台（pager 20 处 cfg(windows)、shell 11 处、Unix 依赖
  全部 cfg 隔离）；zcode 内核 Windows 存在（官方客户端布局
  `%USERPROFILE%\.zcode`，本机内核 db 里就有 Windows 会话的路径证据）。
- agent 侧已适配：`zcode_home()`（HOME→USERPROFILE 回退，18 处调用点）+
  日志走 `std::env::temp_dir()`。
- Windows 步骤：Rust(MSVC)+git → clone `ZouDongj/grok-build` 的 `zgrok` 分支 →
  装 Windows 版 zcode CLI 并登录 coding plan（PATH 有 zcode 或设 ZCODE_BIN）→
  `cargo build --release` → Windows Terminal 里运行。远程仓已推至
  `git@github.com:ZouDongj/grok-build.git`（zgrok 分支，SSH 走
  ssh.github.com:443 过本机代理）。
- 未验证项（如实）：本机 mingw 交叉检查因磁盘不足未完成；我们 fork 的
  cfg(windows) 编译面还没有机器证据；`grok wrap` PTY 功能可能被 cfg 掉。

## 引擎附注（nudge）与 todo 面板（08d0b88 + 45e2756）

- 内核会把 todo 提醒等"引擎附注"存成**独立 user 角色消息**（info.synthetic=true /
  metadata.visibility=model-only；本机库 2692 条）。官方投影 origin=synthetic 折叠
  不显示；我们 replay 曾把它们当用户发言重放（用户看到的"TodoWrite hasn't been
  used"气泡）。修复：replay 按结构标记跳过（勿按文本嗅探）。真实会话只读验证：
  db 25 条 nudge → 重放 0 条，20 条真实消息完整。
- todo 面板链路本来就通（内核 todo 表 → ACP Plan → 面板，回归绿）。看不到列表
  的真实原因：模型没调用 TodoWrite（内核的 nudge 正是在催模型用它）。面板条目
  为空时高度 0 自动隐藏。45e2756：TodoWrite 工具 result 事件即时推 Plan（官方
  同款），不再等回合结束。

## workflows 全链路（41d35e9，实机 e2e 通过）

- **根因**：内核的动态工作流工具集（CreateWorkflow 全家）**fail-closed**——官方
  Host 读 rollout 配置后显式调 `workspace/updateDynamicWorkflowPolicy
  {workspace:{workspacePath,workspaceKey}, enabled:true}`，且只影响**之后**创建的
  会话。不调用的会话模型根本没有这些工具（同 agent 同模型对比实证：官方建的
  AgentENV 会话有全套，我们建的是空的）。
- **对接**：new_session/load_session 在 create/resume 前按会话 cwd 打开闸门。
- **可视化**：v4 workflowRuns.runs → workflow_updated（actors→agents 列表含
  label/phaseName/state；nodes 按 phaseName 分组出 running/complete/failed；
  current phase；节点计数事件行）。e2e：模型用 CreateWorkflow 写了个最小
  dwf（单 actor 问 1+1），面板流 active→running(1 actor, 阶段"提问 1+1")→
  settled→complete，答案返回。
- 用户入口：直接让模型"用 CreateWorkflow 建工作流…"；运行中 `g` 键开 workflows
  视图（runs 非空时）；drafts 在 `<cwd>/.zcode/workflow-drafts/`。
- 未接：startSavedWorkflow（保存的定义启动，workflow_definition 表当前为空，
  用户实际用法是模型动态创建）。

## workflow TUI 界面（5f1c762）

- **状态条（输入框上方）**：有 run 时显示 `⟳ Workflow · 名称 · 阶段 · N
  agents · settled/total phases`，活动时带 spinner；hover 下划线，**鼠标点击或
  g 键**展开 workflows 全屏视图（run 列表/阶段分组/agent roster）。优先显示
  活动 run。
- **agent roster 富化**：v4 run 不带 name/model，从账本补（dwf_run.name、
  dwf_actor.resolved_model 去 provider 前缀、persona_json 的 system 作工作
  内容描述）。description 走 wire→ingest→视图新字段，渲染在 live-activity
  槽位（有 live 子代理活动时优先 live）。
- **actor 进子代理管道**：首次见到 actor 的 sessionId 发 subagent_spawned
  （带 workflow_run_id/persona/model/child_session_id），settled 发
  subagent_finished——点亮 roster、dashboard、workflows 视图的 live map。
- e2e：模型自建"最小问答工作流"，通知流带真实 name/model/desc，
  spawn/finish 事件齐。
- **/workflow 直发命令**：`/workflow <目标>` 在 agent 侧重写为授权提示（不必知道
  工具名）。**授权确认走内核 alwaysAsk 门**：CreateWorkflow 的 interaction 请求
  → ACP RequestPermission → pager 原生确认弹窗，用户批准后才启动（官方同款；
  e2e 实证标题 "Tool CreateWorkflow always requires explicit approval"，9/9 过）。
  重写提示词勿写"不要询问确认"——门是内核级的，措辞误导模型反而多余。

## /quota 套餐额度（最新）

- **数据源**（复刻官方桌面账户页）：`GET bigmodel.cn/api/monitor/usage/quota/limit`
  （`authorization` = **解密后**的 coding-plan api key——v2 凭据库的值是
  `enc:v1:` AES-GCM 密文，直接读原值必 401）；`GET
  zcode.z.ai/api/v1/coding-plan/reset/status`（头：原始 zcode JWT +
  `X-Bigmodel-Authorization` api key + `Bigmodel-Target-Type: PERSONAL`，
  都不加 Bearer 前缀）。
- **用法**：`/quota`（窗口用量/百分比/下次重置（北京时间到分钟）+ 可用重置次数）；
  `/quota reset five_hour|week` 消耗一次重置（POST reset/use 带幂等键）。
- 凭据只在进程内解密使用，不落日志不回显；界面只出数字。
- 旧的 x.ai/billing/check_subscription 桩仍在（pager 兼容），真实数据走 /quota。

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
