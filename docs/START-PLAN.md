# Start Plan（智谱免费 token 桶）使用说明

> 前提：Start Plan 是 bigmodel 账号下的独立额度桶，智谱不定期在其上发放免费
> token。活动结束后该桶的模型会不可用（内核 registry 报 model-not-found），
> 切回 coding plan 即可。

## 切换

`/model` 打开模型选择，目录里同时列出两个桶：

- 裸 id（`GLM-5.3`、`GLM-5.3-Flash`…）→ Coding Plan
- `… (Start)` 标签（`GLM-5.3-Flash@start`、`GLM-5.2@start`、`GLM-5-Turbo@start`、
  `GLM-5.3@start`）→ Start Plan

选 `@start` 条目即完成三件事：偏好落盘（`~/.config/zcode-tui/model.json`）、
账号快照重推（把 bigmodel family 的 `current` 翻到 start-plan——内核 registry
每 family 只保留一个活跃 plan）、`session/setModel` 切换。选回裸 id 同理翻回。
新会话按落盘偏好启动。

## 鉴权（内核交互的要点）

- 请求时内核会发 `interaction/requestProviderRuntimeHeaders` 向宿主要鉴权材料。
  Start Plan 的正确回包是 `requestAuth: {apiKey: <zcode JWT>}`——JWT 走
  anthropic `x-api-key`，与官方桌面端 `resolveCurrent(planKind=start-plan)`
  完全一致。**不能只回 headers**：vercel 适配器在无 apiKey 时直接抛
  `AI_LoadAPIKeyError`，请求发不出去。
- Coding Plan 维持原路径（`requestAuth: {apiKey: <coding-plan API key>}`）。
- 两个 plan 互斥：registry 每 family 只认 `current` 那个 provider，切换必须
  先翻 `current` 再 setModel，否则 model-not-found。

## 推理档位

setModel 要求档位必须落在目标模型的 `reasoningLevel.values` 里：
`GLM-5.3/GLM-5.3-Flash/GLM-5.2` 是 `low/high/max`（GLM-5.2 另有 disabled），
`GLM-5-Turbo` 是 `disabled/enabled`。`normalize_effort` 会把偏好档位映射到
模型支持的值（不支持时取模型最强档），无需手改。

## 验证

```
cargo run -p xai-zcode-agent --example probe_start_plan
```

六个检查：目录含 @start 条目 / coding 回合 / 切换生效 / Flash@start 真实回合 /
GLM-5.3@start 真实回合 / 偏好恢复。2026-10-25 全绿（免费 token 当期有效）。

## 已知边界

- Start Plan 的用量不在 bigmodel.cn monitor API 里，`/quota` 徽章仍显示
  Coding Plan 的额度；Start Plan 余额可后续接
  `zcode.z.ai/api/v1/zcode-plan/billing/current`。
- 同一内核进程内多会话各用不同桶属于未支持场景（family `current` 全局唯一）。
