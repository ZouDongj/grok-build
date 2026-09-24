//! End-to-end verification of the three interactive bridges against the
//! ACTIVE ZCode kernel: plan approval (approve path), resume replay, and
//! session delete. Drives ZcodeAgent as a mini ACP client like `probe`, but
//! answers agent→client requests (exit_plan_mode ext, tool permissions).
//!
//! Usage: cargo run -p xai-zcode-agent --example verify_0169

use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/verify0169";
const PLAN_FILE: &str = "/tmp/verify0169/plan-write.txt";
const MARKER: &str = "暗号紫色河马ZK9173";

#[derive(Default)]
struct Seen {
    prompt_sent_at: Option<std::time::Instant>,
    ack_arrival_ms: Option<u128>,
    usage_updates: std::cell::RefCell<Vec<(u64, u64)>>,
    subagent_events: std::cell::RefCell<Vec<String>>,
    child_sid: RefCell<Option<String>>,
    child_content: std::cell::Cell<usize>,
    child_text: RefCell<String>,
    main_sid: RefCell<Option<String>>,
    plan_approval_ext: bool,
    plan_outcome_sent: Option<String>,
    permissions_answered: usize,
    mode_updates: Vec<String>,
    replay_text: RefCell<String>,
    current_text: RefCell<String>,
    queue_snapshots: RefCell<Vec<(usize, Option<String>)>>,
    plan_entries: RefCell<Vec<usize>>,
    goal_updates: RefCell<Vec<String>>,
    thought_durations_ms: RefCell<Vec<i64>>,
    thought_first_ts: std::cell::Cell<Option<i64>>,
    thought_chunks: RefCell<Vec<(i64, i64, bool)>>,
    interjection_ids: RefCell<Vec<String>>,
}

fn agent_session_is_replay() -> bool {
    false
}

fn mode_name(update: &acp::SessionUpdate) -> Option<String> {
    match update {
        acp::SessionUpdate::CurrentModeUpdate(c) => Some(format!("{:?}", c.current_mode_id)),
        _ => None,
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            std::fs::create_dir_all(WORKDIR)?;
            let _ = std::fs::remove_file(PLAN_FILE);

            let (client, agent_channel) = acp_channels();
            let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
            let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
            let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
            tokio::task::spawn_local(gw_rx.run());

            let seen = Rc::new(RefCell::new(Seen::default()));
            {
                let seen = seen.clone();
                let mut rx = client.rx;
                tokio::task::spawn_local(async move {
                    while let Some(message) = rx.recv().await {
                        match message {
                            AcpClientMessage::ExtMethod(n) => {
                                let method = n.method.clone();
                                if &*method == "x.ai/exit_plan_mode" {
                                    seen.borrow_mut().plan_approval_ext = true;
                                    seen.borrow_mut().plan_outcome_sent = Some("approved".into());
                                    let raw = serde_json::value::to_raw_value(&serde_json::json!({
                                        "outcome": "approved"
                                    }))
                                    .expect("serialize approve");
                                    let _ = n.response_tx.send(Ok(acp::ExtResponse::new(raw.into())));
                                    println!("[verify] PLAN APPROVAL ext seen -> replied approved");
                                } else {
                                    println!("[verify] ext request (auto-reject): {method}");
                                    let _ = n.response_tx.send(Err(acp::Error::method_not_found()));
                                }
                            }
                            AcpClientMessage::RequestPermission(p) => {
                                // Pick the first allow-ish option (plan execution
                                // needs Write allowed once).
                                let pick = p
                                    .options
                                    .iter()
                                    .find(|o| {
                                        matches!(
                                            o.kind,
                                            acp::PermissionOptionKind::AllowOnce
                                                | acp::PermissionOptionKind::AllowAlways
                                        )
                                    })
                                    .or_else(|| p.options.first())
                                    .cloned();
                                if let Some(option) = pick {
                                    seen.borrow_mut().permissions_answered += 1;
                                    println!(
                                        "[verify] PERMISSION -> selecting {:?} ({})",
                                        option.option_id, option.name
                                    );
                                    let _ = p.response_tx.send(Ok(acp::RequestPermissionResponse::new(
                                        acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(
                                            option.option_id.clone(),
                                        )),
                                    )));
                                } else {
                                    let _ = p.response_tx.send(Ok(acp::RequestPermissionResponse::new(
                                        acp::RequestPermissionOutcome::Cancelled,
                                    )));
                                }
                            }
                            AcpClientMessage::SessionNotification(u) => {
                                if let acp::SessionUpdate::AgentThoughtChunk(c) = &u.update {
                                    if let (acp::ContentBlock::Text(t), Some(m)) = (&c.content, u.meta.as_ref()) {
                                        if !t.text.is_empty() {
                                            // Mirror the pager's duration formula:
                                            // elapsed = agentTimestampMs - streamStartMs on
                                            // the LAST chunk of each isReplay thinking block.
                                            let ts = m.get("agentTimestampMs").and_then(|v| v.as_i64());
                                            let start = m.get("streamStartMs").and_then(|v| v.as_i64());
                                            let is_replay = m.get("isReplay").and_then(|v| v.as_bool()).unwrap_or(false);
                                            if let (Some(a), Some(s)) = (ts, start) {
                                                seen
                                                    .borrow_mut()
                                                    .thought_chunks
                                                    .borrow_mut()
                                                    .push((a, s, is_replay));
                                            }
                                            let first = seen.borrow().thought_first_ts.get();
                                            if let (Some(a), Some(first)) = (ts, first) {
                                                seen
                                                    .borrow_mut()
                                                    .thought_durations_ms
                                                    .borrow_mut()
                                                    .push(a - first);
                                            } else if first.is_none() && ts.is_some() {
                                                seen.borrow_mut().thought_first_ts.set(ts);
                                            }
                                        }
                                    }
                                    if let acp::ContentBlock::Text(t) = &c.content {
                                        let is_ack = t.text.is_empty()
                                            && u.meta.as_ref().and_then(|m| m.get("promptId")).is_some()
                                            && seen.borrow().ack_arrival_ms.is_none();
                                        if is_ack {
                                            let t0 = seen.borrow().prompt_sent_at;
                                            if let Some(t0) = t0 {
                                                seen.borrow_mut().ack_arrival_ms =
                                                    Some(t0.elapsed().as_millis());
                                            }
                                        }
                                    }
                                }
                                if let acp::SessionUpdate::UsageUpdate(usage) = &u.update {
                                    seen
                                        .borrow_mut()
                                        .usage_updates
                                        .borrow_mut()
                                        .push((usage.used, usage.size));
                                }
                                let is_child = seen
                                    .borrow()
                                    .main_sid
                                    .borrow()
                                    .as_deref()
                                    .is_some_and(|m| u.request.session_id.0.as_ref() != m);
                                if is_child {
                                    match &u.update {
                                        acp::SessionUpdate::AgentThoughtChunk(_)
                                        | acp::SessionUpdate::AgentMessageChunk(_)
                                        | acp::SessionUpdate::ToolCall(_) => {
                                            let n = seen.borrow().child_content.get();
                                            seen.borrow_mut().child_content.set(n + 1);
                                        }
                                        _ => {}
                                    }
                                    if let acp::SessionUpdate::AgentMessageChunk(c) = &u.update {
                                        if let acp::ContentBlock::Text(t) = &c.content {
                                            seen
                                                .borrow_mut()
                                                .child_text
                                                .borrow_mut()
                                                .push_str(&t.text);
                                        }
                                    }
                                }
                                if let acp::SessionUpdate::Plan(plan) = &u.update {
                                    if !agent_session_is_replay() {
                                        seen.borrow_mut().plan_entries.borrow_mut().push(plan.entries.len());
                                    }
                                }
                                if let Some(mode) = mode_name(&u.update) {
                                    seen.borrow_mut().mode_updates.push(mode.clone());
                                    println!("[verify] MODE UPDATE: {mode}");
                                }
                                if let acp::SessionUpdate::UserMessageChunk(c) = &u.update {
                                    if let acp::ContentBlock::Text(t) = &c.content {
                                        seen.borrow().replay_text.borrow_mut().push_str(&t.text);
                                    }
                                }
                                if let acp::SessionUpdate::AgentMessageChunk(c) = &u.update {
                                    if let acp::ContentBlock::Text(t) = &c.content {
                                        seen.borrow().current_text.borrow_mut().push_str(&t.text);
                                    }
                                }
                            }
                            AcpClientMessage::ExtNotification(n) => {
                                if &*n.method == "x.ai/session/update" {
                                    let v: serde_json::Value =
                                        serde_json::from_str(n.params.get()).unwrap_or(serde_json::json!({}));
                                    if let Some(kind) = v.pointer("/update/sessionUpdate").and_then(|k| k.as_str()) {
                                        if kind.starts_with("subagent_") {
                                            seen.borrow_mut().subagent_events.borrow_mut().push(kind.to_string());
                                        }
                                    }
                                    if v.pointer("/update/sessionUpdate").and_then(|k| k.as_str()) == Some("subagent_spawned") {
                                        if let Some(child) = v.pointer("/update/child_session_id").and_then(|c| c.as_str()) {
                                            let child = child.to_string();
                                            let mut seen_b = seen.borrow_mut();
                                            if seen_b.child_sid.borrow().is_none() {
                                                *seen_b.child_sid.borrow_mut() = Some(child);
                                            }
                                        }
                                    }
                                }
                                if &*n.method == "x.ai/queue/changed" {
                                    let v: serde_json::Value =
                                        serde_json::from_str(n.params.get()).unwrap_or(serde_json::json!({}));
                                    let n_entries = v
                                        .get("entries")
                                        .and_then(|e| e.as_array())
                                        .map(|a| a.len())
                                        .unwrap_or(0);
                                    let running = v
                                        .get("runningPromptId")
                                        .and_then(|r| r.as_str())
                                        .map(|s| s.to_string());
                                    seen.borrow_mut().queue_snapshots.borrow_mut().push((n_entries, running));
                                }
                                if &*n.method == "x.ai/session_notification" {
                                    let v: serde_json::Value =
                                        serde_json::from_str(n.params.get()).unwrap_or(serde_json::json!({}));
                                    if v.pointer("/update/sessionUpdate").and_then(|k| k.as_str()) == Some("goal_updated") {
                                        let status = v.pointer("/update/status").and_then(|k| k.as_str()).unwrap_or("").to_string();
                                        seen.borrow_mut().goal_updates.borrow_mut().push(status);
                                    }
                                }
                                if &*n.method == "x.ai/session/interjection" {
                                    let v: serde_json::Value =
                                        serde_json::from_str(n.params.get()).unwrap_or(serde_json::json!({}));
                                    if let Some(id) = v.get("interjectionId").and_then(|i| i.as_str()) {
                                        seen.borrow_mut().interjection_ids.borrow_mut().push(id.to_string());
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                });
            }

            let mut summary = Vec::new();
            let mut fail = 0usize;
            let mut check = |name: &str, ok: bool, detail: String, summary: &mut Vec<String>, fail: &mut usize| {
                let line = format!("[{}] {name}: {detail}", if ok { "PASS" } else { "FAIL" });
                println!("{line}");
                summary.push(line);
                if !ok {
                    *fail += 1;
                }
            };

            let _init = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
            let session = acp_send(
                acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)),
                &client.tx,
            )
            .await?;
            let sid = session.session_id.clone();
            println!("[verify] session {}", sid.0);
            *seen.borrow_mut().main_sid.borrow_mut() = Some(sid.0.to_string());

            // --- turn 1: establish a context marker for the resume check ---
            seen.borrow_mut().prompt_sent_at = Some(std::time::Instant::now());
            let mut prompt1 = acp::PromptRequest::new(
                sid.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new(format!(
                    "请记住这个暗号，之后我会考你：{MARKER}。只回复：记住了"
                )))],
            );
            let mut p1meta = acp::Meta::new();
            p1meta.insert("promptId".to_string(), serde_json::json!("verify-pid-1"));
            prompt1.meta = Some(p1meta);
            let t1 = acp_send(prompt1, &client.tx).await;
            check(
                "turn1-marker",
                matches!(&t1, Ok(r) if r.stop_reason == acp::StopReason::EndTurn),
                format!("stop={:?}", t1.as_ref().map(|r| r.stop_reason.clone()).map_err(|e| e.to_string())),
                &mut summary,
                &mut fail,
            );

            // --- plan approval: switch to plan mode, prompt for a file write ---
            let _ = acp_send(
                acp::SetSessionModeRequest::new(sid.clone(), acp::SessionModeId::new("plan")),
                &client.tx,
            )
            .await?;
            let plan_prompt = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(format!(
                        "把文本 hello-plan 写入文件 {PLAN_FILE}"
                    )))],
                ),
                &client.tx,
            )
            .await;
            println!(
                "[verify] plan-mode prompt stop={:?}",
                plan_prompt.as_ref().map(|r| r.stop_reason.clone()).map_err(|e| e.to_string())
            );
            // The approved plan queues a continuation turn that performs the
            // write — wait for the file (bounded).
            let mut file_ok = false;
            for _ in 0..60 {
                if std::path::Path::new(PLAN_FILE).exists() {
                    file_ok = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
            // let the continuation turn settle so resume replays a quiet session
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;

            check(
                "plan-approval-ext",
                seen.borrow().plan_approval_ext,
                format!("ext seen={}, outcome={:?}", seen.borrow().plan_approval_ext, seen.borrow().plan_outcome_sent),
                &mut summary,
                &mut fail,
            );
            check(
                "plan-file-written",
                file_ok,
                format!("{PLAN_FILE} exists={file_ok}"),
                &mut summary,
                &mut fail,
            );
            let saw_build = seen.borrow().mode_updates.iter().any(|m| m.contains("build"));
            check(
                "plan-mode-flip-to-build",
                saw_build,
                format!("mode updates={:?}", seen.borrow().mode_updates),
                &mut summary,
                &mut fail,
            );
            check(
                "plan-permissions-answered",
                seen.borrow().permissions_answered > 0,
                format!("answered={}", seen.borrow().permissions_answered),
                &mut summary,
                &mut fail,
            );

            let ack_ms = seen.borrow().ack_arrival_ms;
            check(
                "prompt-ack-immediate",
                ack_ms.is_some_and(|ms| ms < 5000),
                format!("first promptId ack after {ack_ms:?}ms (pager watchdog is 120s)"),
                &mut summary,
                &mut fail,
            );

            // --- mcp/list bridge: the extensions modal's MCP tab ---
            tokio::time::sleep(std::time::Duration::from_secs(4)).await;
            let mcp_resp = acp_send(
                acp::ExtRequest::new(
                    "x.ai/mcp/list",
                    serde_json::value::to_raw_value(&serde_json::json!({"sessionId": sid.0.as_ref(), "cache": true}))
                        .expect("serialize mcp list params")
                        .into(),
                ),
                &client.tx,
            )
            .await;
            let mcp_detail = match &mcp_resp {
                Ok(r) => {
                    let v: serde_json::Value =
                        serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                    let inner = v.get("result").unwrap_or(&v);
                    let servers = inner.get("servers").and_then(|s| s.as_array()).cloned().unwrap_or_default();
                    let ready = servers.iter().filter(|s| {
                        s.pointer("/session/status").and_then(|x| x.as_str()) == Some("ready")
                    }).count();
                    format!("servers={} ready={}", servers.len(), ready)
                }
                Err(e) => format!("err={e}"),
            };
            check(
                "mcp-list-bridge",
                mcp_resp.is_ok() && mcp_detail.starts_with("servers=") && !mcp_detail.contains("servers=0 "),
                mcp_detail,
                &mut summary,
                &mut fail,
            );

            // --- plugins: list + enable/disable round-trip ---
            let plug_resp = acp_send(
                acp::ExtRequest::new(
                    "x.ai/plugins/list",
                    serde_json::value::to_raw_value(&serde_json::json!({"sessionId": sid.0.as_ref()}))
                        .expect("serialize plugins list params")
                        .into(),
                ),
                &client.tx,
            )
            .await;
            let mut plug_detail = String::new();
            let mut toggle_ok = false;
            if let Ok(r) = &plug_resp {
                let v: serde_json::Value =
                    serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                let inner = v.get("result").unwrap_or(&v);
                let plugins = inner
                    .get("plugins")
                    .and_then(|p| p.as_array())
                    .cloned()
                    .unwrap_or_default();
                plug_detail = format!("plugins={}", plugins.len());
                if let Some(first) = plugins.iter().find(|p| {
                    p.get("id").and_then(|x| x.as_str()).is_some_and(|s| !s.is_empty())
                }) {
                    let pid = first.get("id").and_then(|x| x.as_str()).unwrap().to_string();
                    let was_enabled = first.get("enabled").and_then(|x| x.as_bool()).unwrap_or(true);
                    for want in [!was_enabled, was_enabled] {
                        let act = if want { "enable" } else { "disable" };
                        let toggle = acp_send(
                            acp::ExtRequest::new(
                                "x.ai/plugins/action",
                                serde_json::value::to_raw_value(&serde_json::json!({
                                    "sessionId": sid.0.as_ref(),
                                    "action": {"type": act, "pluginId": pid},
                                }))
                                .expect("serialize toggle")
                                .into(),
                            ),
                            &client.tx,
                        )
                        .await;
                        if let Ok(tr) = &toggle {
                            let tv: serde_json::Value =
                                serde_json::from_str(tr.0.get()).unwrap_or(serde_json::json!({}));
                            let status = tv
                                .pointer("/result/status")
                                .and_then(|x| x.as_str())
                                .unwrap_or("");
                            toggle_ok = status == "success";
                        }
                        if !toggle_ok {
                            break;
                        }
                    }
                }
            } else if let Err(e) = &plug_resp {
                plug_detail = format!("err={e}");
            }
            check(
                "plugins-list-bridge",
                plug_resp.is_ok() && plug_detail.starts_with("plugins="),
                plug_detail.clone(),
                &mut summary,
                &mut fail,
            );
            check(
                "plugins-toggle-roundtrip",
                toggle_ok,
                format!("(restore via same toggle) {plug_detail}"),
                &mut summary,
                &mut fail,
            );

            // --- skills list bridge ---
            let skills_resp = acp_send(
                acp::ExtRequest::new(
                    "x.ai/skills/list",
                    serde_json::value::to_raw_value(&serde_json::json!({"cwd": "."}))
                        .expect("serialize skills list params")
                        .into(),
                ),
                &client.tx,
            )
            .await;
            let skills_detail = match &skills_resp {
                Ok(r) => {
                    let v: serde_json::Value =
                        serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                    let inner = v.get("result").unwrap_or(&v);
                    let n = inner
                        .get("skills")
                        .and_then(|s| s.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0);
                    format!("skills={n}")
                }
                Err(e) => format!("err={e}"),
            };
            check(
                "skills-list-bridge",
                skills_resp.is_ok() && !skills_detail.contains("skills=0"),
                skills_detail,
                &mut summary,
                &mut fail,
            );

            // --- hooks / workflows / marketplace tabs ---
            for (method, params, probe) in [
                ("x.ai/hooks/list", serde_json::json!({"sessionId": sid.0.as_ref()}), "hooks"),
                ("x.ai/workflows/list", serde_json::json!({"sessionId": sid.0.as_ref()}), "workflows"),
                ("x.ai/marketplace/list", serde_json::json!({"sessionId": sid.0.as_ref()}), "sources"),
            ] {
                let resp = acp_send(
                    acp::ExtRequest::new(
                        method,
                        serde_json::value::to_raw_value(&params)
                            .expect("serialize tab params")
                            .into(),
                    ),
                    &client.tx,
                )
                .await;
                let detail = match &resp {
                    Ok(r) => {
                        let v: serde_json::Value =
                            serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                        let inner = v.get("result").unwrap_or(&v);
                        let n = inner.get(probe).and_then(|x| x.as_array()).map(|a| a.len());
                        match n {
                            Some(n) => format!("{probe}={n}"),
                            None => format!("{probe}=missing"),
                        }
                    }
                    Err(e) => format!("err={e}"),
                };
                check(
                    &format!("tab-{}", method.strip_prefix("x.ai/").unwrap_or(method)),
                    resp.is_ok() && !detail.contains("missing") && !detail.starts_with("err"),
                    detail,
                    &mut summary,
                    &mut fail,
                );
            }

            // --- misc wiring: subscription poll, billing, session info,
            // prompt history, session usage ---
            for (method, params, probe) in [
                ("x.ai/auth/check_subscription", serde_json::json!({}), "meta"),
                ("x.ai/billing", serde_json::json!({}), "result"),
                ("x.ai/session/info", serde_json::json!({"sessionId": sid.0.as_ref()}), "result"),
                ("x.ai/prompt_history", serde_json::json!({"sessionId": sid.0.as_ref()}), "prompts"),
                ("x.ai/session/usage", serde_json::json!({"sessionId": sid.0.as_ref()}), "usage"),
            ] {
                let resp = acp_send(
                    acp::ExtRequest::new(
                        method,
                        serde_json::value::to_raw_value(&params)
                            .expect("serialize misc params")
                            .into(),
                    ),
                    &client.tx,
                )
                .await;
                let ok = matches!(&resp, Ok(r) if {
                    let v: serde_json::Value =
                        serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                    v.pointer(&format!("/result/{probe}")).or_else(|| v.get(probe)).is_some()
                });
                check(
                    &format!("wire-{}", method.strip_prefix("x.ai/").unwrap_or(method)),
                    ok,
                    format!("{:?}", resp.as_ref().map(|_| "ok").map_err(|e| e.to_string())),
                    &mut summary,
                    &mut fail,
                );
            }
            // prompt_history must contain the turn-1 marker prompt.
            let hist = acp_send(
                acp::ExtRequest::new(
                    "x.ai/prompt_history",
                    serde_json::value::to_raw_value(&serde_json::json!({"sessionId": sid.0.as_ref()}))
                        .expect("serialize history params")
                        .into(),
                ),
                &client.tx,
            )
            .await;
            let hist_has = matches!(&hist, Ok(r) if {
                let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                v.pointer("/result/prompts").and_then(|p| p.as_array())
                    .is_some_and(|a| a.iter().any(|x| {
                        x.as_str().is_some_and(|t| t.contains("暗号紫色河马"))
                    }))
            });
            check(
                "prompt-history-has-marker",
                hist_has,
                "up-arrow recall sees turn-1 text".to_string(),
                &mut summary,
                &mut fail,
            );

            // --- rewind: separate session, two prompts, rewind to point 0,
            // then continue — the fork must be transparent in-place ---
            {
                let rw_session = acp_send(
                    acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)),
                    &client.tx,
                )
                .await;
                let rw_sid = rw_session.expect("rewind session").session_id;
                for text in ["第一句话：香蕉", "第二句话：苹果"] {
                    let _ = acp_send(
                        acp::PromptRequest::new(
                            rw_sid.clone(),
                            vec![acp::ContentBlock::Text(acp::TextContent::new(
                                format!("{text}。只回复：收到{text}"),
                            ))],
                        ),
                        &client.tx,
                    )
                    .await;
                }
                let pts = acp_send(
                    acp::ExtRequest::new(
                        "x.ai/rewind/points",
                        serde_json::value::to_raw_value(&serde_json::json!({"sessionId": rw_sid.0.as_ref()}))
                            .expect("serialize points params")
                            .into(),
                    ),
                    &client.tx,
                )
                .await;
                let pts_count = matches!(&pts, Ok(r) if {
                    let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                    v.pointer("/result/rewindPoints").and_then(|x| x.as_array()).is_some_and(|a| a.len() >= 2)
                });
                check(
                    "rewind-points-listed",
                    pts_count,
                    "two user prompts visible as rewind points".to_string(),
                    &mut summary,
                    &mut fail,
                );
                let exec = acp_send(
                    acp::ExtRequest::new(
                        "x.ai/rewind/execute",
                        serde_json::value::to_raw_value(&serde_json::json!({
                            "sessionId": rw_sid.0.as_ref(),
                            "targetPromptIndex": 1,
                            "force": true,
                            "mode": "conversation_only",
                        }))
                        .expect("serialize rewind exec")
                        .into(),
                    ),
                    &client.tx,
                )
                .await;
                // Truncating rewind (v4 fork path): rewinding to prompt 1
                // must drop everything after turn 0 — the model must NOT
                // know 苹果 afterwards.
                let exec_ok = matches!(&exec, Ok(r) if {
                    let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                    let res = v.get("result").cloned().unwrap_or(v);
                    res.get("success") == Some(&serde_json::json!(true))
                });
                check(
                    "rewind-execute-accepted",
                    exec_ok,
                    format!("{:?}", exec.as_ref().map(|_| "ok").map_err(|e| e.to_string())),
                    &mut summary,
                    &mut fail,
                );
                seen.borrow_mut().current_text.borrow_mut().clear();
                let recall = acp_send(
                    acp::PromptRequest::new(
                        rw_sid.clone(),
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "你听过'第二句话：苹果'吗？只回答：听过 或 没听过。".to_string(),
                        ))],
                    ),
                    &client.tx,
                )
                .await;
                let mut rw_answer = String::new();
                for _ in 0..20 {
                    rw_answer = seen.borrow().current_text.borrow().clone();
                    if rw_answer.contains("过") { break; }
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
                check(
                    "rewind-truncates-history",
                    matches!(&recall, Ok(r) if r.stop_reason == acp::StopReason::EndTurn)
                        && rw_answer.contains("没听"),
                    format!("recall answer: {:?}", rw_answer.chars().take(30).collect::<String>()),
                    &mut summary,
                    &mut fail,
                );
                let _ = acp_send(
                    acp::ExtRequest::new(
                        "x.ai/session/delete",
                        serde_json::value::to_raw_value(&serde_json::json!({
                            "sessionId": rw_sid.0.as_ref(), "cwd": WORKDIR,
                        }))
                        .expect("serialize rw delete")
                        .into(),
                    ),
                    &client.tx,
                )
                .await;
            }

            let usages = seen.borrow().usage_updates.borrow().clone();
            let usage_ok = usages.iter().any(|(used, size)| *used > 0 && *size >= 200_000);
            check(
                "context-usage-updates",
                usage_ok,
                format!("{:?}", usages.last()),
                &mut summary,
                &mut fail,
            );

            // --- subagent lifecycle visualization ---
            seen.borrow_mut().subagent_events.borrow_mut().clear();
            seen.borrow_mut().child_sid.borrow_mut().take();
            seen.borrow_mut().child_content.set(0);
            seen.borrow_mut().child_text.borrow_mut().clear();
            let sub_turn = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "请使用 Agent 工具派一个子代理执行命令 echo subviz-ok 并把输出返回。".to_string(),
                    ))],
                ),
                &client.tx,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let events = seen.borrow().subagent_events.borrow().clone();
            let saw_spawn = events.iter().any(|e| e == "subagent_spawned");
            let saw_finish = events.iter().any(|e| e == "subagent_finished");
            check(
                "subagent-lifecycle-visualized",
                matches!(&sub_turn, Ok(r) if r.stop_reason == acp::StopReason::EndTurn)
                    && saw_spawn && saw_finish,
                format!("events={:?}", events),
                &mut summary,
                &mut fail,
            );
            // Content rides the kernel's transcript files (output.txt byte
            // tails) — the official client channel; one delta per flush.
            let child_events = seen.borrow().child_content.get();
            let child_text = seen.borrow().child_text.borrow().clone();
            check(
                "subagent-content-streamed",
                child_events >= 1 && child_text.contains("subviz-ok"),
                format!(
                    "{child_events} child updates reached the pager view; captured child text: {:?}",
                    &child_text[..child_text.len().min(80)]
                ),
                &mut summary,
                &mut fail,
            );

            // --- kernel RPC boundary on child sessions (documented) ---
            // The official client does NOT read subagent children over RPC:
            // it consumes native subagent.lifecycle events on the parent
            // stream plus transcript FILES under ~/.zcode/cli/agents.
            // session/events, session/messages, session/subscribe are all
            // rejected with "Session is not active" for child sessions.
            let child_sid = seen.borrow().child_sid.borrow().clone();
            if let Some(child) = child_sid {
                let ev_resp = acp_send(
                    acp::ExtRequest::new(
                        "x.ai/session/events",
                        serde_json::value::to_raw_value(&serde_json::json!({
                            "sessionId": child, "afterSeq": 0, "limit": 100,
                        }))
                        .expect("serialize child events req")
                        .into(),
                    ),
                    &client.tx,
                )
                .await;
                let ev_err = match &ev_resp {
                    Err(e) => format!("{:?}", e),
                    Ok(_) => "no-error".to_string(),
                };
                check(
                    "session-events-child-rejected",
                    matches!(&ev_resp, Err(_)) && ev_err.contains("not active"),
                    format!("child RPC read rejected as expected: {ev_err}"),
                    &mut summary,
                    &mut fail,
                );

                let msg_resp = acp_send(
                    acp::ExtRequest::new(
                        "x.ai/session/messages",
                        serde_json::value::to_raw_value(&serde_json::json!({
                            "sessionId": child, "limit": 50,
                        }))
                        .expect("serialize child messages req")
                        .into(),
                    ),
                    &client.tx,
                )
                .await;
                let msg_err = match &msg_resp {
                    Err(e) => format!("{:?}", e),
                    Ok(_) => "no-error".to_string(),
                };
                check(
                    "session-messages-child-rejected",
                    matches!(&msg_resp, Err(_)) && msg_err.contains("not active"),
                    format!("child RPC read rejected as expected: {msg_err}"),
                    &mut summary,
                    &mut fail,
                );
            } else {
                check(
                    "session-events-child-rejected",
                    false,
                    "no child session id captured".to_string(),
                    &mut summary,
                    &mut fail,
                );
            }

            // --- /compact: ext must wait for the REAL compaction turn ---
            // session/compact only accepts (instant {state:"accepted"}) and
            // runs "/compact" as a background prompt turn; during it the
            // kernel rejects session/send with -32010. A prompt fired 1s in
            // must be queued and answered AFTER compaction — not lost and
            // not "answered" by the kernel's "Compacted" text.
            {
                seen.borrow_mut().current_text.borrow_mut().clear();
                let usage_before =
                    seen.borrow().usage_updates.borrow().last().cloned();
                let usage_len_before = seen.borrow().usage_updates.borrow().len();
                let cx_tx = client.tx.clone();
                let cx_sid = sid.clone();
                let compact_task = tokio::task::spawn_local(async move {
                    let t0 = std::time::Instant::now();
                    let resp = acp_send(
                        acp::ExtRequest::new(
                            "x.ai/compact_conversation",
                            serde_json::value::to_raw_value(&serde_json::json!({
                                "sessionId": cx_sid.0,
                            }))
                            .expect("serialize compact verify req")
                            .into(),
                        ),
                        &cx_tx,
                    )
                    .await;
                    (t0.elapsed(), resp.is_ok())
                });
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                let queued_turn = acp_send(
                    acp::PromptRequest::new(
                        sid.clone(),
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "1+1等于几？只回答阿拉伯数字。".to_string(),
                        ))],
                    ),
                    &client.tx,
                )
                .await;
                let (compact_elapsed, compact_ok) = compact_task.await.unwrap_or_default();
                let text_after = seen.borrow().current_text.borrow().clone();
                check(
                    "compact-waits-for-real-completion",
                    compact_ok && compact_elapsed.as_secs() >= 5,
                    format!("compact ext returned after {compact_elapsed:?} (ok={compact_ok}) — must be the real turn, not the instant accept"),
                    &mut summary,
                    &mut fail,
                );
                check(
                    "compact-mid-prompt-queued-and-answered",
                    matches!(&queued_turn, Ok(r) if r.stop_reason == acp::StopReason::EndTurn)
                        && text_after.contains("Compacted")
                        && text_after.contains('2'),
                    format!("queued turn end={:?}, captured text: {:?}", queued_turn.as_ref().map(|r| format!("{:?}", r.stop_reason)).map_err(|e| e.to_string()), &text_after[..text_after.len().min(60)]),
                    &mut summary,
                    &mut fail,
                );
                // Real-time context meter: the kernel's usage patch (v4
                // projection, official conflation channel) must arrive and
                // DROP the meter without waiting for the next user turn.
                let usages = seen.borrow().usage_updates.borrow().clone();
                let dropped = usage_before
                    .filter(|(pre, _)| {
                        usages
                            .iter()
                            .skip(usage_len_before)
                            .any(|(used, _)| *used + 2000 < *pre)
                    })
                    .is_some();
                let last = usages.last().cloned();
                check(
                    "compact-drops-context-meter",
                    dropped,
                    format!(
                        "pre-compact usage {:?}, post-compact stream {:?} — a lowered USAGE push must land without a further user turn",
                        usage_before,
                        last
                    ),
                    &mut summary,
                    &mut fail,
                );
            }

            // --- resume must not rebroadcast historical subagents ---
            {
                seen.borrow_mut().subagent_events.borrow_mut().clear();
                let _ = acp_send(
                    acp::LoadSessionRequest::new(sid.clone(), std::path::PathBuf::from(WORKDIR)),
                    &client.tx,
                )
                .await;
                let first = acp_send(
                    acp::PromptRequest::new(
                        sid.clone(),
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "恢复后的第一句：只回复：好的。".to_string(),
                        ))],
                    ),
                    &client.tx,
                )
                .await;
                let events = seen.borrow().subagent_events.borrow().clone();
                // Replayed thoughts carry ledger timestamps: the pager can
                // compute real durations instead of "Thought for 0.0s".
                let chunks = seen.borrow().thought_chunks.borrow().clone();
                // Per the pager tracker: for each isReplay thinking block the
                // FROZEN duration is the last chunk's agentTimestampMs -
                // streamStartMs. Blocks are delimited by stream-start changes,
                // so assert: replay chunks exist, all carry isReplay, and the
                // last chunk of at least one stream has a >=1s real span.
                let mut all_replay = true;
                for &(_ts, _start, is_replay) in &chunks {
                    if !is_replay {
                        all_replay = false;
                    }
                }
                let spans: Vec<i64> = chunks
                    .windows(2)
                    .filter(|w| w[0].1 == w[1].1)
                    .map(|w| w[1].0 - w[1].1)
                    .collect();
                let replay_durations_real = !chunks.is_empty()
                    && all_replay
                    && spans.iter().any(|d| *d >= 1000)
                    && spans.iter().all(|d| *d >= 0);
                check(
                    "replay-thoughts-have-real-durations",
                    replay_durations_real,
                    format!(
                        "replay thought chunks (agentTs,streamStart,isReplay): {:?}; per-block final spans: {spans:?}",
                        chunks.iter().map(|(a, s, r)| (a - s, r)).collect::<Vec<_>>()
                    ),
                    &mut summary,
                    &mut fail,
                );
                check(
                    "resume-no-historical-subagent-rebroadcast",
                    events.is_empty()
                        && matches!(&first, Ok(r) if r.stop_reason == acp::StopReason::EndTurn),
                    format!("subagent events after resume prompt: {events:?}"),
                    &mut summary,
                    &mut fail,
                );
            }

            // --- goal mode: set via /goal text, panel updates, pause, clear ---
            {
                seen.borrow_mut().goal_updates.borrow_mut().clear();
                let goal_turn = acp_send(
                    acp::PromptRequest::new(
                        sid.clone(),
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "/goal 目标：验证goal面板接线".to_string(),
                        ))],
                    ),
                    &client.tx,
                )
                .await;
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                let goals = seen.borrow().goal_updates.borrow().clone();
                let saw_active = goals.iter().any(|g| g == "active");
                check(
                    "goal-panel-set-visible",
                    matches!(&goal_turn, Ok(r) if r.stop_reason == acp::StopReason::EndTurn) && saw_active,
                    format!("goal updates: {goals:?}"),
                    &mut summary,
                    &mut fail,
                );
                let pause = acp_send(
                    acp::ExtRequest::new(
                        "x.ai/session/goal",
                        serde_json::value::to_raw_value(&serde_json::json!({
                            "sessionId": sid.0, "action": "pause",
                        }))
                        .expect("serialize goal pause")
                        .into(),
                    ),
                    &client.tx,
                )
                .await;
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let goals = seen.borrow().goal_updates.borrow().clone();
                let saw_paused = goals.iter().any(|g| g == "user_paused");
                check(
                    "goal-pause-via-ext",
                    pause.is_ok() && saw_paused,
                    format!("goal updates: {goals:?}"),
                    &mut summary,
                    &mut fail,
                );
                let _ = acp_send(
                    acp::ExtRequest::new(
                        "x.ai/session/goal",
                        serde_json::value::to_raw_value(&serde_json::json!({
                            "sessionId": sid.0, "action": "clear",
                        }))
                        .expect("serialize goal clear")
                        .into(),
                    ),
                    &client.tx,
                )
                .await;
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let goals = seen.borrow().goal_updates.borrow().clone();
                check(
                    "goal-clear-via-ext",
                    goals.iter().any(|g| g == "cleared"),
                    format!("goal updates: {goals:?}"),
                    &mut summary,
                    &mut fail,
                );
            }

            // --- TodoWrite list surfaces in the pager's todos pane ---
            {
                seen.borrow_mut().plan_entries.borrow_mut().clear();
                let todo_turn = acp_send(
                    acp::PromptRequest::new(
                        sid.clone(),
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "请使用 TodoWrite 工具建立两个任务：'部署环境' 和 '运行测试'，然后把 '部署环境' 标记为 in_progress。不要做别的事。".to_string(),
                        ))],
                    ),
                    &client.tx,
                )
                .await;
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                let plans = seen.borrow().plan_entries.borrow().clone();
                check(
                    "todowrite-list-visible",
                    matches!(&todo_turn, Ok(r) if r.stop_reason == acp::StopReason::EndTurn)
                        && plans.iter().any(|n| *n >= 2),
                    format!("plan updates: {plans:?}"),
                    &mut summary,
                    &mut fail,
                );
            }

            // --- interject: mid-turn "send now" queues as a continuation ---
            seen.borrow_mut().current_text.borrow_mut().clear();
            seen.borrow_mut().queue_snapshots.borrow_mut().clear();
            seen.borrow_mut().interjection_ids.borrow_mut().clear();
            let ij_tx = client.tx.clone();
            let ij_sid = sid.clone();
            let ij_prompt = tokio::task::spawn_local(async move {
                acp_send(
                    acp::PromptRequest::new(
                        ij_sid,
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "请从1一个一个慢慢数到100，每个数字单独一行，输出完所有100个数字才能结束，不许提前停止。".to_string(),
                        ))],
                    ),
                    &ij_tx,
                )
                .await
            });
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            let ij_params = serde_json::json!({
                "sessionId": sid.0.as_ref(),
                "text": "插话：不用数了。请只回复两个字：收到",
                "interjectionId": "verify-ij-1",
            });
            let interject = acp_send(
                acp::ExtRequest::new(
                    "x.ai/interject",
                    serde_json::value::to_raw_value(&ij_params).expect("serialize interject").into(),
                ),
                &client.tx,
            )
            .await;
            let ij_steered = matches!(&interject, Ok(r) if {
                let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                v.get("steered") == Some(&serde_json::json!(true))
            });
            let ij_ack = matches!(&interject, Ok(r) if {
                let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                v.get("accepted") == Some(&serde_json::json!(true))
            });
            check(
                "interject-ack",
                ij_ack,
                format!("result={:?}", interject.as_ref().map(|_| "ok").map_err(|e| e.to_string())),
                &mut summary,
                &mut fail,
            );
            let first_stop = ij_prompt.await.ok().and_then(|r| r.ok()).map(|r| r.stop_reason);
            let mut got_echo = false;
            for _ in 0..45 {
                if seen.borrow().current_text.borrow().contains("收到") {
                    got_echo = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
            check(
                "interject-delivered-next-turn",
                got_echo && first_stop == Some(acp::StopReason::EndTurn),
                format!("first turn stop={first_stop:?}, continuation echoed={got_echo}"),
                &mut summary,
                &mut fail,
            );
            // The queue pane reconciles via x.ai/queue/changed snapshots and
            // the interjection delivery broadcast claims the echo block.
            {
                // v4 guide steers INTO the running turn (no queue row, no
                // boundary drain); the legacy queue path remains the
                // fallback on kernels without v4.
                let snapshots = seen.borrow().queue_snapshots.borrow().clone();
                let ij_ids = seen.borrow().interjection_ids.borrow().clone();
                let saw_interjection = ij_ids.iter().any(|id| id == "verify-ij-1");
                let fallback_path = snapshots.iter().any(|(n, _)| *n >= 1)
                    && snapshots.iter().any(|(n, _)| *n == 0)
                    && saw_interjection;
                check(
                    "interject-steered-mid-turn",
                    (ij_steered && saw_interjection) || fallback_path,
                    format!("steered={ij_steered}, interjection ids={ij_ids:?}, queue snapshots={snapshots:?}"),
                    &mut summary,
                    &mut fail,
                );
            }

            // --- send-now (强插): meta.sendNow must cancel the running turn
            // and run the interrupting prompt immediately ---
            // Settle first: the interject continuation turn (unowned by any
            // ACP prompt) may still be finishing.
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            seen.borrow_mut().current_text.borrow_mut().clear();
            let sn_tx = client.tx.clone();
            let sn_sid = sid.clone();
            let slow_turn = tokio::task::spawn_local(async move {
                acp_send(
                    acp::PromptRequest::new(
                        sn_sid,
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "请从1一个一个慢慢数到100，每个数字单独一行，输出完所有100个数字才能结束，不许提前停止。".to_string(),
                        ))],
                    ),
                    &sn_tx,
                )
                .await
            });
            tokio::time::sleep(std::time::Duration::from_millis(2500)).await;
            let mut sn_meta = serde_json::Map::new();
            sn_meta.insert("promptId".to_string(), serde_json::json!("verify-sn-1"));
            sn_meta.insert("sendNow".to_string(), serde_json::json!(true));
            let sn_turn = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "1+1等于几？只回答阿拉伯数字。".to_string(),
                    ))],
                )
                .meta(Some(sn_meta)),
                &client.tx,
            )
            .await;
            let sn_stop = sn_turn.as_ref().map(|r| format!("{:?}", r.stop_reason)).map_err(|e| e.to_string());
            // The interrupted turn must settle (not hang).
            let slow_settled = matches!(
                tokio::time::timeout(std::time::Duration::from_secs(60), slow_turn).await,
                Ok(Ok(_))
            );
            let mut sn_answer = String::new();
            for _ in 0..20 {
                sn_answer = seen.borrow().current_text.borrow().clone();
                if sn_answer.contains('2') {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
            check(
                "send-now-interrupts-running-turn",
                matches!(&sn_turn, Ok(r) if r.stop_reason == acp::StopReason::EndTurn)
                    && sn_answer.contains('2')
                    && slow_settled,
                format!("send-now turn={sn_stop:?}, answer={:?}, interrupted turn settled={slow_settled}", &sn_answer[..sn_answer.len().min(40)]),
                &mut summary,
                &mut fail,
            );

            // Settle: any late continuation/queue delivery from the
            // compact section must finish before the image capture window.
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            // --- image input: switch to Flash (the multimodal model), send a
            // solid-red PNG, expect a color answer that proves the pixels
            // reached the model ---
            let _ = acp_send(
                acp::SetSessionModelRequest::new(
                    sid.clone(),
                    acp::ModelId::new("GLM-5.3-Flash".to_string()),
                ),
                &client.tx,
            )
            .await;
            seen.borrow().current_text.borrow_mut().clear();
            const RED_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP4z8CAFWEXHbQSACj/P8Fu7N9hAAAAAElFTkSuQmCC";
            let img_turn = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![
                        acp::ContentBlock::Text(acp::TextContent::new(
                            "这张图片是什么颜色？只回答一个颜色词".to_string(),
                        )),
                        acp::ContentBlock::Image(acp::ImageContent::new(
                            RED_PNG.to_string(),
                            "image/png",
                        )),
                    ],
                ),
                &client.tx,
            )
            .await;
            let answer = seen.borrow().current_text.borrow().clone();
            let color_ok = answer.contains('红') || answer.to_lowercase().contains("red");
            check(
                "image-input-flash-color",
                matches!(&img_turn, Ok(r) if r.stop_reason == acp::StopReason::EndTurn) && color_ok,
                format!("stop={:?}, answer={:?}", img_turn.as_ref().map(|r| r.stop_reason.clone()).map_err(|e| e.to_string()), answer.chars().take(40).collect::<String>()),
                &mut summary,
                &mut fail,
            );

            // --- resume replay ---
            seen.borrow().replay_text.borrow_mut().clear();
            let loaded = acp_send(
                acp::LoadSessionRequest::new(sid.clone(), std::path::PathBuf::from(WORKDIR)),
                &client.tx,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            let replay = seen.borrow().replay_text.borrow().clone();
            check(
                "resume-load-ok",
                loaded.is_ok(),
                format!("result={:?}", loaded.as_ref().map(|_| "ok").map_err(|e| e.to_string())),
                &mut summary,
                &mut fail,
            );
            check(
                "resume-replay-has-marker",
                replay.contains(MARKER),
                format!("replay chars={} marker_found={}", replay.len(), replay.contains(MARKER)),
                &mut summary,
                &mut fail,
            );

            // --- session delete ---
            let delete_params = serde_json::json!({ "sessionId": sid.0.as_ref(), "cwd": WORKDIR });
            let delete = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/delete",
                    serde_json::value::to_raw_value(&delete_params).expect("serialize delete params").into(),
                ),
                &client.tx,
            )
            .await;
            let delete_detail = match &delete {
                Ok(resp) => {
                    let v: serde_json::Value =
                        serde_json::from_str(resp.0.get()).unwrap_or(serde_json::json!({}));
                    format!("resp={v}")
                }
                Err(e) => format!("err={e}"),
            };
            check(
                "session-delete-ack",
                delete.is_ok(),
                delete_detail,
                &mut summary,
                &mut fail,
            );

            println!("\n=== verify_0169 summary ===");
            for line in &summary {
                println!("{line}");
            }
            println!("=== {} checks failed ===", fail);
            if fail == 0 {
                Ok::<(), anyhow::Error>(())
            } else {
                Err(anyhow::anyhow!("{fail} checks failed"))
            }
        })
        .await?;
    std::process::exit(0)
}
