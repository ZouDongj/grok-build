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
    plan_approval_ext: bool,
    plan_outcome_sent: Option<String>,
    permissions_answered: usize,
    mode_updates: Vec<String>,
    replay_text: RefCell<String>,
    current_text: RefCell<String>,
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

            // --- turn 1: establish a context marker for the resume check ---
            let t1 = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(format!(
                        "请记住这个暗号，之后我会考你：{MARKER}。只回复：记住了"
                    )))],
                ),
                &client.tx,
            )
            .await;
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
