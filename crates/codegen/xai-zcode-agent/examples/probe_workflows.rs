//! E2E: launch a MINIMAL dwf workflow through a probe session (the model
//! writes and starts it via CreateWorkflow), then verify zgrok's
//! workflow_updated rail carries real actors/phases/status to the pager.
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probedwf";

#[derive(Clone, Debug)]
struct WfSnap {
    t_ms: u128,
    status: String,
    agents_used: u64,
    active_agents: u32,
    phases: usize,
    agents: usize,
    current_phase: Option<String>,
    last_event: Option<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local.run_until(async move {
        std::fs::create_dir_all(WORKDIR)?;
        let (client, agent_channel) = acp_channels();
        let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
        let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
        let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
        tokio::task::spawn_local(gw_rx.run());

        let t0 = std::time::Instant::now();
        let snaps: Rc<RefCell<Vec<WfSnap>>> = Rc::new(RefCell::new(Vec::new()));
        let tool_names: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let subagent_events: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let perm_titles: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let meta: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let agent_text: Rc<RefCell<String>> = Rc::new(RefCell::new(String::new()));
        {
            let perm_titles = perm_titles.clone();
            let meta = meta.clone();
            let agent_text = agent_text.clone();
            let snaps = snaps.clone();
            let tool_names = tool_names.clone();
            let subagent_events = subagent_events.clone();
            let t0 = t0.clone();
            let mut rx = client.rx;
            tokio::task::spawn_local(async move {
                while let Some(m) = rx.recv().await {
                    match m {
                        AcpClientMessage::ExtMethod(n) => {
                            let raw = serde_json::value::to_raw_value(&serde_json::json!({
                                "outcome": "approved"
                            }))
                            .unwrap();
                            let _ = n.response_tx.send(Ok(acp::ExtResponse::new(raw.into())));
                        }
                        AcpClientMessage::RequestPermission(p) => {
                            perm_titles.borrow_mut().push(p.tool_call.fields.title.clone().unwrap_or_default());
                            let pick = p
                                .options
                                .iter()
                                .find(|o| matches!(o.kind, acp::PermissionOptionKind::AllowOnce))
                                .or_else(|| p.options.first())
                                .cloned();
                            if let Some(o) = pick {
                                let _ = p.response_tx.send(Ok(
                                    acp::RequestPermissionResponse::new(
                                        acp::RequestPermissionOutcome::Selected(
                                            acp::SelectedPermissionOutcome::new(o.option_id.clone()),
                                        ),
                                    ),
                                ));
                            }
                        }
                        AcpClientMessage::SessionNotification(u) => match &u.update {
                            acp::SessionUpdate::ToolCall(tc) => {
                                tool_names.borrow_mut().push(tc.title.clone());
                            }
                            acp::SessionUpdate::AgentMessageChunk(c) => {
                                if let acp::ContentBlock::Text(t) = &c.content {
                                    agent_text.borrow_mut().push_str(&t.text);
                                }
                            }
                            acp::SessionUpdate::AvailableCommandsUpdate(_) => {}
                            _ => {}
                        },
                        AcpClientMessage::ExtNotification(n) => {
                            let v: serde_json::Value =
                                serde_json::from_str(n.params.get()).unwrap_or(serde_json::json!({}));
                            let upd = &v["update"];
                            if n.method.as_ref() == "x.ai/session_notification" {
                                match upd["sessionUpdate"].as_str() {
                                    Some("workflow_updated") => {
                                        let name = upd["name"].as_str().unwrap_or("").to_string();
                                        let model = upd["agents"].as_array()
                                            .and_then(|a| a.first())
                                            .and_then(|g| g.get("model"))
                                            .and_then(|m| m.as_str())
                                            .map(String::from);
                                        let desc = upd["agents"].as_array()
                                            .and_then(|a| a.first())
                                            .and_then(|g| g.get("description"))
                                            .and_then(|m| m.as_str())
                                            .map(String::from);
                                        if !name.is_empty() && name != "Workflow" {
                                            meta.borrow_mut().push(format!("name={name}"));
                                        }
                                        if let Some(m) = model {
                                            meta.borrow_mut().push(format!("model={m}"));
                                        }
                                        if let Some(d) = desc {
                                            meta.borrow_mut().push(format!("desc={}", d.chars().take(40).collect::<String>()));
                                        }
                                        let steps = upd["steps_settled"].as_u64()
                                            .zip(upd["steps_observed"].as_u64())
                                            .map(|(a, b)| format!("{a}/{b}"));
                                        if let Some(st) = steps {
                                            meta.borrow_mut().push(format!("steps={st}"));
                                        }
                                        if let Some(t) = upd["run_tokens"].as_u64() {
                                            meta.borrow_mut().push(format!("tokens={t}"));
                                        }
                                        if let Some(m) = upd["subagent_model"].as_str() {
                                            meta.borrow_mut().push(format!("submodel={m}"));
                                        }
                                        snaps.borrow_mut().push(WfSnap {
                                        t_ms: t0.elapsed().as_millis(),
                                        status: upd["status"].as_str().unwrap_or("").into(),
                                        agents_used: upd["agents_used"].as_u64().unwrap_or(0),
                                        active_agents: upd["active_agents"].as_u64().unwrap_or(0) as u32,
                                        phases: upd["phases"].as_array().map(|a| a.len()).unwrap_or(0),
                                        agents: upd["agents"].as_array().map(|a| a.len()).unwrap_or(0),
                                        current_phase: upd["current_phase"].as_str().map(String::from),
                                            last_event: upd["last_event"].as_str().map(String::from),
                                        })
                                    }
                                    Some("subagent_spawned") | Some("subagent_finished") => {
                                        subagent_events.borrow_mut().push(
                                            format!(
                                                "{}@{}",
                                                upd["sessionUpdate"].as_str().unwrap_or("?"),
                                                upd.get("description").and_then(|d| d.as_str()).unwrap_or("")
                                            ),
                                        );
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
            });
        }

        let _init =
            acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
        let session = acp_send(
            acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)),
            &client.tx,
        )
        .await?;
        let sid = session.session_id.clone();
        println!("[probe] session {}", sid.0);

        let resp = acp_send(
            acp::PromptRequest::new(
                sid.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "/workflow 单个 actor 回答 1+1 等于几（只回答数字），并用 artifact() 把答案写成一个 markdown 产物交付".to_string(),
                ))],
            ),
            &client.tx,
        )
        .await?;
        println!("[probe] launch turn: {:?}", resp.stop_reason);

        // Watch the workflow rail until a terminal status or timeout.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(420);
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            let last = snaps.borrow().last().cloned();
            if let Some(s) = last {
                println!(
                    "[probe t+{:>6}ms] {} agents={} active={} phases={:?} phase={:?} event={:?}",
                    s.t_ms, s.status, s.agents, s.active_agents, s.phases, s.current_phase, s.last_event
                );
                if s.status == "complete" || s.status == "failed" || s.status == "cancelled" {
                    break;
                }
            }
            if std::time::Instant::now() > deadline {
                println!("[probe] timeout waiting for terminal workflow status");
                break;
            }
        }
        let all = snaps.borrow().clone();
        println!("\n=== workflow_updated timeline ({} snaps) ===", all.len());
        for s in &all {
            println!(
                "t+{:>6}ms {:>10} used={} active={} phases={} agents={} cur={:?} ev={:?}",
                s.t_ms, s.status, s.agents_used, s.active_agents, s.phases, s.agents, s.current_phase, s.last_event
            );
        }
        println!("tools seen: {:?}", tool_names.borrow());
        println!("agent reply: {}", agent_text.borrow().chars().take(300).collect::<String>());
        println!("subagent events: {:?}", subagent_events.borrow());
        println!("meta: {:?}", meta.borrow());
        println!("permission titles: {:?}", perm_titles.borrow());

        // /workflow artifacts: list + preview (second command turn)
        let replies_len = || agent_text.borrow().len();
        let agent_text_after = |from: usize| agent_text.borrow()[from..].to_string();
        {
            let from = replies_len();
            let _ = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "/workflow artifacts".to_string(),
                    ))],
                ),
                &client.tx,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_millis(3000)).await;
            let artifacts_reply = agent_text_after(from);
            println!("[probe] artifacts reply:\n{}", artifacts_reply.chars().take(400).collect::<String>());
        }
        {
            let from = replies_len();
            let _ = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "/workflow artifacts 1".to_string(),
                    ))],
                ),
                &client.tx,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_millis(3000)).await;
            let preview_reply = agent_text_after(from);
            println!("[probe] artifact preview:\n{}", preview_reply.chars().take(300).collect::<String>());
        }

        let verdict = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        verdict(
            "workflow-launched",
            tool_names.borrow().iter().any(|t| t.to_lowercase().contains("workflow")),
        );
        verdict(
            "workflow-updates-flow",
            !all.is_empty(),
        );
        verdict(
            "workflow-actors-visible",
            all.iter().any(|s| s.agents > 0 || s.agents_used > 0),
        );
        verdict(
            "workflow-phases-visible",
            all.iter().any(|s| s.phases > 0),
        );
        verdict(
            "workflow-terminal-status",
            all.iter().any(|s| matches!(s.status.as_str(), "complete" | "failed" | "cancelled")),
        );
        verdict(
            "run-has-real-name",
            meta.borrow().iter().any(|m| m.starts_with("name=") && !m.contains("name=Workflow")),
        );
        verdict(
            "agent-has-model",
            meta.borrow().iter().any(|m| m.starts_with("model=") && m != "model=None"),
        );
        verdict(
            "actor-subagent-events",
            !subagent_events.borrow().is_empty(),
        );
        verdict(
            "summary-fields-flow",
            meta.borrow().iter().any(|m| m.starts_with("steps="))
                && meta.borrow().iter().any(|m| m.starts_with("submodel=")),
        );
        verdict(
            "workflow-approval-asked",
            perm_titles
                .borrow()
                .iter()
                .any(|t| t.to_lowercase().contains("workflow") || t.contains("工作流")),
        );

        let _ = acp_send(
            acp::ExtRequest::new(
                "x.ai/session/delete",
                serde_json::value::to_raw_value(&serde_json::json!({
                    "sessionId": sid.0, "cwd": WORKDIR,
                }))
                .unwrap()
                .into(),
            ),
            &client.tx,
        )
        .await;
        Ok::<(), anyhow::Error>(())
    }).await?;
    Ok(())
}
