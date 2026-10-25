//! Start Plan e2e: the combined picker catalog advertises the start-plan
//! models with the `@start` marker; selecting one routes session/setModel to
//! account:bigmodel-start-plan, the runtime-headers request is answered with
//! the zcode JWT pair, and a REAL turn completes through
//! zcode.z.ai/api/v1/zcode-plan/anthropic. Afterwards the coding-plan
//! preference is restored.
//!
//! Usage: cargo run -p xai-zcode-agent --example probe_start_plan

use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probestart";

#[derive(Default)]
struct Seen {
    model_updates: RefCell<Vec<serde_json::Value>>,
    current_display: RefCell<Option<String>>,
    turn_text: RefCell<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            std::fs::create_dir_all(WORKDIR)?;
            let (client, agent_channel) = acp_channels();
            let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
            let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
            let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
            tokio::task::spawn_local(gw_rx.run());
            let seen = Rc::new(Seen::default());
            {
                let seen = seen.clone();
                let mut rx = client.rx;
                tokio::task::spawn_local(async move {
                    while let Some(m) = rx.recv().await {
                        match m {
                            AcpClientMessage::RequestPermission(p) => {
                                let pick = p
                                    .options
                                    .iter()
                                    .find(|o| matches!(o.kind, acp::PermissionOptionKind::AllowOnce))
                                    .or_else(|| p.options.first())
                                    .cloned();
                                if let Some(o) = pick {
                                    let _ = p.response_tx.send(Ok(acp::RequestPermissionResponse::new(
                                        acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(o.option_id.clone())),
                                    )));
                                }
                            }
                            AcpClientMessage::ExtNotification(n) => {
                                if n.method.as_ref() == "x.ai/models/update" {
                                    let v: serde_json::Value =
                                        serde_json::from_str(n.params.get()).unwrap_or_default();
                                    if let Some(cur) = v.get("currentModelId").and_then(|c| c.as_str()) {
                                        *seen.current_display.borrow_mut() = Some(cur.to_string());
                                    }
                                    seen.model_updates.borrow_mut().push(v);
                                }
                            }
                            AcpClientMessage::SessionNotification(u) => {
                                if let acp::SessionUpdate::AgentMessageChunk(c) = &u.update {
                                    if let acp::ContentBlock::Text(t) = &c.content {
                                        seen.turn_text.borrow_mut().push_str(&t.text);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                });
            }

            let _init = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
            let s = acp_send(acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)), &client.tx)
                .await?
                .session_id;

            // Wait for the combined catalog push.
            for _ in 0..40 {
                if !seen.model_updates.borrow().is_empty() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
            let catalog_ids: Vec<String> = seen
                .model_updates
                .borrow()
                .last()
                .and_then(|v| {
                    v.pointer("/availableModels")
                        .and_then(|a| a.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|m| m.get("modelId").and_then(|x| x.as_str()))
                                .map(str::to_string)
                                .collect()
                        })
                })
                .unwrap_or_default();
            println!("[probe] picker catalog: {catalog_ids:?}");

            // One coding-plan turn first (FK safety for later writes), then switch.
            let coding_turn = acp_send(
                acp::PromptRequest::new(
                    s.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "1+1等于几？只回答阿拉伯数字".to_string(),
                    ))],
                ),
                &client.tx,
            )
            .await;
            let coding_answer = seen.turn_text.borrow().clone();
            println!(
                "[probe] coding-plan turn: stop={:?} answer={coding_answer:?}",
                coding_turn.as_ref().ok().map(|r| r.stop_reason)
            );
            seen.turn_text.borrow_mut().clear();

            // Switch to the start plan (registry-listed model).
            let switched = acp_send(
                acp::SetSessionModelRequest::new(s.clone(), acp::ModelId::new(std::sync::Arc::from("GLM-5.3-Flash@start"))),
                &client.tx,
            )
            .await;
            println!("[probe] setSessionModel(GLM-5.3-Flash@start): accepted={}", switched.is_ok());
            tokio::time::sleep(std::time::Duration::from_millis(1200)).await;
            let after_switch = seen.current_display.borrow().clone();
            println!("[probe] current display id after switch: {after_switch:?}");

            // Real turn through the start-plan endpoint.
            let start_turn = acp_send(
                acp::PromptRequest::new(
                    s.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "2+2等于几？只回答阿拉伯数字".to_string(),
                    ))],
                ),
                &client.tx,
            )
            .await;
            let start_answer = seen.turn_text.borrow().clone();
            println!(
                "[probe] start-plan turn: stop={:?} answer={start_answer:?}",
                start_turn.as_ref().ok().map(|r| r.stop_reason)
            );

            // GLM-5.3 on the start plan (appended id, not registry-listed).
            let switch53 = acp_send(
                acp::SetSessionModelRequest::new(s.clone(), acp::ModelId::new(std::sync::Arc::from("GLM-5.3@start"))),
                &client.tx,
            )
            .await;
            seen.turn_text.borrow_mut().clear();
            let turn53 = acp_send(
                acp::PromptRequest::new(
                    s.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "3+3等于几？只回答阿拉伯数字".to_string(),
                    ))],
                ),
                &client.tx,
            )
            .await;
            let answer53 = seen.turn_text.borrow().clone();
            println!(
                "[probe] GLM-5.3@start: switch={} turn={:?} answer={answer53:?}",
                switch53.is_ok(),
                turn53.as_ref().ok().map(|r| r.stop_reason)
            );
            seen.turn_text.borrow_mut().clear();

            // Restore the coding-plan preference (bare id routes back).
            let _ = acp_send(
                acp::SetSessionModelRequest::new(s.clone(), acp::ModelId::new(std::sync::Arc::from("GLM-5.3"))),
                &client.tx,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_millis(800)).await;
            let restored_pref = std::fs::read_to_string(pref_path().unwrap_or_default())
                .ok()
                .and_then(|r| serde_json::from_str::<serde_json::Value>(&r).ok())
                .map(|v| {
                    (
                        v.get("providerId").and_then(|p| p.as_str()).unwrap_or("").to_string(),
                        v.get("modelId").and_then(|m| m.as_str()).unwrap_or("").to_string(),
                    )
                });
            println!("[probe] restored preference: {restored_pref:?}");

            let _ = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/delete",
                    serde_json::value::to_raw_value(&serde_json::json!({"sessionId": s.0, "cwd": WORKDIR}))
                        .unwrap()
                        .into(),
                ),
                &client.tx,
            )
            .await;

            let v = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
            v(
                "catalog-offers-start-models",
                catalog_ids.iter().any(|id| id.ends_with("@start")),
            );
            v(
                "coding-plan-turn-answered",
                matches!(&coding_turn, Ok(r) if r.stop_reason == acp::StopReason::EndTurn) && !coding_answer.trim().is_empty(),
            );
            v("switch-accepted-current-is-start", after_switch.as_deref() == Some("GLM-5.3-Flash@start"));
            v(
                "start-plan-turn-answered",
                matches!(&start_turn, Ok(r) if r.stop_reason == acp::StopReason::EndTurn) && !start_answer.trim().is_empty(),
            );
            v(
                "glm53-on-start-plan-answered",
                matches!(&turn53, Ok(r) if r.stop_reason == acp::StopReason::EndTurn) && !answer53.trim().is_empty(),
            );
            v(
                "preference-restored-to-coding-plan",
                matches!(restored_pref.as_ref(), Some((p, m)) if p == "account:bigmodel-individual-coding-plan" && m == "GLM-5.3"),
            );

            Ok::<(), anyhow::Error>(())
        })
        .await?;
    Ok(())
}

fn pref_path() -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config")))
        .map(|c| c.join("zcode-tui").join("model.json"))
}
