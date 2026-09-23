//! Alignment probes: v4/conversation/usage vs SQL meter, v4 deleteSession,
//! slash-command parsing via plain session/send.
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};
use serde_json::json;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            std::fs::create_dir_all("/tmp/probealign")?;
            let (client, agent_channel) = acp_channels();
            let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
            let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
            let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
            tokio::task::spawn_local(gw_rx.run());
            let text = Rc::new(RefCell::new(String::new()));
            {
                let text = text.clone();
                let mut rx = client.rx;
                tokio::task::spawn_local(async move {
                    while let Some(m) = rx.recv().await {
                        match m {
                            AcpClientMessage::ExtMethod(n) => {
                                let raw = serde_json::value::to_raw_value(&json!({"outcome": "approved"})).unwrap();
                                let _ = n.response_tx.send(Ok(acp::ExtResponse::new(raw.into())));
                            }
                            AcpClientMessage::RequestPermission(p) => {
                                let opt = p.options.first().map(|o| o.option_id.clone());
                                if let Some(id) = opt {
                                    let tx = p.response_tx;
                                    let _ = tx.send(Ok(acp::RequestPermissionResponse::new(
                                        acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(id)),
                                    )));
                                }
                            }
                            AcpClientMessage::SessionNotification(u) => {
                                if let acp::SessionUpdate::AgentMessageChunk(c) = &u.update {
                                    if let acp::ContentBlock::Text(t) = &c.content {
                                        text.borrow_mut().push_str(&t.text);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                });
            }
            let _ = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
            let session = acp_send(
                acp::NewSessionRequest::new(std::path::PathBuf::from("/tmp/probealign")),
                &client.tx,
            ).await?;
            let sid = session.session_id.clone();
            let _ = acp_send(
                acp::PromptRequest::new(sid.clone(), vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "从1数到5，每行一个。".to_string(),
                ))]),
                &client.tx,
            ).await?;

            // (A) usage: v4/conversation/usage vs our SQL x.ai/session/usage
            let v4u = acp_send(
                acp::ExtRequest::new(
                    "x.ai/v4/usage",
                    serde_json::value::to_raw_value(&json!({"sessionId": sid.0})).unwrap().into(),
                ),
                &client.tx,
            ).await;
            eprintln!("[probe] (A) v4 usage: {}", match &v4u {
                Ok(r) => { let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(json!({})); v.to_string().chars().take(300).collect::<String>() }
                Err(e) => format!("ERR {e}"),
            });

            let sql_usage = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/usage",
                    serde_json::value::to_raw_value(&json!({"sessionId": sid.0})).unwrap().into(),
                ),
                &client.tx,
            ).await;
            eprintln!("[probe] SQL usage: {}", match &sql_usage {
                Ok(r) => { let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(json!({})); v.to_string().chars().take(260).collect::<String>() }
                Err(e) => format!("ERR {e}"),
            });

            // (B) slash command as plain text: /goal
            text.borrow_mut().clear();
            let goal = acp_send(
                acp::PromptRequest::new(sid.clone(), vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "/goal 保持简洁".to_string(),
                ))]),
                &client.tx,
            ).await;
            for _ in 0..15 {
                if !text.borrow().trim().is_empty() { break; }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
            eprintln!("[probe] (B) /goal as plain text -> {:?}; reply: {:?}",
                goal.as_ref().map(|_| "EndTurn").map_err(|e| e.to_string()),
                text.borrow().chars().take(60).collect::<String>());
            // goal show via sessionGoal RPC through the agent? use session/goal passthrough absent; check read
            let read = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/info",
                    serde_json::value::to_raw_value(&json!({"sessionId": sid.0})).unwrap().into(),
                ),
                &client.tx,
            ).await;
            let _ = read;

            // (C) v4 deleteSession on a scratch session
            let scratch = acp_send(
                acp::NewSessionRequest::new(std::path::PathBuf::from("/tmp/probealign")),
                &client.tx,
            ).await?;
            let scratch_sid = scratch.session_id.clone();
            let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
            let del = acp_send(
                acp::ExtRequest::new(
                    "x.ai/v4/command",
                    serde_json::value::to_raw_value(&json!({
                        "commandId": format!("probe-del-{now_ms}"),
                        "clientId": "zgrok-probe",
                        "sessionId": scratch_sid.0,
                        "type": "deleteSession",
                        "payload": {},
                        "issuedAt": now_ms
                    })).unwrap().into(),
                ),
                &client.tx,
            ).await;
            eprintln!("[probe] (C) v4 deleteSession -> {}", match &del {
                Ok(r) => { let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(json!({})); v.to_string().chars().take(200).collect::<String>() }
                Err(e) => format!("ERR {e}"),
            });
            // is it really gone? (db check happens outside)
            let _ = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/delete",
                    serde_json::value::to_raw_value(&json!({"sessionId": sid.0, "cwd": "/tmp/probealign"})).unwrap().into(),
                ),
                &client.tx,
            ).await;
            Ok::<(), anyhow::Error>(())
        })
        .await?;
    Ok(())
}

