//! Decisive probe: does the 3.14.1 release kernel accept v4/command
//! sendText with requestedDelivery "guide" against a legacy session —
//! i.e. is TRUE mid-turn steering reachable on our wire?
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

struct Seen {
    t0: std::time::Instant,
    timeline: RefCell<Vec<(u128, String)>>,
}
impl Default for Seen {
    fn default() -> Seen {
        Seen { t0: std::time::Instant::now(), timeline: RefCell::new(Vec::new()) }
    }
}

fn short(s: &str) -> String {
    let mut t = s.chars().take(50).collect::<String>();
    if s.chars().count() > 50 { t.push('…'); }
    t
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            std::fs::create_dir_all("/tmp/probev4")?;
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
                    while let Some(m) = rx.recv().await {
                        match m {
                            AcpClientMessage::ExtMethod(n) => {
                                let raw = serde_json::value::to_raw_value(&serde_json::json!({"outcome": "approved"})).unwrap();
                                let _ = n.response_tx.send(Ok(acp::ExtResponse::new(raw.into())));
                            }
                            AcpClientMessage::RequestPermission(p) => {
                                let opt = p.options.first().map(|o| o.option_id.clone());
                                if let Some(id) = opt {
                                    let resp_tx = p.response_tx;
                                    let _ = resp_tx.send(Ok(acp::RequestPermissionResponse::new(
                                        acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(id)),
                                    )));
                                }
                            }
                            AcpClientMessage::SessionNotification(u) => {
                                let t = seen.borrow().t0.elapsed().as_millis();
                                let desc = match &u.update {
                                    acp::SessionUpdate::AgentThoughtChunk(c) => {
                                        if let acp::ContentBlock::Text(x) = &c.content { format!("THINK {}", short(&x.text)) } else { "THINK".into() }
                                    }
                                    acp::SessionUpdate::AgentMessageChunk(c) => {
                                        if let acp::ContentBlock::Text(x) = &c.content { format!("TEXT {}", short(&x.text)) } else { "TEXT".into() }
                                    }
                                    acp::SessionUpdate::ToolCall(c) => format!("TOOL {}", &c.title),
                                    acp::SessionUpdate::UsageUpdate(u) => format!("USAGE {}/{}", u.used, u.size),
                                    _ => continue,
                                };
                                println!("[probe t+{t:>6}ms] {desc}");
                                seen.borrow_mut().timeline.borrow_mut().push((t, desc));
                            }
                            _ => {}
                        }
                    }
                });
            }
            let _ = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
            let session = acp_send(
                acp::NewSessionRequest::new(std::path::PathBuf::from("/tmp/probev4")),
                &client.tx,
            ).await?;
            let sid = session.session_id.clone();
            println!("[probe] session {}", sid.0);

            // 0. Introspection: v4/commands/query — does the kernel serve v4 at all?
            let q = acp_send(
                acp::ExtRequest::new(
                    "x.ai/v4/command",
                    serde_json::value::to_raw_value(&serde_json::json!({
                        "__method": "v4/commands/query"
                    })).unwrap().into(),
                ),
                &client.tx,
            ).await;
            println!("[probe] commands/query -> {:?}", q.as_ref().map(|_| "ok").map_err(|e| e.to_string()));

            // 1. Long legacy turn.
            let slow = {
                let tx = client.tx.clone();
                let sid = sid.clone();
                tokio::task::spawn_local(async move {
                    acp_send(
                        acp::PromptRequest::new(sid, vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "请从1一个一个慢慢数到80，每个数字一行，不许提前停止。".to_string(),
                        ))]),
                        &tx,
                    ).await
                })
            };
            tokio::time::sleep(std::time::Duration::from_secs(6)).await;

            // 2. THE EXPERIMENT: v4/command sendText with requestedDelivery guide.
            let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
            let guide = acp_send(
                acp::ExtRequest::new(
                    "x.ai/v4/command",
                    serde_json::value::to_raw_value(&serde_json::json!({
                        "commandId": format!("probe-v4-{now_ms}"),
                        "clientId": "zgrok-probe",
                        "sessionId": sid.0,
                        "type": "sendText",
                        "payload": {
                            "text": "转向测试：请立即停止数数，只回复四个字：转向成功",
                            "requestedDelivery": "guide"
                        },
                        "issuedAt": now_ms
                    })).unwrap().into(),
                ),
                &client.tx,
            ).await;
            let guide_desc = match &guide {
                Ok(r) => {
                    let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({}));
                    format!("ok ack={}", short(&v.to_string()))
                }
                Err(e) => format!("ERR {e}"),
            };
            println!("[probe] v4 sendText(guide) -> {guide_desc}");

            // 3. Wait for the turn to settle and see whether the steer landed.
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            let _ = slow.await;
            let tl = seen.borrow().timeline.borrow().clone();
            let steered: bool = tl.iter().any(|(_, d)| d.contains("转向成功"));
            println!("\n[probe] STEER LANDED: {steered}");
            // 4. cleanup
            let _ = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/delete",
                    serde_json::value::to_raw_value(&serde_json::json!({"sessionId": sid.0, "cwd": "/tmp/probev4"})).unwrap().into(),
                ),
                &client.tx,
            ).await;
            Ok::<(), anyhow::Error>(())
        })
        .await?;
    Ok(())
}
