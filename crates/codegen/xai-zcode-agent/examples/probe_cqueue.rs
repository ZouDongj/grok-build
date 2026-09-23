//! Reproduce the compact + v4-queue wedge with full stderr: does the pump
//! panic, does the compact turn complete, does the queued item drain?
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
            std::fs::create_dir_all("/tmp/probecq")?;
            let (client, agent_channel) = acp_channels();
            let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
            let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
            let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
            tokio::task::spawn_local(gw_rx.run());
            let seen = Rc::new(RefCell::new(String::new()));
            {
                let seen = seen.clone();
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
                                        let txt = t.text.clone();
                                        eprintln!("[probe] chunk: {:?}", &txt[..txt.len().min(30)]);
                                        seen.borrow_mut().push_str(&txt);
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
                acp::NewSessionRequest::new(std::path::PathBuf::from("/tmp/probecq")),
                &client.tx,
            ).await?;
            let sid = session.session_id.clone();
            // small context
            let _ = acp_send(
                acp::PromptRequest::new(sid.clone(), vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "从1数到10，每行一个。".to_string(),
                ))]),
                &client.tx,
            ).await?;
            eprintln!("[probe] context turn done");

            // compact ext (waits for real completion)
            let cx = {
                let tx = client.tx.clone();
                let sid = sid.clone();
                tokio::task::spawn_local(async move {
                    let t0 = std::time::Instant::now();
                    let r = acp_send(
                        acp::ExtRequest::new(
                            "x.ai/compact_conversation",
                            serde_json::value::to_raw_value(&json!({"sessionId": sid.0})).unwrap().into(),
                        ),
                        &tx,
                    ).await;
                    eprintln!("[probe] compact ext returned in {:?} -> {:?}", t0.elapsed(),
                        r.as_ref().map(|_| "ok".to_string()).map_err(|e| e.to_string()));
                })
            };
            tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
            // THE QUEUED PROMPT via v4 (kernel queue)
            let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
            let q = acp_send(
                acp::ExtRequest::new(
                    "x.ai/v4/command",
                    serde_json::value::to_raw_value(&json!({
                        "commandId": format!("probe-cq-{now_ms}"),
                        "clientId": "zgrok-probe",
                        "sessionId": sid.0,
                        "type": "sendText",
                        "payload": {"text": "1+1等于几？只回答阿拉伯数字。", "requestedDelivery": "queue"},
                        "issuedAt": now_ms
                    })).unwrap().into(),
                ),
                &client.tx,
            ).await;
            eprintln!("[probe] v4 queue send -> {}", match &q {
                Ok(r) => { let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(json!({})); v.to_string().chars().take(200).collect::<String>() }
                Err(e) => format!("ERR {e}"),
            });

            // watch for the answer for 120s
            let t0 = std::time::Instant::now();
            while t0.elapsed() < std::time::Duration::from_secs(120) {
                if seen.borrow().contains('2') { break; }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
            eprintln!("[probe] queued item answered: {}", seen.borrow().contains('2'));
            let _ = cx.await;
            let _ = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/delete",
                    serde_json::value::to_raw_value(&json!({"sessionId": sid.0, "cwd": "/tmp/probecq"})).unwrap().into(),
                ),
                &client.tx,
            ).await;
            Ok::<(), anyhow::Error>(())
        })
        .await?;
    Ok(())
}
