//! Truncating-rewind verification via v4 editUserQuery:
//! teach two secrets, rewind to the first message, then ask about the
//! second secret — the model must NOT know it (everything after the
//! target was truncated).
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
            std::fs::create_dir_all("/tmp/proberewind")?;
            let (client, agent_channel) = acp_channels();
            let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
            let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
            let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
            tokio::task::spawn_local(gw_rx.run());
            let text_seen = Rc::new(RefCell::new(String::new()));
            {
                let text_seen = text_seen.clone();
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
                                    let tx = p.response_tx;
                                    let _ = tx.send(Ok(acp::RequestPermissionResponse::new(
                                        acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(id)),
                                    )));
                                }
                            }
                            AcpClientMessage::SessionNotification(u) => {
                                if let acp::SessionUpdate::AgentMessageChunk(c) = &u.update {
                                    if let acp::ContentBlock::Text(t) = &c.content {
                                        text_seen.borrow_mut().push_str(&t.text);
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
                acp::NewSessionRequest::new(std::path::PathBuf::from("/tmp/proberewind")),
                &client.tx,
            ).await?;
            let sid = session.session_id.clone();
            println!("[probe] session {}", sid.0);

            for text in [
                "请记住暗号：紫色河马。只回复：记住了。",
                "再记住第二个暗号：金色猎鹰。只回复：记住了。",
            ] {
                let r = acp_send(
                    acp::PromptRequest::new(sid.clone(), vec![acp::ContentBlock::Text(acp::TextContent::new(text.to_string()))]),
                    &client.tx,
                ).await;
                println!("[probe] teach turn -> {:?}", r.as_ref().map(|_| "EndTurn").map_err(|e| e.to_string()));
            }

            // rewind points (index 0 = first user message)
            let points = acp_send(
                acp::ExtRequest::new(
                    "x.ai/rewind/points",
                    serde_json::value::to_raw_value(&serde_json::json!({"sessionId": sid.0})).unwrap().into(),
                ),
                &client.tx,
            ).await;
            if let Ok(r) = &points {
                let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(json!({}));
                println!("[probe] points: {}", v.to_string().chars().take(300).collect::<String>());
            }

            // REWIND to the first message
            let rw = acp_send(
                acp::ExtRequest::new(
                    "x.ai/rewind/execute",
                    serde_json::value::to_raw_value(&serde_json::json!({
                        "sessionId": sid.0, "targetPromptIndex": 1, "force": true, "mode": "conversation_only"
                    })).unwrap().into(),
                ),
                &client.tx,
            ).await;
            println!("[probe] rewind -> {}", match &rw {
                Ok(r) => { let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or(json!({})); v.to_string().chars().take(200).collect::<String>() }
                Err(e) => format!("ERR {e}"),
            });
            // let the edit rerun settle
            tokio::time::sleep(std::time::Duration::from_secs(25)).await;

            // The decisive question: does the model still know secret #2?
            text_seen.borrow_mut().clear();
            let check = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "你还记得第二个暗号（金色猎鹰）吗？只回答：记得 或 不记得。".to_string(),
                    ))],
                ),
                &client.tx,
            ).await;
            println!("[probe] recall turn -> {:?}", check.as_ref().map(|_| "EndTurn").map_err(|e| e.to_string()));
            // drain
            for _ in 0..20 {
                if text_seen.borrow().contains("记得") { break; }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
            let answer = text_seen.borrow().clone();
            let forgot = answer.contains("不记得") && !answer.contains("记得\n不记得");
            println!("\n[probe] answer: {:?}", answer.chars().take(60).collect::<String>());
            println!("[probe] SECOND SECRET FORGOTTEN (truncation worked): {}", answer.contains("不记得"));

            let _ = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/delete",
                    serde_json::value::to_raw_value(&serde_json::json!({"sessionId": sid.0, "cwd": "/tmp/proberewind"})).unwrap().into(),
                ),
                &client.tx,
            ).await;
            let _ = forgot;
            Ok::<(), anyhow::Error>(())
        })
        .await?;
    Ok(())
}
