//! /quota live check: windows, percentages, reset counts from the real APIs.
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probequota";

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
        let text: Rc<RefCell<String>> = Rc::new(RefCell::new(String::new()));
        {
            let text = text.clone();
            let mut rx = client.rx;
            tokio::task::spawn_local(async move {
                while let Some(m) = rx.recv().await {
                    match m {
                        AcpClientMessage::ExtMethod(n) => {
                            let raw = serde_json::value::to_raw_value(&serde_json::json!({"outcome":"approved"})).unwrap();
                            let _ = n.response_tx.send(Ok(acp::ExtResponse::new(raw.into())));
                        }
                        AcpClientMessage::RequestPermission(p) => {
                            let pick = p.options.iter().find(|o| matches!(o.kind, acp::PermissionOptionKind::AllowOnce)).or_else(|| p.options.first()).cloned();
                            if let Some(o) = pick {
                                let _ = p.response_tx.send(Ok(acp::RequestPermissionResponse::new(
                                    acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(o.option_id.clone())),
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
        let _init = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
        let session = acp_send(acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)), &client.tx).await?;
        let sid = session.session_id.clone();
        let _ = acp_send(
            acp::PromptRequest::new(sid.clone(), vec![acp::ContentBlock::Text(acp::TextContent::new("/quota".to_string()))]),
            &client.tx,
        ).await?;
        std::thread::sleep(std::time::Duration::from_millis(4000));
        let t = text.borrow().clone();
        println!("--- /quota reply ---\n{t}\n--------------------");
        let v = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok {"PASS"} else {"FAIL"});
        v("quota-windows-shown", t.contains("5 小时窗口") && t.contains("7 天窗口"));
        v("quota-percentages", t.contains('%'));
        v("reset-counts-shown", t.contains("可用重置"));
        let _ = acp_send(
            acp::ExtRequest::new(
                "x.ai/session/delete",
                serde_json::value::to_raw_value(&serde_json::json!({"sessionId": sid.0, "cwd": WORKDIR})).unwrap().into(),
            ),
            &client.tx,
        ).await;
        Ok::<(), anyhow::Error>(())
    }).await?;
    Ok(())
}
