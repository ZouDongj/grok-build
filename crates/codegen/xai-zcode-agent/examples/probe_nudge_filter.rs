//! Verify synthetic-nudge filtering on a REAL historical session (read-only
//! resume): the replayed user stream must contain the real prompts but none
//! of the kernel's todo-reminder nudges. The session is NOT modified or
//! deleted — resume only.
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local.run_until(async move {
        let sid = std::fs::read_to_string("/tmp/nudge_test_sid")?.trim().to_string();
        let path = std::fs::read_to_string("/tmp/nudge_test_path")?.trim().to_string();
        println!("[probe] resuming REAL session {sid} in {path} (read-only)");

        let (client, agent_channel) = acp_channels();
        let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
        let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
        let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
        tokio::task::spawn_local(gw_rx.run());

        let user_text: Rc<RefCell<String>> = Rc::new(RefCell::new(String::new()));
        {
            let user_text = user_text.clone();
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
                        AcpClientMessage::SessionNotification(u) => {
                            if let acp::SessionUpdate::UserMessageChunk(c) = &u.update {
                                if let acp::ContentBlock::Text(t) = &c.content {
                                    user_text.borrow_mut().push_str(&t.text);
                                    user_text.borrow_mut().push('\n');
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
        let resp = acp_send(
            acp::LoadSessionRequest::new(
                acp::SessionId::new(sid.clone()),
                std::path::PathBuf::from(path.clone()),
            ),
            &client.tx,
        )
        .await?;
        println!("[probe] load ok: {}", !format!("{resp:?}").contains("Err"));
        tokio::time::sleep(std::time::Duration::from_millis(2500)).await;

        let text = user_text.borrow().clone();
        let nudge_hits = text.matches("TodoWrite tool hasn't been used").count();
        let real_chars = text.chars().count();
        println!(
            "[probe] replayed user stream: {real_chars} chars, nudge occurrences: {nudge_hits}"
        );
        println!("[probe] first 200 chars: {}", text.chars().take(200).collect::<String>());

        let verdict =
            |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        verdict("nudges-hidden-in-replay", nudge_hits == 0);
        verdict("real-user-text-still-replayed", real_chars > 0);

        // NO delete, NO prompt — the user's session stays untouched.
        Ok::<(), anyhow::Error>(())
    }).await?;
    Ok(())
}
