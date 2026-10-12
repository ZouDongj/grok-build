//! /fork e2e: fork via v4 forkAssistant returns a NEW session id, the
//! original session still resumes cleanly afterward, and the fork carries
//! the conversation (marker recallable).
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probefork";
const MARKER: &str = "FORK标记7391";

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

        let replies: Rc<RefCell<String>> = Rc::new(RefCell::new(String::new()));
        {
            let replies = replies.clone();
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
                            if let acp::SessionUpdate::AgentMessageChunk(c) = &u.update {
                                if let acp::ContentBlock::Text(t) = &c.content {
                                    replies.borrow_mut().push_str(&t.text);
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

        let ask = |text: String| {
            let sid = sid.clone();
            let tx = client.tx.clone();
            async move {
                acp_send(
                    acp::PromptRequest::new(
                        sid,
                        vec![acp::ContentBlock::Text(acp::TextContent::new(text))],
                    ),
                    &tx,
                )
                .await
            }
        };

        // A turn with a marker.
        let _ = ask(format!("请原样重复这个词，不要说别的：{MARKER}")).await?;
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        println!("[probe] marker turn done");

        // Fork via the ext.
        let fork_resp = acp_send(
            acp::ExtRequest::new(
                "x.ai/session/fork",
                serde_json::value::to_raw_value(&serde_json::json!({
                    "sourceSessionId": sid.0,
                }))
                .unwrap()
                .into(),
            ),
            &client.tx,
        )
        .await;
        let fork_sid = fork_resp
            .ok()
            .and_then(|r| {
                let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or_default();
                v.pointer("/result/newSessionId")
                    .and_then(|x| x.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();
        println!("[probe] fork -> {fork_sid}");

        // Original session must still resume cleanly.
        let resumed = acp_send(
            acp::LoadSessionRequest::new(sid.clone(), std::path::PathBuf::from(WORKDIR)),
            &client.tx,
        )
        .await;
        println!("[probe] original resume ok: {}", resumed.is_ok());

        // The fork carries the conversation: resume it and recall the marker.
        let mut recall_ok = false;
        if fork_sid.starts_with("sess_") {
            let fork_acp = acp::SessionId::new(fork_sid.clone());
            let loaded = acp_send(
                acp::LoadSessionRequest::new(fork_acp.clone(), std::path::PathBuf::from(WORKDIR)),
                &client.tx,
            )
            .await;
            println!("[probe] fork resume ok: {}", loaded.is_ok());
            replies.borrow_mut().clear();
            let _ = acp_send(
                acp::PromptRequest::new(
                    fork_acp,
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "我刚才让你重复的词是什么？只回答那个词。".to_string(),
                    ))],
                ),
                &client.tx,
            )
            .await;
            tokio::time::sleep(std::time::Duration::from_millis(2500)).await;
            let text = replies.borrow().clone();
            println!("[probe] fork recall reply: {}", text.chars().take(60).collect::<String>());
            recall_ok = text.contains(MARKER);
        }

        let v = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        v("fork-returns-new-session", fork_sid.starts_with("sess_") && fork_sid != sid.0.as_ref());
        v("original-session-still-resumes", resumed.is_ok());
        v("fork-carries-conversation", recall_ok);

        // Cleanup both sessions.
        for s in [sid.0.as_ref().to_string(), fork_sid.clone()] {
            if s.starts_with("sess_") {
                let _ = acp_send(
                    acp::ExtRequest::new(
                        "x.ai/session/delete",
                        serde_json::value::to_raw_value(&serde_json::json!({
                            "sessionId": s, "cwd": WORKDIR,
                        }))
                        .unwrap()
                        .into(),
                    ),
                    &client.tx,
                )
                .await;
            }
        }
        Ok::<(), anyhow::Error>(())
    })
    .await?;
    Ok(())
}
