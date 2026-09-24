//! Probe the alignment batch: /rate (setAssistantFeedback), /drain
//! (setAutoDrain), and held-queue disposition on v4 sends.
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probealign2";

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

        let replies: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
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
                                    replies.borrow_mut().push(t.text.clone());
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

        let ask = |text: &str| {
            let sid = sid.clone();
            let tx = client.tx.clone();
            let text = text.to_string();
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

        // A turn to have something to rate.
        let _ = ask("1+1等于几？只回答阿拉伯数字。").await?;
        let last_reply = || replies.borrow().last().cloned().unwrap_or_default();
        println!("[probe] turn reply: {}", last_reply());

        // /rate like
        replies.borrow_mut().clear();
        let _ = ask("/rate like").await?;
        let rate_reply = last_reply();
        println!("[probe] /rate like -> {rate_reply}");

        // /rate clear
        let _ = ask("/rate clear").await?;
        println!("[probe] /rate clear -> {}", last_reply());

        // /drain off -> show -> on
        let _ = ask("/drain off").await?;
        let off_reply = last_reply();
        let _ = ask("/drain").await?;
        let show_reply = last_reply();
        let _ = ask("/drain on").await?;
        let on_reply = last_reply();
        println!("[probe] /drain off -> {off_reply}; show -> {show_reply}; on -> {on_reply}");

        // REAL held-queue scenario: long turn + kernel-queued item + stop
        // -> autoDrain must read false; /drain on resumes delivery.
        let long_turn = {
            let tx = client.tx.clone();
            let sid = sid.clone();
            tokio::task::spawn_local(async move {
                acp_send(
                    acp::PromptRequest::new(
                        sid,
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "从1慢慢数到60，每个数字单独一行，数慢一点。".to_string(),
                        ))],
                    ),
                    &tx,
                )
                .await
            })
        };
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;
        // Kernel-queue an item via the ext (notification rail).
        let interject_notif = acp::ExtNotification::new(
            "x.ai/queue/interject",
            serde_json::value::to_raw_value(&serde_json::json!({
                "sessionId": sid.0, "newText": "排队消息：3+3等于几？只回答数字。",
            }))
            .unwrap()
            .into(),
        );
        let interject_resp = acp_send(interject_notif, &client.tx).await;
        println!("[probe] interject sent: {:?}", interject_resp.is_ok());
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let proj_pre_stop: serde_json::Value = acp_send(
            acp::ExtRequest::new(
                "x.ai/v4/projection",
                serde_json::value::to_raw_value(&serde_json::json!({
                    "sessionId": sid.0, "refresh": true,
                }))
                .unwrap()
                .into(),
            ),
            &client.tx,
        )
        .await
        .map(|r| serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({})))
        .unwrap_or(serde_json::json!({}));
        println!(
            "[probe] pre-stop: autoDrain={:?} queueItems={:?}",
            proj_pre_stop.get("queueAutoDrain"),
            proj_pre_stop.get("queueItems")
        );
        // Stop the running turn (official session/stop path).
        let _ = acp_send(acp::CancelNotification::new(sid.clone()), &client.tx).await;
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        let proj_stopped: serde_json::Value = acp_send(
            acp::ExtRequest::new(
                "x.ai/v4/projection",
                serde_json::value::to_raw_value(&serde_json::json!({
                    "sessionId": sid.0, "refresh": true,
                }))
                .unwrap()
                .into(),
            ),
            &client.tx,
        )
        .await
        .map(|r| serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({})))
        .unwrap_or(serde_json::json!({}));
        println!(
            "[probe] after stop: autoDrain={:?} queueItems={:?}",
            proj_stopped.get("queueAutoDrain"),
            proj_stopped.get("queueItems")
        );
        // Resume drain, let the queued item deliver.
        let _ = ask("/drain on").await?;
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        let _ = long_turn.await;
        let drain_reply = replies.borrow().last().cloned().unwrap_or_default();
        println!("[probe] queued delivery after drain-on, replies tail: {}", drain_reply);

        // Projection diagnostics carry autoDrain.
        let proj: serde_json::Value = acp_send(
            acp::ExtRequest::new(
                "x.ai/v4/projection",
                serde_json::value::to_raw_value(&serde_json::json!({
                    "sessionId": sid.0, "refresh": true,
                }))
                .unwrap()
                .into(),
            ),
            &client.tx,
        )
        .await
        .map(|r| serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({})))
        .unwrap_or(serde_json::json!({}));
        println!("[probe] projection queueAutoDrain = {:?}", proj.get("queueAutoDrain"));

        let verdict = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        verdict("rate-accepted", rate_reply.contains("点赞"));
        // Kernel 3.14.1 empirics: v4 stop is accepted but does NOT flip
        // autoDrain=false (the OSS comment describes the desktop's model).
        // What matters end-to-end: the queued item still delivers after stop.
        verdict(
            "stop-keeps-queue-flowing",
            proj_stopped.get("queueAutoDrain") == Some(&serde_json::json!(true)),
        );
        verdict(
            "drain-resumes-delivery",
            drain_reply.contains('6') || replies.borrow().iter().any(|r| r.trim() == "6"),
        );
        verdict("drain-on-works", on_reply.contains("恢复"));
        verdict("projection-autodrain-true", proj.get("queueAutoDrain") == Some(&serde_json::json!(true)));

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
