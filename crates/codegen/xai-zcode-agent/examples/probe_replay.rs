//! Probe the resume replay's thought-chunk meta contract. The pager freezes
//! "Thought for Xs" as (last chunk agentTimestampMs - streamStartMs) per
//! isReplay block; without isReplay it runs a LOCAL timer that reads ~0.0s.
//! This probe: turn with thinking -> LoadSession -> dump every replayed
//! AgentThoughtChunk's meta and evaluate the pager formula.
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probereplay";

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

        let chunks: Rc<RefCell<Vec<(i64, i64, bool)>>> = Rc::new(RefCell::new(Vec::new()));
        {
            let chunks = chunks.clone();
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
                            if let acp::SessionUpdate::AgentThoughtChunk(c) = &u.update {
                                if let (acp::ContentBlock::Text(t), Some(m)) =
                                    (&c.content, u.meta.as_ref())
                                {
                                    if !t.text.is_empty() {
                                        let ts =
                                            m.get("agentTimestampMs").and_then(|v| v.as_i64());
                                        let st = m.get("streamStartMs").and_then(|v| v.as_i64());
                                        let rp = m.get("isReplay")
                                            .and_then(|v| v.as_bool())
                                            .unwrap_or(false);
                                        if let (Some(a), Some(s)) = (ts, st) {
                                            chunks.borrow_mut().push((a, s, rp));
                                        } else {
                                            // record bare chunks too, as (0,0,isReplay)
                                            chunks.borrow_mut().push((0, 0, rp));
                                        }
                                    }
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

        // A thinking-heavy turn (GLM-5.3 reasons before answering).
        let resp = acp_send(
            acp::PromptRequest::new(
                sid.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "认真思考一下再回答：13乘17等于多少？只回答数字。".to_string(),
                ))],
            ),
            &client.tx,
        )
        .await?;
        println!("[probe] live turn stop: {:?}", resp.stop_reason);

        // Clear captured LIVE chunks; replay must restamp everything.
        chunks.borrow_mut().clear();

        let _ = acp_send(
            acp::LoadSessionRequest::new(sid.clone(), std::path::PathBuf::from(WORKDIR)),
            &client.tx,
        )
        .await?;
        // Let the replay notifications drain.
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

        let got = chunks.borrow().clone();
        println!("\n=== replayed thought chunks: {} ===", got.len());
        for (a, s, r) in &got {
            println!("agentTs-streamStart={:>8}ms isReplay={r}", a - s);
        }
        // Pager formula: per same-stream block, final chunk delta is the
        // frozen duration.
        let spans: Vec<i64> = got
            .windows(2)
            .filter(|w| w[0].1 == w[1].1)
            .map(|w| w[1].0 - w[1].1)
            .collect();
        let verdict = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        verdict(
            "replay-chunks-carry-full-meta",
            !got.is_empty() && got.iter().all(|(a, _, _)| *a != 0),
        );
        verdict("replay-chunks-marked-isReplay", got.iter().all(|(_, _, r)| *r));
        verdict(
            "pager-formula-yields-real-duration",
            spans.iter().any(|d| *d >= 500) && spans.iter().all(|d| *d >= 0),
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
