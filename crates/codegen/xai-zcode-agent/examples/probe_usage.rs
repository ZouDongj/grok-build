//! Probe the real-time context meter: the v4 projection's usage.contextWindow
//! is the official source (state.updated pushes on every change — "conflation:
//! unchanged values are not sent"). This probe verifies:
//!   1. snapshot usage parses (projection ext reports usedTokens > 0)
//!   2. USAGE notifications stream during turns (not only at turn end)
//!   3. after /compact completes, a USAGE notification with the POST-compact
//!      (much lower) value arrives WITHOUT any further user turn
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probeusage";

struct Seen {
    t0: std::time::Instant,
    usages: RefCell<Vec<(u128, u64, u64)>>,
}

async fn projection(
    client: &tokio::sync::mpsc::UnboundedSender<xai_acp_lib::AcpAgentMessage>,
    sid: &acp::SessionId,
) -> serde_json::Value {
    acp_send(
        acp::ExtRequest::new(
            "x.ai/v4/projection",
            serde_json::value::to_raw_value(&serde_json::json!({
                "sessionId": sid.0, "refresh": true,
            }))
            .expect("serialize projection req")
            .into(),
        ),
        client,
    )
    .await
    .map(|r| serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({})))
    .unwrap_or(serde_json::json!({}))
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

            let seen = Rc::new(Seen { t0: std::time::Instant::now(), usages: RefCell::new(Vec::new()) });
            {
                let seen = seen.clone();
                let mut rx = client.rx;
                tokio::task::spawn_local(async move {
                    while let Some(m) = rx.recv().await {
                        match m {
                            AcpClientMessage::ExtMethod(n) => {
                                let raw = serde_json::value::to_raw_value(&serde_json::json!({
                                    "outcome": "approved"
                                }))
                                .expect("approve");
                                let _ = n.response_tx.send(Ok(acp::ExtResponse::new(raw.into())));
                            }
                            AcpClientMessage::RequestPermission(p) => {
                                let pick = p
                                    .options
                                    .iter()
                                    .find(|o| matches!(o.kind, acp::PermissionOptionKind::AllowOnce))
                                    .or_else(|| p.options.first())
                                    .cloned();
                                if let Some(option) = pick {
                                    let _ = p.response_tx.send(Ok(
                                        acp::RequestPermissionResponse::new(
                                            acp::RequestPermissionOutcome::Selected(
                                                acp::SelectedPermissionOutcome::new(option.option_id.clone()),
                                            ),
                                        ),
                                    ));
                                }
                            }
                            AcpClientMessage::SessionNotification(u) => {
                                if let acp::SessionUpdate::UsageUpdate(u2) = &u.update {
                                    let t = seen.t0.elapsed().as_millis();
                                    println!("[probe t+{t:>6}ms] USAGE {}/{}", u2.used, u2.size);
                                    seen.usages.borrow_mut().push((t, u2.used, u2.size));
                                }
                            }
                            _ => {}
                        }
                    }
                });
            }

            let _init = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
            let session = acp_send(
                acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)),
                &client.tx,
            )
            .await?;
            let sid = session.session_id.clone();
            println!("[probe] session {}", sid.0);

            // Turn 1: build context.
            let t = std::time::Instant::now();
            let resp = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "请写一篇500字的灯塔看守人短文，然后从1数到10，每个一行。".to_string(),
                    ))],
                ),
                &client.tx,
            )
            .await;
            println!("[probe] turn1 {:?}", t.elapsed());
            let _ = resp?;

            let pre = projection(&client.tx, &sid).await;
            let pre_used = pre.pointer("/usage/usedTokens").and_then(|v| v.as_u64());
            println!("[probe] pre-compact projection usage: {:?} (full: {:?})", pre_used, pre.get("usage"));

            // Compact and await real completion.
            let t = std::time::Instant::now();
            let compact = acp_send(
                acp::ExtRequest::new(
                    "x.ai/compact_conversation",
                    serde_json::value::to_raw_value(&serde_json::json!({ "sessionId": sid.0 }))
                        .expect("serialize compact req")
                        .into(),
                ),
                &client.tx,
            )
            .await;
            println!("[probe] compact ext returned in {:?} ({})", t.elapsed(), compact.is_ok());
            compact?;

            // NO further user turn: the meter must already show the drop.
            let compact_done_at = seen.t0.elapsed().as_millis();
            let post = projection(&client.tx, &sid).await;
            let post_used = post.pointer("/usage/usedTokens").and_then(|v| v.as_u64());
            println!("[probe] post-compact projection usage: {:?}", post_used);

            let usages = seen.usages.borrow().clone();
            println!("\n=== usage timeline ({} entries) ===", usages.len());
            for (t, used, size) in usages.iter() {
                println!("t+{t:>6}ms  {used}/{size}");
            }

            let verdict = |name: &str, ok: bool| {
                println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
            };
            verdict("projection-reports-usage", pre_used.unwrap_or(0) > 0);
            let dropped_after = usages
                .iter()
                .any(|(t, used, _)| *t >= compact_done_at && pre_used.unwrap_or(0) > *used + 2000);
            verdict("usage-notification-dropped-after-compact", dropped_after);
            let both = pre_used.zip(post_used);
            verdict(
                "projection-post-compact-much-smaller",
                both.map(|(a, b)| b + 2000 < a).unwrap_or(false),
            );

            // Cleanup: delete session.
            let _ = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/delete",
                    serde_json::value::to_raw_value(&serde_json::json!({
                        "sessionId": sid.0, "cwd": WORKDIR,
                    }))
                    .expect("serialize delete")
                    .into(),
                ),
                &client.tx,
            )
            .await;
            Ok::<(), anyhow::Error>(())
        })
        .await?;
    Ok(())
}
