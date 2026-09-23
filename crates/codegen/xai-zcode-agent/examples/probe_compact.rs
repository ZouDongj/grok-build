//! Reproduce the /compact double-completion + concurrent-prompt issue:
//! kernel session/compact is ASYNC (returns {state:"accepted"} immediately
//! and runs /compact as a background turn). This probe measures the ext
//! return latency, fires a prompt immediately after, and logs the full
//! notification timeline so we can see interleaving.
//!
//! Usage: ZCODE_WIRE_LOG=1 cargo run -p xai-zcode-agent --example probe_compact

use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probecompact";

struct Seen {
    t0: std::time::Instant,
    timeline: RefCell<Vec<(u128, String)>>,
}

impl Default for Seen {
    fn default() -> Seen {
        Seen {
            t0: std::time::Instant::now(),
            timeline: RefCell::new(Vec::new()),
        }
    }
}

fn short(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => {
            let mut t = s.chars().take(40).collect::<String>();
            if s.len() > 40 {
                t.push('…');
            }
            t
        }
        other => {
            let s = other.to_string();
            s.chars().take(60).collect::<String>()
        }
    }
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

            let seen = Rc::new(RefCell::new(Seen::default()));
            {
                let seen = seen.clone();
                let mut rx = client.rx;
                tokio::task::spawn_local(async move {
                    while let Some(message) = rx.recv().await {
                        match message {
                            AcpClientMessage::ExtMethod(n) => {
                                let method = n.method.clone();
                                // auto-approve everything (plan ext, etc.)
                                let raw = serde_json::value::to_raw_value(&serde_json::json!({
                                    "outcome": "approved"
                                }))
                                .expect("approve");
                                let _ = n
                                    .response_tx
                                    .send(Ok(acp::ExtResponse::new(raw.into())));
                                println!("[probe] ext request auto-approved: {method}");
                            }
                            AcpClientMessage::RequestPermission(p) => {
                                let pick = p
                                    .options
                                    .iter()
                                    .find(|o| {
                                        matches!(
                                            o.kind,
                                            acp::PermissionOptionKind::AllowOnce
                                                | acp::PermissionOptionKind::AllowAlways
                                        )
                                    })
                                    .or_else(|| p.options.first())
                                    .cloned();
                                if let Some(option) = pick {
                                    let _ = p.response_tx.send(Ok(
                                        acp::RequestPermissionResponse::new(
                                            acp::RequestPermissionOutcome::Selected(
                                                acp::SelectedPermissionOutcome::new(
                                                    option.option_id.clone(),
                                                ),
                                            ),
                                        ),
                                    ));
                                } else {
                                    let _ = p.response_tx.send(Ok(
                                        acp::RequestPermissionResponse::new(
                                            acp::RequestPermissionOutcome::Cancelled,
                                        ),
                                    ));
                                }
                            }
                            AcpClientMessage::SessionNotification(u) => {
                                let t = seen.borrow().t0.elapsed().as_millis();
                                let desc = match &u.update {
                                    acp::SessionUpdate::AgentThoughtChunk(c) => {
                                        if let acp::ContentBlock::Text(t) = &c.content {
                                            format!("THINK {}", short(&serde_json::Value::String(t.text.clone())))
                                        } else {
                                            "THINK(binary)".into()
                                        }
                                    }
                                    acp::SessionUpdate::AgentMessageChunk(c) => {
                                        if let acp::ContentBlock::Text(t) = &c.content {
                                            format!("TEXT {}", short(&serde_json::Value::String(t.text.clone())))
                                        } else {
                                            "TEXT(binary)".into()
                                        }
                                    }
                                    acp::SessionUpdate::ToolCall(c) => {
                                        format!("TOOL_START {}", &c.title)
                                    }
                                    acp::SessionUpdate::ToolCallUpdate(_) => "TOOL_END".into(),
                                    acp::SessionUpdate::UsageUpdate(u) => {
                                        format!("USAGE {}/{}", u.used, u.size)
                                    }
                                    acp::SessionUpdate::Plan(_) => "PLAN".into(),
                                    acp::SessionUpdate::CurrentModeUpdate(m) => {
                                        format!("MODE {:?}", m.current_mode_id)
                                    }
                                    acp::SessionUpdate::UserMessageChunk(c) => {
                                        if let acp::ContentBlock::Text(t) = &c.content {
                                            format!("USERECHO {}", short(&serde_json::Value::String(t.text.clone())))
                                        } else {
                                            "USERECHO".into()
                                        }
                                    }
                                    other => format!("{:?}", std::mem::discriminant(other))
                                        .replace(')', ""),
                                };
                                println!("[probe t+{t:>6}ms] sess({}) {desc}", u.request.session_id.0);
                                seen.borrow_mut().timeline.borrow_mut().push((t, desc));
                            }
                            AcpClientMessage::ExtNotification(n) => {
                                let t = seen.borrow().t0.elapsed().as_millis();
                                let v: serde_json::Value =
                                    serde_json::from_str(n.params.get()).unwrap_or(serde_json::json!({}));
                                println!(
                                    "[probe t+{t:>6}ms] ext {} {}",
                                    n.method,
                                    short(&v)
                                );
                                seen.borrow_mut().timeline.borrow_mut().push((t, format!("ext:{} {}", n.method, short(&v))));
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

            // Build some context first: two turns.
            for text in [
                "请从1数到30，每行一个数字，不要说别的。",
                "现在把上面你输出的所有数字倒序再输出一遍，不要说别的。",
            ] {
                let t = std::time::Instant::now();
                let resp = acp_send(
                    acp::PromptRequest::new(sid.clone(), vec![acp::ContentBlock::Text(acp::TextContent::new(text.to_string()))]),
                    &client.tx,
                )
                .await;
                println!("[probe] turn done in {:?} -> {:?}", t.elapsed(), resp.as_ref().map(|r| format!("{:?}", r.stop_reason)).map_err(|e| e.to_string()));
            }

            // --- compact: fire it, then 1s later send a prompt that must
            // land while the compaction turn is running ---
            let compact_task = {
                let tx = client.tx.clone();
                let sid = sid.clone();
                tokio::task::spawn_local(async move {
                    let t = std::time::Instant::now();
                    let compact_resp = acp_send(
                        acp::ExtRequest::new(
                            "x.ai/compact_conversation",
                            serde_json::value::to_raw_value(&serde_json::json!({
                                "sessionId": sid.0,
                            }))
                            .expect("serialize compact req")
                            .into(),
                        ),
                        &tx,
                    )
                    .await;
                    println!(
                        "[probe] compact ext returned in {:?} -> {}",
                        t.elapsed(),
                        match &compact_resp {
                            Ok(_) => "Ok".into(),
                            Err(e) => format!("Err({e})"),
                        }
                    );
                })
            };

            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let prompt_task = {
                let tx = client.tx.clone();
                let sid = sid.clone();
                tokio::task::spawn_local(async move {
                    let t = std::time::Instant::now();
                    let resp = acp_send(
                        acp::PromptRequest::new(
                            sid.clone(),
                            vec![acp::ContentBlock::Text(acp::TextContent::new(
                                "1+1等于几？只回答阿拉伯数字。".to_string(),
                            ))],
                        ),
                        &tx,
                    )
                    .await;
                    println!(
                        "[probe] mid-compact prompt returned in {:?} -> {:?}",
                        t.elapsed(),
                        resp.as_ref().map(|r| format!("{:?}", r.stop_reason)).map_err(|e| e.to_string())
                    );
                })
            };

            // Let everything settle: compaction + queued prompt.
            tokio::time::sleep(std::time::Duration::from_secs(120)).await;
            let _ = compact_task.await;
            let _ = prompt_task.await;

            // One more normal turn to see the post-compact state.
            let resp = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new(
                        "现在你的上下文里还有我们数数的对话吗？有就回答有，没有回答没有。".to_string(),
                    ))],
                ),
                &client.tx,
            )
            .await;
            println!(
                "[probe] post-compact recall turn -> {:?}",
                resp.as_ref().map(|r| format!("{:?}", r.stop_reason)).map_err(|e| e.to_string())
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

            let events = seen.borrow().timeline.borrow().clone();
            println!("\n=== timeline ({} entries) ===", events.len());
            for (t, desc) in events.iter() {
                println!("t+{t:>6}ms  {desc}");
            }
            Ok::<(), anyhow::Error>(())
        })
        .await?;
    Ok(())
}
