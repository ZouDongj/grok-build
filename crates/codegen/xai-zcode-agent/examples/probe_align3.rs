//! Probe file-rewind alignment: a turn that edits a file, then
//! /rewind points badge (hasFileChanges), /filerewind preview, and
//! /filerewind apply (content actually reverts; chat history intact).
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probealign3";

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

        // Baseline file, then a turn that rewrites it.
        std::fs::write(format!("{WORKDIR}/target.txt"), "original\n")?;
        let _ = ask("把 target.txt 的内容完全替换为 modified（用 Write 工具）。只回一个 ok。".to_string()).await?;
        let content_after = std::fs::read_to_string(format!("{WORKDIR}/target.txt"))?;
        println!("[probe] after edit turn: {content_after:?}");
        replies.borrow_mut().clear();

        // Rewind points badge.
        let points: serde_json::Value = acp_send(
            acp::ExtRequest::new(
                "x.ai/rewind/points",
                serde_json::value::to_raw_value(&serde_json::json!({ "sessionId": sid.0 }))
                    .unwrap()
                    .into(),
            ),
            &client.tx,
        )
        .await
        .map(|r| serde_json::from_str(r.0.get()).unwrap_or(serde_json::json!({})))
        .unwrap_or(serde_json::json!({}));
        let badges: Vec<bool> = points
            .pointer("/result/rewindPoints")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .map(|p| p.get("hasFileChanges").and_then(|v| v.as_bool()).unwrap_or(false))
                    .collect()
            })
            .unwrap_or_default();
        println!("[probe] rewind badges: {badges:?}");

        // Dump turn headers (actions/fileChanges) for diagnostics.
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
        println!(
            "[probe] turnHeaders: {}",
            serde_json::to_string(proj.get("turnHeaders").unwrap_or(&serde_json::json!([]))).unwrap_or_default()
        );

        // /filerewind preview.
        let _ = ask("/filerewind".to_string()).await?;
        let preview_reply = replies.borrow().last().cloned().unwrap_or_default();
        println!("[probe] preview: {preview_reply}");

        // /filerewind apply.
        let _ = ask("/filerewind apply".to_string()).await?;
        let apply_reply = replies.borrow().last().cloned().unwrap_or_default();
        let content_reverted = std::fs::read_to_string(format!("{WORKDIR}/target.txt"))?;
        println!("[probe] apply: {apply_reply}; file now: {content_reverted:?}");

        // Chat history intact: ask about the earlier turn.
        let recall_from = replies.borrow().len();
        let _ = ask("我们刚才让你改了哪个文件？只回答文件名。".to_string()).await?;
        let recall = replies.borrow()[recall_from..].join("");
        println!("[probe] history recall: {recall}");

        // Title sync: the kernel's generated title must land in the pager's
        // summary.json after a turn (kernel titles are async — settle first).
        tokio::time::sleep(std::time::Duration::from_secs(8)).await;
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        let mut title: Option<String> = None;
        for cand in [
            format!("{home}/.grok/sessions/{WORKDIR}/"),  // encoded name may differ
        ] {
            let _ = &cand;
        }
        // find the session dir regardless of encoding
        fn find_summary(base: &std::path::Path, sid: &str) -> Option<String> {
            for entry in std::fs::read_dir(base).ok()? {
                let entry = entry.ok()?;
                let root = entry.path().join(sid).join("summary.json");
                if root.exists() {
                    return std::fs::read_to_string(root).ok();
                }
            }
            None
        }
        let grok_base = std::path::Path::new(&home).join(".grok/sessions");
        if let Some(raw) = find_summary(&grok_base, sid.0.as_ref()) {
            title = serde_json::from_str::<serde_json::Value>(&raw)
                .ok()
                .and_then(|j| {
                    j.get("session_summary")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                });
        }
        println!("[probe] synced summary title: {title:?}");

        // /retry via the official rail.
        let retry_from = replies.borrow().len();
        let _ = ask("/retry".to_string()).await?;
        // The retried turn runs; give it time and capture text.
        tokio::time::sleep(std::time::Duration::from_secs(25)).await;
        let retry_text = replies.borrow()[retry_from..].join("");
        println!("[probe] /retry reply+turn: {}", retry_text.chars().take(120).collect::<String>());

        // Interrupted-turn retry: cancel a long turn, then the header should
        // offer canRetry and /retry should run on the official rail.
        let long_turn = {
            let tx = client.tx.clone();
            let sid = sid.clone();
            tokio::task::spawn_local(async move {
                acp_send(
                    acp::PromptRequest::new(
                        sid,
                        vec![acp::ContentBlock::Text(acp::TextContent::new(
                            "从1慢慢数到50，每行一个，数慢点。".to_string(),
                        ))],
                    ),
                    &tx,
                )
                .await
            })
        };
        tokio::time::sleep(std::time::Duration::from_secs(4)).await;
        let _ = acp_send(acp::CancelNotification::new(sid.clone()), &client.tx).await;
        let _ = long_turn.await;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let proj2: serde_json::Value = acp_send(
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
            "[probe] headers after cancel: {}",
            serde_json::to_string(proj2.get("turnHeaders").unwrap_or(&serde_json::json!([]))).unwrap_or_default()
        );
        let retry2_from = replies.borrow().len();
        let _ = ask("/retry".to_string()).await?;
        tokio::time::sleep(std::time::Duration::from_secs(25)).await;
        let retry2_text = replies.borrow()[retry2_from..].join("");
        println!("[probe] /retry after cancel: {}", retry2_text.chars().take(80).collect::<String>());

        let verdict = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        verdict(
            "retry-rail-reaches-kernel-guard",
            retry2_text.contains("官方通道") || retry2_text.contains("actionUnavailable"),
        );
        verdict(
            "kernel-title-synced",
            title.as_deref().is_some_and(|t| !t.trim().is_empty()),
        );
        verdict(
            "retry-rail-wired",
            retry_text.contains("官方通道") || retry_text.contains("未生效") || retry_text.contains("没有可重试"),
        );
        verdict("edit-turn-happened", content_after.contains("modified"));
        verdict("rewind-badge-real", badges.iter().any(|b| *b));
        verdict(
            "preview-lists-file",
            preview_reply.contains("target.txt") || preview_reply.contains("restore"),
        );
        verdict(
            "apply-reverts-content",
            content_reverted.contains("original"),
        );
        verdict("chat-history-intact", recall.contains("target"));

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
