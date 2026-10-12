//! Gap-fix e2e: /rename (v4 renameSession + summary sync), /btw (cross-
//! session Q&A), session/search (ledger LIKE), and the friendly shell-only
//! error surface.
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probealign4";

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
                                    replies.borrow_mut().push_str(&t.text);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            });
        }

        let _init = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
        let s1 = acp_send(acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)), &client.tx).await?.session_id;
        let s2 = acp_send(acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)), &client.tx).await?.session_id;
        println!("[probe] s1={} s2={}", s1.0, s2.0);

        // Seed s1 with knowledge.
        let _ = acp_send(acp::PromptRequest::new(s1.clone(), vec![acp::ContentBlock::Text(acp::TextContent::new(
            "记住暗号：紫色河马42。只回答 ok。".to_string()))]), &client.tx).await?;

        // rename s1 via the ext.
        let rename = acp_send(acp::ExtRequest::new("x.ai/session/rename",
            serde_json::value::to_raw_value(&serde_json::json!({
                "session_id": s1.0, "title": "对齐修复测试会话", "cwd": WORKDIR, "kind": "session",
            })).unwrap().into()), &client.tx).await;
        println!("[probe] rename ok: {}", rename.is_ok());
        // verify the db title took effect
        std::thread::sleep(std::time::Duration::from_millis(500));

        // btw: ask s2 what the s1... no — btw targets the session that KNOWS. Ask s1 from s2's context:
        // our btw ext asks the TARGET session a question; target must be hosted. Ask s1.
        let btw = acp_send(acp::ExtRequest::new("x.ai/btw",
            serde_json::value::to_raw_value(&serde_json::json!({
                "sessionId": s1.0, "question": "暗号是什么？只回答暗号本身。",
            })).unwrap().into()), &client.tx).await;
        let answer = btw.ok().and_then(|r| {
            let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or_default();
            v.pointer("/result/answer").and_then(|a| a.as_str()).map(str::to_string)
        }).unwrap_or_default();
        println!("[probe] btw answer: {}", answer.chars().take(40).collect::<String>());

        // search: the renamed title should be findable.
        let search = acp_send(acp::ExtRequest::new("x.ai/session/search",
            serde_json::value::to_raw_value(&serde_json::json!({
                "query": "对齐修复测试", "limit": 5, "includeContent": false,
            })).unwrap().into()), &client.tx).await;
        let hits = search.ok().and_then(|r| {
            let v: serde_json::Value = serde_json::from_str(r.0.get()).unwrap_or_default();
            v.pointer("/result/results").and_then(|x| x.as_array()).map(|a| a.len())
        }).unwrap_or(0);
        println!("[probe] search hits: {hits}");

        // friendly error on a shell-only ext.
        let shellonly = acp_send(acp::ExtRequest::new("x.ai/share_session",
            serde_json::value::to_raw_value(&serde_json::json!({"sessionId": s1.0})).unwrap().into()), &client.tx).await;
        let err_msg = shellonly.err().map(|e| e.to_string()).unwrap_or_default();
        let friendly = err_msg.contains("grok-shell");
        println!("[probe] shell-only error friendly: {friendly} ({})", err_msg.chars().take(60).collect::<String>());

        let v = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        v("rename-accepted", rename.is_ok());
        v("btw-answers-across-sessions", answer.contains("紫色河马") || answer.contains("42"));
        v("search-finds-renamed", hits > 0);
        v("shell-only-friendly-error", friendly);

        for s in [s1, s2] {
            let _ = acp_send(acp::ExtRequest::new("x.ai/session/delete",
                serde_json::value::to_raw_value(&serde_json::json!({"sessionId": s.0, "cwd": WORKDIR})).unwrap().into()), &client.tx).await;
        }
        Ok::<(), anyhow::Error>(())
    }).await?;
    Ok(())
}
