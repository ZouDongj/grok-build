//! Drive ZcodeAgent as a mini ACP client: print every message the agent
//! emits. Usage: cargo run -p xai-zcode-agent --example probe

use agent_client_protocol as acp;
use xai_acp_lib::{AcpGatewayReceiver, acp_channels, acp_send};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    eprintln!("[probe] PROBE START");
    let local = tokio::task::LocalSet::new();
    local.run_until(async move {
        let (client, agent_channel) = acp_channels();
        // Agent side: the gateway pushes AcpClientMessages to our probe.
        let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
        let agent = std::rc::Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
        // Dispatch inbound agent-bound requests to the agent (like spawn.rs).
        let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
        tokio::task::spawn_local(gw_rx.run());

        // Client role: pump everything the agent sends us.
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let printer = tokio::task::spawn_local(async move {
            let mut rx = client.rx;
            let mut turn_seen = false;
            while let Some(message) = rx.recv().await {
                match &message {
                    xai_acp_lib::AcpClientMessage::ExtNotification(n) => {
                        println!("EXT NOTIF: {} params={}", n.method, n.params.get());
                    }
                    xai_acp_lib::AcpClientMessage::SessionNotification(u) => {
                        let kind = match &u.update {
                            acp::SessionUpdate::AgentMessageChunk(_) => "message",
                            acp::SessionUpdate::AgentThoughtChunk(_) => "thought",
                            acp::SessionUpdate::ToolCall(_) => "tool",
                            acp::SessionUpdate::CurrentModeUpdate(_) => "mode",
                            _ => "other",
                        };
                        println!("UPDATE: {kind}");
                        if let acp::SessionUpdate::AgentMessageChunk(c) = &u.update {
                            if let acp::ContentBlock::Text(t) = &c.content {
                                print!("{}", t.text);
                                turn_seen = true;
                            }
                        }
                    }
                    xai_acp_lib::AcpClientMessage::RequestPermission(p) => {
                        println!(
                            "PERMISSION: options={:?}",
                            p.options.iter().map(|o| o.name.clone()).collect::<Vec<_>>()
                        );
                    }
                    other => println!("OTHER: {other:?}"),
                }
            }
            let _ = done_tx.send(());
        });
        let _ = printer;

        // initialize → new_session → prompt
        eprintln!("[probe] sending initialize");
        let init = acp_send(
            acp::InitializeRequest::new(acp::ProtocolVersion::V1),
            &client.tx,
        )
        .await?;
        println!("initialize ok: {:?}", init.auth_methods);

        let cwd = std::env::current_dir()?;
        eprintln!("[probe] sending new_session");
        let session = acp_send(acp::NewSessionRequest::new(cwd), &client.tx).await?;
        println!("new_session ok: {:?}", session.session_id.0);

        // Model switch round-trip test — FULL TUI sequence: a completed turn
        // BEFORE the switch (the TUI repro requires it), effort meta included.
        let prompt1 = acp_send(
            acp::PromptRequest::new(
                session.session_id.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new("hi".to_string()))],
            ),
            &client.tx,
        )
        .await;
        println!("prompt1 stop={:?}", prompt1.map(|r| r.stop_reason).map_err(|e| e.to_string()));

        eprintln!("[probe] turn 1 done; switching model WITH effort meta");
        let mut switch_req = acp::SetSessionModelRequest::new(
            session.session_id.clone(),
            acp::ModelId::new("GLM-5.3-Flash".to_string()),
        );
        let mut meta = acp::Meta::new();
        meta.insert("reasoningEffort".to_string(), serde_json::json!("max"));
        switch_req.meta = Some(meta);
        let switch = acp_send(switch_req, &client.tx).await;
        println!("set_session_model: {}", match &switch { Ok(_) => "OK".into(), Err(e) => format!("ERR {e}") });
        eprintln!("=== post-switch prompt");
        let prompt2 = acp_send(
            acp::PromptRequest::new(
                session.session_id.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new("hi again".to_string()))],
            ),
            &client.tx,
        )
        .await;
        println!("prompt2 stop={:?}", prompt2.map(|r| r.stop_reason).map_err(|e| e.to_string()));

        let prompt = std::env::args().nth(1).unwrap_or_else(|| "say: ok".into());
        let response = acp_send(
            acp::PromptRequest::new(
                session.session_id.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new(prompt))],
            ),
            &client.tx,
        )
        .await?;
        println!("\nprompt done: stop={:?}", response.stop_reason);

        let _ = done_rx.await;
        Ok::<(), anyhow::Error>(())
    })
    .await?;
    Ok(())
}
