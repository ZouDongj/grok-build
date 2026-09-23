//! Minimal send-now reproduction: slow turn + 2.5s-in prompt with
//! meta.sendNow — must cancel the running turn and answer the interrupt.
use agent_client_protocol as acp;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            std::fs::create_dir_all("/tmp/probesendnow")?;
            let (client, agent_channel) = acp_channels();
            let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
            let agent = std::rc::Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
            let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
            tokio::task::spawn_local(gw_rx.run());
            {
                let mut rx = client.rx;
                tokio::task::spawn_local(async move {
                    while let Some(m) = rx.recv().await {
                        if let AcpClientMessage::ExtMethod(n) = m {
                            let raw = serde_json::value::to_raw_value(&serde_json::json!({"outcome": "approved"})).unwrap();
                            let _ = n.response_tx.send(Ok(acp::ExtResponse::new(raw.into())));
                        }
                    }
                });
            }
            let _ = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
            let session = acp_send(
                acp::NewSessionRequest::new(std::path::PathBuf::from("/tmp/probesendnow")),
                &client.tx,
            ).await?;
            let sid = session.session_id.clone();

            let slow = {
                let tx = client.tx.clone();
                let sid = sid.clone();
                tokio::task::spawn_local(async move {
                    acp_send(
                        acp::PromptRequest::new(
                            sid,
                            vec![acp::ContentBlock::Text(acp::TextContent::new(
                                "请从1一个一个慢慢数到100，每个数字一行，不许提前停止。".to_string(),
                            ))],
                        ),
                        &tx,
                    ).await
                })
            };
            tokio::time::sleep(std::time::Duration::from_millis(2500)).await;
            let mut meta = serde_json::Map::new();
            meta.insert("promptId".to_string(), serde_json::json!("probe-sn-1"));
            meta.insert("sendNow".to_string(), serde_json::json!(true));
            let b = acp_send(
                acp::PromptRequest::new(
                    sid.clone(),
                    vec![acp::ContentBlock::Text(acp::TextContent::new("1+1等于几？只回答阿拉伯数字。".to_string()))],
                ).meta(Some(meta)),
                &client.tx,
            ).await;
            println!("[probe] send-now B -> {:?}", b.as_ref().map(|r| format!("{:?}", r.stop_reason)).map_err(|e| e.to_string()));
            let slow_res = tokio::time::timeout(std::time::Duration::from_secs(60), slow).await;
            println!("[probe] slow settled={}", matches!(slow_res, Ok(Ok(_))));
            let _ = acp_send(
                acp::ExtRequest::new(
                    "x.ai/session/delete",
                    serde_json::value::to_raw_value(&serde_json::json!({"sessionId": sid.0, "cwd": "/tmp/probesendnow"})).unwrap().into(),
                ),
                &client.tx,
            ).await;
            Ok::<(), anyhow::Error>(())
        })
        .await?;
    Ok(())
}
