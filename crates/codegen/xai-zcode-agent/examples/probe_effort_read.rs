//! What does session/read actually return for thoughtLevel? Print it.
use agent_client_protocol as acp;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probeeffort";

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local.run_until(async move {
        let (client, agent_channel) = acp_channels();
        let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
        let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
        let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
        tokio::task::spawn_local(gw_rx.run());
        {
            let mut rx = client.rx;
            tokio::task::spawn_local(async move {
                while let Some(m) = rx.recv().await {
                    match m {
                        AcpClientMessage::ExtMethod(n) => {
                            let raw = serde_json::value::to_raw_value(&serde_json::json!({"outcome":"approved"})).unwrap();
                            let _ = n.response_tx.send(Ok(acp::ExtResponse::new(raw.into())));
                        }
                        AcpClientMessage::RequestPermission(p) => {
                            if let Some(o) = p.options.first() {
                                let pick = o.option_id.clone();
                                let _ = p.response_tx.send(Ok(acp::RequestPermissionResponse::new(
                                    acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(pick)),
                                )));
                            }
                        }
                        _ => {}
                    }
                }
            });
        }
        let _init = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
        // Use the user's long-running session — read-only resume
        let sid = acp::SessionId::new("sess_873b4d48-4131-435d-b5f1-220e50598324".to_string());
        let resp = acp_send(
            acp::LoadSessionRequest::new(sid.clone(), std::path::PathBuf::from("/data/zdj/ZCode/grok-build")),
            &client.tx,
        )
        .await;
        println!("load ok: {}", resp.is_ok());
        // Direct kernel read via the projection ext won't show thoughtLevel; instead
        // watch x.ai/models/update notifications from the load
        tokio::time::sleep(std::time::Duration::from_millis(2000)).await;
        Ok::<(), anyhow::Error>(())
    })
    .await?;
    Ok(())
}
