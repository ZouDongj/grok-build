//! Create one session in /tmp/probequota and KEEP it (for resume testing).
use agent_client_protocol as acp;
use std::rc::Rc;
use xai_acp_lib::{AcpGatewayReceiver, acp_channels, acp_send};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local.run_until(async move {
        std::fs::create_dir_all("/tmp/probequota")?;
        let (client, agent_channel) = acp_channels();
        let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
        let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
        let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
        tokio::task::spawn_local(gw_rx.run());
        let _init = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
        let session = acp_send(
            acp::NewSessionRequest::new(std::path::PathBuf::from("/tmp/probequota")),
            &client.tx,
        )
        .await?;
        println!("SID={}", session.session_id.0);
        // keep the process alive briefly so the session settles
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        Ok::<(), anyhow::Error>(())
    }).await?;
    Ok(())
}
