//! Effort persistence: set max via the /model-switch path (setThoughtLevel),
//! then LoadSession — the pushed model state must carry reasoningEffort=max.
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probeeffort";

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
        let model_meta: Rc<RefCell<Option<serde_json::Value>>> = Rc::new(RefCell::new(None));
        {
            let model_meta = model_meta.clone();
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
                        AcpClientMessage::ExtNotification(n) => {
                            if n.method.as_ref() == "x.ai/models/update" {
                                let v: serde_json::Value = serde_json::from_str(n.params.get()).unwrap_or_default();
                                *model_meta.borrow_mut() = Some(v);
                            }
                        }
                        _ => {}
                    }
                }
            });
        }
        let _init = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;
        let session = acp_send(acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)), &client.tx).await?;
        let sid = session.session_id.clone();
        println!("[probe] session {}", sid.0);

        // Run one turn FIRST — the session row (and thus the FK target for
        // model_selection persistence) commits on the first completed turn;
        // setting effort before that hits a kernel FK race.
        let _ = acp_send(
            acp::PromptRequest::new(
                sid.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new("1+1？只回答数字".to_string()))],
            ),
            &client.tx,
        )
        .await?;

        // The /model X max path: SetSessionModel with reasoningEffort meta.
        let mut meta = acp::Meta::new();
        meta.insert("reasoningEffort".to_string(), serde_json::json!("max"));
        let _ = acp_send(
            acp::SetSessionModelRequest::new(
                sid.clone(),
                acp::ModelId::new(std::sync::Arc::from("GLM-5.3")),
            )
            .meta(meta),
            &client.tx,
        )
        .await?;
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

        // Reload: does the model state carry reasoningEffort=max?
        let _ = acp_send(
            acp::LoadSessionRequest::new(sid.clone(), std::path::PathBuf::from(WORKDIR)),
            &client.tx,
        )
        .await?;
        tokio::time::sleep(std::time::Duration::from_millis(2500)).await;
        let v = model_meta.borrow().clone().unwrap_or_default();
        let current = v
            .get("currentModelId")
            .and_then(|m| m.as_str())
            .unwrap_or("?");
        let effort = v
            .pointer("/availableModels")
            .and_then(|a| a.as_array())
            .and_then(|list| {
                list.iter()
                    .find(|m| m.get("modelId").and_then(|x| x.as_str()) == Some(current))
            })
            .and_then(|m| {
                m.pointer("/_meta/reasoningEffort")
                    .or_else(|| m.pointer("/meta/reasoningEffort"))
            })
            .and_then(|e| e.as_str());
        println!("[probe] current={current} effort_after_resume={effort:?}");

        let verdict = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        verdict("effort-restored-after-resume", effort == Some("max"));

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
    })
    .await?;
    Ok(())
}
