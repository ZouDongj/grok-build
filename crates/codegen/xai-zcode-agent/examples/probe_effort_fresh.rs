//! Fresh-session effort flow: set max (persisted) -> NEW session auto-starts
//! with max tag; clear -> NEW session starts with NO tag (kernel default).
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probeeffort2";

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let local = tokio::task::LocalSet::new();
    local.run_until(async move {
        std::fs::create_dir_all(WORKDIR)?;
        // start clean: no persisted effort
        let pref = dirs_pref();
        if let Some(p) = &pref {
            let mut v: serde_json::Value = std::fs::read_to_string(p)
                .ok()
                .and_then(|r| serde_json::from_str(&r).ok())
                .unwrap_or_else(|| serde_json::json!({}));
            if let Some(o) = v.as_object_mut() {
                o.remove("effort");
            }
            std::fs::write(p, v.to_string()).ok();
        }

        let (client, agent_channel) = acp_channels();
        let gateway = xai_acp_lib::AcpGatewaySender::new(agent_channel.tx);
        let agent = Rc::new(xai_zcode_agent::ZcodeAgent::new(gateway, "zcode"));
        let gw_rx = AcpGatewayReceiver::new(agent_channel.rx, agent.clone()).with_tracing(false);
        tokio::task::spawn_local(gw_rx.run());
        let model_state: Rc<RefCell<Option<serde_json::Value>>> = Rc::new(RefCell::new(None));
        {
            let model_state = model_state.clone();
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
                                *model_state.borrow_mut() = Some(v);
                            }
                        }
                        _ => {}
                    }
                }
            });
        }

        let effort_of = |v: &serde_json::Value| -> Option<String> {
            let current = v.get("currentModelId").and_then(|m| m.as_str())?;
            v.pointer("/availableModels")
                .and_then(|a| a.as_array())?
                .iter()
                .find(|m| m.get("modelId").and_then(|x| x.as_str()) == Some(current))?
                .pointer("/_meta/reasoningEffort")
                .and_then(|e| e.as_str())
                .map(str::to_string)
        };

        let _init = acp_send(acp::InitializeRequest::new(acp::ProtocolVersion::V1), &client.tx).await?;

        // 1) fresh session with NO persisted effort: no tag expected
        let s1 = acp_send(acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)), &client.tx).await?.session_id;
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        let e1 = model_state.borrow().clone().and_then(|v| effort_of(&v));
        println!("[probe] fresh (no preference): effort={e1:?}");
        let _ = acp_send(acp::ExtRequest::new("x.ai/session/delete",
            serde_json::value::to_raw_value(&serde_json::json!({"sessionId": s1.0, "cwd": WORKDIR})).unwrap().into()), &client.tx).await;

        // 2) set max on a session (with one turn first for FK safety)
        let s2 = acp_send(acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)), &client.tx).await?.session_id;
        let _ = acp_send(acp::PromptRequest::new(s2.clone(), vec![acp::ContentBlock::Text(acp::TextContent::new("1+1？只回答数字".to_string()))]), &client.tx).await;
        let mut meta = acp::Meta::new();
        meta.insert("reasoningEffort".to_string(), serde_json::json!("max"));
        let _ = acp_send(acp::SetSessionModelRequest::new(s2.clone(), acp::ModelId::new(std::sync::Arc::from("GLM-5.3"))).meta(meta), &client.tx).await;
        tokio::time::sleep(std::time::Duration::from_millis(800)).await;
        let pref_effort = std::fs::read_to_string(pref.clone().unwrap_or_default()).ok()
            .and_then(|r| serde_json::from_str::<serde_json::Value>(&r).ok())
            .and_then(|v| v.get("effort").and_then(|e| e.as_str().map(str::to_string)));
        println!("[probe] after /effort max: model.json effort={pref_effort:?}");
        let _ = acp_send(acp::ExtRequest::new("x.ai/session/delete",
            serde_json::value::to_raw_value(&serde_json::json!({"sessionId": s2.0, "cwd": WORKDIR})).unwrap().into()), &client.tx).await;

        // 3) NEW session: should auto-start with max
        *model_state.borrow_mut() = None;
        let s3 = acp_send(acp::NewSessionRequest::new(std::path::PathBuf::from(WORKDIR)), &client.tx).await?.session_id;
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        let e3 = model_state.borrow().clone().and_then(|v| effort_of(&v));
        println!("[probe] fresh (after max persisted): effort={e3:?}");
        let _ = acp_send(acp::ExtRequest::new("x.ai/session/delete",
            serde_json::value::to_raw_value(&serde_json::json!({"sessionId": s3.0, "cwd": WORKDIR})).unwrap().into()), &client.tx).await;

        let v = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        v("fresh-no-effort-tag-when-no-preference", e1.is_none());
        v("effort-persisted-to-model-json", pref_effort.as_deref() == Some("max"));
        v("fresh-session-inherits-persisted-max", e3.as_deref() == Some("max"));

        Ok::<(), anyhow::Error>(())
    })
    .await?;
    Ok(())
}

fn dirs_pref() -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|v| !v.is_empty())
                .map(|h| std::path::PathBuf::from(h).join(".config"))
        })
        .map(|b| b.join("zcode-tui").join("model.json"))
}
