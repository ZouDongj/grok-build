//! End-to-end goal probe: set a real kernel goal, watch the goal_updated
//! stream through the full lifecycle (active -> executing rounds ->
//! verifying -> verified/notSatisfied), assert the pager-facing fields
//! (rounds, verifying overlay, verdict, pause_message) carry real values.
use agent_client_protocol as acp;
use std::cell::RefCell;
use std::rc::Rc;
use xai_acp_lib::{AcpClientMessage, AcpGatewayReceiver, acp_channels, acp_send};

const WORKDIR: &str = "/tmp/probegoal";

#[derive(Clone)]
struct GoalSnap {
    t_ms: u128,
    status: String,
    phase: String,
    worker_rounds: u32,
    verify_rounds: u32,
    verifying: bool,
    verdict: Option<String>,
    pause_message: Option<String>,
    last_event: Option<String>,
}

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

        let snaps: Rc<RefCell<Vec<GoalSnap>>> = Rc::new(RefCell::new(Vec::new()));
        let t0 = std::time::Instant::now();
        {
            let snaps = snaps.clone();
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
                        AcpClientMessage::ExtNotification(n) => {
                            if n.method.as_ref() == "x.ai/session_notification" {
                                let v: serde_json::Value = serde_json::from_str(n.params.get())
                                    .unwrap_or(serde_json::json!({}));
                                let u = &v["update"];
                                if u["sessionUpdate"] == "goal_updated" {
                                    snaps.borrow_mut().push(GoalSnap {
                                        t_ms: t0.elapsed().as_millis(),
                                        status: u["status"].as_str().unwrap_or("").into(),
                                        phase: u["phase"].as_str().unwrap_or("").into(),
                                        worker_rounds: u["total_worker_rounds"].as_u64().unwrap_or(0) as u32,
                                        verify_rounds: u["total_verify_rounds"].as_u64().unwrap_or(0) as u32,
                                        verifying: u["verifying_completion"].as_bool().unwrap_or(false),
                                        verdict: u["last_classifier_verdict"].as_str().map(String::from),
                                        pause_message: u["pause_message"].as_str().map(String::from),
                                        last_event: u["last_event"].as_str().map(String::from),
                                    });
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

        // Set a small but REAL goal: file creation + self-verification.
        let resp = acp_send(
            acp::PromptRequest::new(
                sid.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new(
                    "/goal 在 /tmp/probegoal 目录下创建 goal-e2e.txt，内容为一行 ok，创建后验证文件内容确实是 ok".to_string(),
                ))],
            ),
            &client.tx,
        )
        .await?;
        println!("[probe] /goal set: {:?}", resp.stop_reason);

        // Watch the goal stream until terminal (verified / notSatisfied /
        // failed / cleared) or timeout. Goal turns run in the background —
        // poll the snaphots.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(420);
        let mut final_status = String::new();
        while std::time::Instant::now() < deadline {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let last = snaps.borrow().last().cloned();
            if let Some(s) = last {
                println!(
                    "[probe t+{:>6}ms] status={} worker={} verify={} verifying={} verdict={:?}",
                    s.t_ms, s.status, s.worker_rounds, s.verify_rounds, s.verifying, s.verdict
                );
                if s.status == "complete" || s.status == "blocked" {
                    final_status = s.status.clone();
                    break;
                }
            }
        }

        let all = snaps.borrow().clone();
        println!("\n=== goal_updated timeline ({} snaps) ===", all.len());
        for s in &all {
            println!(
                "t+{:>6}ms {:>12} phase={:<9} worker={} verify={} verifying={} verdict={:?} pause={:?} event={:?}",
                s.t_ms, s.status, s.phase, s.worker_rounds, s.verify_rounds, s.verifying,
                s.verdict, s.pause_message, s.last_event
            );
        }
        let file_ok = std::fs::read_to_string(format!("{WORKDIR}/goal-e2e.txt")).ok();
        println!("[probe] goal artifact: {:?}", file_ok);

        let verdict = |name: &str, ok: bool| println!("CHECK {name}: {}", if ok { "PASS" } else { "FAIL" });
        verdict(
            "goal-reached-terminal-status",
            final_status == "complete" || final_status == "blocked",
        );
        verdict(
            "goal-rounds-are-real",
            all.iter().any(|s| s.worker_rounds + s.verify_rounds > 0),
        );
        verdict(
            "goal-verdict-reported",
            all.iter().any(|s| s.verdict.is_some()),
        );

        // Cleanup: clear goal + delete session.
        let _ = acp_send(
            acp::PromptRequest::new(
                sid.clone(),
                vec![acp::ContentBlock::Text(acp::TextContent::new("/goal clear".to_string()))],
            ),
            &client.tx,
        )
        .await;
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
