//! `acp::Agent` implementation backed by the ZCode kernel.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use agent_client_protocol as acp;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use xai_acp_lib::AcpGatewaySender;

use crate::kernel::{self, Kernel, KernelMessage, TurnEvent};

const DEFAULT_PROVIDER: &str = "account:bigmodel-individual-coding-plan";
const DEFAULT_MODEL: &str = "GLM-5.3";

pub struct ZcodeAgent {
    gateway: AcpGatewaySender<acp::AgentSide>,
    shared: Rc<Shared>,
}

struct Shared {
    kernel_bin: String,
    state: RefCell<AgentState>,
}

#[derive(Default)]
struct AgentState {
    kernel: Option<Kernel>,
    /// Full official catalog (workspace/readState) cached for pushes.
    catalog: Option<acp::SessionModelState>,
    sessions: HashMap<acp::SessionId, Rc<SessionState>>,
    /// request_id dedupe — the kernel re-sends interactions under fresh
    /// envelope ids with backoff until answered.
    seen_interactions: HashSet<String>,
}

struct SessionState {
    /// Kernel and ACP session ids are the same string (sess_…).
    turn_done: RefCell<Option<oneshot::Sender<acp::StopReason>>>,
    cancelled: std::cell::Cell<bool>,
    /// Set once any text_delta streams this turn: suppresses the duplicate
    /// full `response` echo on turn.completed.
    streamed_text: std::cell::Cell<bool>,
    tools: RefCell<HashMap<String, acp::ToolCallId>>,
    next_tool: std::cell::Cell<u64>,
    /// Prompt to send when the current turn finalizes (plan-approval
    /// continuation: the kernel does not continue on its own after an
    /// approved or revised plan).
    pending_continuation: RefCell<Option<String>>,
}

impl SessionState {
    fn new() -> SessionState {
        SessionState {
            turn_done: RefCell::new(None),
            cancelled: std::cell::Cell::new(false),
            streamed_text: std::cell::Cell::new(false),
            tools: RefCell::new(HashMap::new()),
            next_tool: std::cell::Cell::new(0),
            pending_continuation: RefCell::new(None),
        }
    }

    fn acp_tool_id(&self) -> acp::ToolCallId {
        let n = self.next_tool.replace(self.next_tool.get() + 1);
        acp::ToolCallId::new(format!("tc-{n}"))
    }
}

impl ZcodeAgent {
    pub fn new(gateway: AcpGatewaySender<acp::AgentSide>, kernel_bin: impl Into<String>) -> ZcodeAgent {
        // A panicking pump/gateway task dies silently otherwise (spawn_local
        // JoinHandles are never awaited): capture every panic into the debug
        // log so a wedged turn leaves evidence instead of a mystery hang.
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            debug_log(&format!(
                "PANIC: {info} at {:?}",
                info.location().map(|l| l.to_string())
            ));
            hook(info);
        }));
        ZcodeAgent {
            gateway,
            shared: Rc::new(Shared {
                kernel_bin: kernel_bin.into(),
                state: RefCell::new(AgentState::default()),
            }),
        }
    }

    /// Read the kernel's model catalog for this session and forward it to the
    /// pager as `x.ai/models/update` (params = acp::SessionModelState).
    async fn push_model_state(&self, kernel: &Kernel, session_id: &str) {
        debug_log("push_model_state: session/read");
        let Ok(read) = kernel.call("session/read", kernel::read_params(session_id)).await else {
            debug_log("push_model_state: session/read FAILED");
            return;
        };
        debug_log("push_model_state: session/read ok");
        let Some(mut state) = model_state_from_settings(&read) else {
            return;
        };
        // Prefer the cached full catalog; runtime session/read entries are
        // AUTHORITATIVE for models they list (kernel-resolved context
        // window, modalities, reasoning levels) and override the fold.
        if let Some(catalog) = self.shared.state.borrow().catalog.clone() {
            if catalog.available_models.len() > state.available_models.len() {
                let runtime: std::collections::HashMap<String, acp::ModelInfo> = read
                    .pointer("/settings/model/available")
                    .and_then(Value::as_array)
                    .map(|entries| {
                        entries
                            .iter()
                            .filter_map(crate::catalog::model_info_from_runtime_entry)
                            .map(|m| (m.model_id.0.as_ref().to_string(), m))
                            .collect()
                    })
                    .unwrap_or_default();
                state.available_models = catalog
                    .available_models
                    .iter()
                    .map(|m| match runtime.get(m.model_id.0.as_ref()) {
                        Some(authoritative) => authoritative.clone(),
                        None => m.clone(),
                    })
                    .collect();
            }
        }
        if let Ok(params) = serde_json::value::to_raw_value(&state) {
            self.gateway.forward_fire_and_forget(acp::ExtNotification::new(
                "x.ai/models/update",
                params.into(),
            ));
        }
        // The composer/status model label is SESSION-scoped: it only moves on
        // a ModelChanged session notification, not on the catalog update.
        {
            let current = &state.current_model_id;
            let payload = json!({
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "model_changed",
                    "model_id": current.0.as_ref(),
                },
            });
            if let Ok(params) = serde_json::value::to_raw_value(&payload) {
                self.gateway.forward_fire_and_forget(acp::ExtNotification::new(
                    "x.ai/session_notification",
                    params.into(),
                ));
            }
        }
    }

    /// Full catalog with zero kernel round-trips: the account provider and
    /// current model come from the kernel config's `model/main`; every
    /// enabled official model comes from the kernel's bundled registry.
    async fn refresh_catalog(&self, _kernel: &Kernel) {
        let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) else {
            return;
        };
        let Ok(config_raw) = std::fs::read_to_string(home.join(".zcode/cli/config.json")) else {
            return;
        };
        let Ok(config) = serde_json::from_str::<Value>(&config_raw) else {
            return;
        };
        let Some(main) = config
            .pointer("/model/main")
            .and_then(Value::as_str)
            .or_else(|| config.get("model").and_then(Value::as_str))
            .and_then(|main| main.split_once('/'))
        else {
            return;
        };
        let (provider_id, current) = (main.0, main.1);
        let mut models = crate::catalog::bundled_models(&home, provider_id);
        if models.is_empty() {
            return;
        }
        let current_id = acp::ModelId::new(current.to_string());
        if !models.iter().any(|m| m.model_id == current_id) {
            models.insert(0, acp::ModelInfo::new(current.to_string(), current.to_string()));
        }
        let catalog = acp::SessionModelState::new(current_id, models);
        if let Ok(raw) = serde_json::value::to_raw_value(&catalog) {
            self.gateway.forward_fire_and_forget(acp::ExtNotification::new(
                "x.ai/models/update",
                raw.into(),
            ));
        }
        self.shared.state.borrow_mut().catalog = Some(catalog);
    }

    /// The pager's session listings, answered from the kernel's own session
    /// store. Two consumers, two shapes: `x.ai/session/list` (resume picker)
    /// wants `{sessions:[{sessionId, summary, updatedAt(RFC3339), cwd, …}]}`,
    /// `x.ai/sessions/list` (roster) wants RosterEntry camelCase. Also drops
    /// a `summary.json` stub per session into grok's local store — the pager
    /// refuses to load a session its own store cannot resolve.
    async fn sessions_list(&self, method: &str) -> acp::Result<acp::ExtResponse> {
        let kernel = self.kernel()?;
        let listed = kernel
            .call("session/list", json!({}))
            .await
            .map_err(|e| acp::Error::internal_error().data(format!("session/list failed: {e}")))?;
        let roster = method == "x.ai/sessions/list";
        let mut entries = Vec::new();
        if let Some(sessions) = listed.get("sessions").and_then(Value::as_array) {
            for session in sessions {
                let Some(id) = session.get("sessionId").and_then(Value::as_str) else {
                    continue;
                };
                let cwd = session
                    .pointer("/workspace/workspacePath")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let title = session.get("title").and_then(Value::as_str).unwrap_or("");
                let updated_ms = session.get("updatedAt").and_then(Value::as_i64).unwrap_or(0);
                let created_ms = session.get("createdAt").and_then(Value::as_i64).unwrap_or(updated_ms);
                let running = session.get("status").and_then(Value::as_str) == Some("running");
                if roster {
                    entries.push(json!({
                        "sessionId": id,
                        "title": title,
                        "cwd": cwd,
                        "isWorktree": false,
                        "sessionKind": if session.get("mode").and_then(Value::as_str) == Some("plan") { "plan" } else { "build" },
                        "yolo": false,
                        "activity": if running { "working" } else { "idle" },
                        "resident": false,
                        "lastChangeUnixMs": updated_ms,
                        "origin": {"kind": "local"},
                    }));
                } else {
                    // The picker drops entries without a parseable RFC3339
                    // updatedAt or any display text (summary).
                    entries.push(json!({
                        "sessionId": id,
                        "summary": title,
                        "cwd": cwd,
                        "createdAt": ms_to_rfc3339(created_ms),
                        "updatedAt": ms_to_rfc3339(updated_ms),
                        "lastActiveAt": ms_to_rfc3339(updated_ms),
                        "source": "local",
                        "running": running,
                    }));
                }
                if !cwd.is_empty() {
                    write_summary_stub(id, cwd);
                }
            }
        }
        debug_log(&format!("{method}: {} sessions", entries.len()));
        let body = json!({"result": {"sessions": entries}});
        let params = serde_json::value::to_raw_value(&body).expect("serialize session list");
        Ok(acp::ExtResponse::new(params.into()))
    }

    fn kernel(&self) -> Result<Kernel, acp::Error> {
        let kernel = self
            .shared
            .state
            .borrow()
            .kernel
            .clone()
            .ok_or_else(|| acp::Error::internal_error().data("zcode kernel not running"))?;
        if !kernel.is_alive() {
            return Err(acp::Error::internal_error().data("zcode kernel exited"));
        }
        Ok(kernel)
    }

    /// Spawn the kernel once and start the event pump on this LocalSet.
    /// A kernel that died since the last use is replaced with a fresh one
    /// (its sessions are gone; subsequent calls on them fail visibly).
    async fn ensure_kernel(&self) -> Result<Kernel, acp::Error> {
        let existing = self.shared.state.borrow().kernel.clone();
        if let Some(kernel) = existing {
            if kernel.is_alive() {
                return Ok(kernel);
            }
            debug_log("ensure_kernel: previous kernel dead, respawning");
            self.shared.state.borrow_mut().kernel = None;
        }
        let (kernel, inbound) = Kernel::spawn(&self.shared.kernel_bin)
            .map_err(|e| acp::Error::internal_error().data(format!("zcode app-server spawn failed: {e}")))?;
        self.shared.state.borrow_mut().kernel = Some(kernel.clone());
        // Register the runtime provider (all official models) and cache the
        // full catalog — session/read alone only reports the current model.
        self.refresh_catalog(&kernel).await;
        let pump_shared = Rc::clone(&self.shared);
        let pump_gateway = self.gateway.clone();
        tokio::task::spawn_local(async move {
            let mut inbound = inbound;
            let mut last_model: Option<String> = None;
            debug_log("pump: started");
            while let Some(message) = inbound.recv().await {
                match message {
                    KernelMessage::Event { session_id, payload } => {
                        handle_event(&pump_gateway, &pump_shared, session_id.as_deref(), &payload);
                    }
                    KernelMessage::ServerRequest { id, method, params } => {
                        debug_log(&format!("pump: server request {method}"));
                        handle_server_request(&pump_gateway, &pump_shared, &id, &method, params)
                            .await;
                        debug_log(&format!("pump: server request {method} answered"));
                    }
                    KernelMessage::StateUpdated(params) => {
                        // The full model catalog (all official models, not
                        // just the current one) rides state.updated patches.
                        let Some(state) = model_state_from_patch(&params) else {
                            continue;
                        };
                        let current = state.current_model_id.0.as_ref().to_string();
                        let catalog_grew = state.available_models.len()
                            > last_model.as_ref().map(|_| 1).unwrap_or(0);
                        if last_model.as_deref() != Some(&current) || catalog_grew {
                            if let Ok(raw) = serde_json::value::to_raw_value(&state) {
                                pump_gateway.forward_fire_and_forget(
                                    acp::ExtNotification::new("x.ai/models/update", raw.into()),
                                );
                            }
                            if let Some(session_id) = params.get("sessionId").and_then(Value::as_str) {
                                let payload = json!({
                                    "sessionId": session_id,
                                    "update": {
                                        "sessionUpdate": "model_changed",
                                        "model_id": current,
                                    },
                                });
                                if let Ok(raw) = serde_json::value::to_raw_value(&payload) {
                                    pump_gateway.forward_fire_and_forget(
                                        acp::ExtNotification::new(
                                            "x.ai/session_notification",
                                            raw.into(),
                                        ),
                                    );
                                }
                            }
                            last_model = Some(current);
                        }
                    }
                }
            }
            tracing::info!("zcode kernel inbound stream closed");
            // The kernel is gone: unhang every open turn (its `prompt()`
            // future parks on turn_done) and drop the dead handle so the next
            // `ensure_kernel` respawns a fresh kernel.
            let mut state = pump_shared.state.borrow_mut();
            state.kernel = None;
            for (session, session_state) in state.sessions.iter() {
                notify(
                    &pump_gateway,
                    session,
                    acp::SessionUpdate::AgentMessageChunk(text_chunk(
                        "zcode kernel exited unexpectedly; turn aborted",
                    )),
                );
                finish_turn(session_state, &pump_gateway, session, acp::StopReason::Cancelled);
            }
        });
        Ok(kernel)
    }
}

/// Percent-encode a cwd the way grok's session store lays out directories
/// (`<grok-home>/sessions/<encoded-cwd>/<session-id>/summary.json`):
/// unreserved bytes stay, everything else becomes uppercase `%XX`. The
/// shell's over-length fallback (slug+blake3) is not replicated — sessions
/// under such paths stay resumable only within their own process lifetime.
fn encode_cwd_dirname(cwd: &str) -> Option<String> {
    if cwd.len() > 200 {
        return None;
    }
    let mut out = String::with_capacity(cwd.len());
    for byte in cwd.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    Some(out)
}

/// Grok's resume gate resolves a session through its own on-disk store and
/// only checks that `summary.json` EXISTS — the listing itself comes from the
/// kernel via `x.ai/sessions/list`. Write the stub so picked sessions load.
fn write_summary_stub(session_id: &str, cwd: &str) {
    let Some(home) = std::env::var_os("GROK_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(std::path::PathBuf::from))
    else {
        return;
    };
    let Some(encoded) = encode_cwd_dirname(cwd) else { return };
    let path = home
        .join(".grok")
        .join("sessions")
        .join(encoded)
        .join(session_id)
        .join("summary.json");
    if path.exists() {
        return;
    }
    let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new("")));
    let summary = json!({
        "info": {"id": session_id, "cwd": cwd},
        "session_summary": "",
        "created_at": "1970-01-01T00:00:00Z",
        "updated_at": "1970-01-01T00:00:00Z",
        "num_messages": 0,
        "current_model_id": "GLM-5.3",
    });
    if let Err(error) = std::fs::write(&path, summary.to_string()) {
        tracing::warn!(%error, path = %path.display(), "summary stub write failed");
    }
}

/// Remove one session from the kernel's store. The app-server protocol has
/// no delete method; the desktop app edits this same db. Best-effort per
/// child table (schema varies across kernel versions), but the session row
/// itself must delete (an absent id is already-deleted = success). WAL +
/// busy timeout coexist with live kernels.
fn delete_kernel_session(session_id: &str, cwd: &str) -> Result<(), String> {
    use rusqlite::Connection;

    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .ok_or("no HOME")?;
    let db_path = home.join(".zcode/cli/db/db.sqlite");
    let db = Connection::open(&db_path)
        .and_then(|conn| {
            conn.busy_timeout(std::time::Duration::from_secs(3))?;
            Ok(conn)
        })
        .map_err(|e| format!("open db: {e}"))?;
    let children = [
        "DELETE FROM message WHERE session_id = ?1",
        "DELETE FROM part WHERE session_id = ?1",
        "DELETE FROM todo WHERE session_id = ?1",
        "DELETE FROM session_entry WHERE session_id = ?1",
        "DELETE FROM input_history WHERE session_id = ?1",
        "DELETE FROM session_target WHERE session_id = ?1",
        "DELETE FROM turn_usage WHERE session_id = ?1",
        "DELETE FROM model_usage WHERE session_id = ?1",
        "DELETE FROM tool_usage WHERE session_id = ?1",
        "DELETE FROM session_input WHERE session_id = ?1",
        "DELETE FROM dwf_actor WHERE session_id = ?1",
        "DELETE FROM workflow_run WHERE parent_session_id = ?1",
        "DELETE FROM dwf_run WHERE parent_session_id = ?1",
        "DELETE FROM workflow_activity WHERE child_session_id = ?1",
        "DELETE FROM session_task_link WHERE parent_session_id = ?1 OR child_session_id = ?1",
    ];
    db.execute_batch("BEGIN")
        .map_err(|e| format!("begin: {e}"))?;
    for sql in children {
        // Missing table (older/newer schema) must not abort the whole delete.
        let _ = db.execute(sql, [session_id]);
    }
    let removed = db
        .execute("DELETE FROM session WHERE id = ?1", [session_id])
        .map_err(|e| format!("delete session: {e}"));
    match removed {
        Ok(_) => {
            let _ = db.execute_batch("COMMIT");
        }
        Err(error) => {
            let _ = db.execute_batch("ROLLBACK");
            return Err(error);
        }
    }
    // File-layer traces: model-io rollout, artifacts, and the grok resume
    // stub written for the picker gate.
    let _ = std::fs::remove_file(home.join(format!(
        ".zcode/cli/rollout/model-io-{session_id}.jsonl"
    )));
    let _ = std::fs::remove_dir_all(home.join(format!(".zcode/cli/artifacts/{session_id}")));
    if !cwd.is_empty() {
        if let Some(encoded) = encode_cwd_dirname(cwd) {
            let _ = std::fs::remove_dir_all(
                home.join(".grok/sessions").join(encoded).join(session_id),
            );
        }
    }
    Ok(())
}

/// Epoch-ms → RFC3339 (UTC) without a chrono dependency. The resume picker
/// drops entries whose `updatedAt` does not parse as an RFC3339 string.
fn ms_to_rfc3339(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let millis = ms.rem_euclid(1000);
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (hour, minute, second) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

fn debug_log(text: &str) {
    use std::io::Write;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/zcode-agent-debug.log")
    {
        let _ = writeln!(f, "[{now}] {text}");
    }
}

#[async_trait::async_trait(?Send)]
impl acp::Agent for ZcodeAgent {
    async fn initialize(&self, _args: acp::InitializeRequest) -> acp::Result<acp::InitializeResponse> {
        debug_log("acp: initialize");
        // Advertise a non-grok.com agent method: the pager's welcome screen
        // treats that as "credentials handled outside ACP" and skips its
        // login flow — the ZCode kernel owns auth (config.json / coding plan).
        let method = acp::AuthMethod::Agent(
            acp::AuthMethodAgent::new(
                acp::AuthMethodId::new("zcode.kernel"),
                "ZCode kernel credentials".to_string(),
            )
            .description(Some("configured by the zcode CLI (config.json)".to_string())),
        );
        Ok(acp::InitializeResponse::new(acp::ProtocolVersion::V1)
            .agent_info(
                acp::Implementation::new("zcode", env!("CARGO_PKG_VERSION"))
                    .title("ZCode (official kernel)"),
            )
            .agent_capabilities(acp::AgentCapabilities::new().load_session(true))
            .auth_methods(vec![method]))
    }

    async fn authenticate(&self, _args: acp::AuthenticateRequest) -> acp::Result<acp::AuthenticateResponse> {
        debug_log("acp: authenticate");
        // The kernel owns credentials (config.json / coding plan); no ACP-level auth.
        Ok(acp::AuthenticateResponse::default())
    }

    async fn new_session(&self, args: acp::NewSessionRequest) -> acp::Result<acp::NewSessionResponse> {
        debug_log("acp: new_session");
        let kernel = self.ensure_kernel().await?;
        let cwd = args.cwd.clone();
        let created = kernel
            .call("session/create", kernel::create_params(&cwd))
            .await
            .map_err(|e| acp::Error::internal_error().data(format!("session/create failed: {e}")))?;
        let Some(session_id) = kernel::session_id_from(&created) else {
            return Err(acp::Error::internal_error().data("session/create returned no sessionId"));
        };
        kernel
            .call("session/subscribe", kernel::subscribe_params(&session_id))
            .await
            .map_err(|e| acp::Error::internal_error().data(format!("session/subscribe failed: {e}")))?;
        // Best-effort model preference (same file zcode-tui persists); the
        // kernel default is fine when absent.
        if let Some((provider, model)) = load_model_preference() {
            let _ = kernel
                .call("session/setModel", kernel::set_model_params(&session_id, &provider, &model))
                .await;
        }
        let acp_session = acp::SessionId::new(session_id.clone());
        self.shared
            .state
            .borrow_mut()
            .sessions
            .insert(acp_session.clone(), Rc::new(SessionState::new()));
        // Resumable through the pager's local-store gate from the start.
        write_summary_stub(&session_id, &cwd.to_string_lossy());
        // The OFFICIAL session-scoped model channel: NewSessionResponse.models.
        let initial_models = self.shared.state.borrow().catalog.clone();
        // Belt and braces: the catalog also flows via NewSessionResponse
        // above and the x.ai ext notifications below.
        self.push_model_state(&kernel, &session_id).await;
        tracing::info!(%session_id, "zcode session created");
        let mut response = acp::NewSessionResponse::new(acp_session);
        response.models = initial_models;
        Ok(response)
    }

    async fn load_session(&self, args: acp::LoadSessionRequest) -> acp::Result<acp::LoadSessionResponse> {
        debug_log("acp: load_session");
        let kernel = self.ensure_kernel().await?;
        let session_id = args.session_id.0.as_ref().to_string();
        let resumed = kernel
            .call("session/resume", kernel::resume_params(&session_id))
            .await
            .map_err(|e| acp::Error::internal_error().data(format!("session/resume failed: {e}")))?;
        kernel
            .call("session/subscribe", kernel::subscribe_params(&session_id))
            .await
            .map_err(|e| acp::Error::internal_error().data(format!("session/subscribe failed: {e}")))?;
        // Resume restores the conversation but not the model runtime —
        // revive it or the first send fails with ZCODE_RUNTIME_MODEL_UNAVAILABLE.
        if let Some((provider, model)) = load_model_preference() {
            let _ = kernel
                .call("session/setModel", kernel::set_model_params(&session_id, &provider, &model))
                .await;
        }
        self.shared
            .state
            .borrow_mut()
            .sessions
            .insert(args.session_id.clone(), Rc::new(SessionState::new()));
        let cwd_text = args.cwd.to_string_lossy().to_string();
        write_summary_stub(&session_id, &cwd_text);

        // Transcript replay (full-fidelity, zcode-tui parity): every
        // user/assistant turn with its reasoning parts, sent BEFORE the
        // response so the pager renders history as part of the load.
        if let Some(messages) = resumed.get("messages").and_then(Value::as_array) {
            for message in messages {
                let Some(role) = message.pointer("/info/role").and_then(Value::as_str) else {
                    continue;
                };
                if role != "user" && role != "assistant" {
                    continue;
                }
                let mut reasoning = String::new();
                let mut text = String::new();
                if let Some(parts) = message.get("parts").and_then(Value::as_array) {
                    for part in parts {
                        match part.get("type").and_then(Value::as_str) {
                            Some("text") => {
                                text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""))
                            }
                            Some("reasoning") => {
                                reasoning.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""))
                            }
                            _ => {}
                        }
                    }
                }
                if role == "assistant" && !reasoning.trim().is_empty() {
                    notify(
                        &self.gateway,
                        &args.session_id,
                        acp::SessionUpdate::AgentThoughtChunk(text_chunk(reasoning)),
                    );
                }
                if !text.trim().is_empty() {
                    let update = if role == "user" {
                        acp::SessionUpdate::UserMessageChunk(text_chunk(text))
                    } else {
                        acp::SessionUpdate::AgentMessageChunk(text_chunk(text))
                    };
                    notify(&self.gateway, &args.session_id, update);
                }
            }
        }
        // Initial plan indicator from the resumed session's mode.
        if resumed.pointer("/session/mode").and_then(Value::as_str) == Some("plan") {
            notify(
                &self.gateway,
                &args.session_id,
                acp::SessionUpdate::CurrentModeUpdate(acp::CurrentModeUpdate::new(
                    acp::SessionModeId::new("plan"),
                )),
            );
        }
        let initial_models = self.shared.state.borrow().catalog.clone();
        self.push_model_state(&kernel, &session_id).await;
        tracing::info!(%session_id, "zcode session resumed");
        let mut response = acp::LoadSessionResponse::new();
        response.models = initial_models;
        Ok(response)
    }

    async fn prompt(&self, args: acp::PromptRequest) -> acp::Result<acp::PromptResponse> {
        debug_log("acp: prompt");
        let kernel = self.kernel()?;
        let state = self
            .shared
            .state
            .borrow()
            .sessions
            .get(&args.session_id)
            .cloned()
            .ok_or_else(|| acp::Error::invalid_params().data("unknown session"))?;
        let text = prompt_text(&args.prompt);
        let (tx, rx) = oneshot::channel();
        *state.turn_done.borrow_mut() = Some(tx);
        state.cancelled.set(false);
        state.streamed_text.set(false);
        kernel
            .request("session/send", kernel::send_params(&args.session_id.0, &text))
            .map_err(|e| acp::Error::internal_error().data(e.to_string()))?;
        let stop = rx.await.unwrap_or(acp::StopReason::EndTurn);
        debug_log(&format!("acp: prompt done stop={stop:?}"));
        // Turn-end broadcast (MvpAgent parity): the pager's queue rail
        // finalizes a running turn from `x.ai/session/prompt_complete`, not
        // from the PromptResponse alone, and matches responses to prompts by
        // `meta.promptId`. Emit the broadcast BEFORE responding, like the
        // grok shell does.
        let stop_str = match stop {
            acp::StopReason::Cancelled => "cancelled",
            _ => "end_turn",
        };
        let prompt_id = args.meta.as_ref().and_then(|m| m.get("promptId")).cloned();
        let mut payload = json!({
            "sessionId": args.session_id.0,
            "stopReason": stop_str,
        });
        if let Some(prompt_id) = prompt_id.as_ref() {
            payload["promptId"] = prompt_id.clone();
        }
        if let Ok(params) = serde_json::value::to_raw_value(&payload) {
            self.gateway.forward_fire_and_forget(acp::ExtNotification::new(
                "x.ai/session/prompt_complete",
                params.into(),
            ));
        }
        let mut response = acp::PromptResponse::new(stop);
        response.meta = args.meta.clone();
        Ok(response)
    }

    async fn cancel(&self, args: acp::CancelNotification) -> acp::Result<()> {
        debug_log("acp: cancel");
        let kernel = self.kernel()?;
        if let Some(state) = self.shared.state.borrow().sessions.get(&args.session_id) {
            state.cancelled.set(true);
        }
        let _ = kernel.request("session/stop", kernel::stop_params(&args.session_id.0));
        Ok(())
    }

    async fn set_session_mode(
        &self,
        args: acp::SetSessionModeRequest,
    ) -> acp::Result<acp::SetSessionModeResponse> {
        debug_log("acp: set_session_mode");
        // grok's plan mode maps onto the kernel's plan mode; anything else
        // returns to the default (build) mode. Confirm via CurrentModeUpdate
        // so the pager's optimistic staging settles.
        let kernel = self.kernel()?;
        let kernel_mode = if args.mode_id.0.as_ref() == "plan" { "plan" } else { "build" };
        kernel
            .call("session/setMode", kernel::set_mode_params(&args.session_id.0, kernel_mode))
            .await
            .map_err(|e| acp::Error::internal_error().data(format!("setMode failed: {e}")))?;
        self.gateway.forward_fire_and_forget(acp::SessionNotification::new(
            args.session_id.clone(),
            acp::SessionUpdate::CurrentModeUpdate(acp::CurrentModeUpdate::new(
                args.mode_id.clone(),
            )),
        ));
        Ok(acp::SetSessionModeResponse::default())
    }

    async fn set_session_model(
        &self,
        args: acp::SetSessionModelRequest,
    ) -> acp::Result<acp::SetSessionModelResponse> {
        debug_log("acp: set_session_model");
        let kernel = self.kernel()?;
        let model = &*args.model_id.0;
        let provider = load_model_preference().map(|(p, _)| p).unwrap_or(DEFAULT_PROVIDER.to_string());
        debug_log(&format!("set_session_model: setModel {provider}/{model}"));
        kernel
            .call("session/setModel", kernel::set_model_params(&args.session_id.0, &provider, model))
            .await
            .map_err(|e| {
                debug_log(&format!("set_session_model: setModel FAILED: {e}"));
                acp::Error::internal_error().data(format!("setModel failed: {e}"))
            })?;
        debug_log("set_session_model: setModel ok");
        // Reasoning effort (grok's /effort and the picker's [effort] arg)
        // rides the request meta as `reasoningEffort` — forward to the
        // kernel's thoughtLevel (low/high/max, GLM official levels).
        if let Some(effort) = args
            .meta
            .as_ref()
            .and_then(|meta| meta.get("reasoningEffort"))
            .and_then(Value::as_str)
        {
            debug_log(&format!("set_session_model: setThoughtLevel {effort}"));
            if let Err(error) = kernel
                .call(
                    "session/setThoughtLevel",
                    kernel::set_thought_params(&args.session_id.0, effort),
                )
                .await
            {
                debug_log(&format!("set_session_model: setThoughtLevel FAILED: {error}"));
            }
        }
        debug_log("set_session_model: session/read");
        self.push_model_state(&kernel, &args.session_id.0).await;
        debug_log("set_session_model: done");
        Ok(acp::SetSessionModelResponse::default())
    }

    async fn ext_method(&self, args: acp::ExtRequest) -> acp::Result<acp::ExtResponse> {
        debug_log(&format!("acp: ext_method {}", args.method));
        match args.method.as_ref() {
            // Resume picker (singular) and dashboard roster (plural).
            "x.ai/session/list" | "x.ai/sessions/list" => {
                return self.sessions_list(args.method.as_ref()).await;
            }
            // The picker's delete action. The kernel protocol has no delete
            // method, so this removes the session from the kernel's own
            // store directly (db + rollout/artifacts + grok resume stub).
            "x.ai/session/delete" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params.get("sessionId").and_then(Value::as_str).unwrap_or_default();
                let cwd = params.get("cwd").and_then(Value::as_str).unwrap_or_default();
                if !id.starts_with("sess_") {
                    return Err(acp::Error::invalid_params().data("bad sessionId"));
                }
                delete_kernel_session(id, cwd).map_err(|e| {
                    acp::Error::internal_error().data(format!("session delete failed: {e}"))
                })?;
                debug_log(&format!("session deleted: {id}"));
                let body = json!({"result": {"ok": true, "sessionId": id}});
                let raw = serde_json::value::to_raw_value(&body).expect("serialize delete ack");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            _ => {}
        }
        Err(acp::Error::method_not_found())
    }
}

/// Build the ACP model state from a session/read result: the kernel reports
/// `/model/available[] {label, ref{providerId, modelId}, contextWindow}` and
/// `/model/current` (zcode-tui's controls_from_settings shape).
fn model_state_from_patch(patch: &Value) -> Option<acp::SessionModelState> {
    let model = patch.pointer("/model")?;
    model_state_from_model(model)
}

fn model_state_from_settings(read: &Value) -> Option<acp::SessionModelState> {
    let model = read.pointer("/settings/model")?;
    model_state_from_model(model)
}

pub fn model_state_from_model(model: &Value) -> Option<acp::SessionModelState> {
    let available = model.get("available")?.as_array()?;
    let mut models = Vec::new();
    for entry in available {
        let model_id = entry
            .pointer("/ref/modelId")
            .and_then(Value::as_str)?
            .to_string();
        let mut info = acp::ModelInfo::new(model_id.clone(), {
            let label = entry.get("label").and_then(Value::as_str).unwrap_or(&model_id);
            label.to_string()
        });
        if let Some(window) = entry.get("contextWindow").and_then(Value::as_u64) {
            let mut meta = acp::Meta::new();
            meta.insert("totalContextTokens".to_string(), json!(window));
            info.meta = Some(meta);
        }
        models.push(info);
    }
    if models.is_empty() {
        return None;
    }
    let current = model
        .pointer("/current/modelId")
        .and_then(Value::as_str)
        .map(|id| acp::ModelId::new(id.to_string()))
        .unwrap_or_else(|| models[0].model_id.clone());
    Some(acp::SessionModelState::new(current, models))
}

/// Flatten a prompt's text blocks into one string.
fn prompt_text(prompt: &[acp::ContentBlock]) -> String {
    let mut out = String::new();
    for block in prompt {
        if let acp::ContentBlock::Text(text) = block {
            out.push_str(&text.text);
            out.push('\n');
        }
    }
    out.trim_end().to_string()
}

/// ~/.config/zcode-tui/model.json — the /model preference zcode-tui already
/// persists; reuse it so both frontends agree.
fn load_model_preference() -> Option<(String, String)> {
    let path = dirs_config_file("model.json")?;
    let raw = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&raw).ok()?;
    Some((
        value.get("providerId")?.as_str()?.to_string(),
        value.get("modelId")?.as_str()?.to_string(),
    ))
}

fn dirs_config_file(name: &str) -> Option<std::path::PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(std::path::PathBuf::from))
        .map(|base| base.join("zcode-tui").join(name))
}

fn notify(gateway: &AcpGatewaySender<acp::AgentSide>, session: &acp::SessionId, update: acp::SessionUpdate) {
    gateway.forward_fire_and_forget(acp::SessionNotification::new(session.clone(), update));
}

fn text_chunk(text: impl Into<String>) -> acp::ContentChunk {
    acp::ContentChunk::new(acp::ContentBlock::Text(acp::TextContent::new(text)))
}

/// Translate one kernel `session/event` payload into ACP session updates.
/// Runs on the pump task (single LocalSet thread) — sessions are Rc.
fn handle_event(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    shared: &Rc<Shared>,
    session_id: Option<&str>,
    payload: &Value,
) {
    let Some(session_id) = session_id else { return };
    let acp_session = acp::SessionId::new(session_id.to_string());
    let Some(state) = shared.state.borrow().sessions.get(&acp_session).cloned() else {
        debug_log(&format!("handle_event: UNKNOWN session {session_id}"));
        return;
    };
    let Some(event) = TurnEvent::decode(payload) else {
        debug_log(&format!("handle_event: undecodable payload kind={:?}", payload.get("kind")));
        return;
    };
    match event.kind.as_str() {
        "text_delta" => {
            if !event.delta.is_empty() {
                state.streamed_text.set(true);
                notify(
                    gateway,
                    &acp_session,
                    acp::SessionUpdate::AgentMessageChunk(text_chunk(event.delta)),
                );
            }
        }
        "reasoning_delta" => {
            if !event.delta.is_empty() {
                notify(
                    gateway,
                    &acp_session,
                    acp::SessionUpdate::AgentThoughtChunk(text_chunk(event.delta)),
                );
            }
        }
        "tool_input_start" | "tool_call" => {
            let Some(call_id) = event.tool_call_id.as_deref() else { return };
            let acp_id = state.acp_tool_id();
            state.tools.borrow_mut().insert(call_id.to_string(), acp_id.clone());
            let call = acp::ToolCall::new(acp_id, event.tool_name.clone().unwrap_or_else(|| "tool".into()))
                .status(acp::ToolCallStatus::InProgress);
            notify(gateway, &acp_session, acp::SessionUpdate::ToolCall(call));
        }
        "result" => {
            let Some(call_id) = event.tool_call_id.as_deref() else { return };
            if let Some(acp_id) = state.tools.borrow().get(call_id).cloned() {
                let fields = acp::ToolCallUpdateFields::new()
                    .status(acp::ToolCallStatus::Completed)
                    .content(vec![acp::ToolCallContent::Content(acp::Content::new(
                        acp::ContentBlock::Text(acp::TextContent::new(
                            event.output.clone().unwrap_or_default(),
                        )),
                    ))]);
                notify(
                    gateway,
                    &acp_session,
                    acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate::new(acp_id, fields)),
                );
            }
        }
        "turn.completed" => {
            // The authoritative final text rides `response` only when the
            // kernel batched the turn without streaming it.
            let response = event
                .output
                .as_deref()
                .filter(|t| !t.is_empty() && !state.streamed_text.get());
            if let Some(response) = response {
                notify(
                    gateway,
                    &acp_session,
                    acp::SessionUpdate::AgentMessageChunk(text_chunk(response)),
                );
            }
            finish_turn(&state, gateway, &acp_session, acp::StopReason::EndTurn);
            drain_pending_continuation(shared, &acp_session);
        }
        "turn.failed" => {
            if let Some(why) = event.output.as_deref().filter(|t| !t.is_empty()) {
                notify(
                    gateway,
                    &acp_session,
                    acp::SessionUpdate::AgentMessageChunk(text_chunk(why)),
                );
            }
            finish_turn(&state, gateway, &acp_session, acp::StopReason::EndTurn);
            drain_pending_continuation(shared, &acp_session);
        }
        _ => {}
    }
}

fn finish_turn(state: &Rc<SessionState>, _gateway: &AcpGatewaySender<acp::AgentSide>, _session: &acp::SessionId, done: acp::StopReason) {
    let stop = if state.cancelled.get() { acp::StopReason::Cancelled } else { done };
    if let Some(tx) = state.turn_done.borrow_mut().take() {
        let _ = tx.send(stop);
    }
}

/// Send a plan-approval continuation queued for this session, if any. Runs
/// after finish_turn so the approving turn's ACP prompt settles first.
fn drain_pending_continuation(shared: &Rc<Shared>, session: &acp::SessionId) {
    let state = shared.state.borrow().sessions.get(session).cloned();
    let Some(state) = state else { return };
    let Some(text) = state.pending_continuation.borrow_mut().take() else {
        return;
    };
    let Some(kernel) = shared.state.borrow().kernel.clone() else {
        return;
    };
    debug_log(&format!("plan continuation: {} chars", text.len()));
    state.streamed_text.set(false);
    if let Err(error) = kernel.request("session/send", kernel::send_params(&session.0, &text)) {
        tracing::warn!(%error, "plan continuation send failed");
    }
}

/// A kernel interaction request: forward as an ACP permission request, then
/// translate the user's choice back into the kernel's strict reply shape.
async fn handle_server_request(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    shared: &Rc<Shared>,
    envelope_id: &Value,
    method: &str,
    params: Value,
) {
    let Some(interaction) = parse_interaction(method, &params) else {
        // Unknown reverse request: answer with a benign error so the kernel
        // stops retrying instead of hanging the turn.
        if let Some(kernel) = shared.state.borrow().kernel.clone() {
            let _ = kernel.reply(envelope_id, json!({"ok": false}));
        }
        return;
    };
    if !shared
        .state
        .borrow_mut()
        .seen_interactions
        .insert(interaction.request_id.clone())
    {
        return; // Duplicate envelope of an already-answered request.
    }
    let Some(kernel) = shared.state.borrow().kernel.clone() else { return };

    if interaction.interaction == "plan_approval" {
        handle_plan_approval(gateway, shared, &kernel, envelope_id, &interaction).await;
        return;
    }

    let options: Vec<acp::PermissionOption> = interaction
        .options
        .iter()
        .map(|opt| acp::PermissionOption::new(opt.option_id.clone(), opt.name.clone(), opt.kind))
        .collect();
    let tool_call = acp::ToolCallUpdate::new(
        acp::ToolCallId::new(interaction.request_id.clone()),
        acp::ToolCallUpdateFields::new().title(interaction.title.clone()),
    );
    let request = acp::RequestPermissionRequest::new(
        acp::SessionId::new(interaction.session_id.clone()),
        tool_call,
        options,
    );
    let outcome = match gateway.send(request).await {
        Ok(response) => response.outcome,
        Err(_) => {
            let _ = kernel.reply(envelope_id, json!({"ok": false}));
            return;
        }
    };
    let selected = match outcome {
        acp::RequestPermissionOutcome::Selected(selected) => {
            interaction.options.iter().position(|o| o.option_id == &*selected.option_id.0)
        }
        acp::RequestPermissionOutcome::Cancelled => interaction.deny_index,
        _ => interaction.deny_index,
    };
    let Some(index) = selected else {
        let _ = kernel.reply(envelope_id, json!({"ok": false}));
        return;
    };
    let result = interaction.reply_result(index);
    if let Err(error) = kernel.reply(envelope_id, result) {
        tracing::warn!(%error, "interaction reply to kernel failed");
    }
}

/// A kernel plan-approval interaction, bridged to grok's native plan review:
/// forward as an `x.ai/exit_plan_mode` ext request, then translate the
/// pager's `{outcome, feedback}` back into the kernel contract (pinned via
/// zcode-tui: after "approve" the kernel neither flips the mode nor
/// continues on its own — the client does both).
async fn handle_plan_approval(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    shared: &Rc<Shared>,
    kernel: &Kernel,
    envelope_id: &Value,
    interaction: &InteractionWire,
) {
    let params = json!({
        "sessionId": interaction.session_id,
        "toolCallId": interaction.request_id,
        "planContent": interaction.plan,
    });
    let request = acp::ExtRequest::new(
        "x.ai/exit_plan_mode",
        serde_json::value::to_raw_value(&params)
            .expect("serialize exit_plan_mode params")
            .into(),
    );
    let (outcome, feedback) = match gateway.send(request).await {
        Ok(response) => {
            let parsed: Value = serde_json::from_str(response.0.get()).unwrap_or(json!({}));
            (
                parsed
                    .get("outcome")
                    .and_then(Value::as_str)
                    .unwrap_or("abandoned")
                    .to_string(),
                parsed
                    .get("feedback")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            )
        }
        Err(_) => ("abandoned".to_string(), None),
    };
    debug_log(&format!("plan_approval outcome={outcome} feedback={feedback:?}"));

    let session = acp::SessionId::new(interaction.session_id.clone());
    let state = shared.state.borrow().sessions.get(&session).cloned();
    let queue_continuation = |text: Option<String>| {
        if let Some(text) = text.filter(|t| !t.trim().is_empty()) {
            if let Some(state) = state.as_ref() {
                *state.pending_continuation.borrow_mut() = Some(text);
            }
        }
    };

    match outcome.as_str() {
        "approved" => {
            let index = interaction
                .options
                .iter()
                .position(|o| o.option_id == "approve")
                .unwrap_or(0);
            if let Err(error) = kernel.reply(envelope_id, interaction.reply_result(index)) {
                tracing::warn!(%error, "plan approval reply failed");
            }
            let _ = kernel
                .call(
                    "session/setMode",
                    kernel::set_mode_params(&interaction.session_id, "build"),
                )
                .await;
            queue_continuation(Some("Proceed with the approved plan.".to_string()));
        }
        "cancelled" => {
            // Revise: end this turn via a non-approve option (the kernel has
            // no deny on plan approvals), then send the feedback as the
            // revision prompt once the turn finalizes.
            match interaction.options.iter().position(|o| o.option_id != "approve") {
                Some(index) => {
                    let _ = kernel.reply(envelope_id, interaction.reply_result(index));
                }
                None => {
                    if let Some(state) = state.as_ref() {
                        state.cancelled.set(true);
                    }
                    let _ = kernel.request(
                        "session/stop",
                        kernel::stop_params(&interaction.session_id),
                    );
                }
            }
            queue_continuation(feedback);
        }
        _ => {
            // Abandoned (dismissed or the pager dropped the request): stop
            // the turn; plan mode stays on.
            if let Some(state) = state.as_ref() {
                state.cancelled.set(true);
            }
            let _ = kernel.request("session/stop", kernel::stop_params(&interaction.session_id));
        }
    }
}

// ---- zcode interaction parsing (ported from zcode-tui, pinned live) ----

struct InteractionOptionWire {
    option_id: String,
    name: String,
    kind: acp::PermissionOptionKind,
    /// requestUserInput: the answer value for this option.
    value: Option<String>,
    /// requestPermission: the pre-baked reply `result` object, verbatim.
    response: Option<Value>,
}

struct InteractionWire {
    request_id: String,
    session_id: String,
    title: String,
    header: String,
    options: Vec<InteractionOptionWire>,
    deny_index: Option<usize>,
    permission: bool,
    /// `schema.interaction` — "plan_approval" rides requestUserInput.
    interaction: String,
    /// `input.plan` — the plan text under review (plan_approval).
    plan: Option<String>,
}

impl InteractionWire {
    fn reply_result(&self, index: usize) -> Value {
        let Some(option) = self.options.get(index) else {
            return json!({"ok": false});
        };
        if self.permission {
            return option.response.clone().unwrap_or(json!({"ok": false}));
        }
        let mut answers = serde_json::Map::new();
        if let Some(value) = &option.value {
            answers.insert(self.header.clone(), Value::String(value.clone()));
        }
        json!({"requestId": self.request_id, "answers": Value::Object(answers)})
    }
}

fn parse_interaction(method: &str, params: &Value) -> Option<InteractionWire> {
    let permission = match method {
        "interaction/requestPermission" => true,
        "interaction/requestUserInput" => false,
        _ => return None,
    };
    let request_id = params.get("requestId")?.as_str()?.to_string();
    let session_id = params
        .get("sessionId")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let tool_name = str_at(params, "toolName");
    let mut title = if permission {
        str_at(params, "reason")
    } else {
        str_at(params, "prompt")
    };
    if title.is_empty() {
        title = tool_name.clone();
    }
    if !tool_name.is_empty() && !permission {
        title = format!("{tool_name}: {title}");
    }

    // requestUserInput nests options under questions[]; permission has them
    // at the top level with optionId/name/kind/response.
    let (raw_options, header) = if permission {
        (params.get("options")?.as_array()?, String::new())
    } else {
        let question = params.get("questions")?.as_array()?.first()?;
        (
            question.get("options")?.as_array()?,
            str_at(question, "header"),
        )
    };

    let mut options = Vec::new();
    let mut deny_index = None;
    for (index, raw) in raw_options.iter().enumerate() {
        let option_id = raw
            .get(if permission { "optionId" } else { "value" })
            .and_then(Value::as_str)?
            .to_string();
        let name = {
            let label = raw
                .get(if permission { "name" } else { "label" })
                .and_then(Value::as_str)
                .unwrap_or("");
            if label.is_empty() {
                option_id.clone()
            } else {
                label.to_string()
            }
        };
        let kind = match raw.get("kind").and_then(Value::as_str) {
            Some("deny") => {
                deny_index = Some(index);
                acp::PermissionOptionKind::RejectOnce
            }
            Some("allow_always") => acp::PermissionOptionKind::AllowAlways,
            _ => acp::PermissionOptionKind::AllowOnce,
        };
        options.push(InteractionOptionWire {
            option_id,
            name,
            kind,
            value: raw.get("value").and_then(Value::as_str).map(String::from),
            response: raw.get("response").cloned(),
        });
    }
    if options.is_empty() || session_id.is_empty() {
        return None;
    }
    Some(InteractionWire {
        request_id,
        session_id,
        title,
        header,
        options,
        deny_index,
        permission,
        interaction: params
            .pointer("/schema/interaction")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        plan: params
            .pointer("/input/plan")
            .and_then(Value::as_str)
            .map(String::from),
    })
}

fn str_at(value: &Value, key: &str) -> String {
    value.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}
