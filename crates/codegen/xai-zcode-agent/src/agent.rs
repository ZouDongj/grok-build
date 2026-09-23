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
    /// provider/updateAccountConfig pushed once per kernel boot: the kernel
    /// re-asserts its account state internally, and re-pushing with the
    /// boot-time builtin revision clobbers the newer state.
    account_pushed: std::cell::Cell<bool>,
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
    /// Cancels the pending grace timer after a turn.failed: 0.16.9 emits
    /// turn.failed for a failed ATTEMPT and then retries, so termination must
    /// wait to see whether the turn actually continues.
    fail_grace_cancel: RefCell<Option<oneshot::Sender<()>>>,
    /// Session workspace path — workspace-scoped kernel calls (mcp/list,
    /// plugins/list, …) key off it.
    cwd: RefCell<String>,
    /// The kernel session currently backing this ACP session. Equals the
    /// ACP id until an in-place rewind forks the conversation at an earlier
    /// message; the forked id takes over while the pager keeps its id.
    kernel_id: RefCell<String>,
    /// True while a kernel compaction turn runs. session/compact only
    /// ACCEPTS the job (returns {state:"accepted"} instantly) and runs
    /// "/compact" as a background prompt turn; during it the kernel
    /// rejects session/send with -32010 "A prompt is already running".
    compacting: std::cell::Cell<bool>,
    /// Turn id of the running compaction turn (turn.started with
    /// input "/compact" and inputVisibility "model-only").
    compact_turn_id: RefCell<Option<String>>,
    /// Resolved when the compaction turn ends — x.ai/compact_conversation
    /// awaits this so the pager's "compaction complete" is real.
    compact_wait: RefCell<Option<oneshot::Sender<Result<(), String>>>>,
    /// Subagent lifecycle broadcast state: child id -> (spawn_announced,
    /// finished_announced). Driven natively by the kernel's
    /// `subagent.lifecycle` events on the parent stream; the
    /// session/subagents poll remains as a backstop.
    subagents: RefCell<HashMap<String, (bool, bool)>>,
    /// child session id -> agentId (agent_<uuid>): keys the transcript
    /// directory the kernel writes under ~/.zcode/cli/agents.
    child_agent: RefCell<HashMap<String, String>>,
    /// child session id -> output.txt byte cursor for incremental tails.
    child_file_pos: RefCell<HashMap<String, u64>>,
    /// Per subagent child: message ids already forwarded to the pager —
    /// only used by the SQL fallback when the kernel writes no transcript
    /// files (the official client channel IS the files).
    child_seen: RefCell<HashMap<String, std::collections::HashSet<String>>>,
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
            fail_grace_cancel: RefCell::new(None),
            compacting: std::cell::Cell::new(false),
            compact_turn_id: RefCell::new(None),
            compact_wait: RefCell::new(None),
            cwd: RefCell::new(String::new()),
            kernel_id: RefCell::new(String::new()),
            subagents: RefCell::new(HashMap::new()),
            child_agent: RefCell::new(HashMap::new()),
            child_file_pos: RefCell::new(HashMap::new()),
            child_seen: RefCell::new(HashMap::new()),
        }
    }

    fn acp_tool_id(&self) -> acp::ToolCallId {
        let n = self.next_tool.replace(self.next_tool.get() + 1);
        acp::ToolCallId::new(format!("tc-{n}"))
    }
}

impl ZcodeAgent {
/// The kernel session id currently backing an ACP session (may differ from
/// the ACP id after an in-place rewind forked the conversation).
fn session_kernel_id(&self, session: &acp::SessionId) -> String {
    self.shared
        .state
        .borrow()
        .sessions
        .get(session)
        .map(|s| s.kernel_id.borrow().clone())
        .unwrap_or_else(|| session.0.to_string())
}

/// Session lookup shared by the workspace-scoped ext bridges: the session's
/// cwd (for workspace-keyed kernel calls) plus the live kernel handle.
fn session_cwd_and_kernel(&self, session_id: &str) -> acp::Result<(String, Kernel)> {
    let session = acp::SessionId::new(session_id.to_string());
    let state = self
        .shared
        .state
        .borrow()
        .sessions
        .get(&session)
        .cloned()
        .ok_or_else(|| acp::Error::invalid_params().data("unknown session"))?;
    let cwd = state.cwd.borrow().clone();
    let kernel = self.kernel()?;
    Ok((cwd, kernel))
}

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
                account_pushed: std::cell::Cell::new(false),
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

    /// Push account entitlement to the kernel (0.16.9+): the app-server
    /// worker starts fail-closed and expects its HOST to declare which
    /// account providers are entitled (`provider/updateAccountConfig`) and
    /// to supply request auth (`interaction/requestProviderRuntimeHeaders`).
    /// On 0.16.5 the method does not exist and the push is ignored.
    async fn push_account_config(&self, kernel: &Kernel) {
        if self.shared.account_pushed.get() {
            return;
        }
        self.shared.account_pushed.set(true);
        let provider = DEFAULT_PROVIDER;
        let revision = format!(
            "host:{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );
        // The kernel reconciles pushed account snapshots against its CURRENT
        // builtin revision and drops mismatches — the revision is per-boot, so
        // wait for THIS kernel's own provider_registry.ready log entry.
        let based_on = kernel_builtin_revision(kernel.boot_epoch_ms());
        // The pushed provider entry carries its model list too (zod allows
        // builtinModelIds beside access) — without it the kernel only knows
        // the session's current model. The list MUST match what the kernel's
        // own builtin registry declares (the desktop package the launcher
        // picks), or unknown ids poison the whole entry.
        let model_ids: Vec<String> = std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .map(|home| kernel_builtin_model_ids(&home, provider))
            .unwrap_or_default();
        let params = json!({
            "revision": revision,
            "basedOnZCodeBuiltinRevision": based_on,
            "providers": {
                provider: {
                    "builtinModelIds": model_ids,
                    // zod-strict: access accepts exactly {type, entitled}.
                    "access": {
                        "type": "zhipu-account",
                        "entitled": true,
                    }
                }
            },
            "states": {
                provider: {
                    "availability": "available",
                    "entitled": true,
                    "current": true,
                }
            },
        });
        debug_log(&format!(
            "account push payload: models={model_ids:?} basedOn={based_on}"
        ));
        match kernel.call("provider/updateAccountConfig", params).await {
            Ok(result) => debug_log(&format!(
                "account config pushed: {}",
                serde_json::to_string(&result).unwrap_or_default()
            )),
            Err(error) => debug_log(&format!("account config push skipped: {error}")),
        }
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
        self.push_account_config(&kernel).await;
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
            pump_shared.account_pushed.set(false);
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
/// Map a kernel plugin-operation call result onto the pager's
/// PluginsActionResponse outcome shape.
fn kernel_action_outcome(
    result: Result<Value, String>,
    success_message: String,
) -> Value {
    match result {
        Ok(_) => json!({
            "status": "success",
            "message": success_message,
            "requiresReload": true,
            "requiresRestart": false,
        }),
        Err(error) => json!({
            "status": "internal_error",
            "message": error,
            "requiresReload": false,
            "requiresRestart": false,
        }),
    }
}

/// Rewind points: every real user message in the kernel's ledger, oldest
/// first, with a text preview from its first part.
fn rewind_points(session_id: &str) -> Vec<Value> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let db_path = std::path::Path::new(&home).join(".zcode/cli/db/db.sqlite");
    let Ok(con) = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return Vec::new();
    };
    let Ok(mut stmt) = con.prepare(
        "SELECT m.id, m.time_created, (SELECT p.data FROM part p WHERE p.message_id = m.id \
         AND p.data LIKE '%\"text\"%' ORDER BY p.sequence LIMIT 1) \
         FROM message m WHERE m.session_id = ?1 AND m.data LIKE '%\"role\":\"user\"%' \
         ORDER BY m.sequence",
    ) else {
        return Vec::new();
    };
    let rows = stmt
        .query_map([session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })
        .map(|rows| rows.filter_map(Result::ok).collect::<Vec<_>>())
        .unwrap_or_default();
    rows.into_iter()
        .enumerate()
        .map(|(index, (id, created, preview))| {
            let preview = preview
                .and_then(|data| {
                    serde_json::from_str::<Value>(&data)
                        .ok()
                        .and_then(|j| j.get("text").and_then(Value::as_str).map(str::to_string))
                })
                .unwrap_or_default();
            json!({
                "promptIndex": index,
                "createdAt": ms_to_rfc3339(created),
                "numFileSnapshots": 0,
                "promptPreview": preview.chars().take(160).collect::<String>(),
                "hasFileChanges": false,
                "messageId": id,
            })
        })
        .collect()
}

/// The kernel message id of the Nth real user message, if present.
fn rewind_message_id(session_id: &str, target_prompt_index: usize) -> Option<String> {
    rewind_points(session_id)
        .into_iter()
        .nth(target_prompt_index)
        .and_then(|p| p.get("messageId").and_then(Value::as_str).map(str::to_string))
}

/// Identity headers for official MCP connectors, mirroring the desktop's
/// buildOfficialMcpAuthHeaders: `Authorization` carries the ZCode JWT,
/// `X-Bigmodel-Authorization` the coding-plan maas JWT (the bigmodel OAuth
/// access token). Team-scope headers only apply to team plans — omitted.
fn official_mcp_auth_headers() -> Option<Value> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let path = std::path::Path::new(&home).join(".zcode/v2/credentials.json");
    let store: Value = serde_json::from_str(
        &std::fs::read_to_string(path).ok()?,
    )
    .ok()?;
    let jwt = decrypt_credential(store.get("zcodejwttoken")?.as_str()?);
    let maas = decrypt_credential(store.get("oauth:bigmodel:access_token")?.as_str()?);
    if jwt.is_empty() || maas.is_empty() {
        return None;
    }
    Some(json!({
        "ok": true,
        "headers": {
            "Authorization": format!("Bearer {jwt}"),
            "X-Bigmodel-Authorization": format!("Bearer {maas}"),
        },
    }))
}

/// The kernel's prompt ledger for the composer's up-arrow recall.
fn input_history_prompts(session_id: &str) -> Vec<String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let db_path = std::path::Path::new(&home).join(".zcode/cli/db/db.sqlite");
    let Ok(con) = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return Vec::new();
    };
    let mut stmt = match con.prepare(
        "SELECT text FROM input_history WHERE session_id = ?1 ORDER BY time_created",
    ) {
        Ok(stmt) => stmt,
        Err(_) => return Vec::new(),
    };
    stmt.query_map([session_id], |row| row.get::<_, String>(0))
        .map(|rows| rows.filter_map(Result::ok).collect())
        .unwrap_or_default()
}

/// The current context footprint: the latest MAIN-line model request's
/// input+output. BigModel reports inputTokens inclusive of cache reads, and
/// every request's input contains the whole conversation — so the newest
/// request IS the current context. Turn-level sums overcount (each tool
/// round-trip re-sends the context), and subagent rows describe their own
/// conversations, hence the main_turn filter.
fn last_turn_context_tokens(kernel_session_id: &str) -> u64 {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let db_path = std::path::Path::new(&home).join(".zcode/cli/db/db.sqlite");
    let Ok(con) = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return 0;
    };
    con.query_row(
        "SELECT COALESCE(mu.input_tokens,0) + COALESCE(mu.output_tokens,0) \
         FROM model_usage mu WHERE mu.session_id = ?1 \
         AND mu.query_source = 'main_turn' \
         ORDER BY mu.started_at DESC LIMIT 1",
        [kernel_session_id],
        |r| r.get::<_, i64>(0),
    )
    .map(|v| v.max(0) as u64)
    .unwrap_or(0)
}

/// The current model's context window from the catalog (`_meta
/// .totalContextTokens`), for the UsageUpdate denominator.
fn context_window_tokens(shared: &Rc<Shared>) -> u64 {
    let state = shared.state.borrow();
    let Some(catalog) = state.catalog.as_ref() else {
        return 0;
    };
    let current = catalog.current_model_id.0.as_ref();
    catalog
        .available_models
        .iter()
        .find(|m| m.model_id.0.as_ref() == current)
        .and_then(|m| m.meta.as_ref())
        .and_then(|meta| meta.get("totalContextTokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// Native `subagent.lifecycle` events ride the parent session stream —
/// this is the kernel's own announcement channel (phase spawned/stopped
/// with agentId, childSessionId, agentType, status). The
/// session/subagents poll stays as a backstop for kernels without them.
fn handle_subagent_lifecycle(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    state: &Rc<SessionState>,
    parent: &acp::SessionId,
    payload: &Value,
) {
    let str_of = |k: &str| payload.get(k).and_then(Value::as_str).unwrap_or("");
    let child = str_of("childSessionId");
    if child.is_empty() {
        return;
    }
    let agent_id = str_of("agentId");
    if !agent_id.is_empty() {
        state
            .child_agent
            .borrow_mut()
            .insert(child.to_string(), agent_id.to_string());
    }
    match str_of("phase") {
        "spawned" => {
            let mut subs = state.subagents.borrow_mut();
            if subs.contains_key(child) {
                return;
            }
            subs.insert(child.to_string(), (true, false));
            drop(subs);
            let agent_type = if str_of("agentType").is_empty() {
                "general-purpose"
            } else {
                str_of("agentType")
            };
            emit_subagent_update(
                gateway,
                parent,
                json!({
                    "sessionUpdate": "subagent_spawned",
                    "subagent_id": child,
                    "parent_session_id": parent.0,
                    "child_session_id": child,
                    "subagent_type": agent_type,
                    "description": str_of("description"),
                }),
            );
        }
        "stopped" => {
            let kernel_sid = state.kernel_id.borrow().clone();
            let stats = subagent_metadata(&kernel_sid, agent_id);
            let mut subs = state.subagents.borrow_mut();
            if let Some(entry @ (true, false)) = subs.get_mut(child) {
                entry.1 = true;
                drop(subs);
                emit_subagent_update(
                    gateway,
                    parent,
                    json!({
                        "sessionUpdate": "subagent_finished",
                        "subagent_id": child,
                        "child_session_id": child,
                        "status": str_of("status"),
                        "tool_calls": stats.get("totalToolUseCount").and_then(Value::as_u64).unwrap_or(0),
                        "turns": 0,
                        "duration_ms": stats.get("totalDurationMs").and_then(Value::as_u64).unwrap_or(0),
                        "tokens_used": stats.pointer("/usage/totalTokens").and_then(Value::as_u64).unwrap_or(0),
                    }),
                );
                flush_child_content(gateway, state, child);
                schedule_delayed_child_flush(gateway.clone(), state.clone(), child.to_string());
            }
        }
        _ => {}
    }
}

fn emit_subagent_update(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    parent: &acp::SessionId,
    update: Value,
) {    let payload = json!({"sessionId": parent.0, "update": update});
    if let Ok(raw) = serde_json::value::to_raw_value(&payload) {
        gateway.forward_fire_and_forget(acp::ExtNotification::new(
            "x.ai/session/update",
            raw.into(),
        ));
    }
}

/// One session/subagents diff pass: announce new children as spawned,
/// refresh running ones with a progress tick, and finish the ones that
/// dropped out of `running` (ended items carry the authoritative status).
async fn poll_subagents_once(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    kernel: &Kernel,
    state: &Rc<SessionState>,
    parent: &acp::SessionId,
    final_pass: bool,
) {
    let kernel_sid = state.kernel_id.borrow().clone();
    let Ok(snapshot) = kernel
        .call("session/subagents", json!({"sessionId": kernel_sid}))
        .await
    else {
        return;
    };
    let arr = |pointer: &str| {
        snapshot
            .pointer(pointer)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let running = arr("/running");
    let ended = arr("/ended/items");
    let children = arr("/childSessionIds");
    fn str_of<'a>(v: &'a Value, k: &str) -> &'a str {
        v.get(k).and_then(Value::as_str).unwrap_or("")
    }
    let id_of = |v: &Value| {
        v.as_str()
            .map(str::to_string)
            .or_else(|| v.get("childSessionId").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_default()
    };
    let running_ids: std::collections::HashSet<String> =
        running.iter().map(&id_of).collect();
    let all_ids: std::collections::HashSet<String> = children
        .iter()
        .chain(running.iter())
        .chain(ended.iter())
        .map(&id_of)
        .filter(|id| !id.is_empty())
        .collect();

    // Spawn announcements for children we haven't seen.
    let mut announced_spawn: Vec<String> = Vec::new();
    {
        let mut subs = state.subagents.borrow_mut();
        for id in &all_ids {
            if !subs.contains_key(id) {
                let meta = running
                    .iter()
                    .chain(ended.iter())
                    .find(|v| id_of(v) == *id)
                    .cloned()
                    .unwrap_or(json!({}));
                let subagent_type = if str_of(&meta, "agentId").is_empty() {
                    "general-purpose"
                } else {
                    str_of(&meta, "agentId")
                };
                emit_subagent_update(
                    gateway,
                    parent,
                    json!({
                        "sessionUpdate": "subagent_spawned",
                        "subagent_id": id,
                        "parent_session_id": parent.0,
                        "child_session_id": id,
                        "subagent_type": subagent_type,
                        "description": str_of(&meta, "title"),
                    }),
                );
                subs.insert(id.clone(), (true, false));
                announced_spawn.push(id.clone());

            }
        }
    }
    let _ = announced_spawn;

    // Progress ticks for running children.
    for meta in &running {
        let id = id_of(meta);
        if state
            .subagents
            .borrow()
            .get(&id)
            .is_some_and(|(_, done)| !*done)
        {
            let started = meta
                .get("startedAt")
                .and_then(Value::as_i64)
                .unwrap_or_else(|| std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0));
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            let duration = (now_ms - started).max(0) as u64;
            // Content: incremental tails of the kernel-written transcript
            // files (the official client channel); SQL only when the kernel
            // writes no agent directory.
            {
                let meta_agent = str_of(meta, "agentId");
                if !meta_agent.is_empty() {
                    state
                        .child_agent
                        .borrow_mut()
                        .insert(id.clone(), meta_agent.to_string());
                }
                let fresh = collect_child_content(state, &id);
                debug_log(&format!("subagent content poll {}: {} new parts", id.get(0..24).unwrap_or(&id), fresh.len()));
                for (kind, text) in fresh {
                    let update = if kind == "reasoning" {
                        acp::SessionUpdate::AgentThoughtChunk(text_chunk(text))
                    } else {
                        acp::SessionUpdate::AgentMessageChunk(text_chunk(text))
                    };
                    let child = acp::SessionId::new(id.clone());
                    notify(gateway, &child, update);
                }
            }
            emit_subagent_update(
                gateway,
                parent,
                json!({
                    "sessionUpdate": "subagent_progress",
                    "subagent_id": id,
                    "parent_session_id": parent.0,
                    "child_session_id": id,
                    "duration_ms": duration,
                    "turn_count": 0,
                    "tool_call_count": 0,
                    "tokens_used": 0,
                    "context_window_tokens": 0,
                    "context_usage_pct": 0,
                    "tools_used": [],
                    "error_count": 0,
                }),
            );
        }
    }

    // Finishes: authoritative from ended items; on the final pass anything
    // spawned-but-not-running closes out as completed.
    for meta in &ended {
        let id = id_of(meta);
        let status = if str_of(meta, "status").is_empty() {
            "completed"
        } else {
            str_of(meta, "status")
        };
        let meta_agent = str_of(meta, "agentId");
        if !meta_agent.is_empty() {
            state
                .child_agent
                .borrow_mut()
                .insert(id.clone(), meta_agent.to_string());
        }
        let finish_stats = subagent_metadata(&state.kernel_id.borrow(), meta_agent);
        let mut subs = state.subagents.borrow_mut();
        if let Some(entry @ (true, false)) = subs.get_mut(&id) {
            entry.1 = true;
            drop(subs);
            flush_child_content(gateway, &state, &id);
            schedule_delayed_child_flush(gateway.clone(), state.clone(), id.clone());
            emit_subagent_update(
                gateway,
                parent,
                json!({
                    "sessionUpdate": "subagent_finished",
                    "subagent_id": id,
                    "child_session_id": id,
                    "status": status,
                    "tool_calls": finish_stats.get("totalToolUseCount").and_then(Value::as_u64).unwrap_or(0),
                    "turns": 0,
                    "duration_ms": finish_stats.get("totalDurationMs").and_then(Value::as_u64).unwrap_or(0),
                    "tokens_used": finish_stats.pointer("/usage/totalTokens").and_then(Value::as_u64).unwrap_or(0),
                }),
            );
        }
    }
    if final_pass {
        let ids: Vec<String> = state
            .subagents
            .borrow()
            .iter()
            .filter(|(id, (spawn, done))| *spawn && !*done && !running_ids.contains(*id))
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            if let Some(entry @ (true, false)) = state.subagents.borrow_mut().get_mut(&id) {
                entry.1 = true;
            }
            flush_child_content(gateway, &state, &id);
            schedule_delayed_child_flush(gateway.clone(), state.clone(), id.clone());
            emit_subagent_update(
                gateway,
                parent,
                json!({
                    "sessionUpdate": "subagent_finished",
                    "subagent_id": id,
                    "child_session_id": id,
                    "status": "completed",
                    "tool_calls": 0,
                    "turns": 0,
                    "duration_ms": 0,
                    "tokens_used": 0,
                }),
            );
        }
    }
}

/// While a turn runs, diff the kernel's subagent registry every few seconds
/// and translate changes into the pager's lifecycle notifications.
fn spawn_subagent_poller(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    kernel: Kernel,
    state: Rc<SessionState>,
    parent: acp::SessionId,
) {
    let gateway = gateway.clone();
    tokio::task::spawn_local(async move {
        loop {
            if state.turn_done.borrow().is_none() {
                break;
            }
            poll_subagents_once(&gateway, &kernel, &state, &parent, false).await;
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }
        poll_subagents_once(&gateway, &kernel, &state, &parent, true).await;
    });
}

/// Emit the pager's context meter update for a session.
fn push_context_usage(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    shared: &Rc<Shared>,
    state: &Rc<SessionState>,
    session: &acp::SessionId,
) {
    let used = last_turn_context_tokens(&state.kernel_id.borrow());
    let size = context_window_tokens(shared);
    if used == 0 || size == 0 {
        return;
    }
    notify(
        gateway,
        session,
        acp::SessionUpdate::UsageUpdate(acp::UsageUpdate::new(used, size)),
    );
}

/// Session token totals aggregated from the kernel's per-turn usage ledger,
/// in the pager's PromptUsage wire shape.
fn session_usage_totals(session_id: &str) -> Value {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let db_path = std::path::Path::new(&home).join(".zcode/cli/db/db.sqlite");
    let empty = json!({
        "usage": {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "cached_read_tokens": 0, "cache_creation_tokens": 0,
            "reasoning_tokens": 0, "model_calls": 0, "api_duration_ms": 0,
            "numTurns": 0,
        }
    });
    let Ok(con) = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return empty;
    };
    let Ok(row) = con.query_row(
        "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0),          COALESCE(SUM(computed_total_tokens),0), COALESCE(SUM(cache_read_input_tokens),0),          COALESCE(SUM(cache_creation_input_tokens),0), COALESCE(SUM(reasoning_tokens),0),          COUNT(*) FROM turn_usage WHERE session_id = ?1",
        [session_id],
        |r| {
            Ok((
                r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?, r.get::<_, i64>(4)?, r.get::<_, i64>(5)?,
                r.get::<_, i64>(6)?,
            ))
        },
    ) else {
        return empty;
    };
    json!({
        "usage": {
            "input_tokens": row.0, "output_tokens": row.1, "total_tokens": row.2,
            "cached_read_tokens": row.3, "cache_creation_tokens": row.4,
            "reasoning_tokens": row.5, "model_calls": row.6, "api_duration_ms": 0,
            "numTurns": row.6,
        }
    })
}

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

/// Look up the coding-plan API key for an account provider from the kernel's
/// credential store. Keys are embedded with the account uid:
/// `account-provider:coding-plan:<providerId>:account:<uid>:api-key`.
/// Read in-process only; the value is never logged or forwarded.
fn coding_plan_api_key(provider_id: &str) -> Option<String> {
    let home = std::env::var_os("HOME")?;
    let raw = std::fs::read_to_string(
        std::path::PathBuf::from(home).join(".zcode/v2/credentials.json"),
    )
    .ok()?;
    let creds: Value = serde_json::from_str(&raw).ok()?;
    // Resolve like the kernel's standalone chain: the identity entry names the
    // account the key must belong to (account uid for OAuth, key-hash for a
    // directly configured key) — with several api-key entries present an
    // arbitrary pick answers with a stale key.
    let identity_key = format!("account-provider:{provider_id}:identity");
    let identity = creds
        .get(&identity_key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .map(str::to_string);
    let prefix = format!("account-provider:coding-plan:{provider_id}:account:");
    let suffix = ":api-key";
    let mut identity_match: Option<&str> = None;
    let mut direct_configured: Option<&str> = None;
    let mut oauth_minted: Option<&str> = None;
    let mut any: Option<&str> = None;
    if let Some(map) = creds.as_object() {
        for (key, value) in map {
            if !key.starts_with(&prefix) || !key.ends_with(suffix) {
                continue;
            }
            let Some(text) = value.as_str().filter(|v| !v.trim().is_empty()) else {
                continue;
            };
            let uid = &key[prefix.len()..key.len() - suffix.len()];
            if identity.as_deref() == Some(uid) {
                identity_match = Some(text);
            }
            // "key-<hash>" uids mark DIRECTLY configured keys;
            // account-digit uids mark OAuth-minted ones (desktop's pick).
            if uid.starts_with("key-") {
                direct_configured = Some(text);
            } else if uid.chars().all(|c| c.is_ascii_digit()) && !uid.is_empty() {
                oauth_minted = Some(text);
            }
            any = Some(text);
        }
    }
    // The desktop host resolves the OAuth-minted key (account-uid entry) via
    // the user profile id — match that pick.
    identity_match
        .or(oauth_minted)
        .or(direct_configured)
        .or(any)
        .map(decrypt_credential)
}

/// Shared credential store values are sealed by the kernel as
/// `enc:v1:<iv>.<tag>.<cipher>` — AES-256-GCM with the key derived from
/// `ZCODE_CREDENTIAL_SECRET` (if set) or the kernel's deterministic fallback
/// `zcode-credential-fallback:<platform>:<homedir>:<username>`. Values without
/// the marker are already plaintext. Feeding the sealed form to the kernel
/// looks like a valid one-dot credential to its parser but is ciphertext
/// garbage, which the server rejects as 身份验证失败.
fn decrypt_credential(value: &str) -> String {
    const MARKER: &str = "enc:v1:";
    let Some(sealed) = value.strip_prefix(MARKER) else {
        return value.to_string();
    };
    let mut parts = sealed.split('.');
    let (Some(iv), Some(tag), Some(cipher), None) = (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return value.to_string();
    };
    use aes_gcm::{Aes256Gcm, KeyInit, Nonce, aead::Aead};
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    use sha2::Digest;

    let decode = |s: &str| URL_SAFE_NO_PAD.decode(s).ok();
    let (Some(iv), Some(tag), Some(cipher)) = (decode(iv), decode(tag), decode(cipher)) else {
        return value.to_string();
    };
    let secret = match std::env::var("ZCODE_CREDENTIAL_SECRET") {
        Ok(explicit) if !explicit.trim().is_empty() => explicit.trim().to_string(),
        _ => {
            let username = std::env::var("USER")
                .or_else(|_| std::env::var("LOGNAME"))
                .unwrap_or_else(|_| "unknown".to_string());
            format!(
                "zcode-credential-fallback:{}:{}:{}",
                std::env::consts::OS,
                home_dir_string(),
                username
            )
        }
    };
    let key = sha2::Sha256::digest(secret.as_bytes());
    let Ok(aes) = Aes256Gcm::new_from_slice(&key) else {
        return value.to_string();
    };
    let mut payload = cipher;
    payload.extend_from_slice(&tag);
    match aes.decrypt(Nonce::from_slice(&iv), payload.as_ref()) {
        Ok(plain) => String::from_utf8(plain).unwrap_or_else(|_| value.to_string()),
        Err(_) => {
            debug_log("credential decrypt failed (key mismatch?), passing raw value");
            value.to_string()
        }
    }
}

fn home_dir_string() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/root".to_string())
}

/// setModel with a short retry: 0.16.9 materializes account entitlements
/// asynchronously after kernel start, and a first-touch setModel for an
/// account provider can race that ("Provider Registry 中不存在 Model").
async fn set_model_with_retry(kernel: &Kernel, session_id: &str, provider: &str, model: &str) {
    for attempt in 0..6 {
        match kernel
            .call("session/setModel", kernel::set_model_params(session_id, provider, model))
            .await
        {
            Ok(_) => return,
            Err(error) if attempt == 5 => {
                debug_log(&format!("setModel {provider}/{model} failed: {error}"));
            }
            Err(_) => tokio::time::sleep(std::time::Duration::from_millis(400)).await,
        }
    }
}

/// The model ids the KERNEL's own builtin registry declares for an account
/// provider: resolve the desktop package the launcher selects (highest
/// version under ~/.local/opt/zcode). Pushing ids the kernel does not know
/// poisons the provider entry.
fn kernel_builtin_model_ids(home: &std::path::Path, provider: &str) -> Vec<String> {
    let packages: Vec<std::path::PathBuf> = std::fs::read_dir(home.join(".local/opt/zcode"))
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.path().join("opt/ZCode/resources/config/provider/zcode-builtin.json"))
        .filter(|p| p.is_file())
        .collect();
    let Some(path) = packages.into_iter().max_by_key(|p| version_of(p)) else {
        return Vec::new();
    };
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return Vec::new();
    };
    value
        .pointer("/config/providerConfigRules/providerRules")
        .and_then(Value::as_array)
        .map(|rules| {
            rules
                .iter()
                .filter(|rule| rule.get("providerId").and_then(Value::as_str) == Some(provider))
                .filter_map(|rule| rule.pointer("/config/builtinModelIds").and_then(Value::as_array))
                .flat_map(|ids| ids.iter().filter_map(Value::as_str))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Highest dotted-version path segment (~/.local/opt/zcode/<ver>/opt/...).
fn version_of(path: &std::path::Path) -> Vec<u64> {
    let mut best = Vec::new();
    for comp in path.components() {
        if let Some(seg) = comp.as_os_str().to_str() {
            let parsed: Vec<u64> = seg
                .split('.')
                .map(|p| p.parse::<u64>().ok())
                .collect::<Option<Vec<_>>>()
                .unwrap_or_default();
            if parsed.len() >= 2 {
                best = parsed;
            }
        }
    }
    best
}

/// The kernel's live builtin registry revision, e.g. "zcode-builtin:30:<hash>",
/// as reported by its own provider_registry.ready log line at/after boot.
fn kernel_builtin_revision(boot_epoch_ms: u128) -> String {
    let Some(home) = std::env::var_os("HOME").map(std::path::PathBuf::from) else {
        return "host".to_string();
    };
    let log_dir = home.join(".zcode/cli/log");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Some(revision) = newest_builtin_revision_since(&log_dir, boot_epoch_ms) {
            return revision;
        }
        if std::time::Instant::now() > deadline {
            return "host".to_string();
        }
        std::thread::sleep(std::time::Duration::from_millis(150));
    }
}

fn newest_builtin_revision_since(log_dir: &std::path::Path, since_epoch_ms: u128) -> Option<String> {
    let mut files: Vec<_> = std::fs::read_dir(log_dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".jsonl"))
        .collect();
    files.sort();
    let raw = std::fs::read_to_string(files.last()?).ok()?;
    let mut found: Option<(u128, String)> = None;
    for line in raw.lines() {
        if !line.contains("provider_registry.ready") {
            continue;
        }
        let ts = line
            .split("\"timestamp\":\"")
            .nth(1)
            .and_then(|rest| rest.split('\"').next())
            .and_then(|t| chrono_like_epoch_ms(t))
            .unwrap_or(0);
        if ts < since_epoch_ms {
            continue;
        }
        if let Some(start) = line.find("zcode-builtin:") {
            let rest = &line[start..];
            if let Some(end) = rest.find('"') {
                let candidate = rest[..end].trim_end_matches('\\').to_string();
                if found.as_ref().map(|(t, _)| ts >= *t).unwrap_or(true) {
                    found = Some((ts, candidate));
                }
            }
        }
    }
    found.map(|(_, revision)| revision)
}

/// RFC3339 UTC -> epoch ms (fixed "YYYY-MM-DDTHH:MM:SS.sssZ").
fn chrono_like_epoch_ms(t: &str) -> Option<u128> {
    let year: i64 = t.get(0..4)?.parse().ok()?;
    let month: i64 = t.get(5..7)?.parse().ok()?;
    let day: i64 = t.get(8..10)?.parse().ok()?;
    let hour: i64 = t.get(11..13)?.parse().ok()?;
    let minute: i64 = t.get(14..16)?.parse().ok()?;
    let second: i64 = t.get(17..19)?.parse().ok()?;
    let millis: i64 = t.get(20..23).and_then(|m| m.parse().ok()).unwrap_or(0);
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    let secs = days * 86_400 + hour * 3_600 + minute * 60 + second;
    Some((secs * 1000 + millis).max(0) as u128)
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
        // kernel default is fine when absent. 0.16.9 loads account
        // entitlements asynchronously at startup — an immediate setModel can
        // race that and fail, leaving the session on the default provider,
        // so retry briefly.
        if let Some((provider, model)) = load_model_preference() {
            set_model_with_retry(&kernel, &session_id, &provider, &model).await;
        }
        let acp_session = acp::SessionId::new(session_id.clone());
        let state = Rc::new(SessionState::new());
        *state.cwd.borrow_mut() = cwd.to_string_lossy().to_string();
        *state.kernel_id.borrow_mut() = session_id.clone();
        self.shared
            .state
            .borrow_mut()
            .sessions
            .insert(acp_session.clone(), state);
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
            set_model_with_retry(&kernel, &session_id, &provider, &model).await;
        }
        self.shared
            .state
            .borrow_mut()
            .sessions
            .insert(args.session_id.clone(), {
                let state = Rc::new(SessionState::new());
                *state.cwd.borrow_mut() = args.cwd.to_string_lossy().to_string();
                *state.kernel_id.borrow_mut() = args.session_id.0.to_string();
                state
            });
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
        // Prime the context meter from the ledger so the bar shows real
        // numbers right after resume, instead of waiting for the first
        // completed turn.
        if let Some(state) = self.shared.state.borrow().sessions.get(&args.session_id).cloned() {
            push_context_usage(&self.gateway, &self.shared, &state, &args.session_id);
        }
        tracing::info!(%session_id, "zcode session resumed");
        let mut response = acp::LoadSessionResponse::new();
        response.models = initial_models;
        Ok(response)
    }

    async fn prompt(&self, args: acp::PromptRequest) -> acp::Result<acp::PromptResponse> {
        debug_log("acp: prompt");
        // Instant prompt acknowledgment. The pager arms a 120s prompt-ack
        // watchdog when it sends the prompt and disarms it on the first live
        // session/update stamped with the awaited prompt id. A ZCode turn can
        // legitimately exceed that before its first streamed event (kernel
        // cold boot, or a giant-context prefill taking minutes) — grok-shell
        // acks in under a second via its queue rail; we carry the prompt id
        // on an empty thought chunk, which the renderer drops by design.
        if let Some(prompt_id) = args.meta.as_ref().and_then(|m| m.get("promptId")) {
            let mut ack_meta = acp::Meta::new();
            ack_meta.insert("promptId".to_string(), prompt_id.clone());
            let mut ack = acp::SessionNotification::new(
                args.session_id.clone(),
                acp::SessionUpdate::AgentThoughtChunk(text_chunk("")),
            );
            ack.meta = Some(ack_meta);
            self.gateway.forward_fire_and_forget(ack);
        }
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
        let attachments = prompt_attachments(&args.prompt);
        if !attachments.is_empty() {
            debug_log(&format!("prompt: {} image attachment(s)", attachments.len()));
        }
        let (tx, mut rx) = oneshot::channel();
        *state.turn_done.borrow_mut() = Some(tx);
        state.cancelled.set(false);
        state.streamed_text.set(false);
        // Compaction in flight: the kernel rejects concurrent sends with
        // -32010 "A prompt is already running" (the compact turn IS a
        // prompt). Queue as a continuation — it is sent the moment the
        // compaction turn ends (handle_event drains it). One queued prompt
        // only: a second would overwrite the first's armed turn_done.
        let stop = if state.compacting.get() {
            if state.pending_continuation.borrow().is_some() {
                *state.turn_done.borrow_mut() = None;
                return Err(acp::Error::invalid_request().data(
                    "another prompt is already queued behind the running compaction",
                ));
            }
            if !attachments.is_empty() {
                *state.turn_done.borrow_mut() = None;
                return Err(acp::Error::invalid_request()
                    .data("compaction in progress; resend the image afterwards"));
            }
            debug_log("prompt: queued behind running compaction");
            *state.pending_continuation.borrow_mut() = Some(text);
            rx.await.unwrap_or(acp::StopReason::EndTurn)
        } else {
            spawn_subagent_poller(&self.gateway, kernel.clone(), state.clone(), args.session_id.clone());
            // The host-pushed account snapshot decays when the kernel re-asserts
            // its registry against the builtin revision — re-push before sends.
            self.push_account_config(&kernel).await;
            let send_rx = kernel
                .request(
                    "session/send",
                    kernel::send_params_with_attachments(&state.kernel_id.borrow(), &text, &attachments),
                )
                .map_err(|e| acp::Error::internal_error().data(e.to_string()))?;
            // The send RPC answers only at turn end — EXCEPT when the kernel
            // rejects it outright (-32010 during compaction, bad state). Watch
            // both channels concurrently: an early rejection fails this turn
            // explicitly (the old code dropped the receiver and parked until
            // an unrelated turn's completion event resolved it, losing the
            // prompt).
            tokio::select! {
                sent = send_rx => match sent {
                    Ok(Err(rejection)) => {
                        *state.turn_done.borrow_mut() = None;
                        notify(
                            &self.gateway,
                            &args.session_id,
                            acp::SessionUpdate::AgentMessageChunk(text_chunk(format!(
                                "prompt rejected by kernel: {rejection}"
                            ))),
                        );
                        return Err(acp::Error::internal_error().data(rejection));
                    }
                    // RPC resolved at turn end — the turn.completed event (or
                    // a sibling) fires turn_done; fall through to it for the
                    // real stop reason, with a direct resolve as backstop.
                    _ => rx.await.unwrap_or(acp::StopReason::EndTurn),
                },
                stop = &mut rx => stop.unwrap_or(acp::StopReason::EndTurn),
            }
        };
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
        if let Some(state) = self.shared.state.borrow().sessions.get(&args.session_id) {
            let _ = kernel.request("session/stop", kernel::stop_params(&state.kernel_id.borrow()));
        }
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
            .call("session/setMode", kernel::set_mode_params(&self.session_kernel_id(&args.session_id), kernel_mode))
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
        let mut switched = false;
        for attempt in 0..3 {
            match kernel
                .call("session/setModel", kernel::set_model_params(&self.session_kernel_id(&args.session_id), &provider, model))
                .await
            {
                Ok(_) => {
                    switched = true;
                    break;
                }
                Err(error) => {
                    debug_log(&format!("set_session_model: setModel FAILED (attempt {}): {error}", attempt + 1));
                    // Registry-miss usually means the pushed account snapshot
                    // decayed — re-push before retrying.
                    self.push_account_config(&kernel).await;
                }
            }
        }
        if !switched {
            return Err(acp::Error::internal_error().data("setModel failed after retries"));
        }
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
                    kernel::set_thought_params(&self.session_kernel_id(&args.session_id), effort),
                )
                .await
            {
                debug_log(&format!("set_session_model: setThoughtLevel FAILED: {error}"));
            }
        }
        debug_log("set_session_model: session/read");
        self.push_model_state(&kernel, &self.session_kernel_id(&args.session_id)).await;
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
            // Extensions modal → MCP servers tab. The kernel's workspace-
            // scoped `mcp/list` (mode "status" = read-only, no connecting)
            // reports live per-server state; translate to the pager's
            // session-shaped entries.
            "x.ai/mcp/list" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let session = acp::SessionId::new(id);
                let state = self
                    .shared
                    .state
                    .borrow()
                    .sessions
                    .get(&session)
                    .cloned()
                    .ok_or_else(|| acp::Error::invalid_params().data("unknown session"))?;
                let cwd = state.cwd.borrow().clone();
                let kernel = self.kernel()?;
                let result = kernel
                    .call(
                        "mcp/list",
                        json!({"workspace": {"workspaceKey": cwd, "workspacePath": cwd}, "mode": "status"}),
                    )
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error().data(format!("mcp/list failed: {e}"))
                    })?;
                let statuses = result
                    .get("statuses")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                let servers: Vec<Value> = statuses
                    .iter()
                    .map(|(name, st)| {
                        let kernel_status =
                            st.get("status").and_then(Value::as_str).unwrap_or("disconnected");
                        let tool_count =
                            st.get("toolCount").and_then(Value::as_u64).unwrap_or(0);
                        // Pager status strings: ready / initializing; anything
                        // else lands on Unavailable, which is honest for
                        // failed/disconnected/untrusted. `disabled` config
                        // rows report enabled=false instead.
                        let enabled = kernel_status != "disabled";
                        let session_status = match kernel_status {
                            "connected" => "ready",
                            "connecting" => "initializing",
                            _ => "",
                        };
                        let auth_required = st.get("authorization").is_some();
                        let tools: Vec<Value> = (0..tool_count)
                            .map(|_| json!({"name": "", "enabled": true}))
                            .collect();
                        json!({
                            "name": name,
                            "session": {
                                "enabled": enabled,
                                "status": session_status,
                                "tools": tools,
                                "authRequired": auth_required,
                            },
                        })
                    })
                    .collect();
                let body = json!({
                    "result": {"servers": servers, "sessionMcpResolved": true},
                });
                let raw =
                    serde_json::value::to_raw_value(&body).expect("serialize mcp list");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Extensions modal → Plugins tab. Kernel `plugins/list` returns
            // registry entries; map onto the pager's PluginInfo (camelCase,
            // required fields synthesized where the kernel has no notion).
            "x.ai/plugins/list" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let (cwd, kernel) = self.session_cwd_and_kernel(&id)?;
                let result = kernel
                    .call(
                        "plugins/list",
                        json!({"workspace": {"workspaceKey": cwd, "workspacePath": cwd}}),
                    )
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error().data(format!("plugins/list failed: {e}"))
                    })?;
                let plugins: Vec<Value> = result
                    .get("plugins")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                    .iter()
                    .map(|p| {
                        let mcp_names = p
                            .get("mcpServerNames")
                            .and_then(Value::as_array)
                            .cloned()
                            .unwrap_or_default();
                        let mut entry = json!({
                            "name": p.get("name").cloned().unwrap_or(Value::Null),
                            "id": p.get("id").cloned().unwrap_or(Value::Null),
                            "root": "",
                            "scope": "user",
                            "trusted": true,
                            "enabled": p.get("enabled").and_then(Value::as_bool).unwrap_or(false),
                            "skillCount": p.get("skillCount").and_then(Value::as_u64).unwrap_or(0),
                            "agentCount": 0,
                            "hookStatus": "none",
                            "mcpServerCount": mcp_names.len(),
                            "mcpStatus": if mcp_names.is_empty() { "none" } else { "active" },
                        });
                        for (src, dst) in
                            [("version", "version"), ("description", "description")]
                        {
                            if let Some(v) = p.get(src) {
                                entry[dst] = v.clone();
                            }
                        }
                        if let Some(marketplace) = p.get("marketplace").and_then(Value::as_str) {
                            entry["marketplaceSource"] = json!(marketplace);
                        }
                        entry
                    })
                    .collect();
                let body = json!({"result": {"plugins": plugins}});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize plugins list");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Plugin enable/disable/reload from the Plugins tab.
            "x.ai/plugins/action" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let action = params.get("action").cloned().unwrap_or(json!({}));
                let action_type = action.get("type").and_then(Value::as_str).unwrap_or("");
                let (cwd, kernel) = self.session_cwd_and_kernel(&id)?;
                let outcome = match action_type {
                    "enable" | "disable" => {
                        let plugin_id = action
                            .get("pluginId")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        kernel
                            .call(
                                "plugins/setEnabled",
                                json!({
                                    "workspace": {"workspaceKey": cwd, "workspacePath": cwd},
                                    "pluginId": plugin_id,
                                    "enabled": action_type == "enable",
                                    "scope": "user",
                                }),
                            )
                            .await
                            .map(|_| json!({
                                "status": "success",
                                "message": format!("plugin {plugin_id} {}", action_type),
                                "requiresReload": true,
                                "requiresRestart": false,
                            }))
                            .map_err(|e| {
                                acp::Error::internal_error().data(format!(
                                    "plugins/setEnabled failed: {e}"
                                ))
                            })?
                    }
                    // The kernel IS the plugin host — nothing to reload
                    // client-side; the next plugins/list reads live state.
                    "reload" => json!({
                        "status": "success",
                        "message": "plugin state lives in the zcode kernel",
                        "requiresReload": true,
                        "requiresRestart": false,
                    }),
                    "install" | "add" => {
                        let source = action
                            .get("source")
                            .or_else(|| action.get("path"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        kernel_action_outcome(
                            kernel
                                .call(
                                    "plugins/marketplace/add",
                                    json!({
                                        "workspace": {"workspaceKey": cwd, "workspacePath": cwd},
                                        "source": source,
                                    }),
                                )
                                .await,
                            format!("installed from {source}"),
                        )
                    }
                    "uninstall" => {
                        let plugin_id = action
                            .get("pluginId")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        kernel_action_outcome(
                            kernel
                                .call(
                                    "plugins/uninstall",
                                    json!({
                                        "workspace": {"workspaceKey": cwd, "workspacePath": cwd},
                                        "pluginId": plugin_id,
                                    }),
                                )
                                .await,
                            format!("uninstalled {plugin_id}"),
                        )
                    }
                    "update" => {
                        let plugin_id = action.get("pluginId").and_then(Value::as_str);
                        let mut payload = json!({
                            "workspace": {"workspaceKey": cwd, "workspacePath": cwd},
                        });
                        if let Some(plugin_id) = plugin_id {
                            payload["pluginId"] = json!(plugin_id);
                        }
                        kernel_action_outcome(
                            kernel.call("plugins/update", payload).await,
                            "updated".to_string(),
                        )
                    }
                    "remove" => {
                        let path = action
                            .get("path")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        kernel_action_outcome(
                            kernel
                                .call(
                                    "plugins/marketplace/remove",
                                    json!({
                                        "workspace": {"workspaceKey": cwd, "workspacePath": cwd},
                                        "marketplace": path,
                                    }),
                                )
                                .await,
                            format!("removed {path}"),
                        )
                    }
                    other => {
                        return Err(acp::Error::method_not_found().data(format!(
                            "plugins action {other:?} not bridged yet"
                        )));
                    }
                };
                let body = json!({"result": outcome});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize plugins action");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Extensions modal → Skills tab. The pager sends only {cwd:"."}
            // (the shell resolved the workspace itself); use a live
            // session's cwd, falling back to the grok process's launch
            // directory. Kernel `skills/referenceCatalog` returns exactly
            // what the model's Skill tool sees.
            "x.ai/skills/list" => {
                let cwd = self
                    .shared
                    .state
                    .borrow()
                    .sessions
                    .values()
                    .next()
                    .map(|s| s.cwd.borrow().clone())
                    .filter(|c| !c.is_empty())
                    .or_else(|| {
                        std::env::current_dir().ok().map(|p| p.display().to_string())
                    })
                    .unwrap_or_else(|| ".".to_string());
                let kernel = self.kernel()?;
                let result = kernel
                    .call(
                        "skills/referenceCatalog",
                        json!({"workspace": {"workspaceKey": cwd, "workspacePath": cwd}}),
                    )
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error().data(format!(
                            "skills/referenceCatalog failed: {e}"
                        ))
                    })?;
                let skills: Vec<Value> = result
                    .get("skills")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                    .iter()
                    .map(|s| {
                        let kernel_scope =
                            s.get("scope").and_then(Value::as_str).unwrap_or("user");
                        // Pager scopes: local/repo/user/server. Kernel
                        // "workspace" is the project scope (repo); plugin
                        // skills ride the user scope with their plugin name.
                        let scope = match kernel_scope {
                            "workspace" => "repo",
                            _ => "user",
                        };
                        let mut entry = json!({
                            "name": s.get("name").cloned().unwrap_or(Value::Null),
                            "description": s
                                .get("description")
                                .and_then(Value::as_str)
                                .unwrap_or(""),
                            "hasUserSpecifiedDescription": true,
                            "path": s.get("path").cloned().unwrap_or(Value::Null),
                            "scope": scope,
                        });
                        if let Some(plugin) = s.get("pluginName").and_then(Value::as_str) {
                            entry["pluginName"] = json!(plugin);
                        }
                        entry
                    })
                    .collect();
                let body = json!({"result": {"skills": skills}});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize skills list");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Hooks tab: grok-shell hooks are a shell-side lifecycle system
            // the zcode kernel does not have — answer with a valid empty
            // listing so the tab opens instead of erroring.
            "x.ai/hooks/list" => {
                let body = json!({"result": {"hooks": [], "projectTrusted": true, "loadErrors": []}});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize hooks list");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Workflows tab: kernel workflows/list (project scope) entries.
            "x.ai/workflows/list" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let (cwd, kernel) = self.session_cwd_and_kernel(&id)?;
                let result = kernel
                    .call(
                        "workflows/list",
                        json!({
                            "workspace": {"workspaceKey": cwd, "workspacePath": cwd},
                            "scope": "project",
                        }),
                    )
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error().data(format!(
                            "workflows/list failed: {e}"
                        ))
                    })?;
                let workflows: Vec<Value> = result
                    .get("workflows")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                    .iter()
                    .map(|w| {
                        json!({
                            "name": w.get("name").cloned().unwrap_or(Value::Null),
                            "description": w
                                .get("description")
                                .and_then(Value::as_str)
                                .unwrap_or(""),
                            "when_to_use": w.get("whenToUse").cloned().unwrap_or(Value::Null),
                            "source": w.get("scope").and_then(Value::as_str).unwrap_or("project"),
                            "path": w.get("path").cloned().unwrap_or(Value::Null),
                        })
                    })
                    .collect();
                let body = json!({"result": {"workflows": workflows}});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize workflows list");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Marketplace tab: kernel plugins/overview carries marketplaces
            // plus per-marketplace available plugins with install state.
            "x.ai/marketplace/list" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let (cwd, kernel) = self.session_cwd_and_kernel(&id)?;
                let result = kernel
                    .call(
                        "plugins/overview",
                        json!({"workspace": {"workspaceKey": cwd, "workspacePath": cwd}}),
                    )
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error().data(format!(
                            "plugins/overview failed: {e}"
                        ))
                    })?;
                let marketplaces = result
                    .get("marketplaces")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let available = result
                    .get("availablePlugins")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                let sources: Vec<Value> = marketplaces
                    .iter()
                    .map(|m| {
                        let m_name = m.get("name").and_then(Value::as_str).unwrap_or("");
                        let m_id = m.get("id").and_then(Value::as_str).unwrap_or("");
                        let plugins: Vec<Value> = available
                            .iter()
                            .filter(|p| {
                                p.get("marketplace")
                                    .and_then(Value::as_str)
                                    .map(|mp| mp == m_name || mp == m_id)
                                    .unwrap_or(false)
                            })
                            .map(|p| {
                                json!({
                                    "name": p.get("name").cloned().unwrap_or(Value::Null),
                                    "version": p.get("version").cloned().unwrap_or(Value::Null),
                                    "description": p.get("description").cloned().unwrap_or(Value::Null),
                                    "relativePath": p.get("id").and_then(Value::as_str).unwrap_or(""),
                                    "skillCount": 0,
                                    "hasHooks": false,
                                    "hasAgents": false,
                                    "hasMcp": false,
                                    "installStatus": if p.get("installed").and_then(Value::as_bool).unwrap_or(false)
                                        { "installed" } else { "not_installed" },
                                })
                            })
                            .collect();
                        json!({
                            "sourceName": m_name,
                            "sourceKind": "marketplace",
                            "sourceUrlOrPath": m.get("source").and_then(Value::as_str).unwrap_or(""),
                            "plugins": plugins,
                            "error": m.get("refreshFailure").cloned().unwrap_or(Value::Null),
                        })
                    })
                    .collect();
                let body = json!({"result": {"sources": sources}});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize marketplace list");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Subscription poll (every ~60s): the coding-plan channel has no
            // grok subscription concept — acknowledge so the poll stops
            // erroring; billing mirrors that with a stable tier label.
            "x.ai/auth/check_subscription" => {
                let raw = serde_json::value::to_raw_value(&json!({ "meta": {} }))
                    .expect("serialize check_subscription");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            "x.ai/billing" => {
                let body = json!({"result": {
                    "onDemandEnabled": false,
                    "subscriptionTier": "GLM Coding Plan",
                }});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize billing");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Session-load queries: metadata for the header/feedback flow.
            "x.ai/session/info" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let session = acp::SessionId::new(id.clone());
                let cwd = self
                    .shared
                    .state
                    .borrow()
                    .sessions
                    .get(&session)
                    .map(|s| s.cwd.borrow().clone())
                    .unwrap_or_default();
                let model = self
                    .shared
                    .state
                    .borrow()
                    .catalog
                    .as_ref()
                    .map(|c| c.current_model_id.0.to_string())
                    .unwrap_or_else(|| "GLM-5.3".to_string());
                let body = json!({"result": {
                    "sessionId": id,
                    "cwd": cwd,
                    "agentName": "zcode",
                    "model": model,
                    "modelDisplayName": model,
                    "resolvedModelId": model,
                    "modelFingerprint": null,
                    "turns": 0,
                }});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize session info");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Composer up-arrow recall: the kernel persists every prompt in
            // its input_history store.
            "x.ai/prompt_history" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .or_else(|| params.get("session_id"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let prompts = input_history_prompts(&id);
                let body = json!({"result": {"prompts": prompts}});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize prompt history");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Token usage for the session — aggregated straight from the
            // kernel's turn_usage ledger.
            "x.ai/session/usage" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let usage = session_usage_totals(&id);
                let body = json!(usage);
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize session usage");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Cursor-based event ledger read — unlike session/subscribe this
            // works on subagent child sessions ("Session is not active" only
            // rejects live subscription, not ledger reads).
            "x.ai/session/events" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if !id.starts_with("sess_") {
                    return Err(acp::Error::invalid_params().data("bad sessionId"));
                }
                let kernel = self.kernel()?;
                let mut payload = json!({"sessionId": id});
                if let Some(seq) = params.get("afterSeq").and_then(Value::as_u64) {
                    payload["afterSeq"] = json!(seq);
                }
                if let Some(limit) = params.get("limit").and_then(Value::as_u64) {
                    payload["limit"] = json!(limit);
                }
                let result = kernel
                    .call("session/events", payload)
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error()
                            .data(format!("session/events failed: {e}"))
                    })?;
                let raw = serde_json::value::to_raw_value(&result)
                    .expect("serialize session events");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Message-part ledger read with an afterMessageId cursor — the
            // client's subagentTranscripts store is fed from this family.
            "x.ai/session/messages" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if !id.starts_with("sess_") {
                    return Err(acp::Error::invalid_params().data("bad sessionId"));
                }
                let kernel = self.kernel()?;
                let mut payload = json!({"sessionId": id});
                if let Some(after) = params.get("afterMessageId").and_then(Value::as_str) {
                    payload["afterMessageId"] = json!(after);
                }
                if let Some(limit) = params.get("limit").and_then(Value::as_u64) {
                    payload["limit"] = json!(limit);
                }
                let result = kernel
                    .call("session/messages", payload)
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error()
                            .data(format!("session/messages failed: {e}"))
                    })?;
                let raw = serde_json::value::to_raw_value(&result)
                    .expect("serialize session messages");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // /compact: the kernel folds the transcript itself — but
            // session/compact only ACCEPTS the job ({state:"accepted"} in
            // ~300ms) and runs "/compact" as a background prompt turn that
            // can take minutes. Await the turn's real completion so the
            // pager's "compaction complete" banner and elapsed time are
            // honest; a send rejection would be a lie (the old bug).
            "x.ai/compact_conversation" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let acp_sid = acp::SessionId::new(id.clone());
                let state = self
                    .shared
                    .state
                    .borrow()
                    .sessions
                    .get(&acp_sid)
                    .cloned()
                    .ok_or_else(|| acp::Error::invalid_params().data("unknown session"))?;
                if !id.starts_with("sess_") {
                    return Err(acp::Error::invalid_params().data("bad sessionId"));
                }
                let kernel = self.kernel()?;
                let mut payload = json!({"sessionId": id});
                if let Some(ctx) = params.get("userContext").and_then(Value::as_str) {
                    payload["instructions"] = json!(ctx);
                }
                // Register the waiter BEFORE calling: the kernel fires the
                // background turn before the RPC response lands, and an
                // instant compaction could complete inside that window.
                let (wtx, wrx) = oneshot::channel::<Result<(), String>>();
                state.compacting.set(true);
                *state.compact_wait.borrow_mut() = Some(wtx);
                let result = kernel
                    .call("session/compact", payload)
                    .await
                    .map_err(|e| {
                        state.compacting.set(false);
                        *state.compact_wait.borrow_mut() = None;
                        acp::Error::internal_error().data(format!("compact failed: {e}"))
                    })?;
                let accepted = result
                    .pointer("/compact/state")
                    .and_then(Value::as_str)
                    .unwrap_or("accepted")
                    == "accepted"
                    || result
                        .pointer("/compact/state")
                        .and_then(Value::as_str)
                        == Some("already_running");
                if !accepted {
                    state.compacting.set(false);
                    *state.compact_wait.borrow_mut() = None;
                    let raw = serde_json::value::to_raw_value(&json!({}))
                        .expect("serialize compact ack");
                    return Ok(acp::ExtResponse::new(raw.into()));
                }
                return match tokio::time::timeout(
                    std::time::Duration::from_secs(300),
                    wrx,
                )
                .await
                {
                    Ok(Ok(Ok(()))) => {
                        let raw = serde_json::value::to_raw_value(&json!({}))
                            .expect("serialize compact ack");
                        Ok(acp::ExtResponse::new(raw.into()))
                    }
                    Ok(Ok(Err(msg))) => Err(acp::Error::internal_error()
                        .data(format!("compaction failed: {msg}"))),
                    Ok(Err(_)) => Err(acp::Error::internal_error()
                        .data("compaction state lost (session closed?)")),
                    Err(_) => {
                        // Timed out: clear the flag so prompts stop queueing;
                        // a late completion falls through to normal handling.
                        state.compacting.set(false);
                        Err(acp::Error::internal_error()
                            .data("compaction timed out after 300s"))
                    }
                }
            }
            // Session fork: kernel sessionFork clones at the latest checkpoint.
            "x.ai/session/fork" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sourceSessionId")
                    .or_else(|| params.get("sessionId"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if !id.starts_with("sess_") {
                    return Err(acp::Error::invalid_params().data("bad sessionId"));
                }
                let kernel = self.kernel()?;
                let result = kernel
                    .call("session/fork", json!({"sessionId": id}))
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error().data(format!("fork failed: {e}"))
                    })?;
                let body = json!({"result": result});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize fork result");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Checkpoint rewind (Esc): points from the kernel's message
            // ledger; execute forks the conversation at the target user
            // message and swaps the fork in behind the same ACP session id,
            // so the pager rewinds in place.
            "x.ai/rewind/points" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let kernel_sid = self.session_kernel_id(&acp::SessionId::new(id.clone()));
                let points = rewind_points(&kernel_sid);
                let body = json!({"result": {"rewindPoints": points}});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize rewind points");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            "x.ai/rewind/execute" => {
                // The kernel's wire surface exposes no conversation-
                // truncating rewind: session/fork (message/turn targets)
                // was verified empirically to retain the full message
                // ledger, so the forked model still sees post-target turns.
                // The real rewindToMessage is an internal runtime command
                // without an app-server method. Until a ledger-surgery
                // implementation lands, answer explicitly instead of
                // pretending.
                let _ = args.params.get();
                return Err(acp::Error::method_not_found().data(
                    "rewind execute: kernel exposes no truncating rewind (fork keeps the full ledger); tracked for ledger-surgery implementation",
                ));
            }
            // Mid-turn "send now": the pager queues a follow-up client-side
            // and force-sends it via this method. The kernel's sendText has
            // startNow (preempt) / queue delivery modes, but preempting a
            // long GLM turn throws away its work — deliver as the next turn
            // instead (the proven plan-approval continuation path): the
            // current turn finishes, then this text auto-sends with no
            // further user action.
            "x.ai/interject" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let text = params
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if !id.starts_with("sess_") || text.is_empty() {
                    return Err(acp::Error::invalid_params().data("bad interject"));
                }
                let session = acp::SessionId::new(id.clone());
                let state = self
                    .shared
                    .state
                    .borrow()
                    .sessions
                    .get(&session)
                    .cloned()
                    .ok_or_else(|| acp::Error::invalid_params().data("unknown session"))?;
                let in_flight = state.turn_done.borrow().is_some();
                if in_flight {
                    let mut pending = state.pending_continuation.borrow_mut();
                    match pending.as_mut() {
                        Some(existing) => existing.push_str(&format!("\n\n{text}")),
                        None => *pending = Some(text),
                    }
                    debug_log("interject: queued as next-turn continuation");
                } else {
                    let kernel = self.kernel()?;
                    kernel
                        .call("session/send", kernel::send_params(&id, &text))
                        .await
                        .map_err(|e| {
                            acp::Error::internal_error().data(format!("interject send failed: {e}"))
                        })?;
                    debug_log("interject: sent immediately (no turn in flight)");
                }
                let body = json!({"sessionId": id, "accepted": true});
                let raw =
                    serde_json::value::to_raw_value(&body).expect("serialize interject ack");
                return Ok(acp::ExtResponse::new(raw.into()));
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

/// Image blocks of a prompt as kernel wire attachments. The kernel's
/// `session/send` normalizer takes inline base64 directly
/// (`{kind:"image", dataBase64, mimeType, filename}`), so pasted images need
/// no upload round-trip. Data-URI values are stripped to raw base64.
fn prompt_attachments(prompt: &[acp::ContentBlock]) -> Vec<Value> {
    prompt
        .iter()
        .filter_map(|block| match block {
            acp::ContentBlock::Image(image) => {
                let mut data = image.data.trim();
                if let Some(comma) = data.find(',').filter(|i| data.starts_with("data:")) {
                    data = &data[comma + 1..];
                }
                if data.is_empty() || !image.mime_type.starts_with("image/") {
                    return None;
                }
                let filename = image
                    .uri
                    .as_deref()
                    .and_then(|uri| uri.rsplit('/').next())
                    .filter(|name| !name.is_empty() && !name.contains(':'))
                    .unwrap_or("image.png")
                    .to_string();
                Some(json!({
                    "kind": "image",
                    "dataBase64": data,
                    "mimeType": image.mime_type,
                    "filename": filename,
                }))
            }
            _ => None,
        })
        .collect()
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
/// The ledger often lands a child's tail messages seconds AFTER the
/// parent turn closes — re-flush once more on a delay.
fn schedule_delayed_child_flush(
    gateway: AcpGatewaySender<acp::AgentSide>,
    state: Rc<SessionState>,
    child_id: String,
) {
    tokio::task::spawn_local(async move {
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        flush_child_content(&gateway, &state, &child_id);
    });
}

/// The kernel writes per-agent transcript directories under
/// `~/.zcode/cli/agents/<parent-kernel-session>/<agentId>/` — this is the
/// official client's "subagentTranscripts" channel (its storage-cleanup
/// categories map that exact prefix). RPC reads of child sessions are all
/// rejected ("Session is not active"), files are the real surface.
fn subagent_agent_dir(kernel_parent: &str, agent_id: &str) -> Option<std::path::PathBuf> {
    if !agent_id.starts_with("agent_") || !kernel_parent.starts_with("sess_") {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let dir = std::path::Path::new(&home)
        .join(".zcode/cli/agents")
        .join(kernel_parent)
        .join(agent_id);
    dir.is_dir().then_some(dir)
}

/// Best-effort metadata.json of a subagent run (status, tokens, duration).
fn subagent_metadata(kernel_parent: &str, agent_id: &str) -> Value {
    subagent_agent_dir(kernel_parent, agent_id)
        .and_then(|dir| {
            std::fs::read_to_string(dir.join("metadata.json"))
                .ok()
                .and_then(|raw| serde_json::from_str(&raw).ok())
        })
        .unwrap_or_else(|| json!({}))
}

/// New subagent content since the last poll, as (kind, text) pairs.
/// Primary channel: incremental output.txt tails keyed by a byte cursor.
/// Fallback (kernels that write no transcript files): message-ledger SQL.
fn collect_child_content(state: &Rc<SessionState>, child: &str) -> Vec<(&'static str, String)> {
    let kernel_parent = state.kernel_id.borrow().clone();
    let agent_id = state
        .child_agent
        .borrow()
        .get(child)
        .cloned()
        .unwrap_or_default();
    if let Some(dir) = subagent_agent_dir(&kernel_parent, &agent_id) {
        let mut out = Vec::new();
        if let Ok(bytes) = std::fs::read(dir.join("output.txt")) {
            let mut positions = state.child_file_pos.borrow_mut();
            let pos = positions.entry(child.to_string()).or_insert(0);
            let start = (*pos as usize).min(bytes.len());
            *pos = bytes.len() as u64;
            if bytes.len() > start {
                let text = String::from_utf8_lossy(&bytes[start..]).to_string();
                if !text.trim().is_empty() {
                    out.push(("text", text));
                }
            }
        }
        return out;
    }
    let mut seen_map = state.child_seen.borrow_mut();
    let seen = seen_map.entry(child.to_string()).or_default();
    child_new_messages(child, seen)
        .into_iter()
        .map(|(_, kind, text)| (kind, text))
        .collect()
}

/// Forward any not-yet-delivered child content (the transcript file often
/// lands its tail only as the parent turn closes).
fn flush_child_content(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    state: &Rc<SessionState>,
    child_id: &str,
) {
    if !state.subagents.borrow().contains_key(child_id) {
        return;
    }
    for (kind, text) in collect_child_content(state, child_id) {
        let update = if kind == "reasoning" {
            acp::SessionUpdate::AgentThoughtChunk(text_chunk(text))
        } else {
            acp::SessionUpdate::AgentMessageChunk(text_chunk(text))
        };
        let child = acp::SessionId::new(child_id.to_string());
        notify(gateway, &child, update);
    }
}

/// New assistant messages of a subagent child since the last poll, as
/// (message_id, kind, text) triples — kind is "reasoning" or "text".
fn child_new_messages(child_id: &str, seen: &mut std::collections::HashSet<String>) -> Vec<(String, &'static str, String)> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let db_path = std::path::Path::new(&home).join(".zcode/cli/db/db.sqlite");
    let mut out = Vec::new();
    let Ok(con) = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return out;
    };
    let Ok(mut stmt) = con.prepare(
        "SELECT m.id, p.data FROM message m JOIN part p ON p.message_id = m.id \
         WHERE m.session_id = ?1 AND m.data LIKE '%\"role\":\"assistant\"%' \
         ORDER BY m.sequence, p.sequence",
    ) else {
        return out;
    };
    let rows = stmt
        .query_map([child_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map(|rows| rows.filter_map(Result::ok).collect::<Vec<_>>())
        .unwrap_or_default();
    for (id, data) in rows {
        if seen.contains(&id) {
            continue;
        }
        let Ok(part) = serde_json::from_str::<Value>(&data) else {
            continue;
        };
        let kind = match part.get("type").and_then(Value::as_str) {
            Some("reasoning") => "reasoning",
            Some("text") => "text",
            _ => continue,
        };
        let Some(text) = part.get("text").and_then(Value::as_str) else {
            continue;
        };
        if !text.is_empty() {
            out.push((id.clone(), kind, text.to_string()));
        }
        seen.insert(id);
    }
    out
}

/// Is this kernel session id a subagent child the poller announced?
fn is_known_subagent_child(shared: &Rc<Shared>, session_id: &str) -> bool {
    shared
        .state
        .borrow()
        .sessions
        .values()
        .any(|state| state.subagents.borrow().contains_key(session_id))
}

/// Content-only translation for subagent child events: the pager's
/// subagent view renders these under the child's session id.
fn forward_child_event(gateway: &AcpGatewaySender<acp::AgentSide>, child: &acp::SessionId, payload: &Value) {
    let Some(event) = TurnEvent::decode(payload) else { return };
    match event.kind.as_str() {
        "text_delta" => {
            if !event.delta.is_empty() {
                notify(
                    gateway,
                    child,
                    acp::SessionUpdate::AgentMessageChunk(text_chunk(event.delta)),
                );
            }
        }
        "reasoning_delta" => {
            if !event.delta.is_empty() {
                notify(
                    gateway,
                    child,
                    acp::SessionUpdate::AgentThoughtChunk(text_chunk(event.delta)),
                );
            }
        }
        "tool_input_start" | "tool_call" => {
            let Some(call_id) = event.tool_call_id.as_deref() else { return };
            let call = acp::ToolCall::new(
                acp::ToolCallId::new(call_id.to_string()),
                event.tool_name.clone().unwrap_or_else(|| "tool".into()),
            )
            .status(acp::ToolCallStatus::InProgress);
            notify(gateway, child, acp::SessionUpdate::ToolCall(call));
        }
        "result" => {
            let Some(call_id) = event.tool_call_id.as_deref() else { return };
            let fields = acp::ToolCallUpdateFields::new()
                .status(acp::ToolCallStatus::Completed)
                .content(vec![acp::ToolCallContent::Content(acp::Content::new(
                    acp::ContentBlock::Text(acp::TextContent::new(
                        event.output.clone().unwrap_or_default(),
                    )),
                ))]);
            notify(
                gateway,
                child,
                acp::SessionUpdate::ToolCallUpdate(acp::ToolCallUpdate::new(
                    acp::ToolCallId::new(call_id.to_string()),
                    fields,
                )),
            );
        }
        _ => {}
    }
}

fn handle_event(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    shared: &Rc<Shared>,
    session_id: Option<&str>,
    payload: &Value,
) {
    let Some(session_id) = session_id else { return };
    let acp_session = acp::SessionId::new(session_id.to_string());
    let Some(state) = shared.state.borrow().sessions.get(&acp_session).cloned() else {
        // Not one of OUR sessions — but it may be a subagent child the
        // poller announced: forward its content to the pager's subagent
        // view (thoughts, text, tool calls) without parent turn logic.
        if is_known_subagent_child(shared, session_id) {
            forward_child_event(gateway, &acp_session, payload);
        } else {
            debug_log(&format!("handle_event: UNKNOWN session {session_id}"));
        }
        return;
    };
    let Some(event) = TurnEvent::decode(payload) else {
        debug_log(&format!("handle_event: undecodable payload kind={:?}", payload.get("kind")));
        return;
    };
    // Native subagent lifecycle rides the parent stream — handle before the
    // turn-event arms (it carries no deltas, only phase metadata).
    if event.kind == "subagent.lifecycle" {
        handle_subagent_lifecycle(gateway, &state, &acp_session, payload);
        return;
    }
    // Compaction turns (manual session/compact OR kernel-initiated) start as
    // normal-looking turns whose input is "/compact" with model-only
    // visibility. Track them so their completion can't be mistaken for the
    // user's turn and prompts sent meanwhile can be queued.
    if event.kind == "turn.started" {
        if event
            .input
            .as_deref()
            .is_some_and(|input| input.starts_with("/compact"))
        {
            state.compacting.set(true);
            *state.compact_turn_id.borrow_mut() = event.turn_id.clone();
            debug_log(&format!(
                "compaction turn started: {:?}",
                event.turn_id.as_deref().map(|t| &t[..t.len().min(24)])
            ));
        }
        return;
    }
    // End of a compaction turn: fire the compact waiter, refresh the context
    // meter (usage drops), and drain any prompt queued while compacting.
    // Never finish_turn here — no user turn is in flight (prompts sent during
    // compaction are queued, and the kernel rejects concurrent sends with
    // -32010), so resolving one would fake an answer like the old bug did.
    let is_compact_turn_end = state.compacting.get()
        && event.turn_id.is_some()
        && (event.turn_id == *state.compact_turn_id.borrow()
            || (state.compact_turn_id.borrow().is_none()
                && state.turn_done.borrow().is_none()));
    if is_compact_turn_end && (event.kind == "turn.completed" || event.kind == "turn.failed") {
        state.compacting.set(false);
        *state.compact_turn_id.borrow_mut() = None;
        let outcome = if event.kind == "turn.completed" {
            Ok(())
        } else {
            Err(event.output.clone().unwrap_or_else(|| "compaction failed".into()))
        };
        if let Some(waiter) = state.compact_wait.borrow_mut().take() {
            let _ = waiter.send(outcome);
        }
        if event.kind == "turn.completed" {
            // The kernel's final "Compacted" text lands as a normal message.
            if let Some(response) = event.output.as_deref().filter(|t| !t.is_empty()) {
                notify(
                    gateway,
                    &acp_session,
                    acp::SessionUpdate::AgentMessageChunk(text_chunk(response)),
                );
            }
            push_context_usage(gateway, shared, &state, &acp_session);
            drain_pending_continuation(shared, &acp_session);
        }
        return;
    }
    // Any activity after a turn.failed means the kernel is retrying the turn:
    // cancel the pending grace termination.
    if let Some(cancel) = state.fail_grace_cancel.borrow_mut().take() {
        let _ = cancel.send(());
        debug_log("turn.failed grace cancelled (turn continues)");
    }
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
            if let Some(cancel) = state.fail_grace_cancel.borrow_mut().take() {
                let _ = cancel.send(());
            }
            finish_turn(&state, gateway, &acp_session, acp::StopReason::EndTurn);
            drain_pending_continuation(shared, &acp_session);
            push_context_usage(gateway, shared, &state, &acp_session);
        }
        "turn.failed" => {
            // 0.16.9 emits turn.failed per failed ATTEMPT and may retry the
            // turn; finish only after a grace period with no further events.
            if let Some(why) = event.output.as_deref().filter(|t| !t.is_empty()) {
                notify(
                    gateway,
                    &acp_session,
                    acp::SessionUpdate::AgentMessageChunk(text_chunk(why)),
                );
            }
            let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
            *state.fail_grace_cancel.borrow_mut() = Some(cancel_tx);
            let state = state.clone();
            let session = acp_session.clone();
            let gateway = gateway.clone();
            let shared = Rc::clone(shared);
            tokio::task::spawn_local(async move {
                tokio::select! {
                    () = tokio::time::sleep(std::time::Duration::from_secs(20)) => {
                        state.fail_grace_cancel.borrow_mut().take();
                        debug_log("turn.failed grace expired — finishing turn");
                        finish_turn(&state, &gateway, &session, acp::StopReason::EndTurn);
                        drain_pending_continuation(&shared, &session);
                    }
                    _ = cancel_rx => {}
                }
            });
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
    let kernel_sid = state.kernel_id.borrow().clone();
    match kernel.request("session/send", kernel::send_params(&kernel_sid, &text)) {
        Ok(mut rx) => {
            tokio::task::spawn_local(async move {
                if let Ok(Err(rejection)) = rx.await {
                    debug_log(&format!("continuation send rejected: {rejection}"));
                }
            });
        }
        Err(error) => {
            tracing::warn!(%error, "plan continuation send failed");
        }
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
    // 0.16.9+ request-time account auth: the worker asks its host for the
    // account provider's API key before every model request. We are the
    // host — answer from the kernel's own credential store. The key never
    // leaves this process (no logging, no ACP forwarding).
    if method == "interaction/requestProviderRuntimeHeaders" {
        let provider_id = params
            .get("providerId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let reply = match coding_plan_api_key(&provider_id) {
            Some(api_key) => json!({
                "headersApplied": true,
                "requestAuth": {"apiKey": api_key},
            }),
            None => json!({
                "headersApplied": false,
                "errorMessage": "no coding-plan credential for provider",
            }),
        };
        if let Some(kernel) = shared.state.borrow().kernel.clone() {
            let _ = kernel.reply(envelope_id, reply);
        }
        return;
    }
    // Official MCP connectors (zcode_official auth) ask their host for the
    // identity headers the connector service validates — the desktop sends
    // the ZCode JWT plus the coding-plan maas JWT (oauth access token).
    if method == "interaction/requestOfficialMcpAuthHeaders" {
        let reply = official_mcp_auth_headers()
            .unwrap_or_else(|| json!({"ok": false, "reason": "official_auth_unavailable"}));
        if let Some(kernel) = shared.state.borrow().kernel.clone() {
            let _ = kernel.reply(envelope_id, reply);
        }
        return;
    }
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
            // The kernel switched modes server-side, but the pager's composer
            // tracks the client-side mode from CurrentModeUpdate only — without
            // this it stays "plan" after approval.
            let _ = gateway.forward_fire_and_forget(acp::SessionNotification::new(
                acp::SessionId::new(interaction.session_id.clone()),
                acp::SessionUpdate::CurrentModeUpdate(acp::CurrentModeUpdate::new(
                    acp::SessionModeId::new("build"),
                )),
            ));
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
