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
    /// interjectionIds of queued interjections, in order — used to broadcast
    /// x.ai/session/interjection when the continuation delivers them, so the
    /// pager can claim its optimistic echo blocks.
    pending_interjection_ids: RefCell<Vec<String>>,
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
    /// Monotonic per-session turn counter (bumped on every kernel
    /// turn.started). Lets the turn.failed grace task detect that a NEW
    /// turn replaced the failed one and skip its stale finish.
    turn_epoch: std::cell::Cell<u64>,
    /// After a v4 sendText(startNow) acceptance, the NEXT turn.started is
    /// OUR preempting turn; its turnId is recorded so the PREEMPTED turn's
    /// terminal event can't be mistaken for ours.
    v4_awaiting_turn_start: std::cell::Cell<bool>,
    v4_owned_turn: RefCell<Option<String>>,
    /// v4 conversation projection: latest snapshot revision + logEpoch
    /// (CAS tokens for editUserQuery and other row-targeting commands).
    v4_revision: std::cell::Cell<u64>,
    v4_log_epoch: RefCell<Option<String>>,
    /// Bumped on every v4 snapshot frame — lets callers await freshness.
    v4_snapshot_count: std::cell::Cell<u64>,
    /// realUser userInput rows in conversation order: (rowId, entityId,
    /// turnId, text) — rewind targets.
    v4_user_rows: RefCell<Vec<(i64, String, String, String)>>,
    /// assistantText rows in order: (rowId, entityId, turnId, state) —
    /// forkAssistant targets for deep rewind.
    v4_assistant_rows: RefCell<Vec<(i64, String, String, String)>>,
    /// turnHeader rows in order (raw JSON): file-change summaries and row
    /// actions (canRewindFiles / editDisposition) live here.
    v4_turn_headers: RefCell<Vec<Value>>,
    /// Kernel workflow runs (v4 projection workflowRuns.runs, raw).
    v4_workflow_runs: RefCell<Vec<Value>>,
    /// Kernel-authoritative queue state from the v4 projection (raw items).
    v4_queue_items: RefCell<Vec<(String, String)>>,
    /// Kernel queue autoDrain flag (v4 projection): stop() forces it false —
    /// queued items then wait for setAutoDrain to resume draining.
    v4_queue_autodrain: std::cell::Cell<bool>,
    /// Last-seen kernel goal state (v4 projection) for change detection.
    v4_goal: RefCell<Option<Value>>,
    /// Last-seen kernel background works (v4 projection).
    v4_background_works: RefCell<Vec<Value>>,
    /// Kernel-authoritative context usage from the v4 projection
    /// (snapshot.usage / state.updated usage patches): (usedTokens,
    /// maxTokens). The official client's context meter source — updates in
    /// real time, including the drop right after a compaction completes.
    v4_usage: RefCell<Option<(u64, u64)>>,
    /// Context tokens at goal start — the pager's live goal token line is
    /// (current context - baseline) while active; the frozen delta rides
    /// tokens_used on terminal states.
    goal_token_baseline: std::cell::Cell<Option<u64>>,
    /// True when the RUNNING compaction turn was started by our
    /// x.ai/compact_conversation ext (the pager already shows its Command
    /// state — banners are only for KERNEL-initiated auto compaction).
    compact_by_ext: std::cell::Cell<bool>,
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
            pending_interjection_ids: RefCell::new(Vec::new()),
            turn_epoch: std::cell::Cell::new(0),
            v4_awaiting_turn_start: std::cell::Cell::new(false),
            v4_owned_turn: RefCell::new(None),
            v4_revision: std::cell::Cell::new(0),
            v4_log_epoch: RefCell::new(None),
            v4_snapshot_count: std::cell::Cell::new(0),
            v4_user_rows: RefCell::new(Vec::new()),
            v4_assistant_rows: RefCell::new(Vec::new()),
            v4_turn_headers: RefCell::new(Vec::new()),
            v4_workflow_runs: RefCell::new(Vec::new()),
            v4_queue_items: RefCell::new(Vec::new()),
            v4_queue_autodrain: std::cell::Cell::new(true),
            v4_goal: RefCell::new(None),
            v4_background_works: RefCell::new(Vec::new()),
            v4_usage: RefCell::new(None),
            goal_token_baseline: std::cell::Cell::new(None),
            fail_grace_cancel: RefCell::new(None),
            compacting: std::cell::Cell::new(false),
            compact_by_ext: std::cell::Cell::new(false),
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

    /// Advertise ACP slash commands once the session pane exists. The pager
    /// drops AvailableCommandsUpdate notifications that arrive before the
    /// NewSession/LoadSession response has registered the pane, so fire from
    /// a short delayed task after the response lands.
    fn defer_available_commands(&self, session: acp::SessionId) {
        let gateway = self.gateway.clone();
        tokio::task::spawn_local(async move {
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
            push_available_commands(&gateway, &session);
        });
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
        let Some(home) = Some(std::path::PathBuf::from(zcode_home())) else {
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
        let model_ids: Vec<String> = Some(std::path::PathBuf::from(zcode_home()))
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
                    KernelMessage::V4Frame(params) => {
                        // Wire envelope: {wireVersion, kind, frame: {topic,
                        // payload: {kind: snapshot|deltas, ...}}} — unwrap.
                        let frame = params.get("frame").unwrap_or(&params);
                        if let Some(acp_sid) = frame
                            .get("topic")
                            .and_then(Value::as_str)
                            .and_then(|t| t.strip_prefix("conversation/"))
                        {
                            let session_key = acp::SessionId::new(acp_sid.to_string());
                            // Fork-swapped sessions keep their original ACP id
                            // while the v4 topic carries the fork's kernel id —
                            // fall back to a kernel_id scan so their projection
                            // updates (queue/goal/usage) keep flowing, and
                            // notify the ACP id the pager knows.
                            let found = pump_shared
                                .state
                                .borrow()
                                .sessions
                                .get_key_value(&session_key)
                                .map(|(k, st)| (k.clone(), st.clone()))
                                .or_else(|| {
                                    pump_shared
                                        .state
                                        .borrow()
                                        .sessions
                                        .iter()
                                        .find(|(_, st)| st.kernel_id.borrow().as_str() == acp_sid)
                                        .map(|(k, st)| (k.clone(), st.clone()))
                                });
                            if let Some((notify_sid, state)) = found {
                                let queue_before = state.v4_queue_items.borrow().clone();
                                let goal_before = state.v4_goal.borrow().clone();
                                let works_before = state.v4_background_works.borrow().clone();
                                let runs_before = state.v4_workflow_runs.borrow().clone();
                                let usage_before = *state.v4_usage.borrow();
                                update_v4_projection(&state, frame);
                                let queue_after = state.v4_queue_items.borrow().clone();
                                if queue_before != queue_after {
                                    let rows: Vec<(String, &str, String)> = queue_after
                                        .iter()
                                        .map(|(id, text)| (id.clone(), "prompt", text.clone()))
                                        .collect();
                                    broadcast_queue(&pump_gateway, notify_sid.0.as_ref(), &rows, None);
                                }
                                let goal_after = state.v4_goal.borrow().clone();
                                if goal_before != goal_after {
                                    // Goal token accounting: baseline = context
                                    // usage at goal start (the pager renders the
                                    // live delta while active — the same
                                    // mechanism the grok-native shell uses);
                                    // terminal states freeze the final delta.
                                    let used_now = state.v4_usage.borrow().map(|(u, _)| u);
                                    let was_set = goal_before.is_some();
                                    let is_set = goal_after.is_some();
                                    let terminal = goal_after.as_ref().is_some_and(|g| {
                                        matches!(
                                            g.get("status").and_then(Value::as_str),
                                            Some("verified") | Some("notSatisfied") | Some("failed")
                                        )
                                    });
                                    if !was_set && is_set {
                                        // No usage observation yet (fresh
                                        // session) means nothing was consumed
                                        // before the goal — baseline 0.
                                        state.goal_token_baseline.set(Some(used_now.unwrap_or(0)));
                                    } else if !is_set {
                                        state.goal_token_baseline.set(None);
                                    }
                                    let baseline = state.goal_token_baseline.get().unwrap_or(0);
                                    let tokens_used = if terminal {
                                        used_now.unwrap_or(0).saturating_sub(baseline)
                                    } else {
                                        0
                                    };
                                    emit_goal_updated(
                                        &pump_gateway,
                                        notify_sid.0.as_ref(),
                                        goal_after.as_ref(),
                                        baseline,
                                        tokens_used,
                                    );
                                }
                                let works_after = state.v4_background_works.borrow().clone();
                                if works_before != works_after {
                                    emit_background_tasks(
                                        &pump_gateway,
                                        notify_sid.0.as_ref(),
                                        &works_after,
                                    );
                                }
                                // Workflows: map the kernel's run states onto
                                // the pager's workflow_updated rail (the same
                                // wire the grok-native shell uses).
                                let runs_after = state.v4_workflow_runs.borrow().clone();
                                if runs_before != runs_after {
                                    emit_workflow_updates(
                                        &pump_gateway,
                                        notify_sid.0.as_ref(),
                                        &runs_after,
                                    );
                                }
                                // Real-time context meter: the kernel pushes
                                // usage patches whenever the value changes
                                // (conflation — unchanged values are not
                                // sent), including the drop right after a
                                // compaction completes. Forward each change as
                                // an ACP UsageUpdate so the pager's bar
                                // refreshes immediately, not at next turn end.
                                let usage_after = *state.v4_usage.borrow();
                                if usage_before != usage_after {
                                    if let Some((used, max)) = usage_after {
                                        notify(
                                            &pump_gateway,
                                            &notify_sid,
                                            acp::SessionUpdate::UsageUpdate(acp::UsageUpdate::new(
                                                used, max,
                                            )),
                                        );
                                    }
                                }
                            }
                        }
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
        .or_else(|| Some(std::path::PathBuf::from(zcode_home())))
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

/// Sync the kernel's auto-generated session title (session.title,
/// title_source='generated') into the pager's summary.json so the resume
/// list shows real titles instead of the stub's empty summary. Best-effort;
/// read-only on the kernel db.
fn sync_kernel_title(session_id: &str, cwd: &str) {
    let Some(home) = std::env::var_os("GROK_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| Some(std::path::PathBuf::from(zcode_home())))
    else {
        return;
    };
    let kernel_home = zcode_home();
    let Ok(con) = rusqlite::Connection::open_with_flags(
        std::path::Path::new(&kernel_home).join(".zcode/cli/db/db.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return;
    };
    let Ok((title, updated)) = con.query_row(
        "SELECT title, time_updated FROM session WHERE id = ?1",
        [session_id],
        |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
    ) else {
        return;
    };
    if title.trim().is_empty() {
        return;
    }
    let Some(encoded) = encode_cwd_dirname(cwd) else { return };
    let path = home
        .join(".grok")
        .join("sessions")
        .join(encoded)
        .join(session_id)
        .join("summary.json");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return;
    };
    let Ok(mut summary) = serde_json::from_str::<Value>(&raw) else {
        return;
    };
    let current = summary
        .get("session_summary")
        .and_then(Value::as_str)
        .unwrap_or("");
    let stamp = ms_to_rfc3339(updated);
    if current == title
        && summary.get("updated_at").and_then(Value::as_str) == Some(stamp.as_str())
    {
        return;
    }
    summary["session_summary"] = json!(title);
    summary["updated_at"] = json!(stamp);
    let _ = std::fs::write(&path, summary.to_string());
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
    let home = zcode_home();
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
    let mut rows = stmt
        .query_map([session_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })
        .map(|rows| rows.filter_map(Result::ok).collect::<Vec<_>>())
        .unwrap_or_default();
    if rows.is_empty() {
        return Vec::new();
    }
    // Real per-turn file-change flags: a turn spans from its user message
    // to the next; Write/Edit tool parts inside the span mark it. The
    // picker badges these points (the official turnHeader.fileChanges
    // equivalent reachable from the ledger).
    let edits: Vec<i64> = con
        .prepare(
            "SELECT m.sequence FROM part p JOIN message m ON m.id = p.message_id \
             WHERE p.session_id = ?1 AND json_extract(p.data, '$.type') = 'tool' \
             AND json_extract(p.data, '$.tool') IN ('Write','Edit') \
             ORDER BY m.sequence",
        )
        .and_then(|mut stmt| {
            stmt.query_map([session_id], |row| row.get::<_, i64>(0))
                .map(|rows| rows.filter_map(Result::ok).collect())
        })
        .unwrap_or_default();
    // Boundaries: the leading sequence of each user message.
    let bounds: Vec<i64> = {
        let mut b: Vec<i64> = con
            .prepare(
                "SELECT MIN(m.sequence) FROM message m WHERE m.session_id = ?1 \
                 AND m.data LIKE '%\"role\":\"user\"%' GROUP BY m.id ORDER BY MIN(m.sequence)",
            )
            .and_then(|mut stmt| {
                stmt.query_map([session_id], |row| row.get::<_, i64>(0))
                    .map(|rows| rows.filter_map(Result::ok).collect())
            })
            .unwrap_or_default();
        b.push(i64::MAX);
        b
    };
    rows.sort_by_key(|(_, _, _)| 0);
    let _ = &mut rows;
    let mark = |user_seq_bound_idx: usize| -> bool {
        let from = bounds[user_seq_bound_idx];
        let to = bounds[user_seq_bound_idx + 1];
        edits.iter().any(|seq| *seq > from && *seq < to)
    };
    bounds[..bounds.len() - 1]
        .iter()
        .enumerate()
        .zip(rows.iter())
        .map(|((idx, _), (id, created, preview))| {
            let preview = preview
                .as_deref()
                .and_then(|data| {
                    serde_json::from_str::<Value>(data)
                        .ok()
                        .and_then(|j| j.get("text").and_then(Value::as_str).map(str::to_string))
                })
                .unwrap_or_default();
            json!({
                "promptIndex": idx,
                "createdAt": ms_to_rfc3339(*created),
                "numFileSnapshots": 0,
                "promptPreview": preview.chars().take(160).collect::<String>(),
                "hasFileChanges": mark(idx),
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
    let home = zcode_home();
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
    let home = zcode_home();
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
    let home = zcode_home();
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

/// The kernel's home directory as a String (HOME on Unix, USERPROFILE on
/// Windows — the official Windows client uses %USERPROFILE%\.zcode, same
/// layout as ~/.zcode). Last-resort "/root" matches the old inline default.
fn zcode_home() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/root".to_string())
}

/// The kernel's TodoWrite state for a session, from its own db (the
/// `todo` table: content/status/position). Empty when the model never
/// used TodoWrite.
fn kernel_todos(kernel_sid: &str) -> Vec<(String, String)> {
    let home = zcode_home();
    let db_path = std::path::Path::new(&home).join(".zcode/cli/db/db.sqlite");
    let Ok(con) = rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return Vec::new();
    };
    let Ok(mut stmt) = con.prepare(
        "SELECT content, status FROM todo WHERE session_id = ?1 ORDER BY position",
    ) else {
        return Vec::new();
    };
    stmt.query_map([kernel_sid], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })
    .map(|rows| rows.filter_map(Result::ok).collect())
    .unwrap_or_default()
}

/// Push the model's TodoWrite list to the pager's todos pane. The pane
/// consumes Plan-shaped entries; only sent when there ARE todos, and only
/// when no plan is currently in flight (plan entries own the pane then).
fn push_todos(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    state: &Rc<SessionState>,
    session: &acp::SessionId,
) {
    let todos = kernel_todos(&state.kernel_id.borrow());
    if todos.is_empty() {
        return;
    }
    // Plan entries map 1:1 onto todo rows in the pager; statuses map to
    // the entry states the pane understands.
    let entries: Vec<acp::PlanEntry> = todos
        .iter()
        .map(|(content, status)| {
            let st = match status.as_str() {
                "completed" => acp::PlanEntryStatus::Completed,
                "in_progress" => acp::PlanEntryStatus::InProgress,
                _ => acp::PlanEntryStatus::Pending,
            };
            acp::PlanEntry::new(content.clone(), acp::PlanEntryPriority::Medium, st)
        })
        .collect();
    notify(gateway, session, acp::SessionUpdate::Plan(acp::Plan::new(entries)));
}

/// The /goal command's advertisement JSON (pager `availableCommands` shape).
/// The command itself is intercepted in prompt() and routed to the kernel's
/// sessionGoal RPC — the ad only makes it discoverable in the dropdown.
fn goal_command_json() -> Value {
    json!({
        "name": "goal",
        "description": "Set a long-running goal the kernel pursues autonomously (pause/resume/clear/show)",
        "input": { "hint": "<objective> | pause | resume | clear | show" },
    })
}

fn push_available_commands(gateway: &AcpGatewaySender<acp::AgentSide>, session: &acp::SessionId) {
    let cmd: acp::AvailableCommand =
        serde_json::from_value(goal_command_json()).unwrap_or_else(|_| {
            acp::AvailableCommand::new("goal", "Set a long-running goal")
        });
    let rate = acp::AvailableCommand::new(
        "rate",
        "Rate the last assistant reply (kernel feedback)",
    );
    let drain = acp::AvailableCommand::new(
        "drain",
        "Control kernel queue auto-drain (stopped queues wait)",
    );
    let filerewind = acp::AvailableCommand::new(
        "filerewind",
        "Revert the last turn's file changes (chat history kept)",
    );
    let retry = acp::AvailableCommand::new(
        "retry",
        "Re-run the last user turn (official retryTurn)",
    );
    notify(
        gateway,
        session,
        acp::SessionUpdate::AvailableCommandsUpdate(acp::AvailableCommandsUpdate::new(vec![
            cmd, rate, drain, filerewind, retry,
        ])),
    );
}

/// Emit the pager's context meter update for a session.
fn push_context_usage(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    shared: &Rc<Shared>,
    state: &Rc<SessionState>,
    session: &acp::SessionId,
) {
    // The v4 projection is kernel-authoritative and compact-aware (its
    // usedTokens drops the moment compaction finishes); the db ledger's
    // latest main_turn stays at the pre-compact number until the NEXT user
    // turn, so prefer the projection whenever it has a value.
    let (used, size) = match *state.v4_usage.borrow() {
        Some(v4) => v4,
        None => {
            let used = last_turn_context_tokens(&state.kernel_id.borrow());
            let size = context_window_tokens(shared);
            (used, size)
        }
    };
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
    let home = zcode_home();
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

    let home = std::path::PathBuf::from(zcode_home());
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
    let home = zcode_home();
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
    zcode_home()
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
    let Some(home) = Some(std::path::PathBuf::from(zcode_home())) else {
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
        .open(std::env::temp_dir().join("zcode-agent-debug.log"))
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
            .auth_methods(vec![method])
            .meta({
                // Bootstrap the slash dropdown before any session exists:
                // the pager seeds `availableCommands` from here, and the
                // runtime AvailableCommandsUpdate only lands once a session
                // pane is up (pre-session copies are dropped).
                let mut meta = acp::Meta::new();
                meta.insert(
                    "availableCommands".to_string(),
                    json!([
                        goal_command_json(),
                        json!({
                            "name": "rate",
                            "description": "Rate the last assistant reply (kernel feedback)",
                            "input": { "hint": "like | dislike | clear" },
                        }),
                        json!({
                            "name": "drain",
                            "description": "Control kernel queue auto-drain (stopped queues wait)",
                            "input": { "hint": "on | off | (bare = status)" },
                        }),
                        json!({
                            "name": "filerewind",
                            "description": "Revert the last turn's file changes (chat history kept)",
                            "input": { "hint": "(bare = preview) | apply" },
                        }),
                        json!({
                            "name": "retry",
                            "description": "Re-run the last user turn (official retryTurn)",
                        }),
                    ]),
                );
                meta
            }))
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
        enable_dynamic_workflows(&kernel, &cwd.to_string_lossy()).await;
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
        // v4 conversation projection: CAS tokens + rows for editUserQuery
        // (rewind) and the kernel-authoritative queue.
        v4_subscribe(&kernel, &session_id);
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
        self.defer_available_commands(acp_session.clone());
        let mut response = acp::NewSessionResponse::new(acp_session);
        response.models = initial_models;
        Ok(response)
    }

    async fn load_session(&self, args: acp::LoadSessionRequest) -> acp::Result<acp::LoadSessionResponse> {
        debug_log("acp: load_session");
        let kernel = self.ensure_kernel().await?;
        let session_id = args.session_id.0.as_ref().to_string();
        enable_dynamic_workflows(&kernel, &args.cwd.to_string_lossy()).await;
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
        v4_subscribe(&kernel, &session_id);
        let state = Rc::new(SessionState::new());
        *state.cwd.borrow_mut() = args.cwd.to_string_lossy().to_string();
        *state.kernel_id.borrow_mut() = args.session_id.0.to_string();
        // Seed HISTORICAL subagents (from before the resume) as already
        // announced: session/subagents returns them under ended/children,
        // and the poller would otherwise rebroadcast them as a fresh
        // spawn+finish on the first prompt — surfacing as a spurious
        // "Ran 1 subagent 1 failed" row after every resume.
        if let Ok(snapshot) = kernel
            .call("session/subagents", json!({"sessionId": session_id}))
            .await
        {
            let ids: Vec<String> = ["/childSessionIds", "/running", "/ended/items"]
                .iter()
                .flat_map(|pointer| {
                    snapshot
                        .pointer(pointer)
                        .and_then(Value::as_array)
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| {
                                    v.as_str()
                                        .map(str::to_string)
                                        .or_else(|| {
                                            v.get("childSessionId")
                                                .and_then(Value::as_str)
                                                .map(str::to_string)
                                        })
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .collect();
            let mut subs = state.subagents.borrow_mut();
            for id in ids {
                subs.insert(id, (true, true));
            }
            debug_log(&format!("resume: seeded {} historical subagent(s)", subs.len()));
        }
        self.shared
            .state
            .borrow_mut()
            .sessions
            .insert(args.session_id.clone(), state);
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
                // Engine nudges (todo reminders, etc.) are stored as
                // user-role messages but carry synthetic/model-only markers
                // — the official clients fold them out of the transcript.
                // Never replay them as if the user said them.
                if message.pointer("/info/synthetic").and_then(Value::as_bool) == Some(true)
                    || message.pointer("/info/metadata/visibility").and_then(Value::as_str)
                        == Some("model-only")
                {
                    continue;
                }
                let mut text = String::new();
                // Reasoning parts replayed individually with their own
                // kernel-recorded spans: (text, Option<(start, end)>).
                let mut reasonings: Vec<(String, Option<(u64, u64)>)> = Vec::new();
                if let Some(parts) = message.get("parts").and_then(Value::as_array) {
                    for part in parts {
                        match part.get("type").and_then(Value::as_str) {
                            Some("text") => {
                                text.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""))
                            }
                            Some("reasoning") => {
                                let body = part.get("text").and_then(Value::as_str).unwrap_or("");
                                // Part-level time is the exact thinking span;
                                // tolerate {start,end} (db shape) and
                                // {created,completed} (message-level shape).
                                let span = part
                                    .pointer("/time/start")
                                    .and_then(Value::as_u64)
                                    .zip(part.pointer("/time/end").and_then(Value::as_u64))
                                    .or_else(|| {
                                        part.pointer("/time/created")
                                            .and_then(Value::as_u64)
                                            .zip(part.pointer("/time/completed").and_then(Value::as_u64))
                                    });
                                if !body.trim().is_empty() {
                                    reasonings.push((body.to_string(), span));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // The pager's "Thought for Xs" contract: elapsed = last
                // chunk's agentTimestampMs - streamStartMs, and isReplay must
                // be true or the block runs a LOCAL timer that freezes to ~0ms
                // (the two replay chunks arrive microseconds apart — the old
                // "Thought for 0.0s" bug). Each reasoning part gets its own
                // streamStartMs so multi-segment turns render separate blocks.
                let message_span = message
                    .pointer("/info/time/created")
                    .and_then(Value::as_u64)
                    .zip(message.pointer("/info/time/completed").and_then(Value::as_u64));
                for (body, span) in &reasonings {
                    let span = span.or(message_span);
                    let notify_thought = |chunk_text: String, ts: u64, start: u64| {
                        let mut notif = acp::SessionNotification::new(
                            args.session_id.clone(),
                            acp::SessionUpdate::AgentThoughtChunk(text_chunk(chunk_text)),
                        );
                        let mut meta = acp::Meta::new();
                        meta.insert("agentTimestampMs".to_string(), json!(ts));
                        meta.insert("streamStartMs".to_string(), json!(start));
                        meta.insert("isReplay".to_string(), json!(true));
                        notif.meta = Some(meta);
                        self.gateway.forward_fire_and_forget(notif);
                    };
                    match span {
                        Some((start, end)) if end > start => {
                            // Char-safe midpoint: byte split_at would panic
                            // inside a multi-byte character.
                            let mut split = body.len() / 2;
                            while split < body.len() && !body.is_char_boundary(split) {
                                split += 1;
                            }
                            let (head, tail) = body.split_at(split.max(1));
                            notify_thought(head.to_string(), start, start);
                            notify_thought(tail.to_string(), end, start);
                        }
                        _ => {
                            // No usable span: still mark isReplay so the pager
                            // shows "Thought" without a bogus 0.0s timer.
                            let mut notif = acp::SessionNotification::new(
                                args.session_id.clone(),
                                acp::SessionUpdate::AgentThoughtChunk(text_chunk(body.clone())),
                            );
                            let mut meta = acp::Meta::new();
                            meta.insert("isReplay".to_string(), json!(true));
                            notif.meta = Some(meta);
                            self.gateway.forward_fire_and_forget(notif);
                        }
                    }
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
            sync_kernel_title(state.kernel_id.borrow().as_ref(), &state.cwd.borrow());
        }
        tracing::info!(%session_id, "zcode session resumed");
        self.defer_available_commands(args.session_id.clone());
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
        // /goal text command interception: the kernel does NOT parse slash
        // commands from session/send (they arrive as plain text and confuse
        // the model) — route to the structured sessionGoal RPC instead.
        if text.trim_start().starts_with("/goal") && attachments.is_empty() {
            let kernel_sid = state.kernel_id.borrow().clone();
            let rest = text.trim_start()[5..].trim().to_string();
            let mut action = "set".to_string();
            let mut objective = rest.clone();
            match rest.as_str() {
                "" | "show" => action = "show".to_string(),
                "pause" => action = "pause".to_string(),
                "resume" => action = "resume".to_string(),
                "clear" => action = "clear".to_string(),
                other if other.starts_with("replace ") => {
                    action = "replace".to_string();
                    objective = other[8..].trim().to_string();
                }
                _ => {}
            }
            let mut payload = json!({"sessionId": kernel_sid, "action": action});
            if action == "set" || action == "replace" {
                if objective.is_empty() {
                    notify(
                        &self.gateway,
                        &args.session_id,
                        acp::SessionUpdate::AgentMessageChunk(text_chunk(
                            "用法：/goal <目标> | /goal pause | /goal resume | /goal clear | /goal show",
                        )),
                    );
                    return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
                }
                payload["objective"] = json!(objective);
            }
            debug_log(&format!("goal command: {action}"));
            let result = kernel.call("session/goal", payload).await;
            let reply = match result {
                Ok(v) => v
                    .get("response")
                    .and_then(Value::as_str)
                    .unwrap_or("目标已更新。")
                    .to_string(),
                Err(e) => format!("goal 命令失败: {e}"),
            };
            notify(
                &self.gateway,
                &args.session_id,
                acp::SessionUpdate::AgentMessageChunk(text_chunk(reply)),
            );
            return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
        }
        // /rate like|dislike|clear — message feedback via the v4
        // setAssistantFeedback command (the official desktop's thumbs).
        // Targets the LAST assistant message: a CAS row-targeting command,
        // so re-subscribe for a fresh revision before firing.
        if text.trim_start().starts_with("/rate") && attachments.is_empty() {
            let arg = text.trim_start()[5..].trim();
            let feedback = match arg {
                "like" | "dislike" => Some(arg),
                "clear" | "none" => None,
                _ => {
                    notify(
                        &self.gateway,
                        &args.session_id,
                        acp::SessionUpdate::AgentMessageChunk(text_chunk(
                            "用法：/rate like | /rate dislike | /rate clear（作用于最后一条回复）",
                        )),
                    );
                    return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
                }
            };
            let kernel_sid = state.kernel_id.borrow().clone();
            {
                let before = state.v4_snapshot_count.get();
                v4_subscribe(&kernel, &kernel_sid);
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while state.v4_snapshot_count.get() <= before
                    && std::time::Instant::now() < deadline
                {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
            let rows = state.v4_assistant_rows.borrow().clone();
            let target = rows
                .iter()
                .rev()
                .find(|(_, _, _, st)| st != "interrupted" && st != "failed")
                .map(|(rid, eid, _, _)| (*rid, eid.clone()));
            let reply = match target {
                Some((row_id, entity_id)) => {
                    let payload = json!({
                        "target": {"rowId": row_id, "entityId": entity_id},
                        "feedback": feedback,
                    });
                    let cas = state
                        .v4_log_epoch
                        .borrow()
                        .clone()
                        .map(|epoch| (state.v4_revision.get(), epoch));
                    match v4_command(
                        &kernel,
                        &kernel_sid,
                        "setAssistantFeedback",
                        payload,
                        cas,
                    )
                    .await
                    {
                        Ok(ack) if ack.get("status") == Some(&json!("accepted")) => format!(
                            "已{}最后一条回复。",
                            match feedback {
                                Some("like") => "点赞",
                                Some(_) => "点踩",
                                None => "清除评价",
                            }
                        ),
                        Ok(ack) => format!(
                            "评价未生效: {} ({})",
                            ack.get("reasonCode").and_then(Value::as_str).unwrap_or("?"),
                            ack.get("message").and_then(Value::as_str).unwrap_or("")
                        ),
                        Err(e) => format!("评价失败: {e}"),
                    }
                }
                None => "没有可评价的回复。".to_string(),
            };
            notify(
                &self.gateway,
                &args.session_id,
                acp::SessionUpdate::AgentMessageChunk(text_chunk(reply)),
            );
            return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
        }
        // /drain on|off|show — kernel queue auto-drain control (v4
        // setAutoDrain). stop() forces it false; queued kernel items then
        // wait until drain resumes.
        if text.trim_start().starts_with("/drain") && attachments.is_empty() {
            let arg = text.trim_start()[6..].trim();
            let kernel_sid = state.kernel_id.borrow().clone();
            let reply = match arg {
                "on" | "off" => {
                    let want = arg == "on";
                    // CAS command on this kernel: refresh for a fresh
                    // revision, then fire with the tokens.
                    {
                        let before = state.v4_snapshot_count.get();
                        v4_subscribe(&kernel, &kernel_sid);
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(5);
                        while state.v4_snapshot_count.get() <= before
                            && std::time::Instant::now() < deadline
                        {
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    }
                    let cas = state
                        .v4_log_epoch
                        .borrow()
                        .clone()
                        .map(|epoch| (state.v4_revision.get(), epoch));
                    match v4_command(
                        &kernel,
                        &kernel_sid,
                        "setAutoDrain",
                        json!({"autoDrain": want}),
                        cas,
                    )
                    .await
                    {
                        Ok(ack) if ack.get("status") == Some(&json!("accepted")) => format!(
                            "内核队列排水已{}。",
                            if want { "恢复" } else { "暂停" }
                        ),
                        Ok(ack) => format!(
                            "设置未生效: {} ({})",
                            ack.get("reasonCode").and_then(Value::as_str).unwrap_or("?"),
                            ack.get("message").and_then(Value::as_str).unwrap_or("")
                        ),
                        Err(e) => format!("设置失败: {e}"),
                    }
                }
                _ => format!(
                    "当前排水状态：{}（{}）。用法：/drain on | /drain off",
                    if state.v4_queue_autodrain.get() { "开" } else { "关" },
                    match state
                        .v4_queue_items
                        .borrow()
                        .len() {
                            0 => "队列为空".to_string(),
                            n => format!("内核队列还有 {n} 条"),
                        }
                ),
            };
            notify(
                &self.gateway,
                &args.session_id,
                acp::SessionUpdate::AgentMessageChunk(text_chunk(reply)),
            );
            return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
        }
        // /filerewind [preview|apply] — workspace-only file revert (v4
        // applyFileRewind): restores the files the target turn touched
        // WITHOUT truncating the chat history. Targets the last completed
        // assistant row; preview lists what would happen.
        if text.trim_start().starts_with("/filerewind") && attachments.is_empty() {
            let arg = text.trim_start()[11..].trim();
            let kernel_sid = state.kernel_id.borrow().clone();
            {
                let before = state.v4_snapshot_count.get();
                v4_subscribe(&kernel, &kernel_sid);
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while state.v4_snapshot_count.get() <= before
                    && std::time::Instant::now() < deadline
                {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
            // Target = the turnHeader row (canRewindFiles lives there,
            // not on assistant rows — targeting those gets
            // guard.actionUnavailable).
            let target = state
                .v4_turn_headers
                .borrow()
                .iter()
                .rev()
                .find(|h| {
                    h.pointer("/actions/canRewindFiles").and_then(Value::as_bool) == Some(true)
                        && h.get("state").and_then(Value::as_str) == Some("completedSuccess")
                })
                .and_then(|h| {
                    Some((
                        h.get("rowId").and_then(Value::as_i64)?,
                        h.get("entityId").and_then(Value::as_str)?.to_string(),
                    ))
                });
            let reply = match target {
                Some((row_id, entity_id)) => {
                    let cas = state
                        .v4_log_epoch
                        .borrow()
                        .clone()
                        .map(|epoch| (state.v4_revision.get(), epoch));
                    if arg == "apply" {
                        match cas {
                            Some((rev, epoch)) => {
                                match v4_command(
                                    &kernel,
                                    &kernel_sid,
                                    "applyFileRewind",
                                    json!({"target": {"rowId": row_id, "entityId": entity_id}}),
                                    Some((rev, epoch)),
                                )
                                .await
                                {
                                    Ok(ack) if ack.get("status") == Some(&json!("accepted")) => {
                                        "文件回退已执行（会话历史保留）。".to_string()
                                    }
                                    Ok(ack) => format!(
                                        "文件回退未生效: {}",
                                        ack.get("reasonCode").and_then(Value::as_str).unwrap_or("?")
                                    ),
                                    Err(e) => format!("文件回退失败: {e}"),
                                }
                            }
                            None => "投影尚未就绪，稍后重试。".to_string(),
                        }
                    } else {
                        // Preview: read-only RPC with the same CAS tokens.
                        match cas {
                            Some((rev, epoch)) => {
                                match kernel
                                    .call(
                                        "v4/conversation/fileRewindPreview",
                                        json!({
                                            "sessionId": kernel_sid,
                                            "target": {"rowId": row_id, "entityId": entity_id},
                                            "baseRevision": rev,
                                            "baseLogEpoch": epoch,
                                        }),
                                    )
                                    .await
                                {
                                    Ok(v) => {
                                        let fmt_file = |f: &Value| {
                                            format!(
                                                "  {} {}（{} 次操作）",
                                                f.get("action").and_then(Value::as_str).unwrap_or("?"),
                                                f.get("path").and_then(Value::as_str).unwrap_or("?"),
                                                f.get("operationCount").and_then(Value::as_u64).unwrap_or(0)
                                            )
                                        };
                                        let safe: Vec<String> = v
                                            .get("safeFiles")
                                            .and_then(Value::as_array)
                                            .map(|a| a.iter().map(fmt_file).collect())
                                            .unwrap_or_default();
                                        let unsafe_n = v
                                            .get("unsafeFiles")
                                            .and_then(Value::as_array)
                                            .map(|a| a.len())
                                            .unwrap_or(0);
                                        if safe.is_empty() && unsafe_n == 0 {
                                            "该回合没有可回退的文件改动。".to_string()
                                        } else {
                                            format!(
                                                "将回退:\n{}\n{}（执行：/filerewind apply）",
                                                safe.join("\n"),
                                                if unsafe_n > 0 {
                                                    format!("另有 {unsafe_n} 个文件无法安全回退")
                                                } else {
                                                    String::new()
                                                }
                                            )
                                        }
                                    }
                                    Err(e) => format!("预览失败: {e}"),
                                }
                            }
                            None => "投影尚未就绪，稍后重试。".to_string(),
                        }
                    }
                }
                None => "没有可定位的回复行。".to_string(),
            };
            notify(
                &self.gateway,
                &args.session_id,
                acp::SessionUpdate::AgentMessageChunk(text_chunk(reply)),
            );
            return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
        }
        // /retry — re-run the last user turn via the official v4 retryTurn
        // (row-targeting CAS command; targets the last realUser input row).
        if text.trim_start().starts_with("/retry") && attachments.is_empty() {
            let kernel_sid = state.kernel_id.borrow().clone();
            {
                let before = state.v4_snapshot_count.get();
                v4_subscribe(&kernel, &kernel_sid);
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while state.v4_snapshot_count.get() <= before
                    && std::time::Instant::now() < deadline
                {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            }
            // Row-targeting lesson from applyFileRewind: retryTurn targets
            // the turnHeader row, not the input row. canRetry only rides
            // retryable (failed/interrupted) turns — prefer it, else the
            // last header and let the kernel's guard decide.
            let headers = state.v4_turn_headers.borrow().clone();
            let target = headers
                .iter()
                .rev()
                .find(|h| {
                    h.pointer("/actions/canRetry").and_then(Value::as_bool) == Some(true)
                })
                .or_else(|| headers.iter().rev().find(|h| !h.get("entityId").is_none()))
                .and_then(|h| {
                    Some((
                        h.get("rowId").and_then(Value::as_i64)?,
                        h.get("entityId").and_then(Value::as_str)?.to_string(),
                    ))
                });
            let reply = match target {
                Some((row_id, entity_id)) => {
                    let cas = state
                        .v4_log_epoch
                        .borrow()
                        .clone()
                        .map(|epoch| (state.v4_revision.get(), epoch));
                    match cas {
                        Some((rev, epoch)) => {
                            match v4_command(
                                &kernel,
                                &kernel_sid,
                                "retryTurn",
                                json!({"target": {"rowId": row_id, "entityId": entity_id}}),
                                Some((rev, epoch)),
                            )
                            .await
                            {
                                Ok(ack) if ack.get("status") == Some(&json!("accepted")) => {
                                    "已通过官方通道重试上一轮。".to_string()
                                }
                                Ok(ack) => format!(
                                    "重试未生效: {}",
                                    ack.get("reasonCode").and_then(Value::as_str).unwrap_or("?")
                                ),
                                Err(e) => format!("重试失败: {e}"),
                            }
                        }
                        None => "投影尚未就绪，稍后重试。".to_string(),
                    }
                }
                None => "没有可重试的用户轮次。".to_string(),
            };
            notify(
                &self.gateway,
                &args.session_id,
                acp::SessionUpdate::AgentMessageChunk(text_chunk(reply)),
            );
            return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
        }
        // Send-now (强插): the pager marks interrupting prompts with
        // meta.sendNow — cancel the running kernel turn, wait for the
        // session to free, then send ours. The kernel rejects concurrent
        // sends with -32010, so without this the interrupt fails outright.
        let send_now = args
            .meta
            .as_ref()
            .and_then(|m| m.get("sendNow"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut already_sent = false;
        if send_now && (state.turn_done.borrow().is_some() || state.compacting.get()) {
            // Preferred path: the kernel's v4 command surface — sendText
            // with requestedDelivery startNow atomically preempts the
            // running turn (the official desktop path). Falls back to the
            // legacy stop+retry loop on kernels without v4.
            {
                let kernel_sid = state.kernel_id.borrow().clone();
                let held = held_queue_disposition(&state);
                match v4_send_text_held(&kernel, &kernel_sid, &text, "startNow", held.as_ref().map(|h| (h.0, h.1.clone()))).await {
                    Ok(ack) if ack.get("status") == Some(&json!("accepted")) => {
                        debug_log("prompt: sendNow via v4 startNow accepted");
                        state.v4_awaiting_turn_start.set(true);
                        already_sent = true;
                    }
                    other => {
                        debug_log(&format!(
                            "prompt: v4 startNow unavailable ({:?}), falling back to stop+retry",
                            other.as_ref().map(|a| a.get("status").cloned()).map_err(|e| e.clone())
                        ));
                    }
                }
            }
            if already_sent {
                // The preempted turn will emit its own terminal event; our
                // turn_done below resolves when OUR v4 turn completes.
                if let Some(old) = state.turn_done.borrow_mut().take() {
                    let _ = old.send(acp::StopReason::Cancelled);
                }
                if let Some(cancel) = state.fail_grace_cancel.borrow_mut().take() {
                    let _ = cancel.send(());
                }
            }
        }
        if send_now && !already_sent && (state.turn_done.borrow().is_some() || state.compacting.get()) {
            debug_log("prompt: sendNow — cancelling running turn");
            state.cancelled.set(true);
            {
                let kernel_sid = state.kernel_id.borrow().clone();
                let _ = kernel.request("session/stop", kernel::stop_params(&kernel_sid));
            }
            // Retry until the kernel frees the session: -32010 rejects fast,
            // acceptance answers fast ({status:"prompt_started"}).
            let kernel_sid = state.kernel_id.borrow().clone();
            let send_params = kernel::send_params_with_attachments(&kernel_sid, &text, &attachments);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(25);
            loop {
                match kernel.call("session/send", send_params.clone()).await {
                    Ok(_) => {
                        already_sent = true;
                        break;
                    }
                    Err(e) if e.contains("already running") => {
                        if std::time::Instant::now() >= deadline {
                            break;
                        }
                        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                    }
                    Err(_) => break,
                }
            }
            // Settle the interrupted prompt and cancel its failure grace so
            // neither resolves OUR turn later (the epoch guard also covers a
            // late turn.failed from the stop).
            if let Some(cancel) = state.fail_grace_cancel.borrow_mut().take() {
                let _ = cancel.send(());
            }
            if let Some(old) = state.turn_done.borrow_mut().take() {
                let _ = old.send(acp::StopReason::Cancelled);
            }
            if !already_sent {
                return Err(acp::Error::internal_error()
                    .data("kernel did not free the session for send-now"));
            }
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
            // The kernel queue (v4 sendText queue) WEDGES when the running
            // turn is a COMPACTION turn (empirically verified: the queued
            // item never drains and the compact completion stalls) — the
            // official compact-reentry route is the runtime-internal
            // steerTurn. Use the agent-local continuation slot here; the
            // kernel queue stays available for regular turns.
            {
            let kernel_queue = false;
            debug_log("prompt: queued behind running compaction");
            *state.pending_continuation.borrow_mut() = Some(text.clone());
            // Server-queue snapshot: the queued row must be visible to the
            // pager (and reconciled away when delivered).
            if !kernel_queue {
                let prompt_id = args
                    .meta
                    .as_ref()
                    .and_then(|m| m.get("promptId"))
                    .and_then(Value::as_str)
                    .unwrap_or("queued-prompt")
                    .to_string();
                broadcast_queue(
                    &self.gateway,
                    &args.session_id.0,
                    &[(prompt_id, "prompt", text)],
                    None,
                );
            }
            rx.await.unwrap_or(acp::StopReason::EndTurn)
            }
        } else {
            spawn_subagent_poller(&self.gateway, kernel.clone(), state.clone(), args.session_id.clone());
            // The host-pushed account snapshot decays when the kernel re-asserts
            // its registry against the builtin revision — re-push before sends.
            self.push_account_config(&kernel).await;
            let send_rx = if already_sent {
                // send-now already performed the kernel send during the
                // cancel-wait; nothing to send again.
                None
            } else {
                Some(kernel
                    .request(
                        "session/send",
                        kernel::send_params_with_attachments(&state.kernel_id.borrow(), &text, &attachments),
                    )
                    .map_err(|e| acp::Error::internal_error().data(e.to_string()))?)
            };
            // Reconcile the pager's server queue: clear the send-now echo
            // row, surface this prompt as running, keep any queued
            // continuation visible.
            {
                let prompt_id = args
                    .meta
                    .as_ref()
                    .and_then(|m| m.get("promptId"))
                    .and_then(Value::as_str)
                    .unwrap_or("kernel-turn")
                    .to_string();
                let queued: Vec<(String, &str, String)> = state
                    .pending_continuation
                    .borrow()
                    .as_ref()
                    .map(|t| vec![("queued-prompt".to_string(), "prompt", t.clone())])
                    .unwrap_or_default();
                broadcast_queue(
                    &self.gateway,
                    &args.session_id.0,
                    &queued,
                    Some((&prompt_id, &text)),
                );
            }
            // The send RPC answers only at turn end — EXCEPT when the kernel
            // rejects it outright (-32010 during compaction, bad state). Watch
            // both channels concurrently: an early rejection fails this turn
            // explicitly (the old code dropped the receiver and parked until
            // an unrelated turn's completion event resolved it, losing the
            // prompt).
            let send_wait = async {
                match send_rx {
                    Some(mut rx) => match rx.await {
                        Ok(Err(rejection)) => Err(rejection),
                        _ => Ok(()),
                    },
                    None => Ok(()),
                }
            };
            tokio::select! {
                sent = send_wait => match sent {
                    Err(rejection) if rejection.contains("already running") => {
                        // An UNOWNED kernel turn is running (e.g. a just-
                        // drained continuation no ACP prompt awaits). Cancel
                        // it and retry our send, mirroring send-now.
                        let kernel_sid = state.kernel_id.borrow().clone();
                        let _ = kernel.request("session/stop", kernel::stop_params(&kernel_sid));
                        let retry_params = kernel::send_params_with_attachments(
                            &kernel_sid, &text, &attachments,
                        );
                        let mut accepted = false;
                        let deadline = std::time::Instant::now()
                            + std::time::Duration::from_secs(25);
                        while std::time::Instant::now() < deadline {
                            match kernel.call("session/send", retry_params.clone()).await {
                                Ok(_) => {
                                    accepted = true;
                                    break;
                                }
                                Err(e) if e.contains("already running") => {
                                    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
                                }
                                Err(_) => break,
                            }
                        }
                        if accepted {
                            rx.await.unwrap_or(acp::StopReason::EndTurn)
                        } else {
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
                    }
                    Err(rejection) => {
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
                    Ok(()) => rx.await.unwrap_or(acp::StopReason::EndTurn),
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
            let kernel_sid = state.kernel_id.borrow().clone();
            // Official semantics ride the v4 stop command: it pauses the
            // kernel queue's auto-drain (autoDrain=false) so queued items
            // wait for /drain on. Legacy session/stop only interrupts the
            // running turn and leaves the queue draining.
            match v4_command(&kernel, &kernel_sid, "stop", json!({}), None).await {
                Ok(ack) => debug_log(&format!("v4 stop ack: {ack}")),
                Err(_) => {
                    let _ = kernel.request("session/stop", kernel::stop_params(&kernel_sid));
                }
            }
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
            // Session-level followup delivery default (v4): queue or
            // guide — how mid-turn inputs are admitted by default.
            "x.ai/session/set_followup_mode" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params.get("sessionId").and_then(Value::as_str).unwrap_or_default().to_string();
                let mode = params.get("mode").and_then(Value::as_str).unwrap_or("guide").to_string();
                let acp_sid = acp::SessionId::new(id.clone());
                let state = self
                    .shared
                    .state
                    .borrow()
                    .sessions
                    .get(&acp_sid)
                    .cloned()
                    .ok_or_else(|| acp::Error::invalid_params().data("unknown session"))?;
                let kernel = self.kernel()?;
                let kernel_sid = state.kernel_id.borrow().clone();
                let cas = Some((
                    state.v4_revision.get(),
                    state.v4_log_epoch.borrow().clone().unwrap_or_else(|| "unknown".to_string()),
                ));
                let ack = v4_command(&kernel, &kernel_sid, "setFollowupMode", json!({"mode": mode}), cas)
                    .await
                    .map_err(|e| acp::Error::internal_error().data(format!("setFollowupMode failed: {e}")))?;
                let body = json!({"status": ack.get("status"), "mode": mode});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize followup ack");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Diagnostic: the agent's v4 conversation projection (rows,
            // CAS tokens) — used by probes and troubleshooting.
            "x.ai/v4/projection" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let refresh = params.get("refresh").and_then(Value::as_bool).unwrap_or(false);
                let id = params.get("sessionId").and_then(Value::as_str).unwrap_or_default().to_string();
                let acp_sid = acp::SessionId::new(id.clone());
                if refresh {
                    if let (Some(state), Some(kernel)) = (
                        self.shared.state.borrow().sessions.get(&acp_sid).cloned(),
                        self.shared.state.borrow().kernel.clone(),
                    ) {
                        let kernel_sid = state.kernel_id.borrow().clone();
                        let before = state.v4_snapshot_count.get();
                        v4_subscribe(&kernel, &kernel_sid);
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(3);
                        while state.v4_snapshot_count.get() <= before
                            && std::time::Instant::now() < deadline
                        {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                }
                let state = self
                    .shared
                    .state
                    .borrow()
                    .sessions
                    .get(&acp_sid)
                    .cloned()
                    .ok_or_else(|| acp::Error::invalid_params().data("unknown session"))?;
                let users: Vec<Value> = state
                    .v4_user_rows
                    .borrow()
                    .iter()
                    .map(|(rid, eid, tid, text)| json!({"rowId": rid, "entityId": eid, "turnId": tid, "text": text.chars().take(40).collect::<String>()}))
                    .collect();
                let assistants: Vec<Value> = state
                    .v4_assistant_rows
                    .borrow()
                    .iter()
                    .map(|(rid, eid, tid, st)| json!({"rowId": rid, "entityId": eid, "turnId": tid, "state": st}))
                    .collect();
                let body = json!({
                    "sessionId": id,
                    "revision": state.v4_revision.get(),
                    "logEpoch": state.v4_log_epoch.borrow().clone(),
                    "userRows": users,
                    "assistantRows": assistants,
                    "queueItems": state.v4_queue_items.borrow().len(),
                    "queueAutoDrain": state.v4_queue_autodrain.get(),
                    "workflowRuns": state.v4_workflow_runs.borrow().len(),
                    "turnHeaders": state.v4_turn_headers.borrow().iter().rev().take(5).map(|h| json!({
                        "rowId": h.get("rowId"),
                        "entityId": h.get("entityId"),
                        "state": h.get("state"),
                        "origin": h.get("origin"),
                        "fileChanges": h.get("fileChanges"),
                        "actions": h.get("actions"),
                    })).collect::<Vec<_>>(),
                    "usage": state.v4_usage.borrow().map(|(used, max)| json!({
                        "usedTokens": used,
                        "maxTokens": max,
                    })),
                });
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize projection");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Goal control: the kernel's session/goal RPC (show/pause/
            // resume/clear/replace/set). /goal text prompts already parse;
            // this is the structured control surface.
            "x.ai/session/goal" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params.get("sessionId").and_then(Value::as_str).unwrap_or_default().to_string();
                let action = params.get("action").and_then(Value::as_str).unwrap_or("show").to_string();
                if !id.starts_with("sess_") {
                    return Err(acp::Error::invalid_params().data("bad sessionId"));
                }
                let kernel = self.kernel()?;
                let mut payload = json!({"sessionId": id, "action": action});
                if let Some(objective) = params.get("objective").and_then(Value::as_str) {
                    payload["objective"] = json!(objective);
                }
                let result = kernel
                    .call("session/goal", payload)
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error().data(format!("session/goal failed: {e}"))
                    })?;
                let body = json!({
                    "response": result.get("response").and_then(Value::as_str).unwrap_or_default(),
                    "startedTurn": result.get("startedTurn").and_then(Value::as_bool).unwrap_or(false),
                });
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize goal ack");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // Kill one background work unit (bash tasks here) via the v4
            // cancelBackgroundWork command.
            "x.ai/task/kill" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params.get("sessionId").and_then(Value::as_str).unwrap_or_default().to_string();
                let task_id = params
                    .get("taskId")
                    .or_else(|| params.get("task_id"))
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
                let kernel = self.kernel()?;
                let kernel_sid = state.kernel_id.borrow().clone();
                let ack = v4_command(&kernel, &kernel_sid, "cancelBackgroundWork", json!({"workId": task_id}), None)
                    .await
                    .map_err(|e| acp::Error::internal_error().data(format!("cancelBackgroundWork failed: {e}")))?;
                let body = json!({"status": ack.get("status")});
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize kill ack");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // v4/conversation/usage query — the official session-token
            // meter (same result shape as the legacy session/usage word).
            "x.ai/v4/usage" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params.get("sessionId").and_then(Value::as_str).unwrap_or_default().to_string();
                let kernel = self.kernel()?;
                let result = kernel
                    .call("v4/conversation/usage", json!({"sessionId": id}))
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error().data(format!("v4 usage failed: {e}"))
                    })?;
                let raw = serde_json::value::to_raw_value(&result)
                    .expect("serialize v4 usage");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            // v4/command passthrough: the kernel ships a parallel v4
            // command surface (sendText with requestedDelivery
            // startNow/queue/guide, editUserQuery rewind, queue ops) on
            // the same wire. The legacy session/* methods remain the old
            // face; this exposes the v4 envelope verbatim.
            "x.ai/v4/command" => {
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let kernel = self.kernel()?;
                let result = kernel
                    .call("v4/command", params)
                    .await
                    .map_err(|e| {
                        acp::Error::internal_error().data(format!("v4/command failed: {e}"))
                    })?;
                let raw = serde_json::value::to_raw_value(&result)
                    .expect("serialize v4 ack");
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
                state.compact_by_ext.set(true);
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
                // Truncating rewind on the v4 surface. The pager contract:
                // truncate server-side to BEFORE the target prompt, do NOT
                // re-run anything (the pager truncates its own scrollback,
                // prefills the composer from promptText, the user resends).
                //
                // Mapping (kernel primitives): target N >= 1 forks at the
                // assistant row ending turn N-1 — the fork contains exactly
                // prompts 0..N-1 with their responses (verified: post-target
                // turns absent). target 0 starts a fresh empty session.
                // Both swap kernel_id behind the same ACP session id.
                let params: Value =
                    serde_json::from_str(args.params.get()).unwrap_or(json!({}));
                let id = params
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let target_index = params
                    .get("targetPromptIndex")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                if !id.starts_with("sess_") {
                    return Err(acp::Error::invalid_params().data("bad sessionId"));
                }
                let acp_sid = acp::SessionId::new(id.clone());
                let state = self
                    .shared
                    .state
                    .borrow()
                    .sessions
                    .get(&acp_sid)
                    .cloned()
                    .ok_or_else(|| acp::Error::invalid_params().data("unknown session"))?;
                let kernel = self.kernel()?;
                // Fresh projection (CAS tokens + rows).
                let kernel_sid = state.kernel_id.borrow().clone();
                {
                    let before = state.v4_snapshot_count.get();
                    v4_subscribe(&kernel, &kernel_sid);
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                    while state.v4_snapshot_count.get() <= before
                        && std::time::Instant::now() < deadline
                    {
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    }
                }
                let user_rows = state.v4_user_rows.borrow().clone();
                let Some((_row_id, _entity, _turn, target_text)) = user_rows.get(target_index).cloned() else {
                    let body = json!({
                        "success": false,
                        "targetPromptIndex": target_index,
                        "mode": "conversation_only",
                        "error": format!("rewind target {} not found ({} user rows)", target_index, user_rows.len()),
                    });
                    let raw = serde_json::value::to_raw_value(&body)
                        .expect("serialize rewind miss");
                    return Ok(acp::ExtResponse::new(raw.into()));
                };
                let new_kernel_sid: String = if target_index == 0 {
                    // Rewind to the very start: a fresh empty session in the
                    // same workspace.
                    let cwd = state.cwd.borrow().clone();
                    let created = kernel
                        .call("session/create", kernel::create_params(std::path::Path::new(&cwd)))
                        .await
                        .map_err(|e| acp::Error::internal_error().data(format!("rewind fresh create failed: {e}")))?;
                    let fresh_sid = kernel::session_id_from(&created).unwrap_or_default();
                    kernel
                        .call("session/subscribe", kernel::subscribe_params(&fresh_sid))
                        .await
                        .map_err(|e| acp::Error::internal_error().data(format!("rewind fresh subscribe failed: {e}")))?;
                    fresh_sid
                } else {
                    // Fork at the last assistant row of turn N-1.
                    let prev_turn = user_rows.get(target_index - 1).map(|(_, _, tid, _)| tid.clone()).unwrap_or_default();
                    let fork_target = state
                        .v4_assistant_rows
                        .borrow()
                        .iter()
                        .filter(|(_, _, tid, _)| *tid == prev_turn)
                        .next_back()
                        .map(|(rid, eid, _, _)| (*rid, eid.clone()));
                    let Some((row_id, entity_id)) = fork_target else {
                        let body = json!({
                            "success": false,
                            "targetPromptIndex": target_index,
                            "mode": "conversation_only",
                            "error": "no assistant row before the rewind target",
                        });
                        let raw = serde_json::value::to_raw_value(&body)
                            .expect("serialize rewind miss");
                        return Ok(acp::ExtResponse::new(raw.into()));
                    };
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0);
                    let envelope = json!({
                        "commandId": format!("zgrok-rewind-{now_ms}"),
                        "clientId": "zgrok",
                        "sessionId": kernel_sid,
                        "baseRevision": state.v4_revision.get(),
                        "baseLogEpoch": state.v4_log_epoch.borrow().clone().unwrap_or_else(|| "unknown".to_string()),
                        "type": "forkAssistant",
                        "payload": {"target": {"rowId": row_id, "entityId": entity_id}},
                        "issuedAt": now_ms,
                    });
                    let ack = kernel
                        .call("v4/command", envelope)
                        .await
                        .map_err(|e| acp::Error::internal_error().data(format!("forkAssistant failed: {e}")))?;
                    if ack.get("status").and_then(Value::as_str) != Some("accepted") {
                        let body = json!({
                            "success": false,
                            "targetPromptIndex": target_index,
                            "mode": "conversation_only",
                            "error": format!("forkAssistant {}: {}", ack.get("status").and_then(Value::as_str).unwrap_or("?"), ack.get("reasonCode").and_then(Value::as_str).unwrap_or("?")),
                        });
                        let raw = serde_json::value::to_raw_value(&body)
                            .expect("serialize rewind miss");
                        return Ok(acp::ExtResponse::new(raw.into()));
                    }
                    let fork_sid = ack
                        .pointer("/result/sessionId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    // The fork needs its own legacy event subscription too
                    // (turn events ride session/event per session).
                    if fork_sid.starts_with("sess_") {
                        let _ = kernel
                            .call("session/subscribe", kernel::subscribe_params(&fork_sid))
                            .await;
                    }
                    fork_sid
                };
                if !new_kernel_sid.starts_with("sess_") {
                    return Err(acp::Error::internal_error().data("rewind swap got no session id"));
                }
                // Swap the kernel session behind the same ACP id and re-arm
                // the v4 projection on the new topic.
                *state.kernel_id.borrow_mut() = new_kernel_sid.clone();
                *state.v4_log_epoch.borrow_mut() = None;
                *state.v4_user_rows.borrow_mut() = Vec::new();
                *state.v4_assistant_rows.borrow_mut() = Vec::new();
                *state.v4_usage.borrow_mut() = None;
                state.v4_revision.set(0);
                v4_subscribe(&kernel, &new_kernel_sid);
                write_summary_stub(&new_kernel_sid, &state.cwd.borrow());
                debug_log(&format!("rewind execute idx={target_index} -> {new_kernel_sid}"));
                let body = json!({
                    "success": true,
                    "targetPromptIndex": target_index,
                    "mode": "conversation_only",
                    "promptText": target_text,
                });
                let raw = serde_json::value::to_raw_value(&body)
                    .expect("serialize rewind result");
                return Ok(acp::ExtResponse::new(raw.into()));
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
                let in_flight = state.turn_done.borrow().is_some() || state.compacting.get();
                let interjection_id = params
                    .get("interjectionId")
                    .and_then(Value::as_str)
                    .unwrap_or("interjection")
                    .to_string();
                if in_flight {
                    // Preferred: v4 sendText(guide) — the kernel injects the
                    // interjection INTO the running turn at its next safe
                    // point (official steering). Falls back to the
                    // turn-boundary continuation queue on kernels without v4.
                    let kernel = self.kernel()?;
                    let steered = match v4_send_text(&kernel, &id, &text, "guide").await {
                        Ok(ack) if ack.get("status") == Some(&json!("accepted")) => true,
                        other => {
                            debug_log(&format!(
                                "interject: v4 guide unavailable ({:?}), queueing at turn boundary",
                                other.as_ref().map(|a| a.get("status").cloned()).map_err(|e| e.clone())
                            ));
                            false
                        }
                    };
                    if steered {
                        debug_log("interject: steered into running turn via v4 guide");
                        broadcast_interjection(&self.gateway, &session.0, &text, Some(&interjection_id));
                        let body = json!({"sessionId": id, "accepted": true, "steered": true});
                        let raw = serde_json::value::to_raw_value(&body)
                            .expect("serialize interject ack");
                        return Ok(acp::ExtResponse::new(raw.into()));
                    }
                    let mut pending = state.pending_continuation.borrow_mut();
                    match pending.as_mut() {
                        Some(existing) => existing.push_str(&format!("\n\n{text}")),
                        None => *pending = Some(text.clone()),
                    }
                    state
                        .pending_interjection_ids
                        .borrow_mut()
                        .push(interjection_id.clone());
                    debug_log("interject: queued as next-turn continuation");
                    // Server-queue snapshot: the interjection rides as a
                    // queued row until the turn ends and delivers it.
                    broadcast_queue(
                        &self.gateway,
                        &session.0,
                        &[(interjection_id, "prompt", text)],
                        None,
                    );
                } else {
                    // Idle: send now. Fire-and-forget — the send RPC's Ok
                    // response arrives only at turn end, so awaiting it under
                    // the 30s call timeout would spuriously fail while the
                    // turn actually runs.
                    let kernel = self.kernel()?;
                    kernel
                        .request("session/send", kernel::send_params(&id, &text))
                        .map_err(|e| {
                            acp::Error::internal_error().data(format!("interject send failed: {e}"))
                        })?;
                    debug_log("interject: sent immediately (no turn in flight)");
                    broadcast_interjection(
                        &self.gateway,
                        &session.0,
                        &text,
                        params.get("interjectionId").and_then(Value::as_str),
                    );
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
                // Official path first: v4 deleteSession (kernel-side
                // cleanup incl. artifacts); fall back to the db surgery
                // on kernels without v4.
                let mut deleted_via_v4 = false;
                {
                    let kernel = self.kernel()?;
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0);
                    if let Ok(ack) = v4_command(&kernel, &id, "deleteSession", json!({}), None).await {
                        deleted_via_v4 =
                            ack.get("status").and_then(Value::as_str) == Some("accepted");
                    }
                }
                if !deleted_via_v4 {
                    delete_kernel_session(id, cwd).map_err(|e| {
                        acp::Error::internal_error().data(format!("session delete failed: {e}"))
                    })?;
                }
                debug_log(&format!("session deleted: {id}"));
                let body = json!({"result": {"ok": true, "sessionId": id}});
                let raw = serde_json::value::to_raw_value(&body).expect("serialize delete ack");
                return Ok(acp::ExtResponse::new(raw.into()));
            }
            _ => {}
        }
        Err(acp::Error::method_not_found())
    }

    /// Fire-and-forget pager notifications. The queue pane's edit
    /// operations ride x.ai/queue/* notifications (no response expected)
    /// and map 1:1 onto the kernel's v4 queue commands.
    async fn ext_notification(&self, args: acp::ExtNotification) -> acp::Result<()> {
        let params: Value = serde_json::from_str(args.params.get()).unwrap_or(json!({}));
        let Some(session_id) = params.get("sessionId").and_then(Value::as_str).map(str::to_string) else {
            return Ok(());
        };
        let acp_sid = acp::SessionId::new(session_id.clone());
        let Some(state) = self.shared.state.borrow().sessions.get(&acp_sid).cloned() else {
            return Ok(());
        };
        let Some(kernel) = self.shared.state.borrow().kernel.clone() else {
            return Ok(());
        };
        let kernel_sid = state.kernel_id.borrow().clone();
        match args.method.as_ref() {
            "x.ai/queue/remove" => {
                let Some(id) = params.get("id").and_then(Value::as_str) else { return Ok(()) };
                let _ = v4_command(&kernel, &kernel_sid, "deleteQueueItem", json!({"queueItemId": id}), None).await;
            }
            "x.ai/queue/edit" => {
                let (Some(id), Some(text)) = (
                    params.get("id").and_then(Value::as_str),
                    params.get("newText").and_then(Value::as_str),
                ) else {
                    return Ok(());
                };
                let _ = v4_command(&kernel, &kernel_sid, "editQueueItem", json!({"queueItemId": id, "newText": text}), None).await;
            }
            "x.ai/queue/reorder" => {
                let Some(ordered) = params.get("orderedIds").and_then(Value::as_array) else { return Ok(()) };
                // Rebuild the order by moving each item to the end in the
                // desired sequence (beforeQueueItemId=null → tail).
                for id in ordered {
                    let Some(id) = id.as_str() else { continue };
                    let _ = v4_command(&kernel, &kernel_sid, "reorderQueueItem", json!({"queueItemId": id, "beforeQueueItemId": null}), None).await;
                }
            }
            "x.ai/queue/clear" => {
                let ids: Vec<String> = state
                    .v4_queue_items
                    .borrow()
                    .iter()
                    .map(|(id, _)| id.clone())
                    .collect();
                for id in ids {
                    let _ = v4_command(&kernel, &kernel_sid, "deleteQueueItem", json!({"queueItemId": id}), None).await;
                }
            }
            "x.ai/queue/interject" => {
                // Insert a new queued prompt (after-positioning degrades to
                // append; the kernel queue delivers when the turn ends).
                let text = params
                    .get("newText")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if !text.trim().is_empty() {
                    let held = held_queue_disposition(&state);
                    let _ = v4_send_text_held(&kernel, &kernel_sid, &text, "queue", held.as_ref().map(|h| (h.0, h.1.clone()))).await;
                }
            }
            // Local edit locks (hold/release) have no kernel counterpart.
            _ => {}
        }
        Ok(())
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
        .or_else(|| Some(std::path::PathBuf::from(zcode_home())))
        .map(|base| base.join("zcode-tui").join(name))
}

fn notify(gateway: &AcpGatewaySender<acp::AgentSide>, session: &acp::SessionId, update: acp::SessionUpdate) {
    gateway.forward_fire_and_forget(acp::SessionNotification::new(session.clone(), update));
}

/// Broadcast an `x.ai/queue/changed` snapshot (camelCase QueueChanged wire).
/// The pager replaces its server-queue pane wholesale with `entries` and —
/// critically — its local prompt queue DRAIN is blocked while any
/// non-running server row exists (`server_queue_owns_next_turn`), so every
/// queue transition must reconcile to the truth: queued continuations as
/// rows, delivery as an empty snapshot.
fn broadcast_queue(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    session_id: &str,
    entries: &[(String, &str, String)],
    running: Option<(&str, &str)>,
) {
    let entries: Vec<Value> = entries
        .iter()
        .enumerate()
        .map(|(position, (id, kind, text))| {
            json!({
                "id": id,
                "version": 0,
                "kind": kind,
                "text": text,
                "position": position,
            })
        })
        .collect();
    let mut payload = json!({"sessionId": session_id, "entries": entries});
    if let Some((running_id, running_text)) = running {
        payload["runningPromptId"] = json!(running_id);
        payload["runningText"] = json!(running_text);
        payload["runningKind"] = json!("prompt");
    }
    if let Ok(params) = serde_json::value::to_raw_value(&payload) {
        gateway.forward_fire_and_forget(acp::ExtNotification::new(
            "x.ai/queue/changed",
            params.into(),
        ));
    }
}

/// Submit one v4/command envelope. Returns the ack object.
async fn v4_command(
    kernel: &Kernel,
    session_id: &str,
    command_type: &str,
    payload: Value,
    cas: Option<(u64, String)>,
) -> Result<Value, String> {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    static COMMAND_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let seq = COMMAND_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut envelope = json!({
        "commandId": format!("zgrok-{now_ms}-{seq}"),
        "clientId": "zgrok",
        "sessionId": session_id,
        "type": command_type,
        "payload": payload,
        "issuedAt": now_ms,
    });
    if let Some((revision, log_epoch)) = cas {
        envelope["baseRevision"] = json!(revision);
        envelope["baseLogEpoch"] = json!(log_epoch);
    }
    kernel.call("v4/command", envelope).await
}

/// Build a v4/command sendText envelope and submit it. The kernel's v4
/// surface accepts requestedDelivery startNow (atomic preempt of the
/// running turn), queue (deliver after the turn) and guide (inject into
/// the running turn at its next safe point) — the real steering the
/// legacy session/send refuses with -32010. Returns the ack object.
/// The safe held-queue disposition for a send while the kernel queue is
/// held (items present + autoDrain false): `keepQueueAndSend` plus the
/// expected item ids — the official choice-mode contract, with the
/// non-destructive option as the TUI default (`/drain on` resumes draining).
fn held_queue_disposition(state: &Rc<SessionState>) -> Option<(&'static str, Vec<String>)> {
    let items = state.v4_queue_items.borrow().clone();
    if !items.is_empty() && !state.v4_queue_autodrain.get() {
        Some((
            "keepQueueAndSend",
            items.iter().map(|(id, _)| id.clone()).collect(),
        ))
    } else {
        None
    }
}

async fn v4_send_text(
    kernel: &Kernel,
    session_id: &str,
    text: &str,
    requested_delivery: &str,
) -> Result<Value, String> {
    v4_send_text_held(kernel, session_id, text, requested_delivery, None).await
}

async fn v4_send_text_held(
    kernel: &Kernel,
    session_id: &str,
    text: &str,
    requested_delivery: &str,
    held: Option<(&str, Vec<String>)>,
) -> Result<Value, String> {
    let mut payload = json!({
        "text": text,
        "requestedDelivery": requested_delivery,
    });
    if let Some((disposition, ids)) = held {
        payload["heldQueueDisposition"] = json!(disposition);
        payload["expectedHeldQueueItemIds"] = json!(ids);
    }
    v4_command(kernel, session_id, "sendText", payload, None).await
}

/// Fold one v4 conversation frame (snapshot or deltas) into the session's
/// projection state. Snapshots are authoritative; deltas opportunistically
/// refresh the CAS tokens and queue — anything unparsed keeps the last
/// snapshot's values (rewind re-subscribes for freshness anyway).
/// Parse the v4 sessionUsageState's contextWindow from a snapshot body or a
/// state.updated patch. Returns Some(None) when the field is present but the
/// window is null (no model call yet), None when absent (keep prior value).
fn parse_v4_usage(parent: &Value) -> Option<Option<(u64, u64)>> {
    let window = parent.pointer("/usage/contextWindow")?;
    if window.is_null() {
        return Some(None);
    }
    let used = window.get("usedTokens").and_then(Value::as_u64)?;
    let max = window.get("maxTokens").and_then(Value::as_u64)?;
    Some(Some((used, max)))
}

fn update_v4_projection(state: &Rc<SessionState>, frame: &Value) {
    let payload = frame.pointer("/payload").unwrap_or(&Value::Null);
    match payload.get("kind").and_then(Value::as_str) {
        Some("snapshot") => {
            let Some(snapshot) = payload.get("snapshot") else { return };
            if let Some(rev) = snapshot.get("revision").and_then(Value::as_u64) {
                state.v4_revision.set(rev);
            }
            if let Some(epoch) = snapshot.get("logEpoch").and_then(Value::as_str) {
                *state.v4_log_epoch.borrow_mut() = Some(epoch.to_string());
            }
            state.v4_snapshot_count.set(state.v4_snapshot_count.get() + 1);
            let mut rows = Vec::new();
            if let Some(arr) = snapshot.pointer("/rows/window").and_then(Value::as_array) {
                for row in arr {
                    if row.get("kind").and_then(Value::as_str) != Some("userInput") {
                        continue;
                    }
                    if row.get("origin").and_then(Value::as_str) != Some("realUser") {
                        continue;
                    }
                    let Some(row_id) = row.get("rowId").and_then(Value::as_i64) else { continue };
                    let entity_id = row
                        .get("entityId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let turn_id = row.get("turnId").and_then(Value::as_str).unwrap_or_default().to_string();
                    let text = row.get("text").and_then(Value::as_str).unwrap_or_default().to_string();
                    rows.push((row_id, entity_id, turn_id, text));
                }
            }
            *state.v4_user_rows.borrow_mut() = rows;
            let mut assistant_rows = Vec::new();
            if let Some(arr) = snapshot.pointer("/rows/window").and_then(Value::as_array) {
                for row in arr {
                    if row.get("kind").and_then(Value::as_str) != Some("assistantText") {
                        continue;
                    }
                    let Some(row_id) = row.get("rowId").and_then(Value::as_i64) else { continue };
                    let entity_id = row.get("entityId").and_then(Value::as_str).unwrap_or_default().to_string();
                    let turn_id = row.get("turnId").and_then(Value::as_str).unwrap_or_default().to_string();
                    let st = row.get("state").and_then(Value::as_str).unwrap_or("complete").to_string();
                    assistant_rows.push((row_id, entity_id, turn_id, st));
                }
            }
            *state.v4_assistant_rows.borrow_mut() = assistant_rows;
            let mut headers = Vec::new();
            if let Some(arr) = snapshot.pointer("/rows/window").and_then(Value::as_array) {
                for row in arr {
                    if row.get("kind").and_then(Value::as_str) == Some("turnHeader") {
                        headers.push(row.clone());
                    }
                }
            }
            *state.v4_turn_headers.borrow_mut() = headers;
            *state.v4_workflow_runs.borrow_mut() = snapshot
                .pointer("/workflowRuns/runs")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut items = Vec::new();
            if let Some(arr) = snapshot.pointer("/queue/items").and_then(Value::as_array) {
                for item in arr {
                    let id = item
                        .get("queueItemId")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let text = item
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    if !id.is_empty() {
                        items.push((id, text));
                    }
                }
            }
            *state.v4_queue_items.borrow_mut() = items;
            if let Some(ad) = snapshot.pointer("/queue/autoDrain").and_then(Value::as_bool) {
                state.v4_queue_autodrain.set(ad);
            }
            *state.v4_goal.borrow_mut() = snapshot.get("goal").cloned().filter(|g| !g.is_null());
            *state.v4_background_works.borrow_mut() = snapshot
                .get("backgroundWorks")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if let Some(usage) = parse_v4_usage(snapshot) {
                *state.v4_usage.borrow_mut() = usage;
            }
        }
        Some("deltas") => {
            let Some(deltas) = payload.get("deltas").and_then(Value::as_array) else {
                return;
            };
            for delta in deltas {
                let op = delta.get("op").and_then(Value::as_str).unwrap_or_default();
                match op {
                    "row.appended" | "row.upserted" => {
                        let Some(row) = delta.get("row") else { continue };
                        let Some(row_id) = row.get("rowId").and_then(Value::as_i64) else { continue };
                        let entity_id = row
                            .get("entityId")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        match row.get("kind").and_then(Value::as_str) {
                            Some("userInput") if row.get("origin").and_then(Value::as_str) == Some("realUser") => {
                                let turn_id = row.get("turnId").and_then(Value::as_str).unwrap_or_default().to_string();
                                let text = row
                                    .get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string();
                                let mut rows = state.v4_user_rows.borrow_mut();
                                rows.retain(|(rid, _, _, _)| *rid != row_id);
                                let pos = rows.iter().position(|(rid, _, _, _)| *rid > row_id).unwrap_or(rows.len());
                                rows.insert(pos, (row_id, entity_id, turn_id, text));
                            }
                            Some("assistantText") => {
                                let turn_id = row.get("turnId").and_then(Value::as_str).unwrap_or_default().to_string();
                                let st = row.get("state").and_then(Value::as_str).unwrap_or("complete").to_string();
                                let mut rows = state.v4_assistant_rows.borrow_mut();
                                rows.retain(|(rid, _, _, _)| *rid != row_id);
                                let pos = rows.iter().position(|(rid, _, _, _)| *rid > row_id).unwrap_or(rows.len());
                                rows.insert(pos, (row_id, entity_id, turn_id, st));
                            }
                            Some("turnHeader") => {
                                let mut headers = state.v4_turn_headers.borrow_mut();
                                headers.retain(|h| h.get("rowId").and_then(Value::as_i64) != Some(row_id));
                                headers.push(row.clone());
                            }
                            _ => {}
                        }
                    }
                    // Truncation: the branch cut removes this row and every
                    // later one (edit/retry rewinds).
                    "row.removed" => {
                        if let Some(from) = delta.get("fromRowId").and_then(Value::as_i64) {
                            state.v4_user_rows.borrow_mut().retain(|(rid, _, _, _)| *rid < from);
                            state.v4_assistant_rows.borrow_mut().retain(|(rid, _, _, _)| *rid < from);
                        }
                    }
                    "state.updated" => {
                        let Some(patch) = delta.get("patch") else { continue };
                        if let Some(rev) = patch.get("revision").and_then(Value::as_u64) {
                            state.v4_revision.set(rev);
                        }
                        if let Some(items) = patch.pointer("/queue/items").and_then(Value::as_array) {
                            let mut queue = Vec::new();
                            for item in items {
                                let id = item
                                    .get("queueItemId")
                                    .or_else(|| item.get("id"))
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string();
                                let text = item.get("text").and_then(Value::as_str).unwrap_or_default().to_string();
                                if !id.is_empty() {
                                    queue.push((id, text));
                                }
                            }
                            *state.v4_queue_items.borrow_mut() = queue;
                        }
                        if let Some(ad) = patch.pointer("/queue/autoDrain").and_then(Value::as_bool) {
                            state.v4_queue_autodrain.set(ad);
                        }
                        if let Some(runs) = patch.pointer("/workflowRuns/runs").and_then(Value::as_array) {
                            *state.v4_workflow_runs.borrow_mut() = runs.clone();
                        }
                        if let Some(goal) = patch.get("goal") {
                            *state.v4_goal.borrow_mut() =
                                goal.as_object().map(|_| goal.clone()).or_else(|| {
                                    if goal.is_null() { None } else { Some(goal.clone()) }
                                });
                        }
                        if let Some(works) = patch.get("backgroundWorks").and_then(Value::as_array) {
                            *state.v4_background_works.borrow_mut() = works.clone();
                        }
                        if let Some(usage) = parse_v4_usage(patch) {
                            *state.v4_usage.borrow_mut() = usage;
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// dwf run display facts from the ledger: run name + per-actor
/// (resolved_model, persona summary) keyed by actor siteId.
fn dwf_run_facts(run_id: &str) -> (Option<String>, std::collections::HashMap<String, (Option<String>, Option<String>)>) {
    let home = zcode_home();
    let Ok(con) = rusqlite::Connection::open_with_flags(
        std::path::Path::new(&home).join(".zcode/cli/db/db.sqlite"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) else {
        return (None, Default::default());
    };
    let name = con
        .query_row(
            "SELECT name FROM dwf_run WHERE id = ?1",
            [run_id],
            |r| r.get::<_, Option<String>>(0),
        )
        .ok()
        .flatten();
    let mut map = std::collections::HashMap::new();
    if let Ok(mut stmt) = con.prepare(
        "SELECT site_id, resolved_model, persona_json FROM dwf_actor WHERE run_id = ?1",
    ) {
        let rows = stmt
            .query_map([run_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                ))
            })
            .map(|rows| rows.filter_map(Result::ok).collect::<Vec<_>>())
            .unwrap_or_default();
        for (site, model, persona_json) in rows {
            let persona = persona_json
                .as_deref()
                .and_then(|j| serde_json::from_str::<Value>(j).ok());
            let summary = persona.as_ref().and_then(|p| {
                p.get("system")
                    .and_then(Value::as_str)
                    .or_else(|| p.get("name").and_then(Value::as_str))
                    .map(|t| t.chars().take(80).collect::<String>())
            });
            map.insert(
                site,
                (
                    model.map(|m| {
                        // strip the provider prefix: account:.../GLM-5.3 -> GLM-5.3
                        m.rsplit('/').next().unwrap_or(&m).to_string()
                    }),
                    summary,
                ),
            );
        }
    }
    (name, map)
}

/// Translate the kernel's v4 workflow runs into the pager's
/// workflow_updated notifications (one per run on every change; the pager
/// upserts by run_id).
fn emit_workflow_updates(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    session_id: &str,
    runs: &[Value],
) {
    for run in runs {
        let run_id = run.get("runId").and_then(Value::as_str).unwrap_or_default();
        if run_id.is_empty() {
            continue;
        }
        let (run_name, actor_facts) = dwf_run_facts(run_id);
        let kernel_status = run.get("status").and_then(Value::as_str).unwrap_or("running");
        let status = match kernel_status {
            "pending" | "running" => "active",
            "completed" => "complete",
            "errored" => "failed",
            "stopped" => match run.get("stopReason").and_then(Value::as_str) {
                Some("user") => "cancelled",
                _ => "interrupted",
            },
            _ => "active",
        };
        let nodes_used = run
            .pointer("/usage/nodesUsed")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        // Actors -> the pager's agents list; nodes -> phase progress.
        let actors = run.get("actors").and_then(Value::as_array);
        let nodes = run.get("nodes").and_then(Value::as_array);
        let mut subagent_events: Vec<(String, String, Option<String>, Option<String>, String)> =
            Vec::new(); // (child_sid, name, model, persona, state)
        let agents: Vec<Value> = actors
            .map(|list| {
                list.iter()
                    .map(|a| {
                        let site = a.get("siteId").and_then(Value::as_str).unwrap_or("?");
                        let label = a
                            .get("name")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or_else(|| {
                                format!("actor-{}", a.get("ordinal").and_then(Value::as_u64).unwrap_or(0))
                            });
                        let state =
                            a.get("status").and_then(Value::as_str).unwrap_or("waiting").to_string();
                        let (model, description) = actor_facts
                            .get(site)
                            .cloned()
                            .unwrap_or((None, None));
                        if let Some(child) = a.get("sessionId").and_then(Value::as_str) {
                            subagent_events.push((
                                child.to_string(),
                                label.clone(),
                                model.clone(),
                                description.clone(),
                                state.clone(),
                            ));
                        }
                        json!({
                            "agent_id": site,
                            "label": label,
                            "phase": a.get("phaseName"),
                            "state": state,
                            "model": model,
                            "description": description,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let active_agents = actors
            .map(|list| {
                list.iter()
                    .filter(|a| a.get("status").and_then(Value::as_str) == Some("running"))
                    .count() as u32
            })
            .unwrap_or(0);
        // Phase groups by script phaseName (default "main"): running while
        // any node is pre-settled, failed if any failed, else done.
        let mut phase_order: Vec<String> = Vec::new();
        let mut phase_map: std::collections::HashMap<String, (u64, u64, u64)> =
            std::collections::HashMap::new(); // (active, settled, failed)
        if let Some(list) = nodes {
            for n in list {
                let name = n
                    .get("phaseName")
                    .and_then(Value::as_str)
                    .unwrap_or("main")
                    .to_string();
                let entry = phase_map.entry(name.clone()).or_insert((0, 0, 0));
                match n.get("phase").and_then(Value::as_str) {
                    Some("settled") => {
                        if n.get("outcome").and_then(Value::as_str) == Some("failed") {
                            entry.2 += 1;
                        } else {
                            entry.1 += 1;
                        }
                    }
                    _ => entry.0 += 1,
                }
                if !phase_order.contains(&name) {
                    phase_order.push(name);
                }
            }
        }
        let phases: Vec<Value> = phase_order
            .iter()
            .map(|name| {
                let (active, settled, failed) = phase_map[name];
                json!({
                    "title": name,
                    "state": if failed > 0 && active == 0 { "failed" }
                        else if active > 0 { "running" }
                        else { "complete" },
                    "detail": format!("{settled} settled"),
                })
            })
            .collect();
        let current_phase = phase_order.last().cloned();
        let executing = phase_map.values().map(|(a, _, _)| *a).sum::<u64>();
        let settled = phase_map.values().map(|(_, s, _)| *s).sum::<u64>();
        let failed = phase_map.values().map(|(_, _, f)| *f).sum::<u64>();
        let display_name = run_name.clone().unwrap_or_else(|| "Workflow".to_string());
        let update = json!({
            "sessionUpdate": "workflow_updated",
            "run_id": run_id,
            "name": display_name,
            "objective": run.get("resultPreview").and_then(Value::as_str).unwrap_or(""),
            "status": status,
            "revision": 0,
            "phases": phases,
            "current_phase": current_phase,
            "agents": agents,
            "agents_used": nodes_used,
            "agents_reserved": 0,
            "elapsed_ms": 0,
            "active_agents": active_agents,
            "last_event": format!("nodes: {executing} running / {settled} settled / {failed} failed"),
            "result_summary": run.get("resultPreview"),
            "pause_message": run.get("error"),
        });
        let payload = json!({"sessionId": session_id, "update": update});
        if let Ok(raw) = serde_json::value::to_raw_value(&payload) {
            gateway.forward_fire_and_forget(acp::ExtNotification::new(
                "x.ai/session_notification",
                raw.into(),
            ));
        }
        // Surface actors on the pager's subagent rail (roster + dashboard +
        // live map linkage via workflow_run_id). Spawn on first sight of the
        // child session; finish when the actor settles.
        for (child, label, model, persona, state) in subagent_events {
            let event = if state == "completed" {
                json!({
                    "sessionUpdate": "subagent_finished",
                    "subagent_id": child,
                    "status": "completed",
                    "description": label,
                })
            } else {
                json!({
                    "sessionUpdate": "subagent_spawned",
                    "subagent_id": child,
                    "child_session_id": child,
                    "parent_session_id": session_id,
                    "subagent_type": "workflow-actor",
                    "description": label,
                    "persona": persona,
                    "model": model,
                    "workflow_run_id": run_id,
                })
            };
            let payload = json!({"sessionId": session_id, "update": event});
            if let Ok(raw) = serde_json::value::to_raw_value(&payload) {
                gateway.forward_fire_and_forget(acp::ExtNotification::new(
                    "x.ai/session_notification",
                    raw.into(),
                ));
            }
        }
    }
}

/// Translate the kernel's goal state (v4 projection) into the pager's
/// goal_updated notification (grok goal panel). Null/absent → cleared.
fn emit_goal_updated(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    session_id: &str,
    goal: Option<&Value>,
    token_baseline: u64,
    tokens_used: u64,
) {
    let update = match goal {
        Some(g) => {
            let objective = g.get("objective").and_then(Value::as_str).unwrap_or("");
            let kernel_status = g.get("status").and_then(Value::as_str).unwrap_or("active");
            // Last verification entry: outcome/reason/nextAction drive the
            // verdict chip, the Reason block (paused goals) and the status
            // line's event detail.
            let last_verification = g
                .get("verifications")
                .and_then(Value::as_array)
                .and_then(|v| v.last());
            let verify_reason = last_verification
                .and_then(|v| v.get("reason"))
                .and_then(Value::as_str);
            let verify_next = last_verification
                .and_then(|v| v.get("nextAction"))
                .and_then(Value::as_str);
            let verify_outcome = last_verification
                .and_then(|v| v.get("outcome"))
                .and_then(Value::as_str);
            let (status, pause_message) = match kernel_status {
                "paused" => ("user_paused", None),
                "verified" => ("complete", None),
                "notSatisfied" => (
                    "blocked",
                    Some(verify_reason.unwrap_or("验证结论：目标未达成").to_string()),
                ),
                "failed" => ("blocked", Some("验证过程失败".to_string())),
                _ => ("active", None),
            };
            let phase = if status == "active" { "executing" } else { "idle" };
            let total_verify_rounds = g
                .get("verifications")
                .and_then(Value::as_array)
                .map(|v| v.len() as u32)
                .unwrap_or(0);
            let total_worker_rounds = g
                .get("iterations")
                .and_then(Value::as_array)
                .map(|v| v.len() as u32)
                .unwrap_or(0);
            json!({
                "sessionUpdate": "goal_updated",
                "goal_id": g.get("targetId").and_then(Value::as_str).unwrap_or("kernel-goal"),
                "objective": objective,
                "status": status,
                "phase": phase,
                "elapsed_ms": g.get("timeUsedSeconds").and_then(Value::as_u64).unwrap_or(0) * 1000,
                // Live line = current context - baseline (pager-side);
                // tokens_used only carries the frozen delta on terminal
                // states so a zero never reads as "no data" mid-run. The
                // wire field is a plain i64 — null would fail parsing and
                // drop the whole notification.
                "token_baseline": token_baseline,
                "tokens_used": tokens_used,
                "total_deliverables": 0,
                "completed_deliverables": 0,
                "total_worker_rounds": total_worker_rounds,
                "total_verify_rounds": total_verify_rounds,
                // The pager's dedicated "verifying" overlay (kernel status
                // verifying) and verdict chip (last verification outcome).
                "verifying_completion": if kernel_status == "verifying" { Some(true) } else { None },
                "last_classifier_verdict": match verify_outcome {
                    Some("pass") => Some("achieved"),
                    Some(_) => Some("not_achieved"),
                    None => None,
                },
                "pause_message": pause_message,
                "last_event": verify_outcome.map(|o| format!("verification: {o}")),
                "last_event_detail": verify_next.map(|s| s.to_string()),
            })
        }
        None => json!({
            "sessionUpdate": "goal_updated",
            "goal_id": "kernel-goal",
            "objective": "",
            "status": "cleared",
            "phase": "idle",
            "elapsed_ms": 0,
            "total_deliverables": 0,
            "completed_deliverables": 0,
            "total_worker_rounds": 0,
            "total_verify_rounds": 0,
        }),
    };
    let payload = json!({"sessionId": session_id, "update": update});
    if let Ok(raw) = serde_json::value::to_raw_value(&payload) {
        gateway.forward_fire_and_forget(acp::ExtNotification::new(
            "x.ai/session_notification",
            raw.into(),
        ));
    }
}

/// Translate the kernel's background works (v4 projection) into the
/// pager's background_tasks snapshot. Only bash-kind works are listed as
/// tasks — subagents already ride their own lifecycle notifications.
fn emit_background_tasks(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    session_id: &str,
    works: &[Value],
) {
    let tasks: Vec<Value> = works
        .iter()
        .filter(|w| w.get("kind").and_then(Value::as_str) == Some("bash"))
        .map(|w| {
            let kernel_status = w.get("status").and_then(Value::as_str).unwrap_or("running");
            let status = match kernel_status {
                "failed" => "failed",
                "cancelled" | "resultPending" => "completed",
                _ => "running",
            };
            json!({
                "task_id": w.get("workId").and_then(Value::as_str).unwrap_or_default(),
                "command": w.get("title").and_then(Value::as_str).unwrap_or_default(),
                "cwd": "",
                "kind": "bash",
                "status": status,
                "started_at": w.get("startedAt").and_then(Value::as_u64).unwrap_or(0),
            })
        })
        .collect();
    let payload = json!({
        "sessionId": session_id,
        "update": {"sessionUpdate": "background_tasks", "tasks": tasks, "truncated": false},
    });
    if let Ok(raw) = serde_json::value::to_raw_value(&payload) {
        gateway.forward_fire_and_forget(acp::ExtNotification::new(
            "x.ai/session_notification",
            raw.into(),
        ));
    }
}

/// (Re)subscribe the v4 conversation topic for a session; the snapshot
/// frame lands asynchronously and refreshes the projection state.
/// Open the dynamic-workflow gate. The kernel ships it FAIL-CLOSED: the
/// official host reads its rollout config and calls
/// workspace/updateDynamicWorkflowPolicy; without this call sessions get
/// no CreateWorkflow toolset. Must fire BEFORE session create/resume (the
/// policy only affects records created after it flips).
async fn enable_dynamic_workflows(kernel: &Kernel, cwd: &str) {
    if let Err(e) = kernel
        .call(
            "workspace/updateDynamicWorkflowPolicy",
            json!({
                "workspace": {"workspacePath": cwd, "workspaceKey": cwd},
                "enabled": true,
            }),
        )
        .await
    {
        debug_log(&format!("dynamic workflow policy: {e}"));
    }
}

fn v4_subscribe(kernel: &Kernel, kernel_sid: &str) {
    let _ = kernel.request(
        "v4/conversation/subscribe",
        json!({
            "topic": format!("conversation/{kernel_sid}"),
            "connectionId": "zgrok-host",
            "clientMode": "desktop-continuous",
        }),
    );
}

/// Broadcast `x.ai/session/interjection` — the delivery signal the pager's
/// originator pane uses to claim its optimistic interjection block.
fn broadcast_interjection(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    session_id: &str,
    text: &str,
    interjection_id: Option<&str>,
) {
    let mut payload = json!({"sessionId": session_id, "text": text});
    if let Some(id) = interjection_id {
        payload["interjectionId"] = json!(id);
    }
    if let Ok(params) = serde_json::value::to_raw_value(&payload) {
        gateway.forward_fire_and_forget(acp::ExtNotification::new(
            "x.ai/session/interjection",
            params.into(),
        ));
    }
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
    let home = zcode_home();
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
    let home = zcode_home();
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
    let mut acp_session = acp::SessionId::new(session_id.to_string());
    let mut state = shared.state.borrow().sessions.get(&acp_session).cloned();
    if state.is_none() {
        // Remap by backing kernel session: after a rewind fork swap the
        // kernel id differs from the ACP id the pager knows.
        let remap = {
            let shared_state = shared.state.borrow();
            shared_state
                .sessions
                .iter()
                .find(|(_, st)| st.kernel_id.borrow().as_str() == session_id)
                .map(|(key, _)| key.clone())
        };
        if let Some(key) = remap {
            acp_session = key;
            state = shared.state.borrow().sessions.get(&acp_session).cloned();
        }
    }
    let Some(state) = state else {
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
        // New turn (any kind): advances the epoch so a stale turn.failed
        // grace task knows not to finish it.
        state.turn_epoch.set(state.turn_epoch.get() + 1);
        if state.v4_awaiting_turn_start.replace(false) {
            *state.v4_owned_turn.borrow_mut() = event.turn_id.clone();
            debug_log("v4: claimed preempting turn as ours");
        }
        if event
            .input
            .as_deref()
            .is_some_and(|input| input.starts_with("/compact"))
        {
            state.compacting.set(true);
            *state.compact_turn_id.borrow_mut() = event.turn_id.clone();
            // Kernel-initiated auto compaction: the pager has no Command
            // state for it — surface the official banner pair. (Ext-driven
            // compaction keeps the pager's own /compact UI.)
            if !state.compact_by_ext.get() {
                let (tokens_used, window) = match *state.v4_usage.borrow() {
                    Some((used, max)) => (used, max),
                    None => (last_turn_context_tokens(&state.kernel_id.borrow()), context_window_tokens(shared)),
                };
                let payload = json!({
                    "sessionId": acp_session.0,
                    "update": {
                        "sessionUpdate": "auto_compact_started",
                        "tokens_used": tokens_used,
                        "context_window": window,
                        "percentage": if window > 0 { tokens_used * 100 / window } else { 0 },
                        "reason": "auto",
                    },
                });
                if let Ok(raw) = serde_json::value::to_raw_value(&payload) {
                    gateway.forward_fire_and_forget(acp::ExtNotification::new(
                        "x.ai/session_notification",
                        raw.into(),
                    ));
                }
            }
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
        let was_ext_driven = state.compact_by_ext.replace(false);
        state.compacting.set(false);
        *state.compact_turn_id.borrow_mut() = None;
        if !was_ext_driven && event.kind == "turn.completed" {
            // The post-compaction context size: the v4 projection patch may
            // not have landed yet, so fall back to the ledger (its latest
            // main_turn is still the pre-compact turn — better than 0, which
            // the pager would render as an empty bar).
            let tokens_after = state
                .v4_usage
                .borrow()
                .map(|(used, _)| used)
                .unwrap_or_else(|| last_turn_context_tokens(&state.kernel_id.borrow()));
            let payload = json!({
                "sessionId": acp_session.0,
                "update": {
                    "sessionUpdate": "auto_compact_completed",
                    "tokens_after": tokens_after,
                },
            });
            if let Ok(raw) = serde_json::value::to_raw_value(&payload) {
                gateway.forward_fire_and_forget(acp::ExtNotification::new(
                    "x.ai/session_notification",
                    raw.into(),
                ));
            }
        }
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
            push_todos(gateway, &state, &acp_session);
            sync_kernel_title(state.kernel_id.borrow().as_ref(), &state.cwd.borrow());
            drain_pending_continuation(gateway, shared, &acp_session);
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
                // Live todo pane: the official client updates the plan the
                // moment TodoWrite lands, not at turn end.
                if event.tool_name.as_deref() == Some("TodoWrite") {
                    push_todos(gateway, &state, &acp_session);
                }
            }
        }
        ("turn.completed" | "turn.failed")
            if event.turn_id.is_some()
                && state.v4_owned_turn.borrow().is_some()
                && event.turn_id != *state.v4_owned_turn.borrow() =>
        {
            // Terminal of the turn our v4 startNow preempted — not ours.
            debug_log("v4: ignoring preempted turn's terminal event");
            return;
        }
        "turn.completed" => {
            // Our turn ended: clear the ownership marker.
            *state.v4_owned_turn.borrow_mut() = None;
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
            drain_pending_continuation(gateway, shared, &acp_session);
            push_context_usage(gateway, shared, &state, &acp_session);
            push_todos(gateway, &state, &acp_session);
            sync_kernel_title(state.kernel_id.borrow().as_ref(), &state.cwd.borrow());
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
            let epoch_at_failure = state.turn_epoch.get();
            tokio::task::spawn_local(async move {
                tokio::select! {
                    () = tokio::time::sleep(std::time::Duration::from_secs(20)) => {
                        state.fail_grace_cancel.borrow_mut().take();
                        // A new turn already started (send-now restart, retry,
                        // continuation): the failure belongs to the past —
                        // finishing now would cancel the LIVE turn.
                        if state.turn_epoch.get() != epoch_at_failure {
                            debug_log("turn.failed grace skipped — newer turn already running");
                            return;
                        }
                        debug_log("turn.failed grace expired — finishing turn");
                        finish_turn(&state, &gateway, &session, acp::StopReason::EndTurn);
                        drain_pending_continuation(&gateway, &shared, &session);
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

/// Send a plan-approval/interjection continuation queued for this session,
/// if any. Runs after finish_turn so the settling turn's ACP prompt resolves
/// first. Broadcasts the delivered interjections and reconciles the pager's
/// server queue (empty snapshot unblocks its local queue drain).
fn drain_pending_continuation(
    gateway: &AcpGatewaySender<acp::AgentSide>,
    shared: &Rc<Shared>,
    session: &acp::SessionId,
) {
    let state = shared.state.borrow().sessions.get(session).cloned();
    let Some(state) = state else { return };
    let Some(text) = state.pending_continuation.borrow_mut().take() else {
        return;
    };
    let interjection_ids = std::mem::take(&mut *state.pending_interjection_ids.borrow_mut());
    let Some(kernel) = shared.state.borrow().kernel.clone() else {
        return;
    };
    debug_log(&format!("continuation: {} chars ({} interjection(s))", text.len(), interjection_ids.len()));
    for id in &interjection_ids {
        broadcast_interjection(gateway, &session.0, &text, Some(id));
    }
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
            tracing::warn!(%error, "continuation send failed");
        }
    }
    // The queued row is now the running turn — an empty server queue lets
    // the pager's local queue drain again.
    broadcast_queue(gateway, &session.0, &[], None);
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
