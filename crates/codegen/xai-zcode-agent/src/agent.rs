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
}

impl SessionState {
    fn new() -> SessionState {
        SessionState {
            turn_done: RefCell::new(None),
            cancelled: std::cell::Cell::new(false),
            streamed_text: std::cell::Cell::new(false),
            tools: RefCell::new(HashMap::new()),
            next_tool: std::cell::Cell::new(0),
        }
    }

    fn acp_tool_id(&self) -> acp::ToolCallId {
        let n = self.next_tool.replace(self.next_tool.get() + 1);
        acp::ToolCallId::new(format!("tc-{n}"))
    }
}

impl ZcodeAgent {
    pub fn new(gateway: AcpGatewaySender<acp::AgentSide>, kernel_bin: impl Into<String>) -> ZcodeAgent {
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
        eprintln!("[zcode-agent] session/read for models");
        let Ok(read) = kernel.call("session/read", kernel::read_params(session_id)).await else {
            eprintln!("[zcode-agent] session/read FAILED");
            return;
        };
        eprintln!("[zcode-agent] session/read ok");
        let Some(mut state) = model_state_from_settings(&read) else {
            return;
        };
        // Prefer the cached full catalog, but keep any context-window
        // metadata session/read reports for models it knows.
        if let Some(catalog) = self.shared.state.borrow().catalog.clone() {
            if catalog.available_models.len() > state.available_models.len() {
                let windows: std::collections::HashMap<String, &acp::ModelInfo> =
                    state.available_models.iter().map(|m| (m.model_id.0.as_ref().to_string(), m)).collect();
                state.available_models = catalog
                    .available_models
                    .iter()
                    .map(|m| match windows.get(m.model_id.0.as_ref()) {
                        Some(with_meta) => (*with_meta).clone(),
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
            eprintln!("[zcode-agent] kernel config unreadable");
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
        eprintln!(
            "[zcode-agent] catalog: {} models (current {current})",
            catalog.available_models.len()
        );
        if let Ok(raw) = serde_json::value::to_raw_value(&catalog) {
            self.gateway.forward_fire_and_forget(acp::ExtNotification::new(
                "x.ai/models/update",
                raw.into(),
            ));
        }
        self.shared.state.borrow_mut().catalog = Some(catalog);
    }

    fn kernel(&self) -> Result<Kernel, acp::Error> {
        self.shared
            .state
            .borrow()
            .kernel
            .clone()
            .ok_or_else(|| acp::Error::internal_error().data("zcode kernel not running"))
    }

    /// Spawn the kernel once and start the event pump on this LocalSet.
    async fn ensure_kernel(&self) -> Result<Kernel, acp::Error> {
        if let Some(kernel) = self.shared.state.borrow().kernel.as_ref() {
            return Ok(kernel.clone());
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
            while let Some(message) = inbound.recv().await {
                match message {
                    KernelMessage::Event { session_id, payload } => {
                        handle_event(&pump_gateway, &pump_shared, session_id.as_deref(), &payload);
                    }
                    KernelMessage::ServerRequest { id, method, params } => {
                        handle_server_request(&pump_gateway, &pump_shared, &id, &method, params)
                            .await;
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
        });
        Ok(kernel)
    }
}

fn debug_log(text: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/zcode-agent-debug.log")
    {
        let _ = writeln!(f, "{text}");
    }
}

#[async_trait::async_trait(?Send)]
impl acp::Agent for ZcodeAgent {
    async fn initialize(&self, _args: acp::InitializeRequest) -> acp::Result<acp::InitializeResponse> {
        debug_log("acp: initialize");
        eprintln!("[zcode-agent] initialize called");
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
        eprintln!("[zcode-agent] new_session cwd={:?}", args.cwd);
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
        Ok(acp::PromptResponse::new(stop))
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
        eprintln!("[zcode-agent] set_session_model: {}", args.model_id.0);
        let kernel = self.kernel()?;
        let model = &*args.model_id.0;
        let provider = load_model_preference().map(|(p, _)| p).unwrap_or(DEFAULT_PROVIDER.to_string());
        kernel
            .call("session/setModel", kernel::set_model_params(&args.session_id.0, &provider, model))
            .await
            .map_err(|e| acp::Error::internal_error().data(format!("setModel failed: {e}")))?;
        self.push_model_state(&kernel, &args.session_id.0).await;
        Ok(acp::SetSessionModelResponse::default())
    }

    async fn ext_method(&self, args: acp::ExtRequest) -> acp::Result<acp::ExtResponse> {
        debug_log(&format!("acp: ext_method {}", args.method));
        eprintln!("[zcode-agent] ext_method: {}", args.method);
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
        return;
    };
    let Some(event) = TurnEvent::decode(payload) else {
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
    })
}

fn str_at(value: &Value, key: &str) -> String {
    value.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
}
