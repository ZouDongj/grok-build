//! ZCode kernel client: spawns `zcode app-server` and speaks its
//! newline-delimited JSON protocol.
//!
//! Ported from zcode-tui's proven client (kernel 0.16.5, pinned live):
//! envelope `{id, method, params}` (NOT JSON-RPC — a `jsonrpc` key is
//! rejected), streaming via `session/event`, server→client requests carry
//! STRING envelope ids that must be echoed verbatim in the reply.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};

pub const DELIVERY_KIND: &str = "desktop-continuous";

/// One decoded inbound kernel line.
#[derive(Debug)]
pub enum KernelMessage {
    /// A `session/event` payload plus the kernel session it belongs to.
    Event {
        session_id: Option<String>,
        payload: Value,
    },
    /// Server→client request needing a reply (envelope id is a string).
    ServerRequest {
        id: Value,
        method: String,
        params: Value,
    },
}

/// The event kind: `payload.kind` for streaming payloads, else `params.type`
/// for session-level events.
fn payload_kind(params: &Value) -> Option<String> {
    params
        .pointer("/payload/kind")
        .and_then(Value::as_str)
        .or_else(|| params.get("type").and_then(Value::as_str))
        .map(String::from)
}

/// A streaming turn event, decoded from `session/event` payloads.
#[derive(Debug, Clone)]
pub struct TurnEvent {
    pub kind: String,
    pub delta: String,
    pub tool_call_id: Option<String>,
    pub tool_name: Option<String>,
    pub output: Option<String>,
    pub success: Option<bool>,
}

impl TurnEvent {
    pub fn decode(payload: &Value) -> Option<TurnEvent> {
        let str_field = |key: &str| {
            payload
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        Some(TurnEvent {
            kind: str_field("kind")?,
            delta: str_field("delta").unwrap_or_default(),
            tool_call_id: str_field("toolCallId"),
            tool_name: str_field("toolName"),
            output: payload
                .pointer("/result/content")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| str_field("response")),
            success: payload.pointer("/result/success").and_then(Value::as_bool),
        })
    }
}

/// A live `zcode app-server` child with reader and writer threads. Cloneable;
/// the paired inbound receiver comes back from [`Kernel::spawn`] once and is
/// drained by the agent's pump task.
#[derive(Clone)]
pub struct Kernel {
    child: Arc<Mutex<Child>>,
    writer: std::sync::mpsc::Sender<String>,
    pending: Arc<Pending>,
}

struct Pending {
    next_id: AtomicU64,
    waiting: Mutex<HashMap<u64, oneshot::Sender<Result<Value, String>>>>,
}

impl Kernel {
    /// Spawn the kernel and start the reader/writer threads. The reader
    /// auto-answers the two bootstrap server requests (runtime preferences,
    /// official MCP auth) that otherwise stall `session/create` for 15s.
    pub fn spawn(
        bin: &str,
    ) -> std::io::Result<(Kernel, mpsc::UnboundedReceiver<KernelMessage>)> {
        let mut child = Command::new(bin)
            .arg("app-server")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let (tx, rx) = mpsc::unbounded_channel::<KernelMessage>();
        let (write_tx, write_rx) = std::sync::mpsc::channel::<String>();
        let pending = Arc::new(Pending {
            next_id: AtomicU64::new(1),
            waiting: Mutex::new(HashMap::new()),
        });
        let reader_pending = Arc::clone(&pending);
        let reader_write = write_tx.clone();

        thread::Builder::new()
            .name("zcode-kernel-reader".into())
            .spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    let Ok(value) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };
                    // Server→client request: method AND id together. The kernel
                    // uses string envelope ids here; echo them verbatim.
                    if let (Some(method), Some(id)) = (
                        value.get("method").and_then(Value::as_str),
                        value.get("id"),
                    ) {
                        if let Some(reply) = auto_reply(method) {
                            let _ = reader_write.send(
                                json!({"id": id, "result": reply}).to_string(),
                            );
                        } else {
                            let _ = tx.send(KernelMessage::ServerRequest {
                                id: id.clone(),
                                method: method.to_string(),
                                params: value.get("params").cloned().unwrap_or(Value::Null),
                            });
                        }
                        continue;
                    }
                    if let Some(id) = value.get("id").and_then(Value::as_u64) {
                        if let Some(sender) =
                            reader_pending.waiting.lock().unwrap().remove(&id)
                        {
                            let error = value.get("error").and_then(|err| {
                                err.get("message").and_then(Value::as_str).map(String::from)
                            });
                            let _ = sender.send(match error {
                                Some(error) => Err(error),
                                None => Ok(value.get("result").cloned().unwrap_or(Value::Null)),
                            });
                        }
                        continue;
                    }
                    if value.get("method").and_then(Value::as_str) == Some("session/event") {
                        // Streaming payloads (model.streaming) carry their own
                        // `kind`; session-level events (turn.completed, …) do
                        // NOT — pass `params.type` through as the kind so turn
                        // terminators reach the agent (zcode-tui parity).
                        let params = &value["params"];
                        let kind = payload_kind(params);
                        if let Some(kind) = kind {
                            let mut payload = params
                                .get("payload")
                                .cloned()
                                .unwrap_or_else(|| serde_json::json!({}));
                            if payload.get("kind").is_none() {
                                payload["kind"] = Value::String(kind);
                            }
                            let _ = tx.send(KernelMessage::Event {
                                session_id: params
                                    .get("sessionId")
                                    .and_then(Value::as_str)
                                    .map(String::from),
                                payload,
                            });
                        }
                        continue;
                    }
                    // state.updated and everything else: ignored for now.
                }
                // Kernel gone: wake any pending request and stop the writer.
                reader_pending.waiting.lock().unwrap().clear();
            })?;

        // Single owner of stdin: requests and server-request replies funnel
        // through this writer thread.
        thread::Builder::new()
            .name("zcode-kernel-writer".into())
            .spawn(move || {
                use std::io::Write;
                while let Ok(line) = write_rx.recv() {
                    if stdin
                        .write_all(line.as_bytes())
                        .and_then(|_| stdin.write_all(b"\n"))
                        .and_then(|_| stdin.flush())
                        .is_err()
                    {
                        break;
                    }
                }
            })?;

        Ok((
            Kernel {
                child: Arc::new(Mutex::new(child)),
                writer: write_tx,
                pending,
            },
            rx,
        ))
    }

    pub fn is_alive(&self) -> bool {
        matches!(self.child.lock().unwrap().try_wait(), Ok(None))
    }

    /// Send a request; the reader thread completes the returned receiver.
    pub fn request(
        &self,
        method: &str,
        params: Value,
    ) -> anyhow::Result<oneshot::Receiver<Result<Value, String>>> {
        let id = self.pending.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.waiting.lock().unwrap().insert(id, tx);
        let line = json!({"id": id, "method": method, "params": params}).to_string();
        self.writer
            .send(line)
            .map_err(|_| anyhow::anyhow!("kernel writer is gone"))?;
        Ok(rx)
    }

    /// Reply to a server→client request (echo the envelope id verbatim).
    pub fn reply(&self, id: &Value, result: Value) -> anyhow::Result<()> {
        let line = json!({"id": id, "result": result}).to_string();
        self.writer
            .send(line)
            .map_err(|_| anyhow::anyhow!("kernel writer is gone"))
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Value, String> {
        let rx = self.request(method, params).map_err(|e| e.to_string())?;
        rx.await.map_err(|_| "kernel reader dropped".to_string())?
    }
}

/// The two bootstrap server requests that must be answered for
/// `session/create` to complete (pinned live on kernel 0.16.3+).
fn auto_reply(method: &str) -> Option<Value> {
    match method {
        "session/requestRuntimePreferences" => Some(json!({
            "nativeSearchEnhancementsEnabled": true,
            "memoryEnabled": false,
            "askUserQuestionAutoResolutionEnabled": true,
            "modelContextBudgetStrategy": "preflight-v1",
        })),
        "interaction/requestOfficialMcpAuthHeaders" => {
            Some(json!({"ok": false, "reason": "official_auth_unavailable"}))
        }
        _ => None,
    }
}

pub fn create_params(workspace: &std::path::Path) -> Value {
    let key = workspace.display().to_string();
    json!({"workspace": {"workspaceKey": key, "workspacePath": key}})
}

pub fn subscribe_params(session_id: &str) -> Value {
    json!({"sessionId": session_id, "deliveryKind": DELIVERY_KIND})
}

pub fn send_params(session_id: &str, content: &str) -> Value {
    json!({"sessionId": session_id, "content": content})
}

pub fn stop_params(session_id: &str) -> Value {
    json!({"sessionId": session_id})
}

pub fn set_model_params(session_id: &str, provider_id: &str, model_id: &str) -> Value {
    json!({
        "sessionId": session_id,
        "model": {
            "providerId": provider_id,
            "modelId": model_id,
            "options": {"reasoningLevel": "max"},
        }
    })
}

pub fn session_id_from(result: &Value) -> Option<String> {
    result
        .pointer("/session/sessionId")
        .and_then(Value::as_str)
        .map(String::from)
}
