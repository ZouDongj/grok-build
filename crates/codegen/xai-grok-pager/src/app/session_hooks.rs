//! Client-side `session_start` hooks for the zcode backend.
//!
//! The grok backend fires Claude-compatible hooks server-side (session
//! handle's `DispatchSessionStartHook`); the zcode backend has no host-side
//! runner, so the pager fires the event here. This lights up
//! `~/.grok/hooks/*.json` integrations unchanged — notably herdr's agent
//! registration hook, which reports the pane's agent session to the herdr
//! daemon (GROK_SESSION_ID env + session_start envelope over stdin).

use std::path::Path;
use std::sync::Arc;

/// True when the zcode kernel backend is active (default). The grok backend
/// fires the same hooks server-side — firing here too would double-run them.
pub fn zcode_backend_active() -> bool {
    static ACTIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ACTIVE.get_or_init(|| {
        std::env::var("GROK_BACKEND")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                !(v == "grok" || v == "shell")
            })
            .unwrap_or(true)
    })
}

/// Fire `session_start` for a freshly created/resumed session. Fire-and-
/// forget: the event is observe-only and must never block session startup.
pub fn fire_session_start(session_id: &str, cwd: &str) {
    if !zcode_backend_active() {
        return;
    }
    let global_dir = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .map(|h| h.join(".grok/hooks"));
    let project_dir = Path::new(cwd).join(".grok/hooks");
    let (registry, errors) = xai_grok_hooks::discovery::load_hooks(
        global_dir.as_deref(),
        Some(&project_dir),
    );
    for err in &errors {
        tracing::debug!(%err, "session hook load warning");
    }
    let hooks = registry.hooks_for(xai_grok_hooks::event::HookEventName::SessionStart);
    if hooks.is_empty() {
        return;
    }
    let envelope = xai_grok_hooks::event::HookEventEnvelope {
        hook_event_name: xai_grok_hooks::event::HookEventName::SessionStart,
        session_id: session_id.to_string(),
        cwd: cwd.to_string(),
        workspace_root: cwd.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        transcript_path: None,
        client_identifier: None,
        prompt_id: None,
        permission_mode: None,
        payload: xai_grok_hooks::event::HookPayload::SessionStart {
            source: "startup".to_string(),
            model_id: None,
            agent_type: None,
        },
    };
    let session_id = session_id.to_string();
    let workspace_root = cwd.to_string();
    let registry = Arc::new(registry);
    // Fire-and-forget on a dedicated thread with a one-shot runtime: the
    // dispatch path may run outside any LocalSet (spawn_local panics there),
    // and session startup must never block on hook execution.
    std::thread::Builder::new()
        .name("session-start-hooks".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    tracing::debug!(%e, "session hook runtime spawn failed");
                    return;
                }
            };
            rt.block_on(async {
                let disabled = Arc::new(xai_grok_hooks::trust::DisabledHooks::new([], false));
                let ctx = xai_grok_hooks::runner::RunContext {
                    session_id: &session_id,
                    workspace_root: &workspace_root,
                    process_scope: None,
                    disabled,
                };
                let results = xai_grok_hooks::runner::command::join_hook_group_reaps(async {
                    xai_grok_hooks::dispatcher::dispatch_non_blocking(
                        &registry,
                        xai_grok_hooks::event::HookEventName::SessionStart,
                        &envelope,
                        &ctx,
                    )
                    .await
                })
                .await;
                tracing::debug!(count = results.len(), "session_start hooks fired");
            });
        })
        .map_err(|e| tracing::debug!(%e, "session hook thread spawn failed"))
        .ok();
}
