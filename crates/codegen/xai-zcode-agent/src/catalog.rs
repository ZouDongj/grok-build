//! Model catalog: the kernel's BUNDLED provider registry
//! (~/.zcode/v2/runtime/provider/bundled/zcode-builtin.json) lists every
//! official model enabled for the account's coding-plan provider — the same
//! source the kernel itself uses. No registration handshake needed; per-model
//! context windows come from the session's own `settings.model.available`.

use agent_client_protocol as acp;
use serde_json::Value;

/// All models enabled for `provider_id` from the bundled registry.
/// Falls back to the version-specific dirs if the bundled file is missing.
pub fn bundled_models(home: &std::path::Path, provider_id: &str) -> Vec<acp::ModelInfo> {
    let mut candidates = vec![
        home.join(".zcode/v2/runtime/provider/bundled/zcode-builtin.json"),
    ];
    if let Ok(entries) = std::fs::read_dir(home.join(".zcode/v2/runtime/provider")) {
        let mut dirs: Vec<_> = entries
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .map(|e| e.path().join("zcode-builtin.json"))
            .collect();
        dirs.sort();
        candidates.extend(dirs);
    }
    for path in candidates {
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        let rules = value
            .pointer("/config/modelConfigRules/builtinProviderModelRules")
            .and_then(Value::as_array);
        let Some(rules) = rules else {
            continue;
        };
        let models: Vec<acp::ModelInfo> = rules
            .iter()
            .filter(|rule| {
                rule.get("providerId").and_then(Value::as_str) == Some(provider_id)
                    && rule
                        .pointer("/config/enabled")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
            })
            .filter_map(|rule| {
                let id = rule.get("modelId").and_then(Value::as_str)?;
                Some(acp::ModelInfo::new(id.to_string(), id.to_string()))
            })
            .collect();
        if !models.is_empty() {
            return models;
        }
    }
    Vec::new()
}

pub fn workspace_read_params(workspace: &str) -> serde_json::Value {
    serde_json::json!({
        "workspace": {
            "workspaceKey": workspace,
            "workspacePath": workspace,
        }
    })
}

/// Full catalog from `workspace/readState`.
pub fn catalog_from_read_state(
    result: &Value,
) -> Option<acp::SessionModelState> {
    crate::agent::model_state_from_model(result.pointer("/settings/model")?)
}
