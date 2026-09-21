//! Model catalog: the kernel's BUNDLED provider registry
//! (~/.zcode/v2/runtime/provider/**/zcode-builtin.json) — the same source the
//! kernel itself uses, so context windows, modalities and reasoning levels
//! always match GLM official. `modelRules` form an ordered cascade (later
//! rules override earlier ones per field) matched by regex against the
//! lowercased model id; `builtinProviderModelRules` enable models per
//! provider. Runtime `session/read` entries override the fold for the
//! current model (authoritative once a session exists).

use agent_client_protocol as acp;
use serde_json::Value;

/// Per-model properties resolved from the rule cascade.
#[derive(Default, Clone)]
struct ResolvedProps {
    context_window: Option<u64>,
    max_output_tokens: Option<u64>,
    reasoning_levels: Vec<String>,
    supports_text: Option<bool>,
    supports_image: Option<bool>,
    supports_video: Option<bool>,
    supports_audio: Option<bool>,
    supports_pdf: Option<bool>,
}

impl ResolvedProps {
    fn into_meta(self) -> acp::Meta {
        // Keys consumed by the pager's ModelState: totalContextTokens,
        // acceptsImages/inputModalities, supportsReasoningEffort +
        // reasoningEfforts. maxOutputTokens rides along for completeness.
        let mut meta = acp::Meta::new();
        if let Some(window) = self.context_window {
            meta.insert("totalContextTokens".to_string(), serde_json::json!(window));
        }
        if let Some(max) = self.max_output_tokens {
            meta.insert("maxOutputTokens".to_string(), serde_json::json!(max));
        }
        let mut modalities = Vec::new();
        for (supported, name) in [
            (self.supports_text, "text"),
            (self.supports_image, "image"),
            (self.supports_video, "video"),
            (self.supports_audio, "audio"),
            (self.supports_pdf, "pdf"),
        ] {
            if supported.unwrap_or(false) {
                modalities.push(serde_json::json!(name));
            }
        }
        if let Some(images) = self.supports_image {
            meta.insert("acceptsImages".to_string(), serde_json::json!(images));
        }
        if !modalities.is_empty() {
            meta.insert("inputModalities".to_string(), serde_json::json!(modalities));
        }
        if !self.reasoning_levels.is_empty() {
            meta.insert("supportsReasoningEffort".to_string(), serde_json::json!(true));
            let efforts: Vec<serde_json::Value> = self
                .reasoning_levels
                .iter()
                .filter(|level| !matches!(level.as_str(), "disabled" | "enabled"))
                .map(|level| serde_json::json!({"value": level, "label": level}))
                .collect();
            if !efforts.is_empty() {
                meta.insert("reasoningEfforts".to_string(), serde_json::json!(efforts));
            }
        }
        meta
    }
}

/// Fold the modelRules cascade for one model id (lowercased matching, later
/// rules override earlier per field).
fn fold_rules(rules: &[Value], model_id: &str) -> ResolvedProps {
    let mut props = ResolvedProps::default();
    let id = model_id.to_ascii_lowercase();
    for rule in rules {
        let Some(pattern) = rule.get("modelMatch").and_then(Value::as_str) else {
            continue;
        };
        let Ok(re) = regex::Regex::new(&format!("^(?s:{pattern})$")) else {
            continue;
        };
        if !re.is_match(&id) {
            continue;
        }
        let config = rule.get("config").cloned().unwrap_or(Value::Null);
        if let Some(window) = config.pointer("/properties/contextWindow").and_then(Value::as_u64) {
            props.context_window = Some(window);
        }
        if let Some(max) = config
            .pointer("/optionSpecs/maxOutputTokens/max")
            .and_then(Value::as_u64)
            .or_else(|| config.pointer("/optionSpecs/maxOutputTokens/value").and_then(Value::as_u64))
        {
            props.max_output_tokens = Some(max);
        }
        if let Some(levels) = config
            .pointer("/optionSpecs/reasoningLevel/values")
            .and_then(Value::as_array)
        {
            props.reasoning_levels = levels
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect();
        }
        for (slot, key) in [
            (&mut props.supports_text, "supportsText"),
            (&mut props.supports_image, "supportsImage"),
            (&mut props.supports_video, "supportsVideo"),
            (&mut props.supports_audio, "supportsAudio"),
            (&mut props.supports_pdf, "supportsPdf"),
        ] {
            if let Some(value) =
                config.pointer(&format!("/properties/inputFormat/{key}")).and_then(Value::as_bool)
            {
                *slot = Some(value);
            }
        }
    }
    props
}

fn registry_candidates(home: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut candidates = Vec::new();
    // The launcher (and the desktop) point the kernel at the active registry
    // via ZCODE_BUILTIN_PROVIDER_CONFIG_FILE — that file is authoritative.
    if let Some(path) = std::env::var_os("ZCODE_BUILTIN_PROVIDER_CONFIG_FILE") {
        candidates.push(path.into());
    }
    candidates.push(home.join(".zcode/v2/runtime/provider/bundled/zcode-builtin.json"));
    if let Ok(entries) = std::fs::read_dir(home.join(".zcode/v2/runtime/provider")) {
        let mut dirs: Vec<_> = entries
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir())
            .map(|e| e.path().join("zcode-builtin.json"))
            .collect();
        dirs.sort();
        candidates.extend(dirs);
    }
    candidates
}

/// All models enabled for `provider_id`, each with its cascade-resolved
/// metadata (context window, modalities, reasoning levels).
pub fn bundled_models(home: &std::path::Path, provider_id: &str) -> Vec<acp::ModelInfo> {
    for path in registry_candidates(home) {
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&raw) else {
            continue;
        };
        let model_rules = value
            .pointer("/config/modelConfigRules/modelRules")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        // 0.16.9+ registry: providerConfigRules.providerRules carries each
        // account provider's builtinModelIds (+ access entitlement).
        if let Some(provider_rules) = value
            .pointer("/config/providerConfigRules/providerRules")
            .and_then(Value::as_array)
        {
            let models: Vec<acp::ModelInfo> = provider_rules
                .iter()
                .filter(|rule| rule.get("providerId").and_then(Value::as_str) == Some(provider_id))
                .filter_map(|rule| {
                    rule.pointer("/config/builtinModelIds")
                        .and_then(Value::as_array)
                })
                .flat_map(|ids| ids.iter().filter_map(Value::as_str))
                .map(|id| {
                    let props = fold_rules(&model_rules, id);
                    acp::ModelInfo::new(id.to_string(), id.to_string()).meta(props.into_meta())
                })
                .collect();
            if !models.is_empty() {
                return models;
            }
        }
        // 0.16.5 registry: builtinProviderModelRules enable models per
        // provider ({providerId, modelId, config.enabled}).
        if let Some(rules) = value
            .pointer("/config/modelConfigRules/builtinProviderModelRules")
            .and_then(Value::as_array)
        {
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
                    let props = fold_rules(&model_rules, id);
                    Some(acp::ModelInfo::new(id.to_string(), id.to_string()).meta(props.into_meta()))
                })
                .collect();
            if !models.is_empty() {
                return models;
            }
        }
    }
    Vec::new()
}

/// Fold a runtime `session/read` available[] entry's authoritative metadata
/// (contextWindow, maxOutputTokens, reasoning.levels, properties.inputFormat)
/// into a ModelInfo.
pub fn model_info_from_runtime_entry(entry: &Value) -> Option<acp::ModelInfo> {
    let id = entry.pointer("/ref/modelId").and_then(Value::as_str)?;
    let mut props = ResolvedProps {
        context_window: entry.get("contextWindow").and_then(Value::as_u64),
        max_output_tokens: entry.get("maxOutputTokens").and_then(Value::as_u64),
        ..Default::default()
    };
    if let Some(levels) = entry
        .pointer("/reasoning/levels")
        .and_then(Value::as_array)
    {
        props.reasoning_levels = levels
            .iter()
            .filter_map(|level| level.get("value").and_then(Value::as_str))
            .map(str::to_string)
            .collect();
    }
    for (slot, key) in [
        (&mut props.supports_text, "supportsText"),
        (&mut props.supports_image, "supportsImage"),
        (&mut props.supports_video, "supportsVideo"),
        (&mut props.supports_audio, "supportsAudio"),
        (&mut props.supports_pdf, "supportsPdf"),
    ] {
        if let Some(value) = entry
            .pointer(&format!("/properties/inputFormat/{key}"))
            .and_then(Value::as_bool)
        {
            *slot = Some(value);
        }
    }
    Some(acp::ModelInfo::new(id.to_string(), id.to_string()).meta(props.into_meta()))
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
pub fn catalog_from_read_state(result: &Value) -> Option<acp::SessionModelState> {
    crate::agent::model_state_from_model(result.pointer("/settings/model")?)
}
