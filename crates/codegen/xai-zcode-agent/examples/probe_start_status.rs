//! Dump sanitized start-plan status endpoint responses (diagnostics only —
//! credential-ish fields redacted in the lib before they reach here).
//!
//! Usage: cargo run -p xai-zcode-agent --example probe_start_status

fn main() {
    let v = xai_zcode_agent::start_plan_status_probe();
    println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
    // The lib's own verdict, once the parser sees these bodies.
    println!(
        "verdict: {:?}",
        xai_zcode_agent::start_plan_entitled_probe()
    );
}
