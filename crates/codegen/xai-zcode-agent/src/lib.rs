//! `acp::Agent` implementation backed by the ZCode kernel.
//!
//! Architecture: the grok-build pager (TUI) is an ACP client; this crate is a
//! drop-in agent backend that bridges ACP to ZCode's `zcode app-server`
//! JSON-lines protocol — sessions, streaming turns, permission approval and
//! the model catalog all come from the official kernel, so authentication and
//! model behavior match desktop ZCode by construction.
//!
//! P2 status: skeleton — kernel client and the ACP mapping land next.

pub const KERNEL_BIN_DEFAULT: &str = "zcode";

/// The agent facade the pager constructs (see `xai-grok-pager/src/acp/spawn.rs`).
pub struct ZcodeAgent {
    kernel_bin: String,
}

impl ZcodeAgent {
    pub fn new(kernel_bin: impl Into<String>) -> Self {
        Self {
            kernel_bin: kernel_bin.into(),
        }
    }

    pub fn kernel_bin(&self) -> &str {
        &self.kernel_bin
    }
}
