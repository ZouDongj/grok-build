//! `acp::Agent` implementation backed by the ZCode kernel.
//!
//! Architecture: the grok-build pager (TUI) is an ACP client; this crate is a
//! drop-in agent backend that bridges ACP to ZCode's `zcode app-server`
//! JSON-lines protocol — sessions, streaming turns, permission approval and
//! the model catalog all come from the official kernel, so authentication and
//! model behavior match desktop ZCode by construction.

mod agent;
mod catalog;
mod kernel;

pub use agent::ZcodeAgent;
pub use kernel::{Kernel, KernelMessage};

pub const KERNEL_BIN_DEFAULT: &str = "zcode";
