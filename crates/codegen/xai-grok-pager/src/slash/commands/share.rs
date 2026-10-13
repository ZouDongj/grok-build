use crate::slash::command::{AppCtx, CommandExecCtx, CommandResult, SlashCommand, slash_meta};

/// Share the current session via a public URL.
pub struct ShareCommand;

impl SlashCommand for ShareCommand {
    slash_meta! {
        name: "share",
        description: "Share this session via URL",
        usage: "/share",
        session_scoped: true,
    }

    /// Hidden on the zcode backend: grok-shell server-side feature the
    /// zcode kernel does not host (entrance closed, not erroring).
    fn visible(&self, _ctx: &AppCtx) -> bool {
        !crate::app::session_hooks::zcode_backend_active()
    }

    fn run(&self, ctx: &mut CommandExecCtx, _args: &str) -> CommandResult {
        let _ = ctx;
        CommandResult::Error("Session sharing is temporarily disabled".to_string())
    }
}
