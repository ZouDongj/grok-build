//! `/config-agents`: open the agents modal.

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, CommandExecCtx, CommandResult, SlashCommand, slash_meta};

/// Open the agents modal listing all agent definitions.
pub struct ConfigAgentsCommand;

impl SlashCommand for ConfigAgentsCommand {
    slash_meta! {
        name: "config-agents",
        aliases: ["agents"],
        description: "Manage agent definitions",
        usage: "/config-agents",
    }

    /// Hidden on the zcode backend: grok-shell server-side feature the
    /// zcode kernel does not host (entrance closed, not erroring).
    fn visible(&self, _ctx: &AppCtx) -> bool {
        !crate::app::session_hooks::zcode_backend_active()
    }

    fn run(&self, _ctx: &mut CommandExecCtx, _args: &str) -> CommandResult {
        CommandResult::Action(Action::OpenConfigAgentsModal(None))
    }
}
