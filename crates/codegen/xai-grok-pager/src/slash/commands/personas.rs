//! `/personas` -- open the agents modal on the Personas tab.

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, CommandExecCtx, CommandResult, SlashCommand, slash_meta};
use crate::views::agents_modal::AgentsTab;

/// Open the agents modal directly on the Personas tab.
pub struct PersonasCommand;

impl SlashCommand for PersonasCommand {
    slash_meta! {
        name: "personas",
        description: "Manage personas (create, edit, delete)",
        usage: "/personas",
    }

    /// Hidden on the zcode backend: grok-shell server-side feature the
    /// zcode kernel does not host (entrance closed, not erroring).
    fn visible(&self, _ctx: &AppCtx) -> bool {
        !crate::app::session_hooks::zcode_backend_active()
    }

    fn run(&self, _ctx: &mut CommandExecCtx, _args: &str) -> CommandResult {
        CommandResult::Action(Action::OpenConfigAgentsModal(Some(AgentsTab::Personas)))
    }
}
