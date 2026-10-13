//! `/logout` removes the auth credentials and returns to the login screen.

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, CommandExecCtx, CommandResult, SlashCommand, slash_meta};

pub struct LogoutCommand;

impl SlashCommand for LogoutCommand {
    slash_meta! {
        name: "logout",
        description: "Log out and return to the login screen",
        usage: "/logout",
    }

    /// Hidden on the zcode backend: grok-shell server-side feature the
    /// zcode kernel does not host (entrance closed, not erroring).
    fn visible(&self, _ctx: &AppCtx) -> bool {
        !crate::app::session_hooks::zcode_backend_active()
    }

    fn run(&self, _ctx: &mut CommandExecCtx, _args: &str) -> CommandResult {
        CommandResult::Action(Action::Logout)
    }
}
