use crate::app::actions::Action;
use crate::slash::command::{AppCtx, CommandExecCtx, CommandResult, SlashCommand, slash_meta};

pub struct LoginCommand;

impl SlashCommand for LoginCommand {
    slash_meta! {
        name: "login",
        description: "Log in or re-authenticate with your account",
        usage: "/login",
    }

    /// Hidden on the zcode backend: grok-shell server-side feature the
    /// zcode kernel does not host (entrance closed, not erroring).
    fn visible(&self, _ctx: &AppCtx) -> bool {
        !crate::app::session_hooks::zcode_backend_active()
    }

    fn run(&self, _ctx: &mut CommandExecCtx, _args: &str) -> CommandResult {
        CommandResult::Action(Action::Login)
    }
}
