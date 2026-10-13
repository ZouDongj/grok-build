use crate::app::actions::Action;
use crate::slash::command::{AppCtx, CommandExecCtx, CommandResult, SlashCommand, slash_meta};

/// Save a memory note inline or enter remember mode.
pub struct RememberCommand;

impl SlashCommand for RememberCommand {
    slash_meta! {
        name: "remember",
        description: "Save a memory note",
        usage: "/remember [text]",
        takes_args: true,
        arg_placeholder: "[memory note text]",
    }

    /// Hidden on the zcode backend: grok-shell server-side feature the
    /// zcode kernel does not host (entrance closed, not erroring).
    fn visible(&self, _ctx: &AppCtx) -> bool {
        !crate::app::session_hooks::zcode_backend_active()
    }

    fn run(&self, _ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        let trimmed = args.trim();
        if trimmed.is_empty() {
            CommandResult::Action(Action::EnterRememberMode)
        } else {
            CommandResult::Action(Action::SendRememberNote(trimmed.to_string()))
        }
    }
}
