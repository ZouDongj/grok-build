//! `/memory`: open the memory browser modal.
//!
//! Toggling (`t`) and diagnostics (`s`) live inside the modal, so `/memory` has one meaning.
//! Shadows the shell builtin of the same name so opening the modal goes through
//! `x.ai/memory/list` instead of a prompt turn that echoes into scrollback.
//! The registry hides this command until the shell advertises its own `memory` builtin.

use crate::app::actions::Action;
use crate::slash::command::{AppCtx, CommandExecCtx, CommandResult, SlashCommand, slash_meta};

pub struct MemoryCommand;

impl SlashCommand for MemoryCommand {
    slash_meta! {
        name: "memory",
        aliases: ["mem"],
        description: "Browse, view, and manage your memories",
        usage: "/memory",
        session_scoped: true,
    }

    /// Hidden on the zcode backend: grok-shell server-side feature the
    /// zcode kernel does not host (entrance closed, not erroring).
    fn visible(&self, _ctx: &AppCtx) -> bool {
        !crate::app::session_hooks::zcode_backend_active()
    }

    fn run(&self, _ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        if args.trim().is_empty() {
            CommandResult::Action(Action::OpenMemoryModal)
        } else {
            CommandResult::Error(
                "/memory takes no arguments. Open it, then press t to turn memory on or off and s for status."
                    .to_string(),
            )
        }
    }
}
