//! `/flush` and `/dream`: run a memory operation as a tracked agent command.
//!
//! Both shadow the shell builtin of the same name so they go through `x.ai/memory/flush` and
//! `x.ai/memory/dream` like `/compact`, with a start line, a spinner, and a typed outcome, instead
//! of a silent prompt turn. The registry hides them until the shell advertises its own builtins.

use crate::slash::command::{AppCtx, CommandExecCtx, CommandResult, SlashCommand, slash_meta};

pub struct FlushCommand;

impl SlashCommand for FlushCommand {
    slash_meta! {
        name: "flush",
        description: "Save this session's memory to disk now",
        usage: "/flush",
        session_scoped: true,
    }

    /// Hidden on the zcode backend: grok-shell server-side feature the
    /// zcode kernel does not host (entrance closed, not erroring).
    fn visible(&self, _ctx: &AppCtx) -> bool {
        !crate::app::session_hooks::zcode_backend_active()
    }

    fn run(&self, _ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        no_args("/flush", args)
    }
}

pub struct DreamCommand;

impl SlashCommand for DreamCommand {
    slash_meta! {
        name: "dream",
        description: "Consolidate memory now (merge observations into topics)",
        usage: "/dream",
        session_scoped: true,
    }

    /// Hidden on the zcode backend: grok-shell server-side feature the
    /// zcode kernel does not host (entrance closed, not erroring).
    fn visible(&self, _ctx: &AppCtx) -> bool {
        !crate::app::session_hooks::zcode_backend_active()
    }

    fn run(&self, _ctx: &mut CommandExecCtx, args: &str) -> CommandResult {
        no_args("/dream", args)
    }
}

fn no_args(command: &str, args: &str) -> CommandResult {
    if args.trim().is_empty() {
        CommandResult::QueueCommand(command.to_string())
    } else {
        CommandResult::Error(format!("{command} takes no arguments."))
    }
}
