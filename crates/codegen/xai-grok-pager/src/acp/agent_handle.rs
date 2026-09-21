//! In-process agent backend handle.
//!
//! The pager drives every agent backend over ACP (`acp::Agent`); the concrete
//! type only matters at three lifecycle moments — construction (see
//! `spawn.rs`), skills/config hot-reload, and the shutdown flush. `AgentOps`
//! covers those extras so the rest of the pager can hold the backend-agnostic
//! `AgentHandle` newtype instead of `Rc<MvpAgent>`, letting alternative
//! kernels (e.g. a ZCode app-server adapter) plug in behind the same TUI.
//!
//! The newtype exists because `acp::Agent`'s blanket impls for `Rc<T>`/
//! `Arc<T>` require `T: Sized`, so `Rc<dyn AgentOps>` itself does not
//! implement `acp::Agent`; a local wrapper may implement the foreign trait
//! directly.

use std::fmt;
use std::ops::Deref;
use std::rc::Rc;
use std::time::Duration;

use agent_client_protocol as acp;

/// Lifecycle operations the pager invokes on the in-process agent beyond the
/// ACP request surface.
#[async_trait::async_trait(?Send)]
pub(crate) trait AgentOps: acp::Agent {
    /// Skills directory changed on disk: reload skills for every session.
    fn reload_skills_all_sessions(&self);
    /// Workflow/command definitions changed: re-advertise commands.
    fn advertise_commands_all_sessions(&self);
    /// Flush session journals at shutdown; `grace` bounds the wait.
    async fn flush_all_sessions(&self, grace: Duration);
}

/// The backend-agnostic agent handle kept alive by the agent worker thread.
#[derive(Clone)]
pub(crate) struct AgentHandle {
    inner: Rc<dyn AgentOps + 'static>,
}

impl AgentHandle {
    pub(crate) fn new(agent: Rc<dyn AgentOps + 'static>) -> Self {
        Self { inner: agent }
    }
}

impl Deref for AgentHandle {
    type Target = dyn AgentOps + 'static;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

impl fmt::Debug for AgentHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentHandle").finish_non_exhaustive()
    }
}

#[async_trait::async_trait(?Send)]
impl AgentOps for xai_grok_shell::agent::MvpAgent {
    fn reload_skills_all_sessions(&self) {
        xai_grok_shell::agent::MvpAgent::reload_skills_all_sessions(self);
    }

    fn advertise_commands_all_sessions(&self) {
        xai_grok_shell::agent::MvpAgent::advertise_commands_all_sessions(self);
    }

    async fn flush_all_sessions(&self, grace: Duration) {
        xai_grok_shell::agent::MvpAgent::flush_all_sessions(self, grace).await;
    }
}

// `&AgentHandle` must coerce to `&dyn AgentOps` for the skills watcher.
#[async_trait::async_trait(?Send)]
impl AgentOps for AgentHandle {
    fn reload_skills_all_sessions(&self) {
        self.inner.reload_skills_all_sessions();
    }

    fn advertise_commands_all_sessions(&self) {
        self.inner.advertise_commands_all_sessions();
    }

    async fn flush_all_sessions(&self, grace: Duration) {
        self.inner.flush_all_sessions(grace).await;
    }
}

// The ZCode backend has no in-process skills/workflow registry to hot-reload;
// shutdown flush is unnecessary because the kernel owns session persistence.
#[async_trait::async_trait(?Send)]
impl AgentOps for xai_zcode_agent::ZcodeAgent {
    fn reload_skills_all_sessions(&self) {}

    fn advertise_commands_all_sessions(&self) {}

    async fn flush_all_sessions(&self, _grace: Duration) {}
}

// Delegating `acp::Agent` impl: the default trait bodies answer
// `method_not_found`, so EVERY method forwards explicitly — a lost default
// here would silently break a protocol feature for the whole backend.
#[async_trait::async_trait(?Send)]
impl acp::Agent for AgentHandle {
    async fn initialize(&self, args: acp::InitializeRequest) -> acp::Result<acp::InitializeResponse> {
        self.inner.initialize(args).await
    }

    async fn authenticate(&self, args: acp::AuthenticateRequest) -> acp::Result<acp::AuthenticateResponse> {
        self.inner.authenticate(args).await
    }

    async fn logout(&self, args: acp::LogoutRequest) -> acp::Result<acp::LogoutResponse> {
        self.inner.logout(args).await
    }

    async fn new_session(&self, args: acp::NewSessionRequest) -> acp::Result<acp::NewSessionResponse> {
        self.inner.new_session(args).await
    }

    async fn prompt(&self, args: acp::PromptRequest) -> acp::Result<acp::PromptResponse> {
        self.inner.prompt(args).await
    }

    async fn cancel(&self, args: acp::CancelNotification) -> acp::Result<()> {
        self.inner.cancel(args).await
    }

    async fn load_session(&self, args: acp::LoadSessionRequest) -> acp::Result<acp::LoadSessionResponse> {
        self.inner.load_session(args).await
    }

    async fn set_session_mode(
        &self,
        args: acp::SetSessionModeRequest,
    ) -> acp::Result<acp::SetSessionModeResponse> {
        self.inner.set_session_mode(args).await
    }

    async fn set_session_model(
        &self,
        args: acp::SetSessionModelRequest,
    ) -> acp::Result<acp::SetSessionModelResponse> {
        self.inner.set_session_model(args).await
    }

    async fn set_session_config_option(
        &self,
        args: acp::SetSessionConfigOptionRequest,
    ) -> acp::Result<acp::SetSessionConfigOptionResponse> {
        self.inner.set_session_config_option(args).await
    }

    async fn list_sessions(
        &self,
        args: acp::ListSessionsRequest,
    ) -> acp::Result<acp::ListSessionsResponse> {
        self.inner.list_sessions(args).await
    }

    async fn fork_session(&self, args: acp::ForkSessionRequest) -> acp::Result<acp::ForkSessionResponse> {
        self.inner.fork_session(args).await
    }

    async fn resume_session(
        &self,
        args: acp::ResumeSessionRequest,
    ) -> acp::Result<acp::ResumeSessionResponse> {
        self.inner.resume_session(args).await
    }

    async fn close_session(&self, args: acp::CloseSessionRequest) -> acp::Result<acp::CloseSessionResponse> {
        self.inner.close_session(args).await
    }

    async fn ext_method(&self, args: acp::ExtRequest) -> acp::Result<acp::ExtResponse> {
        self.inner.ext_method(args).await
    }

    async fn ext_notification(&self, args: acp::ExtNotification) -> acp::Result<()> {
        self.inner.ext_notification(args).await
    }
}
