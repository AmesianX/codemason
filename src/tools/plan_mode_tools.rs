//! EnterPlanMode + ExitPlanMode tools for CodeMason.
//! Equivalent to codemason/tools/plan_mode_tools.py — port of
//! EnterPlanModeTool / ExitPlanModeV2Tool from Claude Code, adapted to the
//! local substrate:
//! - Plan-mode state is a process flag (utils/plan_mode) instead of React app
//!   state.
//! - ExitPlanMode takes the plan as a required `plan` parameter and echoes it
//!   back, instead of the V2 file-based flow — codemason has no plan-file /
//!   plan-mode-system-message attachment infra. The V2 SDK-facing extended
//!   schema (_sdkInputSchema) does expose an optional `plan`, so the param
//!   itself is precedented; the difference is that here it is the source of
//!   truth rather than a disk mirror.
//! - Approval uses codemason's terminal interface instead of the Ink dialog.

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde_json::Value;

use crate::tools::base::{CanUseToolFn, PermissionResult, Tool, ToolResult, ToolStream};
use crate::types::query::{ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors, render_markdown};
use crate::utils::plan_mode::{enter_plan_mode, exit_plan_mode, is_plan_mode_active};

pub const ENTER_PLAN_MODE_TOOL_NAME: &str = "EnterPlanMode";
pub const EXIT_PLAN_MODE_TOOL_NAME: &str = "ExitPlanMode";

pub const ENTER_PLAN_MODE_DESCRIPTION: &str = include_str!("prompts/enter_plan_mode_prompt.txt");
pub const EXIT_PLAN_MODE_DESCRIPTION: &str = include_str!("prompts/exit_plan_mode_prompt.txt");

/// Short one-line descriptions — the source's description() returns these; the
/// long text above is the tool prompt() (EnterPlanModeTool.ts:41 /
/// ExitPlanModeV2Tool.ts description()).
pub const ENTER_PLAN_MODE_SHORT_DESCRIPTION: &str = "Requests permission to enter plan mode for complex tasks requiring exploration and design";
pub const EXIT_PLAN_MODE_SHORT_DESCRIPTION: &str =
    "Prompts the user to exit plan mode and start coding";

static SCHEMAS: Lazy<Value> = Lazy::new(|| {
    serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap_or_default()
});

fn schema_for(name: &str) -> Value {
    SCHEMAS
        .get(name)
        .and_then(|v| v.get("schema"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}))
}

pub struct EnterPlanModeTool;

#[async_trait]
impl Tool for EnterPlanModeTool {
    fn name(&self) -> &'static str {
        ENTER_PLAN_MODE_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        schema_for(ENTER_PLAN_MODE_TOOL_NAME)
    }

    async fn description(&self, _input: &Value) -> String {
        ENTER_PLAN_MODE_SHORT_DESCRIPTION.to_string()
    }

    async fn prompt(&self) -> String {
        ENTER_PLAN_MODE_DESCRIPTION.to_string()
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn is_concurrency_safe(&self, _input: Option<&Value>) -> bool {
        true
    }

    async fn is_enabled(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        // Source userFacingName() returns '' for this meta tool.
        String::new()
    }

    fn render_tool_use_message(&self, _input: &Value, _verbose: bool) -> String {
        // EnterPlanModeTool/UI.tsx: no arguments to show.
        String::new()
    }

    // No needs_permissions / check_permissions override: the tool is
    // read-only, so the base check_permissions returns 'allow' (the source's
    // default for EnterPlanMode) — no approval prompt to enter plan mode.

    async fn validate_input(&self, _input: &Value, context: &ToolUseContext) -> ValidationResult {
        // Cannot be used in agent contexts (matches the original agentId guard).
        if context.agent_id.is_some() {
            return ValidationResult::failure("EnterPlanMode tool cannot be used in agent contexts");
        }
        ValidationResult::success()
    }

    fn call(
        &self,
        _input: Value,
        _context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            // handlePlanModeTransition + setAppState(setMode 'plan') — backed
            // by AppState.toolPermissionContext (stashes prior mode as
            // pre_plan_mode).
            enter_plan_mode();
            // The TS tool builds the model-facing text in two stages: call()
            // returns the short `message`, then
            // mapToolResultToToolResultBlockParam appends the workflow block.
            // codemason has no separate result-mapping stage, so the combined
            // non-interview-phase text (EnterPlanModeTool.ts ~99, 108-118) is
            // assembled here verbatim. (The interview-phase branch is omitted
            // because codemason has no plan-file / interview-phase infra.)
            yield Ok(ToolResult::Result {
                result_for_assistant: concat!(
                    "Entered plan mode. You should now focus on exploring the codebase ",
                    "and designing an implementation approach.\n",
                    "\n",
                    "In plan mode, you should:\n",
                    "1. Thoroughly explore the codebase to understand existing patterns\n",
                    "2. Identify similar features and architectural approaches\n",
                    "3. Consider multiple approaches and their trade-offs\n",
                    "4. Use AskUserQuestion if you need to clarify the approach\n",
                    "5. Design a concrete implementation strategy\n",
                    "6. When ready, use ExitPlanMode to present your plan for approval\n",
                    "\n",
                    "Remember: DO NOT write or edit any files yet. This is a read-only ",
                    "exploration and planning phase."
                )
                .to_string(),
                data: Some(serde_json::json!({"plan_mode": true})),
            });
        })
    }
}

pub struct ExitPlanModeTool;

#[async_trait]
impl Tool for ExitPlanModeTool {
    fn name(&self) -> &'static str {
        EXIT_PLAN_MODE_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        schema_for(EXIT_PLAN_MODE_TOOL_NAME)
    }

    async fn description(&self, _input: &Value) -> String {
        EXIT_PLAN_MODE_SHORT_DESCRIPTION.to_string()
    }

    async fn prompt(&self) -> String {
        EXIT_PLAN_MODE_DESCRIPTION.to_string()
    }

    fn is_read_only(&self) -> bool {
        // Source: false ("Now writes to disk"); exiting plan mode mutates the
        // permission context, so it is not read-only.
        false
    }

    fn is_concurrency_safe(&self, _input: Option<&Value>) -> bool {
        true
    }

    async fn is_enabled(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        // Source userFacingName() returns '' for this meta tool.
        String::new()
    }

    fn render_tool_use_message(&self, _input: &Value, _verbose: bool) -> String {
        // ExitPlanModeTool/UI.tsx renderToolUseMessage returns null: the plan
        // is shown as the RESULT (below), not echoed again at call time.
        String::new()
    }

    /// The approved plan, rendered as the markdown it is
    /// (ExitPlanModeTool/UI.tsx).
    fn render_tool_result_message(&self, data: &Value, _verbose: bool) -> Option<String> {
        if !data.is_object() {
            return None;
        }
        let plan = data.get("plan").and_then(|p| p.as_str()).unwrap_or("").trim();
        if plan.is_empty() {
            return Some("Plan approved — exiting plan mode.".to_string());
        }
        let mut lines = vec![colorize("Plan approved:", colors::BOLD), String::new()];
        lines.extend(render_markdown(plan).split('\n').map(|s| s.to_string()));
        Some(lines.join("\n"))
    }

    fn always_ask(&self) -> bool {
        // A plan is the user's call. Approving one is not the same as
        // authorising a tool the model was going to run anyway, so -dsp must
        // not swallow it — the whole point of plan mode is the look before
        // the leap.
        true
    }

    async fn check_permissions(&self, _input: &Value, _context: &ToolUseContext) -> PermissionResult {
        // Source checkPermissions returns { behavior: 'ask', message: 'Exit
        // plan mode?' } for non-teammates — route approval through codemason's
        // real permission flow (can_use_tool), NOT a raw input() in call().
        // call() then runs only when the user approves.
        PermissionResult::ask_with("Exit plan mode?")
    }

    async fn validate_input(&self, _input: &Value, _context: &ToolUseContext) -> ValidationResult {
        // Source: reject before the approval dialog when not in plan mode
        // (errorCode 1). The deferred-tool list announces this tool regardless
        // of mode, so the model can call it spuriously.
        if !is_plan_mode_active() {
            let mut v = ValidationResult::failure(concat!(
                "You are not in plan mode. This tool is only for exiting plan ",
                "mode after writing a plan. If your plan was already approved, ",
                "continue with implementation."
            ));
            v.meta = Some(serde_json::json!({"errorCode": 1}));
            return v;
        }
        ValidationResult::success()
    }

    fn call(
        &self,
        input: Value,
        _context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            // Reaching call() means approval already happened
            // (check_permissions 'ask' -> can_use_tool, or
            // dangerously_skip_permissions). Rejection is handled by the
            // permission flow, which yields its own denial result — so there
            // is no rejection branch here (the source has none either).
            let plan = input.get("plan").and_then(|p| p.as_str()).unwrap_or("").to_string();

            // Exit plan mode: restore the prior mode (ExitPlanModeV2Tool.ts
            // call()).
            exit_plan_mode();

            if plan.trim().is_empty() {
                // Source: empty-plan branch of mapToolResultToToolResultBlockParam.
                yield Ok(ToolResult::Result {
                    result_for_assistant:
                        "User has approved exiting plan mode. You can now proceed.".to_string(),
                    data: Some(serde_json::json!({"approved": true, "plan": null})),
                });
                return;
            }

            // Mirrors ExitPlanModeV2Tool.ts mapToolResultToToolResultBlockParam
            // (~481-491): the lead "You can now start coding…" line and the
            // echoed plan under an "## Approved Plan:" heading. The file-path
            // and TeamCreate lines are dropped as a substrate adaptation
            // (codemason has no plan-file infra and no TeamCreate tool).
            yield Ok(ToolResult::Result {
                result_for_assistant: format!(
                    "User has approved your plan. You can now start coding. Start \
                     with updating your todo list if applicable\n\
                     \n\
                     ## Approved Plan:\n\
                     {plan}"
                ),
                data: Some(serde_json::json!({"approved": true, "plan": plan})),
            });
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn plan_mode_transitions() {
        // enter: flips the mode flag; exit tool refuses when not in plan mode.
        let _guard = crate::state::app_state::test_lock();
        crate::state::app_state::reset_app_state();
        assert!(!is_plan_mode_active());
        let v = ExitPlanModeTool
            .validate_input(&serde_json::json!({"plan": "x"}), &ToolUseContext::default())
            .await;
        assert!(!v.result);
        assert_eq!(v.meta.unwrap()["errorCode"], 1);

        let mut s = EnterPlanModeTool.call(
            serde_json::json!({}),
            ToolUseContext::default(),
            None,
        );
        let r = s.next().await.unwrap().unwrap();
        match r {
            ToolResult::Result { result_for_assistant, .. } => {
                assert!(result_for_assistant.starts_with("Entered plan mode."));
            }
            _ => panic!(),
        }
        assert!(is_plan_mode_active());

        // exit with a plan restores the mode and echoes the plan.
        let mut s2 = ExitPlanModeTool.call(
            serde_json::json!({"plan": "## step 1"}),
            ToolUseContext::default(),
            None,
        );
        let r2 = s2.next().await.unwrap().unwrap();
        match r2 {
            ToolResult::Result { result_for_assistant, data } => {
                assert!(result_for_assistant.contains("## Approved Plan:\n## step 1"));
                assert_eq!(data.unwrap()["approved"], true);
            }
            _ => panic!(),
        }
        assert!(!is_plan_mode_active());
    }

    #[tokio::test]
    async fn enter_refused_in_agent_context() {
        let ctx = ToolUseContext { agent_id: Some("a1".into()), ..Default::default() };
        let v = EnterPlanModeTool.validate_input(&serde_json::json!({}), &ctx).await;
        assert!(!v.result);
        assert_eq!(v.message, "EnterPlanMode tool cannot be used in agent contexts");
    }

    #[test]
    fn exit_always_asks() {
        assert!(ExitPlanModeTool.always_ask());
        assert!(!EnterPlanModeTool.always_ask());
    }
}
