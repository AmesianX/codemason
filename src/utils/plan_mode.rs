//! Plan mode state for CodeMason.
//! Equivalent to codemason/utils/plan_mode.py.
//!
//! Based on the permission-mode 'plan' concept from Claude Code. The original
//! keeps this on `AppState.toolPermissionContext.mode`; codemason has a real
//! AppState (state/app_state.rs), so plan mode lives there too — NOT a
//! separate bool flag.
//!
//! - enter_plan_mode(): prepareContextForPlanMode (stash current mode as
//!   pre_plan_mode) + applyPermissionUpdate({setMode: 'plan'}).
//! - exit_plan_mode(): restore mode = pre_plan_mode ?? 'default', clear
//!   pre_plan_mode (ExitPlanModeV2Tool.ts call()).
//!
//! Only the setMode behavior of applyPermissionUpdate is ported — its full
//! permission-rule update machinery backs a rule engine codemason doesn't
//! have. While mode == 'plan', write tools are blocked in the permission check
//! (query.rs) and the model explores + plans.

use crate::state::app_state::{get_app_state, set_app_state};

/// True when the current permission mode is 'plan'.
pub fn is_plan_mode_active() -> bool {
    get_app_state().tool_permission_context.mode == "plan"
}

/// Enter plan mode (EnterPlanModeTool.call()).
///
/// Stashes the current mode as pre_plan_mode (prepareContextForPlanMode) then
/// sets mode='plan' (applyPermissionUpdate setMode). No-op if already in plan
/// mode, preserving the original pre_plan_mode.
pub fn enter_plan_mode() {
    set_app_state(|mut prev| {
        let ctx = &mut prev.tool_permission_context;
        if ctx.mode == "plan" {
            return prev;
        }
        ctx.pre_plan_mode = Some(ctx.mode.clone());
        ctx.mode = "plan".to_string();
        prev
    });
}

/// Exit plan mode, restoring the prior mode (ExitPlanModeV2Tool.ts call()).
///
/// Restore target is pre_plan_mode, falling back to 'default'; pre_plan_mode
/// is cleared. No-op when not in plan mode.
pub fn exit_plan_mode() {
    set_app_state(|mut prev| {
        let ctx = &mut prev.tool_permission_context;
        if ctx.mode != "plan" {
            return prev;
        }
        ctx.mode = ctx
            .pre_plan_mode
            .take()
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| "default".to_string());
        prev
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_mode_roundtrip() {
        let _guard = crate::state::app_state::test_lock();
        crate::state::app_state::reset_app_state();
        assert!(!is_plan_mode_active());
        enter_plan_mode();
        assert!(is_plan_mode_active());
        // Re-entering preserves the stash.
        enter_plan_mode();
        exit_plan_mode();
        assert!(!is_plan_mode_active());
        // Python stashed the empty-string default mode and restored it as
        // falsy -> 'default'; here mode_or_default handles the '' case.
        exit_plan_mode(); // no-op when not in plan mode
        assert!(!is_plan_mode_active());
        crate::state::app_state::reset_app_state();
    }
}
