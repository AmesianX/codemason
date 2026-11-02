//! LlamaCPP service — llama.cpp server variant of the model service.
//! Equivalent to codemason/services/llamacpp.py (src/services/llamacpp.ts).
//!
//! All OpenAI-compatible logic lives in services/openai_common.rs. llama.cpp
//! and vLLM differ by exactly one request field: vLLM adds
//! stream_options.continuous_usage_stats for a running per-chunk token count,
//! while llama.cpp emits usage only in the final chunk. This shim pins that
//! flag OFF and otherwise delegates everything.
//!
//! Selected at runtime by services/backend.rs when the server is llama.cpp.

use crate::types::query::AssistantMessage;

use super::openai_common::{query_openai_compatible, QueryModelParams};

// Back-compat re-exports. External code only calls query_model (via
// services::backend), but this module historically exposed these helpers, so
// keep them importable from here to avoid surprising any in-tree caller.
#[allow(unused_imports)]
pub use super::openai_common::{
    build_system_prompt_with_cwd_emphasis, clean_special_tokens, convert_to_chat_messages,
    get_simple_description, recover_leaked_tool_calls, truncate_tool_result, ChatMessage,
};

/// Query a llama.cpp server. Usage arrives only in the final stream chunk, so
/// continuous_usage_stats is left off (the live count uses a per-chunk tally
/// until the final exact count lands).
pub async fn query_llamacpp(params: QueryModelParams) -> AssistantMessage {
    query_openai_compatible(
        params, /*continuous_usage_stats=*/ false, /*repetition_detection=*/ false,
        /*reasoning_roundtrip=*/ false,
        /*max_images=*/ 0,
    )
    .await
}
