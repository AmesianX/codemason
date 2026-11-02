//! vLLM service — OpenAI-compatible vLLM server variant of the model service.
//! Equivalent to codemason/services/vllm.py.
//!
//! All OpenAI-compatible logic lives in services/openai_common.rs. llama.cpp
//! and vLLM differ by two request fields, both pinned on here:
//!
//!   - stream_options.continuous_usage_stats — a running completion_tokens on
//!     every chunk, so the live spinner shows the server's exact count instead
//!     of a per-chunk tally.
//!   - repetition_detection — vLLM's own n-gram loop detector, which ends a
//!     generation that has begun repeating itself and says so in
//!     finish_reason. llama.cpp has no equivalent field.
//!
//! Otherwise this shim delegates everything.
//!
//! Selected at runtime by services/backend.rs when the server is vLLM.

use crate::types::query::AssistantMessage;

use super::openai_common::{query_openai_compatible, QueryModelParams};

// Back-compat re-exports (see llamacpp.rs for the rationale).
#[allow(unused_imports)]
pub use super::openai_common::{
    build_system_prompt_with_cwd_emphasis, clean_special_tokens, convert_to_chat_messages,
    get_simple_description, recover_leaked_tool_calls, truncate_tool_result, ChatMessage,
};

/// Query a vLLM server. Enables continuous_usage_stats so the live token count
/// is driven by the server's exact running completion_tokens (counting both
/// answer and reasoning deltas) rather than a per-chunk tally.
///
/// services/backend.rs dispatches here when the server is vLLM; the neutral
/// public entry point is services::backend::query_model.
pub async fn query_vllm(params: QueryModelParams) -> AssistantMessage {
    query_openai_compatible(
        params, /*continuous_usage_stats=*/ true, /*repetition_detection=*/ true,
        /*reasoning_roundtrip=*/ true,
        // Images go out as image_url parts only when the server says it takes
        // them, and never more than it takes (asked once, cached —
        // services/vision.rs).
        /*max_images=*/ super::vision::enabled_limit().await,
    )
    .await
}
