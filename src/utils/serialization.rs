//! Shared JSON-serialization helper.
//! Equivalent to codemason/utils/serialization.py.
//!
//! In Python, `to_jsonable` reflected over dataclasses/pydantic models to
//! produce plain JSON. In Rust every message type already derives Serialize,
//! so the equivalent is serde_json::to_value with a lossy fallback — used by
//! print_runner (stream-json output) and session_storage (transcript JSONL).

use serde::Serialize;
use serde_json::Value;

/// Convert any Serialize into a plain JSON value; falls back to the Debug/
/// Display string form for values that fail to serialize (the Python str(v)
/// tail case).
pub fn to_jsonable<T: Serialize + std::fmt::Debug>(v: &T) -> Value {
    serde_json::to_value(v).unwrap_or_else(|_| Value::String(format!("{v:?}")))
}
