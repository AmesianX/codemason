//! Environment Utilities for CodeMason
//! Equivalent to codemason/utils/env.py (src/utils/env.ts)
//!
//! NOTE: paths are NOT defined here. The single source of truth is config.rs
//! (CONFIG_DIR / MEMORY_DIR, derived from $CODEMASON_CONFIG_DIR). This module
//! only re-exports them and provides the ensure_* helpers.

use std::path::PathBuf;

pub use super::config::{CONFIG_DIR, MEMORY_DIR};

/// Ensure memory directory exists and return its path.
pub fn ensure_memory_dir() -> PathBuf {
    let _ = std::fs::create_dir_all(&*MEMORY_DIR);
    MEMORY_DIR.clone()
}

/// Ensure config directory exists and return its path.
pub fn ensure_config_dir() -> PathBuf {
    let _ = std::fs::create_dir_all(&*CONFIG_DIR);
    CONFIG_DIR.clone()
}
