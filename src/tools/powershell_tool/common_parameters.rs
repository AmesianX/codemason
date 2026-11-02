//! PowerShell common parameters (commonParameters.ts).
//!
//! Available on all cmdlets via [CmdletBinding()] (about_CommonParameters +
//! Get-Command). Merged by path validation (per-cmdlet known-param sets) and
//! read-only validation (safe-flags check). Stored lowercase with leading dash
//! — callers lowercase input.

use once_cell::sync::Lazy;
use std::collections::HashSet;

pub const COMMON_SWITCHES: &[&str] = &["-verbose", "-debug"];

pub const COMMON_VALUE_PARAMS: &[&str] = &[
    "-erroraction",
    "-warningaction",
    "-informationaction",
    "-progressaction",
    "-errorvariable",
    "-warningvariable",
    "-informationvariable",
    "-outvariable",
    "-outbuffer",
    "-pipelinevariable",
];

pub static COMMON_PARAMETERS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    COMMON_SWITCHES
        .iter()
        .chain(COMMON_VALUE_PARAMS.iter())
        .copied()
        .collect()
});
