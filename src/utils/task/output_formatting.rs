//! Task output formatting — adapted from src/utils/task/outputFormatting.ts.
//! Equivalent to codemason/utils/task/output_formatting.py.
//!
//! Truncates task output for API consumption. When too large, returns the
//! LAST N chars that fit, prefixed with a header naming the full-output file
//! path.
//!
//! (Written by cluster TM — the file was still a stub and TaskOutput needs
//! format_task_output; the module is small enough that the "minimal
//! extension" is the faithful whole. Cluster TB may overwrite freely.)

use super::disk_output::get_task_output_path;

pub const TASK_MAX_OUTPUT_UPPER_LIMIT: usize = 160_000;
pub const TASK_MAX_OUTPUT_DEFAULT: usize = 32_000;

/// Port of validateBoundedIntEnvVar (envValidation.ts): parse a positive int
/// from an env value, fall back to `default` on missing/invalid/<=0, and cap
/// at `upper_limit`.
fn validate_bounded_int_env_var(
    value: Option<String>,
    default: usize,
    upper_limit: usize,
) -> usize {
    let Some(value) = value else {
        return default;
    };
    if value.is_empty() {
        return default;
    }
    let Ok(parsed) = value.trim().parse::<i64>() else {
        return default;
    };
    if parsed <= 0 {
        return default;
    }
    let parsed = parsed as usize;
    if parsed > upper_limit {
        return upper_limit;
    }
    parsed
}

/// Effective max output length (getMaxTaskOutputLength).
pub fn get_max_task_output_length() -> usize {
    validate_bounded_int_env_var(
        std::env::var("TASK_MAX_OUTPUT_LENGTH").ok(),
        TASK_MAX_OUTPUT_DEFAULT,
        TASK_MAX_OUTPUT_UPPER_LIMIT,
    )
}

pub struct FormattedTaskOutput {
    pub content: String,
    pub was_truncated: bool,
}

/// Truncate output if too large (formatTaskOutput).
///
/// When truncated, the header names the full-output file path and the
/// returned content is the last chars that fit. Char-based like the Python
/// (len() counts chars, output[-n:] slices chars).
pub fn format_task_output(output: &str, task_id: &str) -> FormattedTaskOutput {
    let max_len = get_max_task_output_length();

    let char_count = output.chars().count();
    if char_count <= max_len {
        return FormattedTaskOutput { content: output.to_string(), was_truncated: false };
    }

    let file_path = get_task_output_path(task_id);
    let header = format!("[Truncated. Full output: {}]\n\n", file_path.to_string_lossy());
    // Clamp to >= 0: a very small TASK_MAX_OUTPUT_LENGTH could make this
    // negative, and a negative tail slice would return the FRONT of the output
    // (the source has this same latent slice bug). 0 → just the header.
    let header_chars = header.chars().count();
    let available_space = max_len.saturating_sub(header_chars);
    let truncated: String = if available_space > 0 {
        output.chars().skip(char_count - available_space.min(char_count)).collect()
    } else {
        String::new()
    };

    FormattedTaskOutput { content: format!("{header}{truncated}"), was_truncated: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_output_untouched() {
        let r = format_task_output("hello", "t1");
        assert!(!r.was_truncated);
        assert_eq!(r.content, "hello");
    }

    #[test]
    fn env_bounds() {
        assert_eq!(
            validate_bounded_int_env_var(None, TASK_MAX_OUTPUT_DEFAULT, TASK_MAX_OUTPUT_UPPER_LIMIT),
            TASK_MAX_OUTPUT_DEFAULT
        );
        assert_eq!(
            validate_bounded_int_env_var(Some("-5".into()), 100, 200),
            100
        );
        assert_eq!(validate_bounded_int_env_var(Some("999999".into()), 100, 200), 200);
        assert_eq!(validate_bounded_int_env_var(Some("junk".into()), 100, 200), 100);
    }

    #[test]
    fn truncation_keeps_tail_and_names_file() {
        std::env::set_var("TASK_MAX_OUTPUT_LENGTH", "100");
        let long: String = "x".repeat(50) + &"y".repeat(100);
        let r = format_task_output(&long, "tabc");
        std::env::remove_var("TASK_MAX_OUTPUT_LENGTH");
        assert!(r.was_truncated);
        assert!(r.content.starts_with("[Truncated. Full output: "));
        assert!(r.content.ends_with('y'));
        // header + tail fits the cap
        assert!(r.content.chars().count() <= 100);
    }
}
