//! FileWrite Tool for CodeMason
//! Equivalent to codemason/tools/file_write_tool.py — based on
//! src/tools/FileWriteTool/ from Claude Code.
//!
//! Tool name is 'Write'. Schema, description, and prompt are ported verbatim.
//! Behavior reproduced: overwrite/create file, read-before-edit requirement
//! for EXISTING files, parent directory creation, and the modified-since-read
//! staleness check (with the full-read content-equality fallback). The read
//! state is updated after writing.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::tools::base::{CanUseToolFn, Tool, ToolError, ToolResult, ToolStream};
use crate::types::query::{ReadFileState, ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors};

use super::file_read_tool::{embedded_tool_schema, mtime_seconds, resolve_path};

pub const FILE_WRITE_TOOL_NAME: &str = "Write";

/// Ported verbatim from prompt.ts (DESCRIPTION) and FileWriteTool.ts description().
pub const DESCRIPTION: &str = "Write a file to the local filesystem.";

/// Ported verbatim from prompt.ts getWriteToolDescription().
pub const PROMPT: &str = "Writes a file to the local filesystem.\n\nUsage:\n- This tool will overwrite the existing file if there is one at the provided path.\n- If this is an existing file, you MUST use the Read tool first to read the file's contents. This tool will fail if you did not read the file first.\n- Prefer the Edit tool for modifying existing files — it only sends the diff. Only use this tool to create new files or for complete rewrites.\n- NEVER create documentation files (*.md) or README files unless explicitly requested by the User.\n- Only use emojis if the user explicitly requests it. Avoid writing emojis to files unless asked.";

/// Lines of new content echoed under the path at tool-use time (verbose: all).
const PREVIEW_LINES: usize = 10;

/// Python `content.encode(encoding)` for the encodings detect_file_encoding
/// can return. utf-16-le/-be add no BOM of their own — but decode_bytes keeps
/// the file's U+FEFF in the content (Python codec parity), so a BOM'd file
/// round-trips its BOM through here. utf-8-sig PREPENDS one.
/// latin-1 raises on non-Latin-1 chars in Python; here those become an Err.
pub(crate) fn encode_with_encoding(content: &str, encoding: &str) -> Result<Vec<u8>, String> {
    match encoding {
        "utf-16-le" => Ok(content.encode_utf16().flat_map(|u| u.to_le_bytes()).collect()),
        "utf-16-be" => Ok(content.encode_utf16().flat_map(|u| u.to_be_bytes()).collect()),
        "utf-8-sig" => {
            let mut out = vec![0xEF, 0xBB, 0xBF];
            out.extend_from_slice(content.as_bytes());
            Ok(out)
        }
        "latin-1" => content
            .chars()
            .map(|c| {
                let cp = c as u32;
                if cp <= 0xFF {
                    Ok(cp as u8)
                } else {
                    Err(format!(
                        "'latin-1' codec can't encode character {c:?}"
                    ))
                }
            })
            .collect(),
        // Mirror of the read path: a legacy label is only in play because the
        // detector proved decode-then-encode is lossless for this file, so the
        // write-back reproduces the original bytes outside the edit.
        other => match encoding_rs::Encoding::for_label(other.as_bytes()) {
            Some(enc) if !other.eq_ignore_ascii_case("utf-8") => {
                let (bytes, _, had_errors) = enc.encode(content);
                if had_errors {
                    return Err(format!("'{other}' codec can't encode this content"));
                }
                Ok(bytes.into_owned())
            }
            _ => Ok(content.as_bytes().to_vec()),
        },
    }
}

/// Write a file to the local filesystem (create or overwrite).
pub struct FileWriteTool;

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &'static str {
        FILE_WRITE_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        embedded_tool_schema(FILE_WRITE_TOOL_NAME)
    }

    /// The API description is the FULL write prompt (the Python class attr
    /// DESCRIPTION = PROMPT; the original API sends tool.prompt()).
    async fn description(&self, _input: &Value) -> String {
        PROMPT.to_string()
    }

    async fn prompt(&self) -> String {
        PROMPT.to_string()
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn needs_permissions(&self) -> bool {
        true
    }

    fn user_facing_name(&self, _input: Option<&Value>) -> String {
        "Write".to_string()
    }

    async fn validate_input(&self, input: &Value, context: &ToolUseContext) -> ValidationResult {
        let file_path = input.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
        let full_file_path = resolve_path(file_path);

        // SECURITY: Skip filesystem operations for UNC paths to prevent NTLM
        // credential leaks. Let the permission check handle UNC paths.
        if full_file_path.starts_with("\\\\") || full_file_path.starts_with("//") {
            return ValidationResult::success();
        }

        // Stat the file. ENOENT (new file) => no read-before-edit requirement.
        let Some(file_mtime_s) = mtime_seconds(&full_file_path) else {
            return ValidationResult::success();
        };
        let file_mtime_ms = file_mtime_s * 1000.0;

        // Read-before-edit for EXISTING files + partial-view rejection
        // (FileWriteTool.ts:198-206).
        let read_state = context.read_file_state.lock().get(&full_file_path).cloned();
        let Some(read_state) = read_state.filter(|s| !s.is_partial_view) else {
            return ValidationResult::failure(
                "File has not been read yet. Read it first before writing to it.",
            );
        };

        // Modified-since-read: mtime bump, with the SAME content-equality
        // fallback Edit uses — otherwise Write spuriously blocked on a
        // linter/formatter/sync touch that left the content identical.
        // timestamp is stored in seconds, so compare in ms.
        let last_write_time = file_mtime_ms.floor();
        if last_write_time > (read_state.timestamp * 1000.0).floor() {
            let mut content_unchanged = false;
            if read_state.is_full_read() {
                if let Ok(bytes) = std::fs::read(&full_file_path) {
                    let on_disk = String::from_utf8_lossy(&bytes).into_owned();
                    content_unchanged = on_disk == read_state.content;
                }
            }
            if !content_unchanged {
                return ValidationResult::failure(
                    "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.",
                );
            }
        }

        ValidationResult::success()
    }

    /// Path + a preview of what is about to be written (FileWriteTool/UI.tsx).
    /// The display path (UI.tsx getToolUseSummary).
    fn get_tool_use_summary(&self, input: Option<&Value>) -> Option<String> {
        let value = input?.get("file_path")?.as_str()?;
        if value.is_empty() {
            return None;
        }
        Some(crate::utils::file::get_display_path(value))
    }

    /// "Writing <path>" (getActivityDescription).
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        Some(match self.get_tool_use_summary(input) {
            Some(summary) => format!("Writing {summary}"),
            None => "Writing file".to_string(),
        })
    }

    fn render_tool_use_message(&self, input: &Value, verbose: bool) -> String {
        let file_path = input.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
        let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");
        if file_path.is_empty() {
            return String::new();
        }
        let mut lines = vec![colorize(file_path, colors::DIM)];
        let content_lines: Vec<&str> = content.split('\n').collect();
        let shown: Vec<&str> = if verbose {
            content_lines.clone()
        } else {
            content_lines.iter().take(PREVIEW_LINES).copied().collect()
        };
        for line in &shown {
            // Mark a cut line. Only the LINE-count overflow was annotated
            // below, so a line trimmed at 100 chars looked complete: 99 benign
            // characters followed by anything at all showed as clean, and this
            // rendering is what the user approves.
            if line.chars().count() > 100 {
                let head: String = line.chars().take(100).collect();
                lines.push(format!("{head} …"));
            } else {
                lines.push(line.to_string());
            }
        }
        if content_lines.len() > shown.len() {
            let hidden = content_lines.len() - shown.len();
            lines.push(colorize(&format!("... ({hidden} more lines)"), colors::DIM));
        }
        lines.join("\n")
    }

    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let file_path = input.get("file_path").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let full_file_path = resolve_path(&file_path);
            let dir_name = std::path::Path::new(&full_file_path)
                .parent()
                .map(|p| p.to_path_buf());

            // Faithful: write failures (modified-since-read, I/O errors)
            // PROPAGATE as a tool error — the original throws. The query loop
            // turns it into an is_error tool_result.
            let outcome: Result<ToolResult, String> = (|| {
                // Ensure parent directory exists before the read-modify-write section.
                if let Some(dir) = &dir_name {
                    if !dir.as_os_str().is_empty() {
                        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                    }
                }

                // Load current state. ENOENT => new file (old_content is None).
                let old_content: Option<String> = match std::fs::read(&full_file_path) {
                    Ok(bytes) => Some(String::from_utf8_lossy(&bytes).into_owned()),
                    Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => None,
                    Err(e) => return Err(e.to_string()),
                };

                // Staleness check: confirm no changes since last read, with the
                // full-read content-equality fallback (FileWriteTool.ts:279-294).
                if let Some(old) = &old_content {
                    let last_read = context.read_file_state.lock().get(&full_file_path).cloned();
                    let last_write_time = mtime_seconds(&full_file_path);
                    let stale = match (&last_read, last_write_time) {
                        (None, _) => true,
                        (Some(lr), Some(lw)) => lw > lr.timestamp,
                        (Some(_), None) => false,
                    };
                    if stale {
                        let content_unchanged = last_read
                            .as_ref()
                            .map(|lr| lr.is_full_read() && *old == lr.content)
                            .unwrap_or(false);
                        if !content_unchanged {
                            return Err(
                                "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it."
                                    .to_string(),
                            );
                        }
                    }
                }

                // Write is a full content replacement — the model's line endings
                // are meant; do not rewrite them. But the existing file's
                // ENCODING is preserved on overwrite (e.g. UTF-16LE), matching
                // Edit — plain utf-8 would silently transcode the file.
                let write_encoding = if std::path::Path::new(&full_file_path).exists() {
                    crate::utils::file::detect_file_encoding(&full_file_path)
                } else {
                    "utf-8".to_string()
                };
                // Re-attach the UTF-16 BOM the model never saw. decode_bytes
                // KEEPS a file's U+FEFF in the string, which is what lets Edit
                // round-trip it — but Read strips it before showing the text
                // (file_read_tool.rs), so a full replacement arrives here with
                // none, and encode_with_encoding adds none of its own. The file
                // came out BOM-less and was then detected as utf-8 forever after
                // (NULs are valid UTF-8): it read back as NUL-laced garbage and
                // Grep skipped it as binary. Measured end to end. utf-8-sig
                // already prepends its own, so only the UTF-16 pair needs this.
                let content = if matches!(write_encoding.as_str(), "utf-16-le" | "utf-16-be")
                    && !content.starts_with('\u{feff}')
                {
                    format!("\u{feff}{content}")
                } else {
                    content
                };
                let bytes = encode_with_encoding(&content, &write_encoding)?;
                // Atomic (temp + rename) so a crash mid-write can't leave a
                // truncated file.
                crate::utils::file::atomic_write_bytes(&full_file_path, &bytes)
                    .map_err(|e| e.to_string())?;

                // Refresh readFileState post-write (full read: store the written
                // content for the equality fallback), to invalidate stale writes.
                if let Some(mtime) = mtime_seconds(&full_file_path) {
                    context.read_file_state.lock().insert(full_file_path.clone(), ReadFileState {
                        timestamp: mtime,
                        content: content.clone(),
                        offset: None,
                        limit: None,
                        is_partial_view: false,
                    });
                }

                let (data, result_msg) = if let Some(old) = old_content {
                    (
                        json!({
                            "type": "update",
                            "filePath": file_path,
                            "content": content,
                            "originalFile": old,
                        }),
                        format!("The file {file_path} has been updated successfully."),
                    )
                } else {
                    (
                        json!({
                            "type": "create",
                            "filePath": file_path,
                            "content": content,
                            "originalFile": null,
                        }),
                        format!("File created successfully at: {file_path}"),
                    )
                };

                Ok(ToolResult::Result { result_for_assistant: result_msg, data: Some(data) })
            })();

            match outcome {
                Ok(result) => yield Ok(result),
                Err(message) => {
                    crate::utils::log::log_error(&message, None);
                    yield Err(ToolError::new(message));
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("cm-write-test-{tag}-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        d
    }

    #[tokio::test]
    async fn create_new_file_no_read_needed() {
        let dir = tmpdir("create");
        let path = dir.join("new.txt").to_string_lossy().into_owned();
        let ctx = ToolUseContext::default();
        let tool = FileWriteTool;
        let v = tool
            .validate_input(&json!({"file_path": path, "content": "x"}), &ctx)
            .await;
        assert!(v.result);
        let mut s = tool.call(json!({"file_path": path, "content": "hello"}), ctx, None);
        let r = s.next().await.unwrap().unwrap();
        if let ToolResult::Result { result_for_assistant, .. } = r {
            assert!(result_for_assistant.starts_with("File created successfully at:"));
        } else {
            panic!();
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn existing_file_requires_read() {
        let dir = tmpdir("rbe");
        let path = dir.join("f.txt").to_string_lossy().into_owned();
        std::fs::write(&path, "old").unwrap();
        let ctx = ToolUseContext::default();
        let tool = FileWriteTool;
        let v = tool
            .validate_input(&json!({"file_path": path, "content": "x"}), &ctx)
            .await;
        assert!(!v.result);
        assert_eq!(
            v.message,
            "File has not been read yet. Read it first before writing to it."
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn modified_since_read_blocks_but_equal_content_passes() {
        let dir = tmpdir("msr");
        let path = dir.join("f.txt").to_string_lossy().into_owned();
        std::fs::write(&path, "same").unwrap();
        let ctx = ToolUseContext::default();
        // Simulate a FULL read recorded long ago (old timestamp) with matching content.
        ctx.read_file_state.lock().insert(path.clone(), ReadFileState {
            timestamp: 0.0,
            content: "same".into(),
            offset: None,
            limit: None,
            is_partial_view: false,
        });
        let tool = FileWriteTool;
        let v = tool.validate_input(&json!({"file_path": path, "content": "x"}), &ctx).await;
        assert!(v.result, "equal-content fallback should pass: {}", v.message);

        // Now with non-matching recorded content: blocked.
        ctx.read_file_state.lock().insert(path.clone(), ReadFileState {
            timestamp: 0.0,
            content: "different".into(),
            offset: None,
            limit: None,
            is_partial_view: false,
        });
        let v2 = tool.validate_input(&json!({"file_path": path, "content": "x"}), &ctx).await;
        assert!(!v2.result);
        assert!(v2.message.starts_with("File has been modified since read"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn encoding_roundtrip() {
        let b = encode_with_encoding("한글", "utf-16-le").unwrap();
        assert_eq!(super::super::file_read_tool::decode_bytes(&b, "utf-16-le"), "한글");
        let sig = encode_with_encoding("x", "utf-8-sig").unwrap();
        assert_eq!(&sig[..3], &[0xEF, 0xBB, 0xBF]);
        assert!(encode_with_encoding("한", "latin-1").is_err());
    }
}

#[cfg(test)]
mod schema_dump_equality {
    use super::*;
    use crate::tools::base::Tool as _;

    /// The input schema must be byte-equal to the pydantic model_json_schema()
    /// dump (schemas/tool_schemas.json) — the model-facing contract.
    #[test]
    fn schema_matches_python_dump() {
        let dump: serde_json::Value =
            serde_json::from_str(include_str!("../../schemas/tool_schemas.json")).unwrap();
        let expected = &dump["Write"]["schema"];
        let actual = FileWriteTool.input_json_schema();
        assert_eq!(&actual, expected, "schema for Write diverges from the Python dump");
    }
}
