//! NotebookEdit Tool for CodeMason
//! Equivalent to codemason/tools/notebook_edit_tool.py — based on
//! src/tools/NotebookEditTool/ from Claude Code.
//!
//! Schema, description, prompt, validation (abs path, .ipynb, edit modes,
//! read-before-edit, cell-id resolution) and the cell replace/insert/delete
//! logic are ported verbatim. Read-before-edit uses
//! ToolUseContext.read_file_state (populated by Read).

use async_trait::async_trait;
use once_cell::sync::Lazy;
use serde_json::{json, Value};

use crate::tools::base::{CanUseToolFn, Tool, ToolResult, ToolStream};
use crate::types::query::{ReadFileState, ToolUseContext, ValidationResult};
use crate::utils::formatting::{colorize, colors};

use super::file_read_tool::{embedded_tool_schema, ext_of, mtime_seconds, resolve_path};

pub const NOTEBOOK_EDIT_TOOL_NAME: &str = "NotebookEdit";

/// Lines of the new cell source echoed at tool-use time (verbose: all).
const PREVIEW_LINES: usize = 10;

pub const DESCRIPTION: &str =
    "Replace the contents of a specific cell in a Jupyter notebook.";

pub const PROMPT: &str = "Completely replaces the contents of a specific cell in a Jupyter notebook (.ipynb file) with new source. Jupyter notebooks are interactive documents that combine code, text, and visualizations, commonly used for data analysis and scientific computing. The notebook_path parameter must be an absolute path, not a relative path. The cell_number is 0-indexed. Use edit_mode=insert to add a new cell at the index specified by cell_number. Use edit_mode=delete to delete the cell at the index specified by cell_number.";

/// Detect 'CRLF' vs 'LF' by majority (utils/fileRead.detectLineEndingsForString).
fn detect_line_endings_str(content: &str) -> &'static str {
    let mut crlf_count = 0usize;
    let mut lf_count = 0usize;
    let chars: Vec<char> = content.chars().collect();
    for (i, ch) in chars.iter().enumerate() {
        if *ch == '\n' {
            if i > 0 && chars[i - 1] == '\r' {
                crlf_count += 1;
            } else {
                lf_count += 1;
            }
        }
    }
    if crlf_count > lf_count {
        "CRLF"
    } else {
        "LF"
    }
}

/// Parse a 'cell-N' style id to its numeric index (utils/notebook.parseCellId).
fn parse_cell_id(cell_id: &str) -> Option<i64> {
    static RE: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"^cell-(\d+)$").unwrap());
    RE.captures(cell_id).and_then(|c| c[1].parse().ok())
}

/// Python json.dumps(notebook, indent=1, ensure_ascii=False): pretty JSON with
/// a ONE-space indent (serde's default pretty is two).
fn dumps_indent1(v: &Value) -> String {
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, formatter);
    use serde::Serialize;
    if v.serialize(&mut ser).is_ok() {
        String::from_utf8_lossy(&out).into_owned()
    } else {
        v.to_string()
    }
}

/// Edit Jupyter notebook cells (.ipynb).
pub struct NotebookEditTool;

fn s_of<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(|v| v.as_str()).filter(|s| !s.is_empty())
}

#[async_trait]
impl Tool for NotebookEditTool {
    fn name(&self) -> &'static str {
        NOTEBOOK_EDIT_TOOL_NAME
    }

    fn input_json_schema(&self) -> Value {
        embedded_tool_schema(NOTEBOOK_EDIT_TOOL_NAME)
    }

    /// Full notebook prompt (the Python class attr DESCRIPTION = PROMPT; the
    /// original API sends tool.prompt()).
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
        "Edit Notebook".to_string()
    }

    /// Cell location + the new source (NotebookEditTool/UI.tsx).
    /// The display path (UI.tsx getToolUseSummary).
    fn get_tool_use_summary(&self, input: Option<&Value>) -> Option<String> {
        let value = input?.get("notebook_path")?.as_str()?;
        if value.is_empty() {
            return None;
        }
        Some(crate::utils::file::get_display_path(value))
    }

    /// "Editing notebook <path>" (getActivityDescription).
    fn get_activity_description(&self, input: Option<&Value>) -> Option<String> {
        Some(match self.get_tool_use_summary(input) {
            Some(summary) => format!("Editing notebook {summary}"),
            None => "Editing notebook".to_string(),
        })
    }

    fn render_tool_use_message(&self, input: &Value, verbose: bool) -> String {
        let Some(path) = s_of(input, "notebook_path") else {
            return String::new();
        };
        let mut location = path.to_string();
        if let Some(cell_id) = s_of(input, "cell_id") {
            location.push_str(&format!(" · cell {cell_id}"));
        }
        let edit_mode = s_of(input, "edit_mode").unwrap_or("replace");
        if edit_mode != "replace" {
            location.push_str(&format!(" · {edit_mode}"));
        }
        if let Some(cell_type) = s_of(input, "cell_type") {
            location.push_str(&format!(" · {cell_type}"));
        }

        let mut lines = vec![colorize(&location, colors::DIM)];
        let source_text = input.get("new_source").and_then(|v| v.as_str()).unwrap_or("");
        let source: Vec<&str> = source_text.split('\n').collect();
        let shown: Vec<&str> = if verbose {
            source.clone()
        } else {
            source.iter().take(PREVIEW_LINES).copied().collect()
        };
        for line in &shown {
            lines.push(line.chars().take(100).collect());
        }
        if source.len() > shown.len() {
            lines.push(colorize(
                &format!("... ({} more lines)", source.len() - shown.len()),
                colors::DIM,
            ));
        }
        lines.join("\n")
    }

    async fn validate_input(&self, input: &Value, context: &ToolUseContext) -> ValidationResult {
        let notebook_path = input.get("notebook_path").and_then(|v| v.as_str()).unwrap_or("");
        let cell_type = s_of(input, "cell_type");
        let cell_id = s_of(input, "cell_id");
        let edit_mode = s_of(input, "edit_mode").unwrap_or("replace");
        let full_path = resolve_path(notebook_path);

        if full_path.starts_with("\\\\") || full_path.starts_with("//") {
            return ValidationResult::success();
        }

        if ext_of(&full_path) != "ipynb" {
            return ValidationResult::failure(
                "File must be a Jupyter notebook (.ipynb file). For editing other file types, use the FileEdit tool.",
            );
        }

        if !matches!(edit_mode, "replace" | "insert" | "delete") {
            return ValidationResult::failure("Edit mode must be replace, insert, or delete.");
        }

        if edit_mode == "insert" && cell_type.is_none() {
            return ValidationResult::failure("Cell type is required when using edit_mode=insert.");
        }

        // Read-before-edit + partial-view rejection (matches FileEditTool/
        // FileWriteTool). Notebooks are JSON, so no text content-equality
        // fallback applies — the mtime check is authoritative.
        {
            let read_state = context.read_file_state.lock().get(&full_path).cloned();
            let Some(read_state) = read_state.filter(|s| !s.is_partial_view) else {
                return ValidationResult::failure(
                    "File has not been read yet. Read it first before writing to it.",
                );
            };
            if let Some(mtime) = mtime_seconds(&full_path) {
                if mtime > read_state.timestamp {
                    return ValidationResult::failure(
                        "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.",
                    );
                }
            }
        }

        // STRICT, not lossy. The edit path re-serializes whatever it decoded
        // and writes it back, so a replaced byte is not a display glitch — it is
        // permanent loss of the original file's content, the same way the Edit
        // tool used to destroy CP949 sources. A notebook is JSON and JSON is
        // UTF-8, so anything else is a malformed file we refuse rather than
        // silently rewrite.
        let content = match std::fs::read(&full_path) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => {
                    return ValidationResult::failure(
                        "Notebook is not valid UTF-8 (JSON must be UTF-8); editing it would replace the undecodable bytes and corrupt the file.",
                    )
                }
            },
            Err(_) => return ValidationResult::failure("Notebook file does not exist."),
        };

        let notebook: Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return ValidationResult::failure("Notebook is not valid JSON."),
        };

        let cells = notebook.get("cells").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        match cell_id {
            None => {
                if edit_mode != "insert" {
                    return ValidationResult::failure(
                        "Cell ID must be specified when not inserting a new cell.",
                    );
                }
            }
            Some(cell_id) => {
                let idx = cells
                    .iter()
                    .position(|c| c.get("id").and_then(|v| v.as_str()) == Some(cell_id));
                if idx.is_none() {
                    match parse_cell_id(cell_id) {
                        Some(parsed) => {
                            if parsed < 0 || parsed as usize >= cells.len() {
                                return ValidationResult::failure(format!(
                                    "Cell with index {parsed} does not exist in notebook."
                                ));
                            }
                        }
                        None => {
                            return ValidationResult::failure(format!(
                                "Cell with ID \"{cell_id}\" not found in notebook."
                            ));
                        }
                    }
                }
            }
        }

        ValidationResult::success()
    }

    fn call(
        &self,
        input: Value,
        context: ToolUseContext,
        _can_use_tool: Option<CanUseToolFn>,
    ) -> ToolStream {
        Box::pin(async_stream::stream! {
            let notebook_path = input.get("notebook_path").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let new_source = input.get("new_source").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let cell_id = s_of(&input, "cell_id").map(|s| s.to_string());
            let mut cell_type = s_of(&input, "cell_type").map(|s| s.to_string());
            let mut edit_mode = s_of(&input, "edit_mode").unwrap_or("replace").to_string();
            let full_path = resolve_path(&notebook_path);

            let outcome: Result<ToolResult, String> = (|| {
                // Read raw to detect the original line-ending style (CRLF vs
                // LF) before parsing, so an edit writes it back unchanged
                // instead of silently normalizing CRLF notebooks to LF.
                // A notebook is JSON, so it IS UTF-8 — anything else was refused
                // in validate_input.
                let raw_bytes = std::fs::read(&full_path).map_err(|e| e.to_string())?;
                // Strict as well — validate_input already refused a non-UTF-8
                // notebook, and if that ever stops being true this must fail
                // loudly rather than write U+FFFD over the user's file.
                let raw = String::from_utf8(raw_bytes).map_err(|_| {
                    "Notebook is not valid UTF-8 (JSON must be UTF-8); editing it would replace the undecodable bytes and corrupt the file.".to_string()
                })?;
                let head: String = raw.chars().take(4096).collect();
                let line_endings = detect_line_endings_str(&head);
                let content = raw.replace("\r\n", "\n");
                let mut notebook: Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;
                if notebook.get("cells").map(|c| !c.is_array()).unwrap_or(true) {
                    notebook["cells"] = json!([]);
                }

                // Resolve target index
                let cells_len = notebook["cells"].as_array().map(|a| a.len()).unwrap_or(0);
                let mut cell_index: i64 = match &cell_id {
                    None => 0,
                    Some(cid) => {
                        let mut idx = notebook["cells"]
                            .as_array()
                            .and_then(|cells| {
                                cells.iter().position(|c| {
                                    c.get("id").and_then(|v| v.as_str()) == Some(cid.as_str())
                                })
                            })
                            .map(|i| i as i64)
                            .unwrap_or(-1);
                        if idx == -1 {
                            if let Some(parsed) = parse_cell_id(cid) {
                                idx = parsed;
                            }
                        }
                        if edit_mode == "insert" {
                            idx += 1;
                        }
                        idx
                    }
                };

                // Replace one-past-the-end becomes insert
                if edit_mode == "replace" && cell_index == cells_len as i64 {
                    edit_mode = "insert".to_string();
                    if cell_type.is_none() {
                        cell_type = Some("code".to_string());
                    }
                }
                let _ = &mut cell_index;

                let nbformat = notebook.get("nbformat").and_then(|v| v.as_i64()).unwrap_or(4);
                let nbformat_minor =
                    notebook.get("nbformat_minor").and_then(|v| v.as_i64()).unwrap_or(0);
                let new_cell_id: Option<String> =
                    if nbformat > 4 || (nbformat == 4 && nbformat_minor >= 5) {
                        if edit_mode == "insert" {
                            let hex = uuid::Uuid::new_v4().simple().to_string();
                            Some(hex.chars().take(13).collect())
                        } else {
                            cell_id.clone()
                        }
                    } else {
                        None
                    };

                let cells = notebook["cells"].as_array_mut().ok_or("cells is not an array")?;

                if edit_mode == "delete" {
                    if cell_index >= 0 && (cell_index as usize) < cells.len() {
                        cells.remove(cell_index as usize);
                    }
                } else if edit_mode == "insert" {
                    let mut new_cell = if cell_type.as_deref() == Some("markdown") {
                        json!({"cell_type": "markdown", "source": new_source, "metadata": {}})
                    } else {
                        json!({"cell_type": "code", "source": new_source,
                               "metadata": {}, "execution_count": null, "outputs": []})
                    };
                    // nbformat < 4.5 cells carry NO `id` field; only add it when
                    // the format supports ids (writing "id": null produced an
                    // off-spec cell).
                    if let Some(ncid) = &new_cell_id {
                        new_cell["id"] = json!(ncid);
                    }
                    let at = cell_index.clamp(0, cells.len() as i64) as usize;
                    cells.insert(at, new_cell);
                } else {
                    // Guard against an unresolved cell_id (cell_index == -1):
                    // without this, a negative index would silently edit the
                    // wrong cell. validate_input rejects such ids today, but
                    // call() must not depend on that.
                    if !(cell_index >= 0 && (cell_index as usize) < cells.len()) {
                        return Err(format!(
                            "cell_id {:?} did not resolve to a valid cell to replace",
                            cell_id.as_deref().unwrap_or("None")
                        ));
                    }
                    let target = &mut cells[cell_index as usize];
                    target["source"] = json!(new_source);
                    if target.get("cell_type").and_then(|v| v.as_str()) == Some("code") {
                        target["execution_count"] = Value::Null;
                        target["outputs"] = json!([]);
                    }
                    if let Some(ct) = &cell_type {
                        if target.get("cell_type").and_then(|v| v.as_str()) != Some(ct.as_str()) {
                            target["cell_type"] = json!(ct);
                        }
                    }
                }

                let mut updated = dumps_indent1(&notebook);
                if line_endings == "CRLF" {
                    // Normalize any CRLF first so the join doesn't produce
                    // \r\r\n, then restore CRLF (matches writeTextContent).
                    updated = updated.replace("\r\n", "\n").replace('\n', "\r\n");
                }
                crate::utils::file::atomic_write_bytes(&full_path, updated.as_bytes())
                    .map_err(|e| e.to_string())?;

                // Refresh readFileState post-write (matches the original).
                if let Some(mtime) = mtime_seconds(&full_path) {
                    context.read_file_state.lock().insert(full_path.clone(), ReadFileState {
                        timestamp: mtime,
                        content: String::new(),
                        offset: None,
                        limit: None,
                        is_partial_view: false,
                    });
                }

                // Python f-string prints None for a missing id.
                let disp_id = new_cell_id
                    .or(cell_id.clone())
                    .unwrap_or_else(|| "None".to_string());
                let result_text = if edit_mode == "delete" {
                    format!("Deleted cell {disp_id}")
                } else if edit_mode == "insert" {
                    format!("Inserted cell {disp_id} with {new_source}")
                } else {
                    format!("Updated cell {disp_id} with {new_source}")
                };

                Ok(ToolResult::Result {
                    result_for_assistant: result_text,
                    data: Some(json!({"notebook_path": full_path, "edit_mode": edit_mode,
                                      "cell_id": disp_id})),
                })
            })();

            match outcome {
                Ok(result) => yield Ok(result),
                Err(e) => {
                    // Python catches everything here and reports as a normal
                    // (non-raised) result, unlike Edit/Write.
                    yield Ok(ToolResult::Result {
                        result_for_assistant: format!("Error editing notebook: {e}"),
                        data: Some(json!({"error": e})),
                    });
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    fn notebook_json() -> String {
        dumps_indent1(&json!({
            "cells": [
                {"id": "aaa", "cell_type": "code", "source": "print(1)",
                 "metadata": {}, "execution_count": 1, "outputs": []},
                {"id": "bbb", "cell_type": "markdown", "source": "# hi", "metadata": {}}
            ],
            "metadata": {"language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5
        }))
    }

    fn setup(tag: &str) -> (std::path::PathBuf, String, ToolUseContext) {
        let dir = std::env::temp_dir().join(format!("cm-nb-{tag}-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("n.ipynb");
        std::fs::write(&path, notebook_json()).unwrap();
        let p = path.to_string_lossy().into_owned();
        let ctx = ToolUseContext::default();
        ctx.read_file_state.lock().insert(p.clone(), ReadFileState {
            timestamp: f64::MAX,
            content: String::new(),
            offset: Some(1),
            limit: None,
            is_partial_view: false,
        });
        (dir, p, ctx)
    }

    #[tokio::test]
    async fn replace_cell_clears_outputs() {
        let (dir, p, ctx) = setup("replace");
        let tool = NotebookEditTool;
        let mut s = tool.call(
            json!({"notebook_path": p, "cell_id": "aaa", "new_source": "print(2)"}),
            ctx,
            None,
        );
        let r = s.next().await.unwrap().unwrap();
        if let ToolResult::Result { result_for_assistant, .. } = r {
            assert_eq!(result_for_assistant, "Updated cell aaa with print(2)");
        } else {
            panic!();
        }
        let nb: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(nb["cells"][0]["source"], "print(2)");
        assert!(nb["cells"][0]["execution_count"].is_null());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn insert_after_cell_and_delete() {
        let (dir, p, ctx) = setup("insert");
        let tool = NotebookEditTool;
        let mut s = tool.call(
            json!({"notebook_path": p, "cell_id": "aaa", "new_source": "x=1",
                   "edit_mode": "insert", "cell_type": "code"}),
            ctx.clone(),
            None,
        );
        let _ = s.next().await.unwrap().unwrap();
        let nb: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(nb["cells"].as_array().unwrap().len(), 3);
        assert_eq!(nb["cells"][1]["source"], "x=1");
        assert!(nb["cells"][1]["id"].is_string());

        // refresh read state after external mtime change
        ctx.read_file_state.lock().get_mut(&p).unwrap().timestamp = f64::MAX;
        let mut s2 = tool.call(
            json!({"notebook_path": p, "cell_id": "cell-1", "new_source": "",
                   "edit_mode": "delete"}),
            ctx,
            None,
        );
        let _ = s2.next().await.unwrap().unwrap();
        let nb2: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(nb2["cells"].as_array().unwrap().len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn validate_rejects_unknown_id_and_non_ipynb() {
        let (dir, p, ctx) = setup("validate");
        let tool = NotebookEditTool;
        let v = tool
            .validate_input(&json!({"notebook_path": p, "cell_id": "zzz", "new_source": "x"}), &ctx)
            .await;
        assert!(!v.result);
        assert_eq!(v.message, "Cell with ID \"zzz\" not found in notebook.");
        let v2 = tool
            .validate_input(&json!({"notebook_path": "/tmp/x.py", "new_source": "x"}), &ctx)
            .await;
        assert!(!v2.result);
        assert!(v2.message.starts_with("File must be a Jupyter notebook"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn indent1_matches_python_shape() {
        let s = dumps_indent1(&json!({"a": [1, 2], "b": {"c": null}}));
        assert_eq!(s, "{\n \"a\": [\n  1,\n  2\n ],\n \"b\": {\n  \"c\": null\n }\n}");
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
        let expected = &dump["NotebookEdit"]["schema"];
        let actual = NotebookEditTool.input_json_schema();
        assert_eq!(&actual, expected, "schema for NotebookEdit diverges from the Python dump");
    }
}
