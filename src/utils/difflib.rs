//! Python difflib.unified_diff equivalent, over the `similar` crate (which
//! implements Python-difflib-compatible sequence matching).
//!
//! Returns the same line shapes Python's unified_diff(lineterm='') yields:
//! '--- ' / '+++ ' headers (empty names, as the Python callers pass none),
//! '@@ -a,b +c,d @@' hunk headers, then ' ctx' / '-old' / '+new' lines.

use similar::TextDiff;

pub fn unified_diff(old_lines: &[&str], new_lines: &[&str], context: usize) -> Vec<String> {
    let old_joined = old_lines.join("\n");
    let new_joined = new_lines.join("\n");
    let diff = TextDiff::configure().newline_terminated(false).diff_lines(&old_joined, &new_joined);

    let mut out: Vec<String> = Vec::new();
    let mut wrote_header = false;

    for hunk in diff.unified_diff().context_radius(context).iter_hunks() {
        if !wrote_header {
            out.push("--- ".to_string());
            out.push("+++ ".to_string());
            wrote_header = true;
        }
        out.push(hunk.header().to_string());
        for change in hunk.iter_changes() {
            let sign = match change.tag() {
                similar::ChangeTag::Delete => '-',
                similar::ChangeTag::Insert => '+',
                similar::ChangeTag::Equal => ' ',
            };
            let mut line = change.value().to_string();
            if line.ends_with('\n') {
                line.pop();
            }
            out.push(format!("{sign}{line}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_diff() {
        let old = ["a", "b", "c"];
        let new = ["a", "x", "c"];
        let d = unified_diff(&old, &new, 3);
        assert!(d.iter().any(|l| l == "-b"));
        assert!(d.iter().any(|l| l == "+x"));
        assert!(d.iter().any(|l| l.starts_with("@@ ")));
    }

    #[test]
    fn no_changes_is_empty() {
        let old = ["same"];
        let d = unified_diff(&old, &old, 3);
        assert!(d.is_empty());
    }
}
