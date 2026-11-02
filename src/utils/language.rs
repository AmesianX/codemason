//! Which language the model should write in.
//! Equivalent to codemason/utils/language.py — see that file for the design
//! rationale (derived from the conversation script, config only as fallback).

use once_cell::sync::Lazy;
use parking_lot::Mutex;

/// Script ranges, checked in order; kana before Han so Japanese isn't read as
/// Chinese.
const SCRIPTS: &[(&str, &[(u32, u32)])] = &[
    ("Korean", &[(0xAC00, 0xD7A3), (0x1100, 0x11FF), (0x3130, 0x318F)]),
    ("Japanese", &[(0x3040, 0x309F), (0x30A0, 0x30FF)]),
    ("Chinese", &[(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]),
    ("Russian", &[(0x0400, 0x04FF)]),
];

/// A stray CJK identifier in an otherwise English question ("what is 中文
/// mode?") shouldn't flip the language. Require the script to carry real weight.
const MIN_SHARE: f64 = 0.10;
const MIN_CHARS: usize = 2;

/// Text in a message that the user did not write in any language: the label a
/// photo from the phone travels as (remote/poller.rs) and file paths/URLs. A
/// 64-hex attachment path is ~80 Latin letters, enough to sink a short Korean
/// caption under MIN_SHARE.
const PHOTO_LABELS: &[&str] = &["[Photo from the phone", "[A photo was sent from the phone"];

fn is_photo_label(line: &str) -> bool {
    line.ends_with(']')
        && PHOTO_LABELS.iter().any(|p| {
            line.starts_with(p)
                && !line[p.len()..].chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_')
        })
}

fn written_text(text: &str) -> String {
    text.lines()
        .filter(|l| !is_photo_label(l))
        .flat_map(str::split_whitespace)
        .filter(|w| !w.contains('/') && !w.contains('\\'))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Set per request from the conversation (services build the tool schemas and
/// the system prompt in the same pass).
static REQUEST_LANGUAGE: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

/// The language `text` is written in, or None for Latin script / no signal.
pub fn detect_language(text: &str) -> Option<&'static str> {
    if text.is_empty() {
        return None;
    }
    let letters: Vec<char> = written_text(text).chars().filter(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        return None;
    }
    for (name, ranges) in SCRIPTS {
        let hits = letters
            .iter()
            .filter(|c| {
                let cp = **c as u32;
                ranges.iter().any(|(lo, hi)| *lo <= cp && cp <= *hi)
            })
            .count();
        if hits < MIN_CHARS {
            continue;
        }
        // Kana is used ONLY by Japanese — Chinese never has it — so a couple of
        // kana chars are a definitive Japanese signal even in kanji-heavy text
        // whose kana share dips below MIN_SHARE.
        if *name == "Japanese" || (hits as f64 / letters.len() as f64) >= MIN_SHARE {
            return Some(name);
        }
    }
    None
}

pub fn set_request_language(language: Option<String>) {
    *REQUEST_LANGUAGE.lock() = language;
}

/// The language to instruct the model in.
///
/// The CONVERSATION decides; config.language is only a fallback for when the
/// text carries no script signal (Latin), where it still serves its upstream
/// purpose ("always answer in X").
pub fn effective_language() -> Option<String> {
    if let Some(l) = REQUEST_LANGUAGE.lock().clone() {
        return Some(l);
    }
    crate::utils::config::get_config().language
}

/// The language pinned for this SESSION — what the sub-agent and forked-skill
/// system prompts are built from.
///
/// They must not use effective_language(). That value follows the last
/// message of the conversation, which inside a tool loop is a tool RESULT, so
/// it swings with whatever a command printed. A sub-agent's system prompt is
/// built once per spawn, so a swing did not break the cache within one
/// sub-agent run — but it made every spawn's prefix (tool block + agent prompt,
/// ~20k tokens) a coin flip against the previous one, so consecutive
/// sub-agents could not share it. On a KV pool sized for 1.62 conversations
/// (measured 2026-09-07: 847,533 tokens) every needless cold prefill also
/// evicts the parent's blocks.
///
/// Pinned by the first user prompt that carries a script signal; config.language
/// until then. First pin wins for the life of the process.
static SESSION_LANGUAGE: once_cell::sync::OnceCell<String> = once_cell::sync::OnceCell::new();

pub fn pin_session_language(language: Option<&str>) {
    if let Some(l) = language {
        let _ = SESSION_LANGUAGE.set(l.to_string());
    }
}

pub fn session_language() -> Option<String> {
    SESSION_LANGUAGE
        .get()
        .cloned()
        .or_else(|| crate::utils::config::get_config().language)
}

#[cfg(test)]
mod written_text_tests {
    use super::detect_language;

    /// The photo label and file paths are not the user's language
    /// (tests/test_vision_followups.py test_language_ignores_label_and_paths).
    #[test]
    fn ignores_photo_label_and_paths() {
        let label = format!(
            "[Photo from the phone: /home/u/.codemason/remote/attachments/{}.jpg]",
            "ab".repeat(32)
        );
        assert_eq!(detect_language(&format!("이 사진 설명해줘\n{label}")), Some("Korean"));
        assert_eq!(detect_language(&format!("뭐야\n{label}\n{label}\n{label}")), Some("Korean"));
        assert_eq!(
            detect_language("~/work/codemason_v2.0/src/services/openai_common.rs 이거 봐"),
            Some("Korean")
        );
        assert_eq!(detect_language("look at /tmp/a.png please"), None);
    }
}
