//! UI strings, translated.
//! Equivalent to codemason/utils/i18n.py — see that file for the full design
//! notes. The source text is the key: `tr("Remote connected")` returns the
//! translation when there is one and the string it was handed when there is
//! not, so an empty catalog is a fully English program.
//!
//! The UI language resolves in this order, first hit wins:
//!     1. config.ui_language   2. config.language   3. the conversation
//!     4. OS locale (LANG/LC_ALL, Windows UI language)   5. English

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;

/// Supported languages: catalog code -> (display name, accepted spellings).
pub static LANGUAGES: &[(&str, &str, &[&str])] = &[
    // English has no catalog — it is the source. Registered anyway so that
    // "show me English" can be said out loud.
    ("en", "English", &["en", "eng", "english", "en_us", "en_gb", "c", "posix"]),
    ("ko", "한국어", &["ko", "kor", "korean", "한국어", "ko_kr"]),
    ("ja", "日本語", &["ja", "jpn", "japanese", "日本語", "ja_jp"]),
    ("zh", "中文", &["zh", "chi", "zho", "chinese", "中文", "zh_cn", "zh_hans"]),
    ("ru", "Русский", &["ru", "rus", "russian", "русский", "ru_ru"]),
    ("de", "Deutsch", &["de", "deu", "ger", "german", "deutsch", "de_de"]),
    ("es", "Español", &["es", "spa", "spanish", "español", "espanol", "es_es"]),
    ("fr", "Français", &["fr", "fra", "french", "français", "francais", "fr_fr"]),
    ("pt", "Português", &["pt", "por", "portuguese", "português", "portugues", "pt_br"]),
];

static ALIASES: Lazy<HashMap<String, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for (code, _name, aliases) in LANGUAGES {
        for a in *aliases {
            m.insert(a.to_string(), *code);
        }
    }
    m
});

/// How many self-identifying turns settle it. Two, so a single quoted line does
/// not flip the UI mid-session — unless the OS agrees, and then one is plenty.
const TURNS_TO_SETTLE: i32 = 2;
const TURNS_TO_SETTLE_WITH_LOCALE: i32 = 1;

struct I18nState {
    active: Option<&'static str>,
    resolved: bool,
    /// Where `active` came from: 'config' and 'explicit' are the user speaking
    /// and are final; 'locale' is a guess the conversation may correct.
    source: Option<&'static str>,
    /// Conversation evidence, until it decides. Counting is by TURN.
    turns: HashMap<&'static str, i32>,
    conversation_settled: bool,
}

static STATE: Lazy<Mutex<I18nState>> = Lazy::new(|| {
    Mutex::new(I18nState {
        active: None,
        resolved: false,
        source: None,
        turns: HashMap::new(),
        conversation_settled: false,
    })
});

/// 'Korean' / 'ko_KR.UTF-8' / '한국어' -> 'ko'. None for anything unknown.
pub fn normalize(name: Option<&str>) -> Option<&'static str> {
    let name = name?;
    if name.trim().is_empty() {
        return None;
    }
    let key = name.trim().to_lowercase().replace('-', "_");
    let key = key.split('.').next().unwrap_or("").split('@').next().unwrap_or("");
    if let Some(code) = ALIASES.get(key) {
        return Some(code);
    }
    let head = key.split('_').next().unwrap_or("");
    ALIASES.get(head).copied()
}

fn from_config() -> Option<&'static str> {
    let cfg = crate::utils::config::get_config();
    normalize(cfg.ui_language.as_deref()).or_else(|| normalize(cfg.language.as_deref()))
}

fn from_locale() -> Option<&'static str> {
    for var in ["LC_ALL", "LC_MESSAGES", "LANG"] {
        if let Ok(v) = std::env::var(var) {
            if let Some(code) = normalize(Some(&v)) {
                return Some(code);
            }
        }
    }
    from_os_locale()
}

/// The OS's own idea of the UI language, where the environment is silent.
/// On Windows GetUserDefaultUILanguage is the language Windows itself is
/// displayed in; POSIX has nothing beyond the env vars checked above.
fn from_os_locale() -> Option<&'static str> {
    #[cfg(windows)]
    {
        extern "system" {
            fn GetUserDefaultUILanguage() -> u16;
        }
        let lcid = unsafe { GetUserDefaultUILanguage() } as u32;
        if let Some(tag) = lcid_to_tag(lcid) {
            return normalize(Some(tag));
        }
    }
    None
}

/// Windows LCID -> a name normalize() understands. Only the primary language
/// matters (ko-KR and ko-KP read the same catalog): the low 10 bits.
#[allow(dead_code)]
fn lcid_to_tag(lcid: u32) -> Option<&'static str> {
    if lcid == 0 {
        return None;
    }
    match lcid & 0x3FF {
        0x12 => Some("ko"),
        0x11 => Some("ja"),
        0x04 => Some("zh"),
        0x19 => Some("ru"),
        0x07 => Some("de"),
        0x0A => Some("es"),
        0x0C => Some("fr"),
        0x16 => Some("pt"),
        0x09 => Some("en"),
        _ => None,
    }
}

/// The active language code, or None when the UI is in English.
pub fn active() -> Option<&'static str> {
    {
        let s = STATE.lock();
        if s.resolved {
            return s.active;
        }
    }
    // Resolve outside the lock: from_config() -> get_config() may log/lock.
    let cfg = from_config();
    let act = cfg.or_else(from_locale);
    let mut s = STATE.lock();
    if !s.resolved {
        s.active = act;
        s.source = if cfg.is_some() {
            Some("config")
        } else if act.is_some() {
            Some("locale")
        } else {
            None
        };
        s.resolved = true;
    }
    s.active
}

/// Switch at runtime. Returns the code; an unknown name falls back to English.
pub fn set_language(name: Option<&str>) -> Option<&'static str> {
    let code = normalize(name);
    let mut s = STATE.lock();
    s.active = code;
    s.resolved = true;
    s.source = Some("explicit");
    s.conversation_settled = true; // said out loud; stop guessing
    code
}

/// Re-resolve on the next call — after a config change, and in tests.
pub fn reset() {
    let mut s = STATE.lock();
    s.active = None;
    s.resolved = false;
    s.source = None;
    s.turns.clear();
    s.conversation_settled = false;
}

static NOISE: Lazy<regex::Regex> = Lazy::new(|| {
    regex::Regex::new(
        r"(?si)```.*?```|`[^`]*`|\[Pasted text[^\]]*\]|<[^>\s]+>|\b[a-z]+://\S+|(?:^|\s)[~./][^\s]*",
    )
    .unwrap()
});

/// `text` with the parts that are not the user's own writing removed.
/// Code blocks, inline code, URLs and paths are Latin whatever the author
/// speaks; a [Pasted text #N] placeholder stands in for something they did not
/// type at all.
fn typed_only(text: &str) -> String {
    NOISE.replace_all(text, " ").into_owned()
}

/// Take one user turn as evidence of which language to show the UI in.
/// See the Python reference for when this is (and is not) called.
pub fn observe_user_text(text: &str) {
    {
        let s = STATE.lock();
        if s.conversation_settled {
            return;
        }
    }
    if active().is_some() {
        let mut s = STATE.lock();
        if matches!(s.source, Some("config") | Some("explicit")) {
            s.conversation_settled = true; // the user already said so
            return;
        }
    }

    let detected = crate::utils::language::detect_language(&typed_only(text));
    let Some(code) = normalize(detected) else {
        return; // Latin script: no signal
    };

    let locale_agrees = from_locale() == Some(code);
    let mut s = STATE.lock();
    if s.conversation_settled {
        return;
    }
    let count = s.turns.entry(code).or_insert(0);
    *count += 1;
    let needed = if locale_agrees {
        TURNS_TO_SETTLE_WITH_LOCALE
    } else {
        TURNS_TO_SETTLE
    };
    if *count >= needed {
        // Adopt the conversation's language, once, and stop looking.
        s.active = Some(code);
        s.resolved = true;
        s.source = Some("conversation");
        s.conversation_settled = true;
    }
}

fn catalog(code: &str) -> &'static HashMap<&'static str, &'static str> {
    crate::locales::catalog(code)
}

/// A named format argument for `trf` — the stand-in for Python's **kwargs.
#[derive(Debug, Clone)]
pub enum FmtArg {
    Str(String),
    Int(i64),
    Float(f64),
}

impl From<&str> for FmtArg {
    fn from(v: &str) -> Self {
        FmtArg::Str(v.to_string())
    }
}
impl From<String> for FmtArg {
    fn from(v: String) -> Self {
        FmtArg::Str(v)
    }
}
impl From<i64> for FmtArg {
    fn from(v: i64) -> Self {
        FmtArg::Int(v)
    }
}
impl From<usize> for FmtArg {
    fn from(v: usize) -> Self {
        FmtArg::Int(v as i64)
    }
}
impl From<f64> for FmtArg {
    fn from(v: f64) -> Self {
        FmtArg::Float(v)
    }
}

impl FmtArg {
    /// Render with a Python format spec subset: '' (str), 'g' (general float),
    /// ',' (thousands separators), ',.0f' etc. Unknown specs fall back to the
    /// plain rendering — a typo in a catalog must never take the program down.
    fn render(&self, spec: &str) -> String {
        match (self, spec) {
            (FmtArg::Str(s), _) => s.clone(),
            (FmtArg::Int(n), "") => n.to_string(),
            (FmtArg::Int(n), ",") => thousands(&n.to_string()),
            (FmtArg::Int(n), "g") => n.to_string(),
            (FmtArg::Float(f), "") => {
                // Python str(float): integral values keep '.0'
                if f.fract() == 0.0 && f.abs() < 1e16 {
                    format!("{:.1}", f)
                } else {
                    format!("{}", f)
                }
            }
            (FmtArg::Float(f), "g") => {
                // Python '{:g}': shortest of fixed/scientific, trailing zeros cut
                let s = format!("{}", f);
                if f.fract() == 0.0 && f.abs() < 1e16 {
                    format!("{}", *f as i64)
                } else {
                    s
                }
            }
            (FmtArg::Float(f), ",") => thousands(&format!("{}", f)),
            (FmtArg::Int(n), _) => n.to_string(),
            (FmtArg::Float(f), _) => format!("{}", f),
        }
    }
}

fn thousands(digits: &str) -> String {
    let (sign, body) = digits.strip_prefix('-').map_or(("", digits), |b| ("-", b));
    let (int_part, frac) = match body.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (body, None),
    };
    let mut out = String::new();
    let chars: Vec<char> = int_part.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*c);
    }
    let mut s = format!("{sign}{out}");
    if let Some(f) = frac {
        s.push('.');
        s.push_str(f);
    }
    s
}

/// Substitute named `{placeholder}` / `{placeholder:spec}` fields. `{{`/`}}`
/// are literal braces, as in Python str.format. Errors (unknown name) fail the
/// whole formatting, mirroring Python's KeyError -> fallback path.
fn format_named(template: &str, args: &[(&str, FmtArg)]) -> Result<String, ()> {
    let mut out = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                    out.push('{');
                    continue;
                }
                let mut field = String::new();
                let mut closed = false;
                for c2 in chars.by_ref() {
                    if c2 == '}' {
                        closed = true;
                        break;
                    }
                    field.push(c2);
                }
                if !closed {
                    return Err(());
                }
                let (name, spec) = match field.split_once(':') {
                    Some((n, s)) => (n, s),
                    None => (field.as_str(), ""),
                };
                match args.iter().find(|(k, _)| *k == name) {
                    Some((_, v)) => out.push_str(&v.render(spec)),
                    None => return Err(()),
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                }
                out.push('}');
            }
            c => out.push(c),
        }
    }
    Ok(out)
}

/// The translated line, or the English it was handed (gettext / `_`).
pub fn tr(text: &str) -> String {
    match active() {
        Some(code) => catalog(code).get(text).map(|s| s.to_string()).unwrap_or_else(|| text.to_string()),
        None => text.to_string(),
    }
}

/// gettext with named placeholders. A formatting failure — a typo in someone's
/// catalog — falls back to the source rather than raising, because one line of
/// UI is never worth taking the program down.
pub fn trf(text: &str, args: &[(&str, FmtArg)]) -> String {
    let out = tr(text);
    if args.is_empty() {
        return out;
    }
    match format_named(&out, args) {
        Ok(s) => s,
        Err(_) => format_named(text, args).unwrap_or_else(|_| text.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_variants() {
        assert_eq!(normalize(Some("Korean")), Some("ko"));
        assert_eq!(normalize(Some("ko_KR.UTF-8")), Some("ko"));
        assert_eq!(normalize(Some("한국어")), Some("ko"));
        assert_eq!(normalize(Some("nope")), None);
    }

    #[test]
    fn format_named_specs() {
        assert_eq!(
            format_named("{n}m ago", &[("n", FmtArg::Int(5))]).unwrap(),
            "5m ago"
        );
        assert_eq!(
            format_named("{secs:g}s", &[("secs", FmtArg::Float(30.0))]).unwrap(),
            "30s"
        );
        assert_eq!(
            format_named("{chars:,}", &[("chars", FmtArg::Int(1234567))]).unwrap(),
            "1,234,567"
        );
        assert!(format_named("{missing}", &[]).is_err());
        assert_eq!(format_named("{{literal}}", &[]).unwrap(), "{literal}");
    }
}
