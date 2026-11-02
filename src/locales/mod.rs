//! UI translations; one module per language.
//! Equivalent to codemason/locales/. Adding a language is adding a module with
//! a MESSAGES table and registering the code in utils::i18n::LANGUAGES plus the
//! dispatch below — the English source is the key, so a partial catalog is safe.

use once_cell::sync::Lazy;
use std::collections::HashMap;

pub mod ko;

static EMPTY: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(HashMap::new);

/// The catalog for a language code; empty (=> run in English) when there is
/// none, exactly like the Python import-failure path.
pub fn catalog(code: &str) -> &'static HashMap<&'static str, &'static str> {
    match code {
        "ko" => &ko::MESSAGES,
        _ => &EMPTY,
    }
}
