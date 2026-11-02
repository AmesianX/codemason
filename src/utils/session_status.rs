//! Short status of the conversation for the screen and the phone: how full the
//! context is and how many images ride along (utils/session_status.py). One
//! place, so the PC footer and the Telegram message cannot disagree.

use crate::types::query::Message;
use crate::utils::i18n::{trf, FmtArg};

/// A limit no conversation reaches, so `attached_image_count` counts every image
/// the history still holds instead of the number one request carries.
const COUNT_ALL: i64 = 1 << 30;

/// (attached, held), or None when the server takes no images.
///
/// `attached` is what the next request carries, `held` is every image the
/// conversation still has, so the gap between them is what the server's
/// per-request limit pushed out. Reads the limit the system prompt build last
/// found (services/vision.rs), so it is sync. Both counts come from
/// attached_image_count, so neither can disagree with the request itself.
pub fn image_fill(messages: &[Message]) -> Option<(usize, usize)> {
    let n = crate::services::vision::last_enabled_limit();
    if n <= 0 {
        return None;
    }
    Some((
        crate::services::openai_common::attached_image_count(messages, n),
        crate::services::openai_common::attached_image_count(messages, COUNT_ALL),
    ))
}

/// The denominator is every image the conversation holds, so the two numbers part
/// the moment the per-request limit pushes the oldest out -- and the words say
/// what parting them means, since the figure alone cannot. Nothing else on screen
/// tells the user: the model is given the rule in its system prompt
/// (constants/prompts.rs) and the eviction goes only to the debug log.
fn image_line(attached: usize, held: usize) -> String {
    let (k, n) = (FmtArg::Int(attached as i64), FmtArg::Int(held as i64));
    if attached == held {
        return trf("images {k}/{n}", &[("k", k), ("n", n)]);
    }
    trf(
        "images {k}/{n} ({d} dropped)",
        &[("k", k), ("n", n), ("d", FmtArg::Int((held - attached) as i64))],
    )
}

pub fn image_text(messages: &[Message]) -> String {
    match image_fill(messages) {
        Some((attached, held)) => image_line(attached, held),
        None => String::new(),
    }
}

/// Whole percent of `limit`, half up, at most 100 — the input-line gauge and the
/// phone line both, so they never show two numbers (same formula as the Python
/// port, whose round() is half-even).
pub fn fill_percent(used: i64, limit: i64) -> i64 {
    std::cmp::min(100, (used as f64 * 100.0 / limit as f64 + 0.5).floor() as i64)
}

/// Fill against the auto-compact threshold, the same figure as the footer.
pub fn context_percent(messages: &[Message]) -> Option<i64> {
    let limit = crate::services::compaction::get_auto_compact_threshold()?;
    if limit <= 0 {
        return None;
    }
    let used = crate::utils::tokens::token_count_with_estimation(messages);
    Some(fill_percent(used, limit))
}

/// "Context 22% · images 6/8" — either part left out when unknown.
pub fn status_line(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    if let Some(pct) = context_percent(messages) {
        parts.push(trf("Context {pct}%", &[("pct", FmtArg::Int(pct))]));
    }
    let img = image_text(messages);
    if !img.is_empty() {
        parts.push(img);
    }
    parts.join(" · ")
}

#[cfg(test)]
mod image_line_tests {
    use super::image_line;

    /// tests/test_session_status.py test_dropped_images_are_said, which asserts
    /// both texts outright: it owns its process, this one shares it.
    #[test]
    fn dropped_is_said_only_when_there_is_one() {
        // Not asserted in one language. tr() follows the active one, which this
        // machine decides, and setting it here would race every other test in
        // the process. The catalog key is checked the way
        // adapters/telegram.rs::scaffold_key_matches_ko_catalog does, and the
        // numbers are read out of whatever language came back.
        assert!(crate::locales::ko::MESSAGES.contains_key("images {k}/{n} ({d} dropped)"));

        let plain = image_line(5, 5);
        assert!(plain.contains("5/5"), "{plain}");
        assert!(!plain.contains('('), "nothing was dropped: {plain}");

        // 4, 9 and 5 are three different digits, so the dropped count cannot be
        // read out of the fraction: {d} has to be wired to see it.
        let dropped = image_line(4, 9);
        assert!(dropped.contains("4/9") && dropped.contains('5'), "{dropped}");
    }
}

#[cfg(test)]
mod fill_percent_tests {
    use super::fill_percent;

    /// tests/test_vision_followups.py test_status_rules.
    #[test]
    fn half_up_and_capped() {
        assert_eq!(fill_percent(125, 1000), 13);
        assert_eq!(fill_percent(5, 1000), 1);
        assert_eq!(fill_percent(1500, 1000), 100);
    }
}
