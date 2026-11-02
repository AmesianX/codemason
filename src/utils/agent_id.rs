//! Agent ids — adapted from utils/uuid.ts (createAgentId) and types/ids.ts (isAgentId).
//! Equivalent to codemason/utils/agent_id.py.
//!
//!     createAgentId(label?) ->  `a{label}-{hex16}` with a label, else `a{hex16}`
//!     isAgentId(value)      ->  the value if it matches /^a(?:.+-)?[0-9a-f]{16}$/
//!
//! The id doubles as the sub-agent's task id (the source builds the task with
//! `id: agentId`), and the task id names the agent's output file — so the
//! format is not cosmetic: a value that fails the pattern is not an agent id.

use rand::RngCore;

/// Generate an agent id, optionally carrying a label (createAgentId).
///
/// From the OS entropy source, not a thread-local PRNG: the id names a file
/// under a shared output directory, so it has to be unguessable for the same
/// reason task ids are.
pub fn create_agent_id(label: Option<&str>) -> String {
    let mut bytes = [0u8; 8];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    match label {
        Some(n) if !n.is_empty() => format!("a{n}-{hex}"),
        _ => format!("a{hex}"),
    }
}

/// `Some(value)` when it is a well-formed agent id, else None (isAgentId).
///
/// Hand-checked rather than regex: /^a(?:.+-)?[0-9a-f]{16}$/ is "a", then an
/// optional name ending in '-', then exactly 16 lowercase hex digits.
pub fn is_agent_id(value: &str) -> Option<&str> {
    let rest = value.strip_prefix('a')?;
    let tail = match rest.rfind('-') {
        // The name part must be non-empty (the source's `.+`).
        Some(i) if i > 0 => &rest[i + 1..],
        Some(_) => return None,
        None => rest,
    };
    if tail.len() == 16 && tail.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_id_is_a_plus_sixteen_hex() {
        let id = create_agent_id(None);
        assert_eq!(id.len(), 17);
        assert_eq!(is_agent_id(&id), Some(id.as_str()));
    }

    #[test]
    fn a_named_id_keeps_the_name() {
        let id = create_agent_id(Some("explore"));
        assert!(id.starts_with("aexplore-"));
        assert_eq!(is_agent_id(&id), Some(id.as_str()));
    }

    #[test]
    fn two_ids_differ() {
        assert_ne!(create_agent_id(None), create_agent_id(None));
    }

    #[test]
    fn anything_else_is_not_an_agent_id() {
        for bad in ["deadbeef", "a", "abeef", "a-0123456789abcdef", "a0123456789ABCDEF", "b0123456789abcdef"] {
            assert_eq!(is_agent_id(bad), None, "{bad} should not parse");
        }
    }
}
