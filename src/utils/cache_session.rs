//! This install's prefix-cache session id.
//!
//! Sent only inside the encrypted request body (`cache_session`); the gateway
//! turns it into this session's cache salt with a server-only secret and
//! removes it. Everything after our public system prompt is cached under that
//! salt, so no other session can reuse -- or time -- it.
//!
//! One random id per install, kept in `<config dir>/cache_session` (0600):
//! stable across runs, so `/resume` and later turns still hit this user's own
//! cached conversation; never derived from anything guessable.

use once_cell::sync::Lazy;

static ID: Lazy<String> = Lazy::new(load_or_create);

/// The id, created on first use.
pub fn id() -> &'static str {
    ID.as_str()
}

fn valid(s: &str) -> bool {
    (16..=128).contains(&s.len())
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn load_or_create() -> String {
    let path = crate::utils::config::CONFIG_DIR.join("cache_session");
    if let Ok(existing) = std::fs::read_to_string(&path) {
        let existing = existing.trim().to_string();
        if valid(&existing) {
            return existing;
        }
    }
    let fresh = uuid::Uuid::new_v4().simple().to_string();
    let _ = std::fs::create_dir_all(&*crate::utils::config::CONFIG_DIR);
    if std::fs::write(&path, &fresh).is_ok() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
    }
    fresh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_id_is_what_the_gateway_accepts() {
        let fresh = uuid::Uuid::new_v4().simple().to_string();
        assert!(valid(&fresh), "{fresh}");
        assert!(!valid("short"));
        assert!(!valid("has space in it......"));
    }
}
