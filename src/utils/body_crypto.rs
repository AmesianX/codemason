//! Symmetric body encryption for the codemason ↔ vLLM channel.
//! Equivalent to codemason/utils/body_crypto.py — see that file for the threat
//! model. Wire format: base64( nonce(12) || AES-256-GCM ciphertext || tag(16) ).

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::Engine;

pub const NONCE_LEN: usize = 12; // 96-bit nonce, the AES-GCM standard size

/// Key is malformed, or a message failed to decrypt/authenticate.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct BodyCryptoError(pub String);

/// Parse a configured key string into 32 raw bytes (AES-256).
///
/// Accepts base64 (with or without padding) or hex — whichever the user
/// pasted. Must decode to exactly 32 bytes; anything else is a configuration
/// error, not something to silently pad or truncate.
pub fn load_key(raw: &str) -> Result<[u8; 32], BodyCryptoError> {
    let s = raw.trim();
    if s.is_empty() {
        return Err(BodyCryptoError("empty AES key".into()));
    }
    // Try BOTH hex and base64 and take whichever yields exactly 32 bytes. A
    // valid 32-byte key is 64 hex chars XOR 44 base64 chars — the lengths never
    // overlap, so there is no ambiguity to resolve by guessing.
    let mut candidates: Vec<Vec<u8>> = Vec::new();
    if s.len() % 2 == 0 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        if let Ok(b) = hex::decode(s) {
            candidates.push(b);
        }
    }
    let padded = format!("{}{}", s, "=".repeat((4 - s.len() % 4) % 4));
    if let Ok(b) = base64::engine::general_purpose::STANDARD.decode(padded) {
        candidates.push(b);
    }
    for key in &candidates {
        if key.len() == 32 {
            let mut out = [0u8; 32];
            out.copy_from_slice(key);
            return Ok(out);
        }
    }
    if candidates.is_empty() {
        return Err(BodyCryptoError(
            "AES key is neither valid base64 nor hex".into(),
        ));
    }
    Err(BodyCryptoError(format!(
        "AES key must be 32 bytes (256-bit); got {} after decoding",
        candidates[0].len()
    )))
}

/// Encrypt one message. Returns base64(nonce || ciphertext || tag).
pub fn encrypt(key: &[u8; 32], plaintext: &[u8]) -> String {
    let cipher = Aes256Gcm::new(key.into());
    let mut nonce = [0u8; NONCE_LEN];
    use rand::RngCore;
    rand::rngs::OsRng.fill_bytes(&mut nonce);
    // ct includes the 16-byte tag
    let ct = cipher
        .encrypt(Nonce::from_slice(&nonce), Payload { msg: plaintext, aad: &[] })
        .expect("AES-GCM encryption cannot fail with a valid key/nonce");
    let mut blob = Vec::with_capacity(NONCE_LEN + ct.len());
    blob.extend_from_slice(&nonce);
    blob.extend_from_slice(&ct);
    base64::engine::general_purpose::STANDARD.encode(blob)
}

/// Reverse encrypt(). Errors on a wrong key or tampering.
pub fn decrypt(key: &[u8; 32], blob: &str) -> Result<Vec<u8>, BodyCryptoError> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(blob.trim())
        .map_err(|e| BodyCryptoError(format!("ciphertext is not valid base64: {e}")))?;
    if raw.len() < NONCE_LEN + 16 {
        return Err(BodyCryptoError(
            "ciphertext too short to contain nonce + tag".into(),
        ));
    }
    let (nonce, ct) = raw.split_at(NONCE_LEN);
    let cipher = Aes256Gcm::new(key.into());
    cipher
        .decrypt(Nonce::from_slice(nonce), Payload { msg: ct, aad: &[] })
        .map_err(|_| BodyCryptoError("decryption failed: wrong key or tampered message".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let key = load_key(&"a".repeat(64).replace('a', "3")).unwrap();
        let ct = encrypt(&key, "안녕 world".as_bytes());
        assert_eq!(decrypt(&key, &ct).unwrap(), "안녕 world".as_bytes());
    }

    #[test]
    fn wrong_key_fails() {
        let k1 = load_key(&"11".repeat(32)).unwrap();
        let k2 = load_key(&"22".repeat(32)).unwrap();
        let ct = encrypt(&k1, b"msg");
        assert!(decrypt(&k2, &ct).is_err());
    }

    #[test]
    fn base64_key_accepted() {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode([7u8; 32]);
        assert_eq!(load_key(&b64).unwrap(), [7u8; 32]);
        // unpadded too
        let unpadded = b64.trim_end_matches('=').to_string();
        assert_eq!(load_key(&unpadded).unwrap(), [7u8; 32]);
    }
}
