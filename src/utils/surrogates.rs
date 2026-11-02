//! Repair UTF-16 surrogates that leak into input text on the Windows console.
//! Equivalent to codemason/utils/surrogates.py.
//!
//! In Python, prompt_toolkit's per-batch surrogate merging could leave lone
//! UTF-16 surrogate code points inside a `str`, which blew up later at the
//! HTTP layer; `repair_surrogates` rejoined split pairs at the submit boundary.
//!
//! In Rust a `String` is guaranteed well-formed UTF-8 and CANNOT hold a lone
//! surrogate — the console input layer (crossterm) decodes UTF-16 pairs before
//! they ever reach us, and any unpairable half arrives as U+FFFD already. The
//! function is therefore an identity kept for call-site parity: the submit
//! boundary still calls it, and if a future input source hands us bytes we
//! repair them in `repair_surrogate_bytes`.

/// Rejoin split surrogate pairs; replace unpairable halves with U+FFFD.
/// A no-op for well-formed text — which in Rust is all `&str` text.
pub fn repair_surrogates(text: &str) -> String {
    text.to_string()
}

/// Byte-level variant for raw input sources: decodes as UTF-8, replacing any
/// ill-formed sequence (including CESU-8-encoded lone surrogates) with U+FFFD,
/// after first rejoining CESU-8 surrogate PAIRS into the character they encode.
pub fn repair_surrogate_bytes(bytes: &[u8]) -> String {
    // CESU-8 encodes each UTF-16 surrogate as a 3-byte sequence ED A0..AF xx
    // (high) / ED B0..BF xx (low). Rejoin adjacent high+low pairs first, so a
    // pair that straddled an input-batch boundary comes back as its character.
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 6 <= bytes.len() && bytes[i] == 0xED && (0xA0..=0xAF).contains(&bytes[i + 1]) {
            if bytes[i + 3] == 0xED && (0xB0..=0xBF).contains(&bytes[i + 4]) {
                let high = 0xD800
                    + (((bytes[i + 1] as u32 - 0xA0) << 6) | (bytes[i + 2] as u32 & 0x3F));
                let low = 0xDC00
                    + (((bytes[i + 4] as u32 - 0xB0) << 6) | (bytes[i + 5] as u32 & 0x3F));
                let cp = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                if let Some(c) = char::from_u32(cp) {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    i += 6;
                    continue;
                }
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejoins_cesu8_pair() {
        // 😀 U+1F600 as CESU-8: high D83D low DE00
        let cesu = [0xED, 0xA0, 0xBD, 0xED, 0xB8, 0x80];
        assert_eq!(repair_surrogate_bytes(&cesu), "😀");
    }

    #[test]
    fn lone_half_becomes_replacement() {
        let lone = [0xED, 0xA0, 0xBD];
        assert_eq!(repair_surrogate_bytes(&lone), "\u{FFFD}\u{FFFD}\u{FFFD}");
    }
}
