//! UTF-8 stdio on Windows.
//! Equivalent to codemason/utils/stdio_encoding.py.
//!
//! On a Korean/Japanese/Chinese Windows the process locale encoding is a
//! legacy codepage (cp949, cp932, cp936); redirected Python streams died with
//! UnicodeEncodeError on non-latin1 glyphs. Rust always writes UTF-8 bytes, so
//! the Python failure mode cannot occur — the remaining Windows concern is the
//! CONSOLE codepage for interactive display, which we set to UTF-8 (65001),
//! the same end state the Python reconfigure achieved.

/// Reconfigure stdio to UTF-8. No-op off Windows.
pub fn ensure_utf8_stdio() {
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn SetConsoleOutputCP(cp: u32) -> i32;
            fn SetConsoleCP(cp: u32) -> i32;
        }
        // 65001 = CP_UTF8. Best-effort: a detached/pythonw-style process has
        // no console and the calls just fail quietly.
        let _ = SetConsoleOutputCP(65001);
        let _ = SetConsoleCP(65001);
    }
}
