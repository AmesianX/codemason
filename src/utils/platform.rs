//! Platform detection for CodeMason.
//! Equivalent to codemason/utils/platform.py (src/utils/platform.ts).
//!
//! Tier-1 Windows support ("git-bash / MSYS2 / WSL2"): this module only
//! classifies the host. WSL2 is a real Linux kernel, so the shell substrate
//! already works there unchanged — we detect it purely to label the platform.

use once_cell::sync::Lazy;
use parking_lot::Mutex;

/// 'macos' | 'windows' | 'wsl' | 'linux' | 'unknown' (Platform in platform.ts)
static PLATFORM_CACHE: Lazy<Mutex<Option<&'static str>>> = Lazy::new(|| Mutex::new(None));
static WSL_VERSION_CACHE: Lazy<Mutex<(bool, Option<String>)>> =
    Lazy::new(|| Mutex::new((false, None)));

/// Read /proc/version (empty string if unavailable). WSL markers live here.
fn read_proc_version() -> String {
    std::fs::read_to_string("/proc/version").unwrap_or_default()
}

/// Classify the host (memoized). Mirrors getPlatform() in platform.ts.
///
/// Returns one of 'macos', 'windows', 'wsl', 'linux', 'unknown'. WSL is Linux
/// with a 'microsoft'/'wsl' marker in /proc/version.
pub fn get_platform() -> &'static str {
    let mut cache = PLATFORM_CACHE.lock();
    if let Some(p) = *cache {
        return p;
    }
    let p = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "linux") {
        let pv = read_proc_version().to_lowercase();
        if pv.contains("microsoft") || pv.contains("wsl") {
            "wsl"
        } else {
            "linux"
        }
    } else {
        "unknown"
    };
    *cache = Some(p);
    p
}

/// True only on native Windows (not WSL). WSL is 'wsl', handled as POSIX.
pub fn is_windows() -> bool {
    get_platform() == "windows"
}

/// WSL major version ('1'/'2'/...) or None (getWslVersion() in platform.ts).
pub fn get_wsl_version() -> Option<String> {
    let mut cache = WSL_VERSION_CACHE.lock();
    if cache.0 {
        return cache.1.clone();
    }
    cache.0 = true;
    if !cfg!(target_os = "linux") {
        cache.1 = None;
        return None;
    }
    let pv = read_proc_version();
    let re = regex::Regex::new(r"(?i)WSL(\d+)").unwrap();
    cache.1 = if let Some(m) = re.captures(&pv) {
        Some(m[1].to_string())
    } else if pv.to_lowercase().contains("microsoft") {
        // Original WSL1 format ("4.4.0-19041-Microsoft") carries no explicit version.
        Some("1".to_string())
    } else {
        None
    };
    cache.1.clone()
}

/// Clear memoized results (tests only).
pub fn reset_cache() {
    *PLATFORM_CACHE.lock() = None;
    *WSL_VERSION_CACHE.lock() = (false, None);
}

/// Is this pid running? Portable.
///
/// POSIX spells the test `kill(pid, 0)` — signal 0 checks permission to signal
/// without sending one. On Windows, OpenProcess + GetExitCodeProcess asks for
/// STILL_ACTIVE instead (there are no signals to probe with).
pub fn pid_alive(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    #[cfg(unix)]
    {
        let r = unsafe { libc::kill(pid as libc::pid_t, 0) };
        if r == 0 {
            return true;
        }
        let err = std::io::Error::last_os_error();
        // EPERM: exists, owned by someone else. ESRCH: gone.
        return err.raw_os_error() == Some(libc::EPERM);
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::RawHandle;
        // PROCESS_QUERY_LIMITED_INFORMATION = 0x1000, STILL_ACTIVE = 259
        extern "system" {
            fn OpenProcess(access: u32, inherit: i32, pid: u32) -> RawHandle;
            fn GetExitCodeProcess(h: RawHandle, code: *mut u32) -> i32;
            fn CloseHandle(h: RawHandle) -> i32;
        }
        unsafe {
            let h = OpenProcess(0x1000, 0, pid as u32);
            if h.is_null() {
                return false; // gone, or not ours to look at
            }
            let mut code: u32 = 0;
            let ok = GetExitCodeProcess(h, &mut code);
            CloseHandle(h);
            ok != 0 && code == 259
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}
