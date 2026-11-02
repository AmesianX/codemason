//! Images of a session, on disk next to its transcript
//! (codemason/utils/image_store.py; keep the two in step):
//!
//!     <CONFIG_DIR>/projects/<sanitized-cwd>/<sessionId>/images/
//!         <sha256>.<ext>        an image as it was sent to the model
//!         orig-<sha256>.<ext>   the original it was made from (debugging)
//!
//! The transcript does not carry the base64 payload. When an entry is written,
//! an image block's base64 source is stored as a file and replaced by a
//! reference
//!
//!     {"type": "codemason_file", "path": "<sha256>.<ext>", "sha256": ..., "media_type": ...}
//!
//! and the reverse happens when a session is resumed. Only the copy written to
//! disk changes; the in-memory messages keep their base64. A reference whose
//! file is gone comes back as a text note instead of an image, so a resumed
//! session never sends a broken image part. Deleting a session in the resume
//! picker deletes this folder with it (session_storage::delete_session).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use base64::Engine;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

/// Base64 the way Python's b64decode reads it: a payload wrapped with line
/// breaks is still the same image.
pub fn decode_b64(data: &str) -> Option<Vec<u8>> {
    let engine = &base64::engine::general_purpose::STANDARD;
    engine.decode(data).ok().or_else(|| {
        let compact: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
        engine.decode(compact).ok()
    })
}

/// What dehydrate writes as a reference: a bare file name, nothing that walks.
fn is_ref_name(name: &str) -> bool {
    let Some((stem, ext)) = name.split_once('.') else { return false };
    stem.len() == 64
        && stem.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        && (1..=5).contains(&ext.len())
        && ext.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
}

fn ext_for(media_type: &str) -> &'static str {
    match media_type {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => "bin",
    }
}

pub fn images_dir(session_id: Option<&str>) -> PathBuf {
    let sid = session_id
        .map(String::from)
        .unwrap_or_else(crate::utils::state::get_session_id);
    super::session_storage::get_project_dir(&crate::utils::state::get_original_cwd())
        .join(sid)
        .join("images")
}

fn write_once(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // A writer-unique temp name: two writers of the same image (parallel
    // subagents, parallel tests) must not share one, or the second rename finds
    // it gone. Whoever renames last wins with identical bytes.
    let tmp = path.with_extension(format!(
        "{}.{}.tmp",
        path.extension().and_then(|e| e.to_str()).unwrap_or(""),
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path).or_else(|e| {
        let _ = std::fs::remove_file(&tmp);
        if path.exists() { Ok(()) } else { Err(e) }
    })
}

/// True for a file codemason itself stored by content hash (a phone
/// attachment, a session image): never edited in place, so a hard link to it
/// is as good as a copy.
fn in_own_store(path: &Path) -> bool {
    let Ok(p) = path.canonicalize() else { return false };
    [crate::utils::config::CONFIG_DIR.clone(), crate::remote::adapters::telegram::attachments_dir()]
        .iter()
        .filter_map(|root| root.canonicalize().ok())
        .any(|root| p.starts_with(root))
}

/// Keep the original file an image was made from; returns its path.
///
/// `source` is the file `data` was read from. When that is one of codemason's
/// own stores (a photo from the phone, an image Read back from this store),
/// the copy is a hard link: same bytes, no second copy on disk.
pub fn save_original(data: &[u8], ext: &str, session_id: Option<&str>, source: Option<&Path>) -> PathBuf {
    let sha = hex::encode(Sha256::digest(data));
    let ext = ext.trim_start_matches('.').to_lowercase();
    let ext = if ext.is_empty() { "bin".to_string() } else { ext };
    let path = images_dir(session_id).join(format!("orig-{sha}.{ext}"));
    if path.exists() {
        return path;
    }
    if let Some(src) = source.filter(|s| in_own_store(s)) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if std::fs::hard_link(src, &path).is_ok() {
            return path;
        } // another filesystem, or no links: copy
    }
    let _ = write_once(&path, data);
    path
}

fn dehydrate_source(source: &Value, session_id: Option<&str>) -> Value {
    let data = source.get("data").and_then(|d| d.as_str()).unwrap_or("");
    let media_type = source.get("media_type").and_then(|m| m.as_str()).unwrap_or("image/png");
    let Some(raw) = decode_b64(data) else {
        return source.clone();
    };
    let sha = hex::encode(Sha256::digest(&raw));
    let name = format!("{sha}.{}", ext_for(media_type));
    if write_once(&images_dir(session_id).join(&name), &raw).is_err() {
        return source.clone(); // keep the payload inline rather than lose the image
    }
    json!({"type": "codemason_file", "path": name, "sha256": sha, "media_type": media_type})
}

/// stored_path answers, by (store, payload length, payload hash). Evicted
/// images are labelled on every request and each answer costs a full base64
/// decode and SHA-256, so it is worked out once. A failure is remembered too:
/// the label must not gain a path on a later request, which would move the
/// request prefix.
static STORED: Lazy<Mutex<HashMap<(PathBuf, usize, u64), Option<PathBuf>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
const STORED_MAX: usize = 4096;

/// Where a base64 image source lives in the session store, writing it there
/// first if it is not yet (dehydrate normally has). None when it cannot be
/// stored or is not base64.
pub fn stored_path(source: &Value, session_id: Option<&str>) -> Option<PathBuf> {
    if source.get("type").and_then(|t| t.as_str()) != Some("base64") {
        return None;
    }
    let data = source.get("data").and_then(|d| d.as_str()).unwrap_or("");
    let dir = images_dir(session_id);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut h);
    let key = (dir.clone(), data.len(), h.finish());
    if let Some(hit) = STORED.lock().get(&key) {
        return hit.clone();
    }
    let r = dehydrate_source(source, session_id);
    let path = (r.get("type").and_then(|t| t.as_str()) == Some("codemason_file"))
        .then(|| r.get("path").and_then(|p| p.as_str()).map(|p| dir.join(p)))
        .flatten();
    let mut cache = STORED.lock();
    if cache.len() >= STORED_MAX {
        cache.clear();
    }
    cache.insert(key, path.clone());
    path
}

fn is_image_with(obj: &Map<String, Value>, source_type: &str) -> bool {
    obj.get("type").and_then(|t| t.as_str()) == Some("image")
        && obj
            .get("source")
            .and_then(|s| s.get("type"))
            .and_then(|t| t.as_str())
            == Some(source_type)
}

/// Copy of a JSON message with base64 image sources moved to files.
pub fn dehydrate(v: &Value, session_id: Option<&str>) -> Value {
    match v {
        Value::Array(items) => Value::Array(items.iter().map(|x| dehydrate(x, session_id)).collect()),
        Value::Object(obj) if is_image_with(obj, "base64") => {
            let mut out = obj.clone();
            out.insert("source".into(), dehydrate_source(&obj["source"], session_id));
            Value::Object(out)
        }
        Value::Object(obj) => Value::Object(
            obj.iter().map(|(k, x)| (k.clone(), dehydrate(x, session_id))).collect(),
        ),
        other => other.clone(),
    }
}

/// Reverse of dehydrate. A missing file becomes a text note.
pub fn hydrate(v: &Value, session_id: Option<&str>) -> Value {
    match v {
        Value::Array(items) => Value::Array(items.iter().map(|x| hydrate(x, session_id)).collect()),
        Value::Object(obj) if is_image_with(obj, "codemason_file") => {
            let src = &obj["source"];
            let name = src.get("path").and_then(|p| p.as_str()).unwrap_or("");
            if !is_ref_name(name) {
                // Not something dehydrate wrote: never follow it out of the store.
                let shown: String = name.chars().take(80).collect();
                return json!({"type": "text", "text": format!("[Image reference not valid: {shown}]")});
            }
            let path = images_dir(session_id).join(name);
            match std::fs::read(&path) {
                Ok(raw) => {
                    let mut out = obj.clone();
                    out.insert(
                        "source".into(),
                        json!({
                            "type": "base64",
                            "media_type": src.get("media_type").and_then(|m| m.as_str()).unwrap_or("image/png"),
                            "data": base64::engine::general_purpose::STANDARD.encode(raw),
                        }),
                    );
                    Value::Object(out)
                }
                Err(_) => json!({"type": "text", "text": format!("[Image file missing: {}]", path.display())}),
            }
        }
        Value::Object(obj) => Value::Object(
            obj.iter().map(|(k, x)| (k.clone(), hydrate(x, session_id))).collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";

    fn msg() -> Value {
        json!({"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "tool_result", "tool_use_id": "r1", "content": [
                {"type": "text", "text": "[Image file: /x.png]"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": B64}}]}]})
    }

    #[test]
    fn round_trip_and_missing_file() {
        let sid = format!("img-store-{}", uuid::Uuid::new_v4());
        let disk = dehydrate(&msg(), Some(&sid));
        let s = disk.to_string();
        assert!(!s.contains(B64) && s.contains("codemason_file"), "{s}");
        let files: Vec<_> = std::fs::read_dir(images_dir(Some(&sid))).unwrap().collect();
        assert_eq!(files.len(), 1);
        assert_eq!(hydrate(&disk, Some(&sid)), msg());
        for f in std::fs::read_dir(images_dir(Some(&sid))).unwrap() {
            std::fs::remove_file(f.unwrap().path()).unwrap();
        }
        let back = hydrate(&disk, Some(&sid));
        let img = &back["content"][1]["content"][1];
        assert_eq!(img["type"], "text");
        assert!(img["text"].as_str().unwrap().contains("Image file missing"));
        let _ = std::fs::remove_dir_all(images_dir(Some(&sid)).parent().unwrap());
    }

    #[test]
    fn reference_out_of_the_store_refused() {
        let bad = json!({"type": "image", "source": {"type": "codemason_file",
                         "path": "../../../etc/passwd", "media_type": "image/png"}});
        let out = hydrate(&bad, Some("img-store-ref"));
        assert_eq!(out["type"], "text");
        assert!(out["text"].as_str().unwrap().contains("not valid"), "{out}");
        assert!(is_ref_name(&format!("{}.png", "a".repeat(64))));
        assert!(!is_ref_name(&format!("{}.png/x", "a".repeat(64))));
    }

    #[test]
    fn stored_path_once_and_lenient_base64() {
        let sid = format!("img-store-{}", uuid::Uuid::new_v4());
        let src = json!({"type": "base64", "media_type": "image/png", "data": B64});
        let p1 = stored_path(&src, Some(&sid)).unwrap();
        std::fs::remove_file(&p1).unwrap();
        // Remembered: not written again, same answer (the label stays put).
        assert_eq!(stored_path(&src, Some(&sid)), Some(p1.clone()));
        assert!(!p1.exists());
        let wrapped = format!("{}\n{}", &B64[..40], &B64[40..]);
        assert_eq!(decode_b64(&wrapped), decode_b64(B64));
        let _ = std::fs::remove_dir_all(images_dir(Some(&sid)).parent().unwrap());
    }
}
