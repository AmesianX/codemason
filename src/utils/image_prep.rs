//! Prepare an image for the model before it is sent — the client half of the
//! server's own geometry, so the server only pads and never resamples again
//! (codemason/utils/image_prep.py; keep the two in step).
//!
//! The server (GLM-5.3-Flash processor, pad mode) gives one image a budget of
//! VISION_MAX_TOKENS tokens; one token covers a 28x28 pixel cell, so the budget
//! is an AREA: 4096 tokens = 3,211,264 px = 1792^2, in any aspect ratio. Inside
//! the budget the server does not resample at all — it pads right/bottom up to
//! the next multiple of 28. Over the budget it shrinks the image proportionally
//! to the largest size whose 28-aligned canvas fits.
//!
//! So, per image ("within budget" is the 28-aligned canvas, the area the server
//! itself measures — an image whose own area fits but whose padded canvas does
//! not is shrunk by the server):
//!   - EXIF orientation is applied to the pixels first, so the size we reason
//!     about is the size the model sees.
//!   - Within budget: sent as-is, never enlarged (the server enlarges only
//!     images under its 16-token minimum). The original pixels when no
//!     rotation was needed: a PNG or JPEG is passed through with only its
//!     metadata removed (EXIF with GPS and camera, XMP, text chunks),
//!     losslessly; anything else is re-encoded.
//!   - Area over budget: shrunk ONCE, with Lanczos, to exactly the content size
//!     the server would pick. The server then pads and does not resample again.
//!   - Extreme aspect ratio (a whole-page capture): shrinking to the area budget
//!     would leave the short side under MIN_READABLE_SHORT_SIDE, so the image is
//!     cut along its long side into tiles that each fit the budget at readable
//!     scale — at most `slots_left` tiles.
//!   - An animated GIF/WebP/PNG is sent as its first frame (`animated` says so).

use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};

/// Pixels per token side (patch 14 x merge 2).
pub const CELL: u32 = 28;
pub const MAX_TOKENS_PER_IMAGE: u32 = 4096;
pub const MIN_TOKENS_PER_IMAGE: u32 = 16;
/// 3,211,264 = 1792^2.
pub const BUDGET_PIXELS: u64 = MAX_TOKENS_PER_IMAGE as u64 * CELL as u64 * CELL as u64;
pub const MIN_READABLE_SHORT_SIDE: u32 = 768;
pub const MAX_TILE_SHORT_SIDE: u32 = 1792;

#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub data: Vec<u8>,
    pub media_type: &'static str,
    pub width: u32,
    pub height: u32,
    pub tokens: u32,
    /// The source had more frames; this is the first.
    pub animated: bool,
}

fn ceil_to(v: u32, f: u32) -> u32 {
    v.div_ceil(f) * f
}

fn padded_area(w: u32, h: u32) -> u64 {
    ceil_to(w, CELL) as u64 * ceil_to(h, CELL) as u64
}

/// JPEG segments kept when the metadata goes: APP0 JFIF/JFXX (pixel density),
/// APP2 ICC_PROFILE (colours), APP14 Adobe (colour transform). Every other APPn
/// and COM is metadata — APP1 alone carries EXIF (camera, GPS) and XMP.
fn jpeg_segment_kept(marker: u8, body: &[u8]) -> bool {
    let keep: &[&[u8]] = match marker {
        0xE0 => &[b"JFIF\0", b"JFXX\0"],
        0xE2 => &[b"ICC_PROFILE\0"],
        0xEE => &[b"Adobe"],
        0xE1..=0xEF | 0xFE => &[],
        _ => return true,
    };
    keep.iter().any(|sig| body.starts_with(sig))
}

fn strip_jpeg(data: &[u8]) -> Option<Vec<u8>> {
    if !data.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut out = data[..2].to_vec();
    let mut pos = 2;
    while pos + 1 < data.len() {
        if data[pos] != 0xFF {
            return None;
        }
        while pos + 1 < data.len() && data[pos + 1] == 0xFF {
            pos += 1; // fill bytes
        }
        let marker = *data.get(pos + 1)?;
        if marker == 0xDA {
            out.extend_from_slice(&data[pos..]); // start of scan: the rest is pixels
            return Some(out);
        }
        if (0xD0..=0xD7).contains(&marker) || marker == 0x01 {
            out.extend_from_slice(&data[pos..pos + 2]);
            pos += 2;
            continue;
        }
        let len = u16::from_be_bytes([*data.get(pos + 2)?, *data.get(pos + 3)?]) as usize;
        let end = pos + 2 + len;
        if len < 2 || end > data.len() {
            return None;
        }
        if jpeg_segment_kept(marker, &data[pos + 4..end]) {
            out.extend_from_slice(&data[pos..end]);
        }
        pos = end;
    }
    None
}

/// PNG chunks that are metadata, not pixels or colour.
const PNG_DROP: &[&[u8; 4]] = &[b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"tIME"];

fn strip_png(data: &[u8]) -> Option<Vec<u8>> {
    const SIG: &[u8] = b"\x89PNG\r\n\x1a\n";
    if !data.starts_with(SIG) {
        return None;
    }
    let mut out = SIG.to_vec();
    let mut pos = SIG.len();
    while pos + 8 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        let ctype = &data[pos + 4..pos + 8];
        let end = pos.checked_add(12)?.checked_add(len)?;
        if end > data.len() {
            return None;
        }
        if !PNG_DROP.iter().any(|d| d.as_slice() == ctype) {
            out.extend_from_slice(&data[pos..end]);
        }
        pos = end;
        if ctype == b"IEND" {
            return Some(out);
        }
    }
    None
}

/// The same file without its metadata, pixels untouched (no re-encode). None
/// for a file it cannot walk.
pub fn strip_metadata(data: &[u8], jpeg: bool) -> Option<Vec<u8>> {
    if jpeg { strip_jpeg(data) } else { strip_png(data) }
}

/// More than one frame (GIF, WebP, APNG) — only the first is sent.
fn is_animated(data: &[u8], fmt: Option<ImageFormat>) -> bool {
    use image::AnimationDecoder;
    match fmt {
        Some(ImageFormat::Gif) => image::codecs::gif::GifDecoder::new(Cursor::new(data))
            .map(|d| d.into_frames().take(2).count() > 1)
            .unwrap_or(false),
        Some(ImageFormat::WebP) => image::codecs::webp::WebPDecoder::new(Cursor::new(data))
            .map(|d| d.has_animation())
            .unwrap_or(false),
        Some(ImageFormat::Png) => image::codecs::png::PngDecoder::new(Cursor::new(data))
            .and_then(|d| d.is_apng())
            .unwrap_or(false),
        _ => false,
    }
}

/// Server `_fit_aligned_size_within_budget` for a still image -> (h, w) canvas.
fn fit_canvas(h: u32, w: u32) -> (u32, u32) {
    let (mut low, mut high) = (1i64, h as i64);
    let mut best = (CELL, CELL);
    while low <= high {
        let ch = ((low + high) / 2) as u32;
        let cw = ((w as u64 * ch as u64) / h as u64).max(1) as u32;
        let (ah, aw) = (ceil_to(ch, CELL), ceil_to(cw, CELL));
        if ah as u64 * aw as u64 <= BUDGET_PIXELS {
            best = (ah, aw);
            low = ch as i64 + 1;
        } else {
            high = ch as i64 - 1;
        }
    }
    best
}

/// Tokens the server spends on an image of this size (pad mode).
pub fn image_tokens(width: u32, height: u32) -> u32 {
    let (mut cw, mut ch) = (ceil_to(width, CELL), ceil_to(height, CELL));
    if cw as u64 * ch as u64 > BUDGET_PIXELS {
        let (h2, w2) = fit_canvas(height, width);
        ch = h2;
        cw = w2;
    }
    ((cw / CELL) * (ch / CELL)).max(MIN_TOKENS_PER_IMAGE)
}

/// Content size (w, h) the server would shrink an over-budget image to.
pub fn fit_size(width: u32, height: u32) -> (u32, u32) {
    let (canvas_h, canvas_w) = fit_canvas(height, width);
    let scale = 1.0f64
        .min(canvas_h as f64 / height as f64)
        .min(canvas_w as f64 / width as f64);
    let fw = ((width as f64 * scale).floor() as u32).min(canvas_w).max(1);
    let fh = ((height as f64 * scale).floor() as u32).min(canvas_h).max(1);
    (fw, fh)
}

fn encode(img: &DynamicImage, jpeg: bool) -> Result<Vec<u8>, String> {
    let mut buf = Cursor::new(Vec::new());
    if jpeg {
        JpegEncoder::new_with_quality(&mut buf, 92)
            .encode_image(&img.to_rgb8())
            .map_err(|e| e.to_string())?;
    } else {
        img.write_to(&mut buf, ImageFormat::Png).map_err(|e| e.to_string())?;
    }
    Ok(buf.into_inner())
}

fn prepared(img: &DynamicImage, jpeg: bool, animated: bool) -> Result<PreparedImage, String> {
    let (w, h) = (img.width(), img.height());
    Ok(PreparedImage {
        data: encode(img, jpeg)?,
        media_type: if jpeg { "image/jpeg" } else { "image/png" },
        width: w,
        height: h,
        tokens: image_tokens(w, h),
        animated,
    })
}

/// Turn one original image into what is sent (one image, or tiles).
pub fn prepare_image(data: &[u8], slots_left: usize) -> Result<Vec<PreparedImage>, String> {
    let reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let fmt_in = reader.format();
    let mut decoder = reader.into_decoder().map_err(|e| e.to_string())?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).map_err(|e| e.to_string())?;
    let rotated = orientation != Orientation::NoTransforms;
    if rotated {
        img.apply_orientation(orientation);
    }
    let jpeg = fmt_in == Some(ImageFormat::Jpeg);
    let animated = is_animated(data, fmt_in);
    let (w, h) = (img.width(), img.height());
    let slots_left = slots_left.max(1);

    if padded_area(w, h) <= BUDGET_PIXELS {
        if !rotated && matches!(fmt_in, Some(ImageFormat::Png) | Some(ImageFormat::Jpeg)) {
            if let Some(stripped) = strip_metadata(data, jpeg) {
                return Ok(vec![PreparedImage {
                    data: stripped,
                    media_type: if jpeg { "image/jpeg" } else { "image/png" },
                    width: w,
                    height: h,
                    tokens: image_tokens(w, h),
                    animated,
                }]);
            } // cannot walk it: re-encode
        }
        return Ok(vec![prepared(&img, jpeg, animated)?]);
    }

    let (fw, fh) = fit_size(w, h);
    let short = w.min(h);
    let long = w.max(h);
    if fw.min(fh) >= MIN_READABLE_SHORT_SIDE || slots_left <= 1 {
        let small = img.resize_exact(fw, fh, FilterType::Lanczos3);
        return Ok(vec![prepared(&small, jpeg, animated)?]);
    }

    // Tile along the long side. Keep the short side readable (capped), and if
    // that needs more tiles than slots, shrink until it fits the slots.
    let mut s = short.min(MAX_TILE_SHORT_SIDE);
    let (mut f, mut n);
    loop {
        f = s as f64 / short as f64;
        // The server measures the padded canvas, so the tile length is what
        // fits next to the short side rounded UP to the cell.
        let tile_len = (BUDGET_PIXELS / ceil_to(s, CELL) as u64) as u32 / CELL * CELL;
        n = (long as f64 * f / tile_len as f64).ceil() as usize;
        if n <= slots_left || s <= MIN_READABLE_SHORT_SIDE / 2 {
            break;
        }
        s = (MIN_READABLE_SHORT_SIDE / 2)
            .max((s as f64 * (slots_left as f64 / n as f64).sqrt()) as u32);
    }
    let n = n.min(slots_left).max(1);
    if f < 1.0 {
        let nw = (w as f64 * f).round_ties_even() as u32;
        let nh = (h as f64 * f).round_ties_even() as u32;
        img = img.resize_exact(nw, nh, FilterType::Lanczos3);
    }
    let (big_w, big_h) = (img.width(), img.height());
    let vertical = big_h >= big_w;
    let step = (if vertical { big_h } else { big_w }).div_ceil(n as u32);
    let mut tiles = Vec::with_capacity(n);
    for i in 0..n as u32 {
        let mut t = if vertical {
            let y0 = (i * step).min(big_h);
            img.crop_imm(0, y0, big_w, big_h.min((i + 1) * step).saturating_sub(y0))
        } else {
            let x0 = (i * step).min(big_w);
            img.crop_imm(x0, 0, big_w.min((i + 1) * step).saturating_sub(x0), big_h)
        };
        if t.width() == 0 || t.height() == 0 {
            continue;
        }
        if padded_area(t.width(), t.height()) > BUDGET_PIXELS {
            // rounding guard
            let (tw, th) = fit_size(t.width(), t.height());
            t = t.resize_exact(tw, th, FilterType::Lanczos3);
        }
        tiles.push(prepared(&t, jpeg, animated)?);
    }
    Ok(tiles)
}

// === Token cost of an image block already in the conversation ===
//
// Counting runs on every turn, so read only the header: the first ~96 KB of the
// image covers a PNG IHDR and a JPEG SOF behind its EXIF (an EXIF larger than
// that — a big embedded thumbnail — falls back to the whole image). Blocks made by
// prepare_image never carry a rotation tag (rotation is baked into the pixels),
// so the header size is the size the server sees.
const HEADER_B64_CHARS: usize = 131072; // multiple of 4

static TOKEN_CACHE: once_cell::sync::Lazy<parking_lot::Mutex<std::collections::HashMap<(usize, String, String), u32>>> =
    once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(std::collections::HashMap::new()));

/// Server token cost of an Anthropic image `source`; MAX when it cannot tell.
pub fn image_source_tokens(source: &serde_json::Value) -> u32 {
    let Some(data) = source.get("data").and_then(|d| d.as_str()).filter(|d| !d.is_empty()) else {
        return MAX_TOKENS_PER_IMAGE;
    };
    let head: String = data.chars().take(48).collect();
    let tail: String = data.chars().rev().take(48).collect();
    let key = (data.len(), head, tail);
    if let Some(t) = TOKEN_CACHE.lock().get(&key) {
        return *t;
    }
    let size_of = |b64: &str| {
        crate::utils::image_store::decode_b64(b64).and_then(|raw| {
            ImageReader::new(Cursor::new(raw)).with_guessed_format().ok()?.into_dimensions().ok()
        })
    };
    let cut = data.len().min(HEADER_B64_CHARS) / 4 * 4;
    let mut dims = data.get(..cut).and_then(size_of);
    if dims.is_none() && data.len() > cut {
        dims = size_of(data);
    }
    let tokens = dims.map(|(w, h)| image_tokens(w, h)).unwrap_or(MAX_TOKENS_PER_IMAGE);
    let mut cache = TOKEN_CACHE.lock();
    if cache.len() > 4096 {
        cache.clear();
    }
    cache.insert(key, tokens);
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_match_server_measurements() {
        // Prompt-token deltas measured on the live server (glm53 vision lane).
        assert_eq!(image_tokens(1792, 1792), 4096);
        assert_eq!(image_tokens(896, 896), 1024);
        assert_eq!(image_tokens(4000, 3000), 4070);
        assert_eq!(image_tokens(8000, 6000), 4070);
        assert_eq!(image_tokens(6000, 4000), 4056);
        assert_eq!(image_tokens(1920, 1080), 2691);
        assert_eq!(image_tokens(3840, 2160), 4080);
    }

    #[test]
    fn fit_size_keeps_token_count() {
        for (w, h) in [(3840u32, 2160u32), (4000, 3000), (6000, 4000), (8000, 6000)] {
            let (fw, fh) = fit_size(w, h);
            assert!(fw as u64 * fh as u64 <= BUDGET_PIXELS);
            assert_eq!(image_tokens(fw, fh), image_tokens(w, h), "{w}x{h} -> {fw}x{fh}");
        }
        assert_eq!(fit_size(3840, 2160), (2380, 1338));
        assert_eq!(fit_size(4000, 3000), (2053, 1540));
    }

    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        DynamicImage::new_rgb8(w, h).write_to(&mut buf, ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// Same inputs, same outputs as codemason/utils/image_prep.py.
    #[test]
    fn tiling_matches_python() {
        let dims = |v: &Vec<PreparedImage>| v.iter().map(|p| (p.width, p.height)).collect::<Vec<_>>();
        let long = prepare_image(&png(1080, 20000), 8).unwrap();
        assert_eq!(long.len(), 7);
        assert_eq!(dims(&long)[0], (1080, 2858));
        let few = prepare_image(&png(1080, 20000), 3).unwrap();
        assert_eq!(dims(&few), vec![(707, 4365), (707, 4365), (707, 4363)]);
        let wide = prepare_image(&png(30000, 1200), 8).unwrap();
        assert_eq!(wide.len(), 8);
        assert_eq!(dims(&wide)[0], (3060, 979));
        let within = png(2560, 1080);
        let one = prepare_image(&within, 8).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].data, within, "within budget, no metadata: the original bytes");
        assert_eq!(dims(&prepare_image(&png(300, 200), 8).unwrap()), vec![(300, 200)]);
        // Area fits, padded canvas does not: shrunk here, once.
        assert_eq!(dims(&prepare_image(&png(2000, 1600), 8).unwrap()), vec![(1988, 1590)]);
        // Short side not a multiple of 28: tiles still fit padded.
        let odd = prepare_image(&png(1000, 30000), 8).unwrap();
        assert_eq!(odd.len(), 8);
        assert_eq!(dims(&odd)[0], (894, 3353));
        let four = prepare_image(&png(1080, 30000), 4).unwrap();
        assert_eq!(dims(&four)[0], (651, 4521));
        for t in long.iter().chain(&few).chain(&wide).chain(&odd).chain(&four) {
            assert!(padded_area(t.width, t.height) <= BUDGET_PIXELS, "{}x{}", t.width, t.height);
        }
    }

    /// EXIF/XMP/text chunks go, the scan data stays byte for byte.
    #[test]
    fn metadata_stripped_losslessly() {
        let mut buf = Cursor::new(Vec::new());
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90)
            .encode_image(&DynamicImage::new_rgb8(64, 48).to_rgb8())
            .unwrap();
        let plain = buf.into_inner();
        let exif = b"Exif\0\0MM\0*\0\0\0\x08CameraMaker GPS";
        let mut seg = vec![0xFF, 0xE1];
        seg.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
        seg.extend_from_slice(exif);
        let with: Vec<u8> = [&plain[..2], &seg[..], &plain[2..]].concat();
        let out = prepare_image(&with, 8).unwrap().remove(0);
        let find = |h: &[u8], n: &[u8]| h.windows(n.len()).position(|w| w == n);
        assert!(find(&out.data, b"CameraMaker").is_none());
        let sos = |d: &[u8]| find(d, &[0xFF, 0xDA]).unwrap();
        assert_eq!(out.data[sos(&out.data)..], with[sos(&with)..]);
        let mut png_text = png(50, 40);
        let iend = png_text.len() - 12;
        let mut chunk = Vec::new();
        let body = b"Comment\0secret place";
        chunk.extend_from_slice(&(body.len() as u32).to_be_bytes());
        chunk.extend_from_slice(b"tEXt");
        chunk.extend_from_slice(body);
        chunk.extend_from_slice(&[0, 0, 0, 0]); // CRC is not checked by the walker
        png_text.splice(iend..iend, chunk);
        let out = strip_metadata(&png_text, false).unwrap();
        assert!(find(&out, b"secret place").is_none() && out.ends_with(b"IEND\xaeB`\x82"));
    }

    #[test]
    fn animated_gif_first_frame() {
        use image::codecs::gif::GifEncoder;
        let mut buf = Vec::new();
        {
            let mut enc = GifEncoder::new(&mut buf);
            for c in [255u8, 0] {
                let f = image::RgbaImage::from_pixel(40, 30, image::Rgba([c, 255 - c, 0, 255]));
                enc.encode_frame(image::Frame::new(f)).unwrap();
            }
        }
        let out = prepare_image(&buf, 8).unwrap();
        assert!(out.len() == 1 && out[0].animated && out[0].media_type == "image/png");
        assert!(!prepare_image(&png(40, 30), 8).unwrap()[0].animated);
    }

    #[test]
    fn block_tokens_from_header() {
        use base64::Engine;
        let img = DynamicImage::new_rgb8(1920, 1080);
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, ImageFormat::Png).unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        let src = serde_json::json!({"type": "base64", "media_type": "image/png", "data": b64});
        assert_eq!(image_source_tokens(&src), 2691);
        assert_eq!(image_source_tokens(&serde_json::json!({"data": "AAAA"})), MAX_TOKENS_PER_IMAGE);
        // SOF behind APP segments bigger than the header window.
        let mut jb = Cursor::new(Vec::new());
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jb, 90)
            .encode_image(&DynamicImage::new_rgb8(640, 480).to_rgb8())
            .unwrap();
        let small = jb.into_inner();
        let mut xmp = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
        xmp.resize(xmp.len() + 60000, b' ');
        let mut seg = vec![0xFF, 0xE1];
        seg.extend_from_slice(&((xmp.len() + 2) as u16).to_be_bytes());
        seg.extend_from_slice(&xmp);
        let big: Vec<u8> = [&small[..2], &seg[..], &seg[..], &seg[..], &small[2..]].concat();
        let src = serde_json::json!({"data": base64::engine::general_purpose::STANDARD.encode(&big)});
        assert_eq!(image_source_tokens(&src), image_tokens(640, 480));
    }
}
