//! Verify the compressed HEVC encoder against ffmpeg: the decoded frame must
//! exactly match the encoder's reconstruction (a conformant decoder reproduces
//! the encoder's samples bit-for-bit), and the stream must be far smaller than
//! raw. Uses high-contrast content so the intra boundary filters are exercised.
#![cfg(feature = "native-codec")]

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use threers::codec::hevc::{CompressedEncoder, Yuv420Frame};

fn ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn tmp(name: &str) -> std::path::PathBuf {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("threers_c_{}_{n}_{name}", std::process::id()))
}

/// High-contrast content (sharp edges + gradient) — exercises DC/H/V boundary
/// filters, planar, and angular modes.
fn test_image(w: u32, h: u32) -> Yuv420Frame {
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let block = ((x / 5) + (y / 3)) % 2 == 0;
            rgba[i] = if block { 235 } else { (x * 255 / w) as u8 };
            rgba[i + 1] = if block { 20 } else { (y * 255 / h) as u8 };
            rgba[i + 2] = if (x / 7) % 2 == 0 { 200 } else { 40 };
            rgba[i + 3] = 255;
        }
    }
    Yuv420Frame::from_rgba(w, h, &rgba)
}

/// A smooth low-frequency gradient. Residual coefficients stay sparse and
/// low-frequency, which the transform-coefficient path codes conformantly.
fn smooth_image(w: u32, h: u32) -> Yuv420Frame {
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            rgba[i] = (40 + x * 3) as u8;
            rgba[i + 1] = (60 + y * 2) as u8;
            rgba[i + 2] = (80 + (x + y)) as u8;
            rgba[i + 3] = 255;
        }
    }
    Yuv420Frame::from_rgba(w, h, &rgba)
}

fn check(w: u32, h: u32) {
    check_frame(w, h, test_image(w, h), false);
}

/// Encode `frame`, assert it is smaller than raw, and — when ffmpeg is present —
/// assert the decode reproduces the encoder's reconstruction bit-for-bit.
fn check_frame(w: u32, h: u32, frame: Yuv420Frame, residual: bool) {
    let mut enc = CompressedEncoder::new(w, h, 26).residual(residual);
    let (au, recon) = enc.encode_frame(&frame);

    let raw = (w * h + 2 * (w / 2) * (h / 2)) as usize;
    let ratio = raw as f64 / au.len() as f64;
    eprintln!(
        "compressed {w}x{h} (residual={residual}): {} bytes vs {raw} raw ({ratio:.1}x smaller)",
        au.len()
    );
    assert!(au.len() < raw, "must be smaller than raw at {w}x{h}");

    if !ffmpeg() {
        eprintln!("skipping decode check ({w}x{h}): ffmpeg not found");
        return;
    }
    let in265 = tmp("in.265");
    let out = tmp("out.yuv");
    std::fs::write(&in265, &au).unwrap();
    let ok = Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "hevc",
            "-i",
        ])
        .arg(&in265)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&out)
        .status()
        .expect("ffmpeg")
        .success();
    assert!(ok, "ffmpeg failed to decode ({w}x{h})");
    let decoded = std::fs::read(&out).unwrap();
    let _ = std::fs::remove_file(&in265);
    let _ = std::fs::remove_file(&out);

    // Compare against the reconstruction, cropped to display size (conformance
    // window). recon is at coded (CTB-aligned) size.
    let (cw, ch) = (recon.coded_width, recon.coded_height);
    let mut expected = Vec::new();
    for row in 0..h {
        expected.extend_from_slice(&recon.y[(row * cw) as usize..(row * cw + w) as usize]);
    }
    let (cwc, chc, wc, hc) = (cw / 2, ch / 2, w / 2, h / 2);
    let _ = chc;
    for plane in [&recon.u, &recon.v] {
        for row in 0..hc {
            expected.extend_from_slice(&plane[(row * cwc) as usize..(row * cwc + wc) as usize]);
        }
    }
    assert_eq!(decoded.len(), expected.len(), "plane size {w}x{h}");
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "decoder != reconstruction at byte {pos}: {} != {} ({w}x{h})",
            decoded[pos], expected[pos]
        );
    }
    eprintln!("conformant: {w}x{h} decoder == reconstruction");
}

#[test]
fn compressed_prediction_only_is_conformant() {
    // Aligned and non-aligned (conformance window) sizes, high-contrast content.
    check(128, 128);
    check(64, 64);
    check(80, 48);
}

#[test]
fn compressed_residual_smooth_is_conformant() {
    // Transform-coefficient residual coding (experimental path) on smooth,
    // low-frequency content: ffmpeg decodes to exactly the encoder's
    // reconstruction. Guards the chroma residual fixes (sig-context table and
    // luma-only reference smoothing) against regression.
    check_frame(16, 16, smooth_image(16, 16), true);
    check_frame(32, 32, smooth_image(32, 32), true);
    check_frame(64, 64, smooth_image(64, 64), true);
}
