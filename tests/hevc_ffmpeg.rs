//! Conformance oracle for the native HEVC encoder.
//!
//! The runtime encoder has **no** ffmpeg dependency — but in dev we use ffmpeg
//! purely as an *independent decoder* to prove our from-scratch bitstream is
//! actually decodable and that the `I_PCM` path is lossless in the YUV domain
//! (decoded planar YUV must equal the samples we fed in, byte for byte).
//!
//! Runs only with `--features native-codec`; silently skips if ffmpeg is absent.
//!
//! ```text
//! cargo test --features native-codec --test hevc_ffmpeg -- --nocapture
//! ```
#![cfg(feature = "native-codec")]

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use threers::codec::hevc::{HevcEncoder, Yuv420Frame};

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn tmp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "threers_hevc_{}_{}_{name}",
        std::process::id(),
        nanos
    ));
    p
}

/// Build a deterministic, high-frequency test frame — the strongest input for
/// catching sample-order or alignment bugs (uniform data would mask them).
fn checker(width: u32, height: u32) -> Yuv420Frame {
    let mut f = Yuv420Frame::new(width, height);
    for j in 0..height {
        for i in 0..width {
            // A busy pattern spanning the full 0..255 range.
            f.y[(j * width + i) as usize] =
                ((i.wrapping_mul(37) ^ j.wrapping_mul(101)) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (width / 2, height / 2);
    for j in 0..ch {
        for i in 0..cw {
            f.u[(j * cw + i) as usize] = ((i * 5 + j * 3 + 20) & 0xFF) as u8;
            f.v[(j * cw + i) as usize] = ((i * 2 + j * 9 + 200) & 0xFF) as u8;
        }
    }
    f
}

/// Encode one frame, decode it with ffmpeg to raw yuv420p, and assert an exact
/// match against the (display-size) input planes.
fn roundtrip_lossless(width: u32, height: u32) {
    if !ffmpeg_available() {
        eprintln!("skipping HEVC ffmpeg roundtrip ({width}x{height}): ffmpeg not found");
        return;
    }

    let frame = checker(width, height);
    let mut enc = HevcEncoder::new(width, height);
    let au = enc.encode_frame(&frame);

    let in265 = tmp_path("in.265");
    let out_yuv = tmp_path("out.yuv");
    std::fs::write(&in265, &au).unwrap();

    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "hevc", "-i"])
        .arg(&in265)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&out_yuv)
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg failed to decode our stream ({width}x{height})"
    );

    let decoded = std::fs::read(&out_yuv).unwrap();
    let _ = std::fs::remove_file(&in265);
    let _ = std::fs::remove_file(&out_yuv);

    let mut expected = Vec::with_capacity(frame.y.len() + frame.u.len() + frame.v.len());
    expected.extend_from_slice(&frame.y);
    expected.extend_from_slice(&frame.u);
    expected.extend_from_slice(&frame.v);

    assert_eq!(
        decoded.len(),
        expected.len(),
        "decoded plane size mismatch at {width}x{height} (got {}, want {})",
        decoded.len(),
        expected.len()
    );
    // Locate the first differing byte for a useful message if it ever breaks.
    if let Some(pos) = decoded.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "sample mismatch at {width}x{height}, byte {pos}: decoded {} != expected {}",
            decoded[pos], expected[pos]
        );
    }
    eprintln!(
        "HEVC ffmpeg roundtrip OK: {width}x{height} lossless ({} bytes .265)",
        au.len()
    );
}

/// Encode a monochrome frame (the alpha auxiliary layer's format), decode it
/// with ffmpeg as a standalone `gray` stream, and assert an exact match. This is
/// the "standalone" half of transparent-video verification — it proves the alpha
/// channel is coded losslessly, independent of multi-layer compositing.
fn roundtrip_gray_lossless(width: u32, height: u32) {
    if !ffmpeg_available() {
        eprintln!("skipping HEVC mono roundtrip ({width}x{height}): ffmpeg not found");
        return;
    }
    // A busy alpha ramp: fully transparent → opaque across the frame.
    let gray: Vec<u8> = (0..width * height)
        .map(|i| ((i.wrapping_mul(53) ^ (i >> 3)) & 0xFF) as u8)
        .collect();

    let mut enc = HevcEncoder::new_monochrome(width, height);
    let au = enc.encode_gray_frame(&gray);

    let in265 = tmp_path("mono.265");
    let out_gray = tmp_path("mono.gray");
    std::fs::write(&in265, &au).unwrap();
    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "hevc", "-i"])
        .arg(&in265)
        .args(["-f", "rawvideo", "-pix_fmt", "gray"])
        .arg(&out_gray)
        .status()
        .expect("run ffmpeg");
    assert!(
        status.success(),
        "ffmpeg failed to decode our monochrome stream ({width}x{height})"
    );
    let decoded = std::fs::read(&out_gray).unwrap();
    let _ = std::fs::remove_file(&in265);
    let _ = std::fs::remove_file(&out_gray);

    assert_eq!(
        decoded.len(),
        gray.len(),
        "mono plane size mismatch {width}x{height}"
    );
    if let Some(pos) = decoded.iter().zip(&gray).position(|(a, b)| a != b) {
        panic!(
            "mono mismatch at {width}x{height} byte {pos}: {} != {}",
            decoded[pos], gray[pos]
        );
    }
    eprintln!(
        "HEVC mono (alpha-layer) roundtrip OK: {width}x{height} lossless ({} bytes)",
        au.len()
    );
}

#[test]
fn ctb_aligned_is_lossless() {
    // 64x64 → 16 CTBs, no conformance window.
    roundtrip_lossless(64, 64);
}

#[test]
fn monochrome_alpha_layer_is_lossless() {
    roundtrip_gray_lossless(64, 64);
    roundtrip_gray_lossless(100, 100); // exercises conformance window in mono units
}

#[test]
fn non_square_aligned_is_lossless() {
    roundtrip_lossless(128, 48);
}

#[test]
fn conformance_window_crops_correctly() {
    // 66x66 → coded 80x80, cropped back to 66x66 via the conformance window.
    roundtrip_lossless(66, 66);
    // A realistic odd height: 640x360 is aligned, 640x358 exercises padding.
    roundtrip_lossless(640, 358);
}
