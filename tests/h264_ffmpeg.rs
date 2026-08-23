//! Conformance oracle for the native H.264 encoder.
//!
//! Uses ffmpeg purely as an independent decoder to prove the from-scratch
//! bitstream is decodable and that the I_PCM path is lossless in the YUV domain.
//!
//! ```text
//! cargo test --features native-codec --test h264_ffmpeg -- --nocapture
//! ```
#![cfg(feature = "native-codec")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use threers::codec::h264::{encode_mp4, H264Encoder};
use threers::codec::hevc::Yuv420Frame;
use threers::{encode_animation_rgba, AnimationEncodeOptions, BrowserCodec};

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn ffprobe_available() -> bool {
    Command::new("ffprobe")
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
        "threers_h264_{}_{}_{name}",
        std::process::id(),
        nanos
    ));
    p
}

fn checker(width: u32, height: u32) -> Yuv420Frame {
    checker_offset(width, height, 0)
}

/// Deterministic YUV420 picture; `seed` shifts the pattern per frame.
fn checker_offset(width: u32, height: u32, seed: u32) -> Yuv420Frame {
    let mut f = Yuv420Frame::new(width, height);
    let s = seed.wrapping_mul(17);
    for j in 0..height {
        for i in 0..width {
            f.y[(j * width + i) as usize] =
                ((i.wrapping_mul(37) ^ j.wrapping_mul(101) ^ s) & 0xFF) as u8;
        }
    }
    let (cw, ch) = (width / 2, height / 2);
    for j in 0..ch {
        for i in 0..cw {
            f.u[(j * cw + i) as usize] =
                ((i.wrapping_mul(5) + j.wrapping_mul(3) + 20 + s) & 0xFF) as u8;
            f.v[(j * cw + i) as usize] =
                ((i.wrapping_mul(2) + j.wrapping_mul(9) + 200 + s) & 0xFF) as u8;
        }
    }
    f
}

fn planes_to_vec(frames: &[Yuv420Frame]) -> Vec<u8> {
    let mut out = Vec::new();
    for f in frames {
        out.extend_from_slice(&f.y);
        out.extend_from_slice(&f.u);
        out.extend_from_slice(&f.v);
    }
    out
}

/// Decode any ffmpeg-readable H.264 bitstream or MP4 to raw yuv420p.
fn decode_to_yuv(input: &Path) -> Option<Vec<u8>> {
    let out = tmp_path("dec.yuv");
    let ok = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&out)
        .status()
        .ok()?
        .success();
    let data = if ok { std::fs::read(&out).ok() } else { None };
    let _ = std::fs::remove_file(&out);
    data
}

fn assert_planes_match(decoded: &[u8], expected: &[u8], label: &str) {
    assert_eq!(
        decoded.len(),
        expected.len(),
        "{label}: decoded plane size mismatch (got {}, want {})",
        decoded.len(),
        expected.len()
    );
    if let Some(pos) = decoded.iter().zip(expected).position(|(a, b)| a != b) {
        panic!(
            "{label}: sample mismatch at byte {pos}: decoded {} != expected {}",
            decoded[pos], expected[pos]
        );
    }
}

fn roundtrip_lossless(width: u32, height: u32) {
    if !ffmpeg_available() {
        eprintln!("skipping H.264 ffmpeg roundtrip ({width}x{height}): ffmpeg not found");
        return;
    }

    let frame = checker(width, height);
    let mut enc = H264Encoder::new(width, height);
    let au = enc.encode_frame(&frame);

    let in264 = tmp_path("in.264");
    std::fs::write(&in264, &au).unwrap();

    let decoded = decode_to_yuv(&in264)
        .unwrap_or_else(|| panic!("ffmpeg failed to decode our Annex-B stream ({width}x{height})"));
    let _ = std::fs::remove_file(&in264);

    assert_planes_match(
        &decoded,
        &planes_to_vec(&[frame]),
        &format!("Annex-B {width}x{height}"),
    );
    eprintln!(
        "H.264 ffmpeg roundtrip OK: {width}x{height} lossless ({} bytes .264)",
        au.len()
    );
}

fn mp4_roundtrip_lossless(width: u32, height: u32, frames: &[Yuv420Frame]) {
    if !ffmpeg_available() {
        eprintln!("skipping H.264 MP4 ffmpeg roundtrip ({width}x{height}): ffmpeg not found");
        return;
    }

    let mp4 = encode_mp4(width, height, 30, frames);
    let path = tmp_path("roundtrip.mp4");
    std::fs::write(&path, &mp4).unwrap();

    let decoded = decode_to_yuv(&path).unwrap_or_else(|| {
        panic!(
            "ffmpeg failed to decode our MP4 ({width}x{height}, {} frames)",
            frames.len()
        )
    });
    let _ = std::fs::remove_file(&path);

    assert_planes_match(
        &decoded,
        &planes_to_vec(frames),
        &format!("MP4 {width}x{height} x{}", frames.len()),
    );
    eprintln!(
        "H.264 MP4 ffmpeg roundtrip OK: {width}x{height} x{} frames ({} bytes .mp4)",
        frames.len(),
        mp4.len()
    );
}

#[test]
fn mb_aligned_is_lossless() {
    roundtrip_lossless(64, 64);
}

#[test]
fn non_square_aligned_is_lossless() {
    roundtrip_lossless(128, 48);
}

#[test]
fn frame_cropping_is_lossless() {
    roundtrip_lossless(66, 66);
    roundtrip_lossless(640, 358);
}

#[test]
fn mp4_single_frame_matches_source_via_ffmpeg() {
    mp4_roundtrip_lossless(64, 64, &[checker(64, 64)]);
    mp4_roundtrip_lossless(66, 66, &[checker(66, 66)]);
}

#[test]
fn mp4_multi_frame_matches_source_via_ffmpeg() {
    let frames = [
        checker_offset(128, 48, 0),
        checker_offset(128, 48, 1),
        checker_offset(128, 48, 2),
    ];
    mp4_roundtrip_lossless(128, 48, &frames);
}

#[test]
fn annexb_and_mp4_decode_identically_via_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("skipping Annex-B vs MP4 parity: ffmpeg not found");
        return;
    }

    let width = 64;
    let height = 64;
    let frame = checker(width, height);
    let mut enc = H264Encoder::new(width, height);
    let au = enc.encode_frame(&frame);
    let mp4 = encode_mp4(width, height, 30, std::slice::from_ref(&frame));

    let h264_path = tmp_path("parity.264");
    let mp4_path = tmp_path("parity.mp4");
    std::fs::write(&h264_path, &au).unwrap();
    std::fs::write(&mp4_path, &mp4).unwrap();

    let from_annexb = decode_to_yuv(&h264_path).expect("ffmpeg decode Annex-B");
    let from_mp4 = decode_to_yuv(&mp4_path).expect("ffmpeg decode MP4");
    let _ = std::fs::remove_file(&h264_path);
    let _ = std::fs::remove_file(&mp4_path);

    assert_planes_match(&from_annexb, &planes_to_vec(std::slice::from_ref(&frame)), "Annex-B");
    assert_planes_match(&from_mp4, &planes_to_vec(&[frame]), "MP4");
    assert_eq!(
        from_annexb, from_mp4,
        "Annex-B and MP4 must decode to identical pixels via ffmpeg"
    );
    eprintln!("H.264 Annex-B / MP4 ffmpeg parity OK ({width}x{height})");
}

#[test]
fn mp4_reports_h264_video_stream() {
    if !ffmpeg_available() || !ffprobe_available() {
        eprintln!("skipping H.264 MP4 probe: ffmpeg/ffprobe not found");
        return;
    }
    let mp4 = encode_mp4(64, 64, 30, &[checker(64, 64)]);
    let path = tmp_path("probe.mp4");
    std::fs::write(&path, &mp4).unwrap();

    let status = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(&path)
        .args(["-f", "null", "-"])
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg failed to open our MP4");

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,codec_tag_string,width,height,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&path)
        .output()
        .expect("run ffprobe");
    let _ = std::fs::remove_file(&path);

    assert!(probe.status.success(), "ffprobe failed on our MP4");
    let text = String::from_utf8_lossy(&probe.stdout);
    assert!(text.contains("codec_name=h264"), "expected h264: {text}");
    assert!(
        text.contains("codec_tag_string=avc1"),
        "expected avc1: {text}"
    );
    assert!(text.contains("width=64"), "expected width=64: {text}");
    assert!(text.contains("height=64"), "expected height=64: {text}");
    assert!(text.contains("pix_fmt=yuv420p"), "expected yuv420p: {text}");
}

fn solid_rgba(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let mut out = vec![0; (width * height * 4) as usize];
    for px in out.chunks_exact_mut(4) {
        px.copy_from_slice(&[r, g, b, a]);
    }
    out
}

fn gradient_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            out[i] = ((x * 255) / width.max(1)) as u8;
            out[i + 1] = ((y * 255) / height.max(1)) as u8;
            out[i + 2] = (((x ^ y) * 37) & 0xFF) as u8;
            out[i + 3] = 255;
        }
    }
    out
}

fn encode_mp4_from_rgba(
    width: u32,
    height: u32,
    fps: u32,
    frames: impl IntoIterator<Item = Vec<u8>>,
) -> Vec<u8> {
    let opts = AnimationEncodeOptions {
        width,
        height,
        fps,
        codec: BrowserCodec::Mp4,
        transparent: false,
        gif_colors: 256,
    };
    encode_animation_rgba(&opts, frames).expect("rgba mp4 encode")
}

fn rgba_mp4_roundtrip_lossless(width: u32, height: u32, fps: u32, rgba_frames: &[Vec<u8>]) {
    if !ffmpeg_available() {
        eprintln!("skipping RGBA MP4 ffmpeg roundtrip ({width}x{height}): ffmpeg not found");
        return;
    }
    let mp4 = encode_mp4_from_rgba(width, height, fps, rgba_frames.iter().cloned());
    let path = tmp_path("rgba_roundtrip.mp4");
    std::fs::write(&path, &mp4).unwrap();

    let decoded = decode_to_yuv(&path).unwrap_or_else(|| {
        panic!(
            "ffmpeg failed to decode RGBA-sourced MP4 ({width}x{height}, {} frames)",
            rgba_frames.len()
        )
    });
    let _ = std::fs::remove_file(&path);

    let yuv: Vec<Yuv420Frame> = rgba_frames
        .iter()
        .map(|rgba| Yuv420Frame::from_rgba(width, height, rgba))
        .collect();
    assert_planes_match(
        &decoded,
        &planes_to_vec(&yuv),
        &format!(
            "RGBA MP4 {width}x{height} x{} @ {fps}fps",
            rgba_frames.len()
        ),
    );
}

fn ffprobe_field(path: &Path, field: &str) -> Option<String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            &format!("stream={field}"),
            "-of",
            "default=noprint_wrappers=1:nokey=0",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(&format!("{field}=")) {
            return Some(rest.to_string());
        }
    }
    None
}

#[test]
fn resolution_matrix_is_lossless_via_ffmpeg() {
    for (width, height) in [(64, 48), (128, 72), (320, 240), (640, 360)] {
        mp4_roundtrip_lossless(width, height, &[checker(width, height)]);
    }
}

#[test]
fn rgba_solid_frames_are_lossless_via_ffmpeg() {
    let width = 64;
    let height = 48;
    let frames = vec![
        solid_rgba(width, height, 255, 0, 0, 255),
        solid_rgba(width, height, 0, 255, 0, 255),
        solid_rgba(width, height, 0, 0, 255, 255),
        solid_rgba(width, height, 255, 255, 0, 255),
    ];
    rgba_mp4_roundtrip_lossless(width, height, 10, &frames);
}

#[test]
fn rgba_gradient_and_low_bitness_are_lossless_via_ffmpeg() {
    let width = 128;
    let height = 72;
    let mut low = solid_rgba(width, height, 0, 0, 0, 255);
    for (i, px) in low.chunks_exact_mut(4).enumerate() {
        let q = ((i % 4) * 64) as u8;
        px[0] = q;
        px[1] = q.wrapping_add(16);
        px[2] = q.wrapping_add(32);
    }
    rgba_mp4_roundtrip_lossless(width, height, 24, &[gradient_rgba(width, height), low]);
}

#[test]
fn mp4_fps_reports_expected_rate_via_ffprobe() {
    if !ffmpeg_available() || !ffprobe_available() {
        eprintln!("skipping fps ffprobe: ffmpeg/ffprobe not found");
        return;
    }
    for (fps, expect) in [(1, "1/1"), (24, "24/1"), (30, "30/1"), (60, "60/1")] {
        let mp4 = encode_mp4(64, 64, fps, &[checker(64, 64)]);
        let path = tmp_path("fps_probe.mp4");
        std::fs::write(&path, &mp4).unwrap();
        let rate = ffprobe_field(&path, "r_frame_rate").unwrap_or_default();
        let _ = std::fs::remove_file(&path);
        assert_eq!(rate, expect, "fps={fps}");
    }
}

#[test]
fn multi_frame_varying_seed_is_lossless() {
    let width = 640;
    let height = 358;
    let frames: Vec<_> = (0..6)
        .map(|seed| checker_offset(width, height, seed))
        .collect();
    mp4_roundtrip_lossless(width, height, &frames);
}
