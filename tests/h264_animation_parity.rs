//! RGBA animation-path parity for H.264 / MP4 (wasm-facing `encodeMp4Rgba`).
//!
//! Golden checksums are shared with `web/examples/video-export-test.html` so
//! native and browser encoders stay byte-identical for fixed inputs.
//!
//! ```text
//! cargo test --features native-codec --test h264_animation_parity -- --nocapture
//! ```
#![cfg(feature = "native-codec")]

use threers::codec::hevc::Yuv420Frame;
use threers::codec::h264::{encode_compressed_mp4, encode_mp4 as encode_pcm_mp4};
use threers::{encode_animation_rgba, AnimationEncodeOptions, BrowserCodec};

/// FNV-1a 32-bit — must match `web/scripts/video-export-test-lib.mjs`.
fn fnv1a(data: &[u8]) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in data {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h
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

fn checker_rgba(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut out = vec![0; (width * height * 4) as usize];
    let s = seed.wrapping_mul(17);
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            let v = ((x.wrapping_mul(37) ^ y.wrapping_mul(101) ^ s) & 0xFF) as u8;
            out[i] = v;
            out[i + 1] = v.wrapping_add(40);
            out[i + 2] = v.wrapping_add(80);
            out[i + 3] = 255;
        }
    }
    out
}

fn gray_ramp_rgba(width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            let g = (((x + y) * 255) / (width + height).max(1)) as u8;
            out[i..i + 3].fill(g);
            out[i + 3] = 255;
        }
    }
    out
}

fn encode_mp4_rgba(
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
    encode_animation_rgba(&opts, frames).expect("encode_animation_rgba mp4")
}

fn synthetic_solid_frames(width: u32, height: u32) -> Vec<Vec<u8>> {
    vec![
        solid_rgba(width, height, 255, 0, 0, 255),
        solid_rgba(width, height, 0, 255, 0, 255),
        solid_rgba(width, height, 0, 0, 255, 255),
        solid_rgba(width, height, 255, 255, 0, 255),
    ]
}

#[test]
fn rgba_path_matches_direct_yuv_mp4_bytes() {
    let width = 64;
    let height = 48;
    let rgba_frames = synthetic_solid_frames(width, height);
    let from_rgba = encode_mp4_rgba(width, height, 10, rgba_frames.clone());

    let yuv: Vec<Yuv420Frame> = rgba_frames
        .iter()
        .map(|rgba| Yuv420Frame::from_rgba(width, height, rgba))
        .collect();
    // The browser path uses the compressed encoder at its default QP; this
    // pins the two to the same bytes so the RGBA wrapper stays a pure
    // convenience over the direct API.
    let from_yuv = encode_compressed_mp4(width, height, 10, 26, &yuv);

    assert_eq!(
        from_rgba, from_yuv,
        "RGBA animation path must match direct YUV encode_compressed_mp4"
    );
    assert!(from_rgba.len() > 100);
    assert_eq!(&from_rgba[4..8], b"ftyp");
}

#[test]
fn golden_checksum_64x48_solid_4f_10fps() {
    let width = 64;
    let height = 48;
    let mp4 = encode_mp4_rgba(width, height, 10, synthetic_solid_frames(width, height));
    let sum = fnv1a(&mp4);
    // Shared with browser `video-export-test.html`.
    assert_eq!(
        sum, 0xf5ca7103,
        "golden mp4 checksum — changed when the MP4 path moved from I_PCM to the \
         compressed intra encoder; update the browser test to match"
    );
}

#[test]
fn golden_checksum_128x72_gradient_3f_30fps() {
    let width = 128;
    let height = 72;
    let frames = [
        gradient_rgba(width, height),
        checker_rgba(width, height, 1),
        gray_ramp_rgba(width, height),
    ];
    let mp4 = encode_mp4_rgba(width, height, 30, frames);
    let sum = fnv1a(&mp4);
    assert_eq!(sum, 0x1c091ef4, "golden gradient mp4 checksum");
}

#[test]
fn golden_checksum_320x240_checker_2f_60fps() {
    let width = 320;
    let height = 240;
    let frames = [
        checker_rgba(width, height, 0),
        checker_rgba(width, height, 2),
    ];
    let mp4 = encode_mp4_rgba(width, height, 60, frames);
    let sum = fnv1a(&mp4);
    assert_eq!(sum, 0x06597030, "golden 320x240 mp4 checksum");
}

#[test]
fn fps_does_not_change_sample_payload_for_identical_frames() {
    let width = 64;
    let height = 64;
    let frame = solid_rgba(width, height, 128, 64, 32, 255);
    let at_10 = encode_mp4_rgba(width, height, 10, [frame.clone()]);
    let at_60 = encode_mp4_rgba(width, height, 60, [frame]);
    // Mux timing differs; video sample NAL payloads should match.
    assert_ne!(at_10, at_60, "container timing metadata differs by fps");
    // Both must be valid MP4 with identical ftyp brand layout start.
    assert_eq!(&at_10[4..8], b"ftyp");
    assert_eq!(&at_60[4..8], b"ftyp");
    // Sample bytes (mdat) should be same length — I_PCM is fps-independent.
    assert_eq!(
        at_10.len(),
        at_60.len(),
        "I_PCM payload size should not depend on fps"
    );
}

#[test]
fn color_schemes_produce_distinct_outputs() {
    let width = 64;
    let height = 64;
    let schemes = [
        ("solid-red", solid_rgba(width, height, 255, 0, 0, 255)),
        ("gradient", gradient_rgba(width, height)),
        ("checker", checker_rgba(width, height, 3)),
        ("gray-ramp", gray_ramp_rgba(width, height)),
        ("low-bitness", {
            let mut f = solid_rgba(width, height, 0, 0, 0, 255);
            for (i, px) in f.chunks_exact_mut(4).enumerate() {
                let q = ((i % 4) * 64) as u8;
                px[0] = q;
                px[1] = q.wrapping_add(16);
                px[2] = q.wrapping_add(32);
            }
            f
        }),
    ];
    let mut sums = Vec::new();
    for (name, frame) in schemes {
        let mp4 = encode_mp4_rgba(width, height, 24, [frame]);
        assert_eq!(&mp4[4..8], b"ftyp", "{name}: missing ftyp");
        sums.push((name, fnv1a(&mp4)));
    }
    for i in 0..sums.len() {
        for j in (i + 1)..sums.len() {
            assert_ne!(
                sums[i].1, sums[j].1,
                "{} and {} must produce distinct mp4 bytes",
                sums[i].0, sums[j].0
            );
        }
    }
}

#[test]
fn resolution_matrix_encodes() {
    for (width, height) in [(64, 48), (128, 72), (320, 240), (640, 360), (1280, 720)] {
        let frames = synthetic_solid_frames(width, height);
        let mp4 = encode_mp4_rgba(width, height, 30, frames);
        assert_eq!(&mp4[4..8], b"ftyp", "{width}x{height}");
        assert!(mp4.len() > 200, "{width}x{height} too small");
    }
}

/// The `I_PCM` encoder is still reachable and still lossless, even though the
/// export paths no longer default to it.
#[test]
fn pcm_encoder_remains_available_and_is_far_larger() {
    let (width, height) = (64u32, 48u32);
    let rgba = synthetic_solid_frames(width, height);
    let yuv: Vec<Yuv420Frame> = rgba
        .iter()
        .map(|f| Yuv420Frame::from_rgba(width, height, f))
        .collect();
    let pcm = encode_pcm_mp4(width, height, 10, &yuv);
    let compressed = encode_compressed_mp4(width, height, 10, 26, &yuv);
    assert!(
        compressed.len() * 4 < pcm.len(),
        "compressed {} vs I_PCM {}",
        compressed.len(),
        pcm.len()
    );
}
