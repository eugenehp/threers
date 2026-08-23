//! GIF encoder/decoder integration + ffmpeg RGBA conformance.
#![cfg(feature = "native-codec")]

use threers::codec::gif::{
    decode_gif, encode_gif, DisposalMode, GifDecoder, GifEncoder, GifOptions, GifWriter,
    LzwClearMode, PaletteMode, QuantizerKind,
};

fn solid(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
    let mut v = vec![0u8; (w * h * 4) as usize];
    for p in v.chunks_mut(4) {
        p[0] = r;
        p[1] = g;
        p[2] = b;
        p[3] = a;
    }
    v
}

fn ffmpeg_ok() -> bool {
    std::process::Command::new("ffmpeg")
        .args(["-version"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Decode GIF to raw RGBA via ffmpeg (`rgba`, one output frame per GIF frame).
fn decode_gif_rgba(path: &std::path::Path, w: u32, h: u32) -> Option<Vec<u8>> {
    let out = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-i",
            path.to_str()?,
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let frame = (w * h * 4) as usize;
    if out.stdout.is_empty() || out.stdout.len() % frame != 0 {
        return None;
    }
    Some(out.stdout)
}

fn write_temp(name: &str, gif: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, gif).unwrap();
    path
}

#[test]
fn write_gif() {
    let (w, h) = (32u32, 32u32);
    let mut enc = GifEncoder::new(w, h, 0)
        .colors(64)
        .transparency(true)
        .dither(true)
        .diff_rects(true);
    for f in 0..3u32 {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                rgba[i] = (x * 8 + f * 40) as u8;
                rgba[i + 1] = (y * 8) as u8;
                rgba[i + 2] = 200;
                rgba[i + 3] = (x * 255 / (w - 1)) as u8;
            }
        }
        enc.add_frame_owned(rgba, 1, 10);
    }
    let gif = enc.finish();
    assert_eq!(&gif[..6], b"GIF89a");
    assert!(gif.len() > 32);
}

#[test]
fn write_gif_16_color_transparent() {
    let (w, h) = (48u32, 24u32);
    let mut enc = GifEncoder::new(w, h, 0).colors(16).transparency(true);
    for f in 0..2u32 {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let in_circle = {
                    let cx = w as i32 / 2 + (f as i32 - 1) * 8;
                    let cy = h as i32 / 2;
                    let dx = x as i32 - cx;
                    let dy = y as i32 - cy;
                    dx * dx + dy * dy < 80
                };
                if in_circle {
                    rgba[i] = 220;
                    rgba[i + 1] = 40;
                    rgba[i + 2] = 40;
                    rgba[i + 3] = 255;
                }
            }
        }
        enc.add_frame_owned(rgba, 1, 10);
    }
    let gif = enc.finish();
    assert_eq!(&gif[..6], b"GIF89a");
}

#[test]
fn gif_ffmpeg_decodes() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    let (w, h) = (16u32, 16u32);
    let mut enc = GifEncoder::new(w, h, 0)
        .colors(32)
        .dither(false)
        .diff_rects(false)
        .transparency(false);
    for f in 0..2u32 {
        enc.add_frame_owned(solid(w, h, (f * 80) as u8, 40, 200, 255), 1, 10);
    }
    let gif = enc.finish();
    let path = write_temp("threers_gif_ff.gif", &gif);
    let decoded = decode_gif_rgba(&path, w, h).expect("ffmpeg decode");
    let _ = std::fs::remove_file(&path);
    assert_eq!(decoded.len() % ((w * h * 4) as usize), 0);
    assert!(
        decoded.len() >= (w * h * 4 * 2) as usize,
        "expected ≥2 frames, got {} bytes",
        decoded.len()
    );
}

/// Transparent holes must decode near-black (or zero) alpha / background.
#[test]
fn gif_ffmpeg_transparency_holes() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    let (w, h) = (16u32, 16u32);
    let mut enc = GifEncoder::new(w, h, 1)
        .colors(8)
        .dither(false)
        .diff_rects(false)
        .transparency(true)
        .alpha_threshold(128)
        .palette_mode(PaletteMode::Global);
    let mut rgba = solid(w, h, 255, 0, 0, 255);
    // Left half transparent.
    for y in 0..h {
        for x in 0..(w / 2) {
            let i = ((y * w + x) * 4) as usize;
            rgba[i + 3] = 0;
        }
    }
    enc.add_frame_owned(rgba, 1, 10);
    let gif = enc.finish();
    let path = write_temp("threers_gif_alpha.gif", &gif);
    let decoded = decode_gif_rgba(&path, w, h).expect("ffmpeg decode");
    let _ = std::fs::remove_file(&path);

    // Check alpha: left half should be much more transparent than the right.
    let mut left_a = 0u32;
    let mut right_a = 0u32;
    let mut left_n = 0u32;
    let mut right_n = 0u32;
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            if x < w / 2 {
                left_a += decoded[i + 3] as u32;
                left_n += 1;
            } else {
                right_a += decoded[i + 3] as u32;
                right_n += 1;
            }
        }
    }
    let left_avg = left_a / left_n;
    let right_avg = right_a / right_n;
    assert!(
        right_avg > 200,
        "right half should be opaque (avg alpha {right_avg})"
    );
    assert!(
        left_avg + 80 < right_avg,
        "left (transparent) alpha should be lower (left {left_avg} right {right_avg})"
    );
}

/// Quantized solid color should be close after ffmpeg round-trip.
#[test]
fn gif_ffmpeg_color_approx() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    let (w, h) = (24u32, 24u32);
    let target = [40u8, 180, 90];
    let mut enc = GifEncoder::new(w, h, 1)
        .colors(64)
        .dither(false)
        .diff_rects(false)
        .transparency(false)
        .quantizer(QuantizerKind::Octree)
        .palette_mode(PaletteMode::Global);
    enc.add_frame_owned(solid(w, h, target[0], target[1], target[2], 255), 1, 10);
    let gif = enc.finish();
    let path = write_temp("threers_gif_color.gif", &gif);
    let decoded = decode_gif_rgba(&path, w, h).expect("ffmpeg decode");
    let _ = std::fs::remove_file(&path);

    let i = ((h / 2 * w + w / 2) * 4) as usize;
    let dr = (decoded[i] as i32 - target[0] as i32).unsigned_abs();
    let dg = (decoded[i + 1] as i32 - target[1] as i32).unsigned_abs();
    let db = (decoded[i + 2] as i32 - target[2] as i32).unsigned_abs();
    assert!(
        dr < 20 && dg < 20 && db < 20,
        "color drift too large: got [{},{},{}] want {target:?}",
        decoded[i],
        decoded[i + 1],
        decoded[i + 2]
    );
}

#[test]
fn options_builder_roundtrip() {
    let opts = GifOptions::default()
        .colors(32)
        .dither(false)
        .diff_rects(true)
        .palette_mode(PaletteMode::Auto)
        .quantizer(QuantizerKind::MedianCut);
    let frames = vec![(solid(8, 8, 1, 2, 3, 255), 1u16, 10u16)];
    let gif = encode_gif(8, 8, 1, &opts, frames);
    assert_eq!(&gif[..6], b"GIF89a");
    let (info, decoded) = decode_gif(&gif).expect("native decode");
    assert_eq!(info.width, 8);
    assert_eq!(info.height, 8);
    assert_eq!(decoded.len(), 1);
}

#[test]
fn native_encode_decode_roundtrip() {
    let (w, h) = (12u32, 10u32);
    let mut enc = GifEncoder::new(w, h, 0)
        .colors(32)
        .dither(false)
        .diff_rects(true)
        .transparency(true)
        .disposal(DisposalMode::Auto);
    let f0 = solid(w, h, 20, 40, 60, 255);
    let mut f1 = f0.clone();
    for y in 2..6 {
        for x in 3..8 {
            let i = ((y * w + x) * 4) as usize;
            f1[i] = 200;
            f1[i + 1] = 10;
            f1[i + 2] = 10;
        }
    }
    enc.add_frame_owned(f0.clone(), 1, 10);
    enc.add_frame_owned(f1.clone(), 1, 10);
    let gif = enc.finish();
    let (info, frames) = decode_gif(&gif).unwrap();
    assert_eq!(info.frame_count, 2);
    assert_eq!(frames.len(), 2);
    // Center of first frame roughly source color.
    let i = ((h / 2 * w + w / 2) * 4) as usize;
    assert!((frames[0].rgba[i] as i32 - 20).unsigned_abs() < 25);
    // Dirty patch on second frame.
    let j = ((4 * w + 5) * 4) as usize;
    assert!(frames[1].rgba[j] > 150);
    assert_eq!(frames[1].delay_den, 100);
}

#[test]
fn gif_writer_matches_encode_gif() {
    let (w, h) = (8u32, 8u32);
    let opts = GifOptions::default()
        .colors(16)
        .dither(false)
        .diff_rects(false)
        .transparency(false)
        .palette_mode(PaletteMode::Local);
    let frames = [solid(w, h, 10, 20, 30, 255), solid(w, h, 200, 50, 50, 255)];
    let via_enc = encode_gif(
        w,
        h,
        0,
        &opts,
        frames
            .iter()
            .cloned()
            .map(|f| (f, 1u16, 10u16))
            .collect::<Vec<_>>(),
    );
    let mut buf = Vec::new();
    let mut writer = GifWriter::new(&mut buf, w, h, 0, opts.clone()).unwrap();
    for f in &frames {
        writer.write_frame(f, 1, 10).unwrap();
    }
    writer.finish().unwrap();
    // Both must decode to equivalent canvases (exact bitstream may differ if
    // disposal lookahead differs slightly, so compare decoded pixels).
    let (_, a) = decode_gif(&via_enc).unwrap();
    let (_, b) = decode_gif(&buf).unwrap();
    assert_eq!(a.len(), b.len());
    for (fa, fb) in a.iter().zip(b.iter()) {
        assert_eq!(fa.rgba, fb.rgba);
    }
}

#[test]
fn gif_ffmpeg_comment_and_interlace() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    let (w, h) = (16u32, 16u32);
    let opts = GifOptions::default()
        .colors(16)
        .dither(false)
        .diff_rects(false)
        .transparency(false)
        .interlace(true)
        .comment(b"threers");
    let gif = encode_gif(w, h, 1, &opts, vec![(solid(w, h, 90, 120, 40, 255), 1, 10)]);
    assert!(gif.windows(2).any(|w| w == [0x21, 0xFE]));
    assert!(gif.windows(7).any(|w| w == b"threers"));
    let path = write_temp("threers_gif_interlace.gif", &gif);
    let decoded = decode_gif_rgba(&path, w, h).expect("ffmpeg decode interlaced");
    let _ = std::fs::remove_file(&path);
    let i = ((h / 2 * w + w / 2) * 4) as usize;
    assert!((decoded[i] as i32 - 90).unsigned_abs() < 30);

    let (_, native) = decode_gif(&gif).unwrap();
    assert_eq!(native.len(), 1);
    assert!((native[0].rgba[i] as i32 - 90).unsigned_abs() < 30);
}

#[test]
fn gif_ffmpeg_lossy_and_lzw_modes() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    let (w, h) = (24u32, 24u32);
    let mut base = solid(w, h, 30, 30, 30, 255);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            base[i] = (30 + (x % 8) * 3) as u8;
            base[i + 1] = (30 + (y % 8) * 3) as u8;
        }
    }
    let mut next = base.clone();
    for y in 8..16 {
        for x in 8..16 {
            let i = ((y * w + x) * 4) as usize;
            next[i] = 180;
            next[i + 1] = 40;
            next[i + 2] = 40;
        }
    }

    let lossless = encode_gif(
        w,
        h,
        0,
        &GifOptions::default()
            .colors(64)
            .dither(false)
            .diff_rects(true)
            .lossy(0)
            .lzw_clear(LzwClearMode::WhenFull),
        vec![(base.clone(), 1, 10), (next.clone(), 1, 10)],
    );
    let lossy = encode_gif(
        w,
        h,
        0,
        &GifOptions::default()
            .colors(64)
            .dither(false)
            .diff_rects(true)
            .lossy(40)
            .lzw_clear(LzwClearMode::Deferred),
        vec![(base.clone(), 1, 10), (next.clone(), 1, 10)],
    );
    let adaptive = encode_gif(
        w,
        h,
        0,
        &GifOptions::default()
            .colors(64)
            .dither(false)
            .diff_rects(true)
            .lossy(20)
            .lzw_clear(LzwClearMode::Adaptive),
        vec![(base, 1, 10), (next, 1, 10)],
    );
    assert!(
        lossy.len() <= lossless.len() + 64,
        "lossy should not inflate much: {} vs {}",
        lossy.len(),
        lossless.len()
    );

    for (name, bytes) in [
        ("lossless", &lossless),
        ("lossy", &lossy),
        ("adaptive", &adaptive),
    ] {
        let path = write_temp(&format!("threers_gif_{name}.gif"), bytes);
        let decoded = decode_gif_rgba(&path, w, h).unwrap_or_else(|| panic!("ffmpeg {name}"));
        let _ = std::fs::remove_file(&path);
        assert!(decoded.len() >= (w * h * 4 * 2) as usize);
        let (_, native) = decode_gif(bytes).unwrap();
        assert_eq!(native.len(), 2);
    }
}

#[test]
fn gif_ffmpeg_auto_disposal_punch() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }
    let (w, h) = (20u32, 20u32);
    let f0 = solid(w, h, 255, 0, 0, 255);
    let mut f1 = f0.clone();
    // Punch a transparent hole.
    for y in 6..14 {
        for x in 6..14 {
            let i = ((y * w + x) * 4) as usize;
            f1[i + 3] = 0;
        }
    }
    let gif = encode_gif(
        w,
        h,
        0,
        &GifOptions::default()
            .colors(8)
            .dither(false)
            .diff_rects(true)
            .transparency(true)
            .disposal(DisposalMode::Auto),
        vec![(f0, 1, 10), (f1, 1, 10)],
    );
    let path = write_temp("threers_gif_disposal.gif", &gif);
    let ff = decode_gif_rgba(&path, w, h).expect("ffmpeg disposal");
    let _ = std::fs::remove_file(&path);
    let frame = (w * h * 4) as usize;
    assert!(ff.len() >= frame * 2);
    let second = &ff[frame..frame * 2];
    let hole = ((10 * w + 10) * 4) as usize;
    assert!(
        second[hole + 3] < 40,
        "hole should be transparent, alpha={}",
        second[hole + 3]
    );

    let (_, native) = decode_gif(&gif).unwrap();
    assert!(native[1].rgba[hole + 3] < 40);
}

#[test]
fn gif_malformed_inputs() {
    assert!(GifDecoder::open(b"not a gif").is_err());
    assert!(GifDecoder::open(b"GIF89a").is_err()); // truncated
    assert!(decode_gif(&[0u8; 16]).is_err());
}

#[test]
fn lossy_zero_size_regression() {
    let (w, h) = (16u32, 16u32);
    let mut frames = Vec::new();
    for f in 0..3u32 {
        let mut rgba = solid(w, h, 50, 60, 70, 255);
        rgba[((f * 17) as usize % (w * h) as usize) * 4] = 200;
        frames.push((rgba, 1u16, 10u16));
    }
    let a = encode_gif(
        w,
        h,
        0,
        &GifOptions::default().lossy(0).dither(false),
        frames.clone(),
    );
    let b = encode_gif(w, h, 0, &GifOptions::default().dither(false), frames);
    assert_eq!(a, b, "lossy(0) must match default bitstream");
}
