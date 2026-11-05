//! The AV1 encoder against two decoders: `decode::av1`, whose tables it
//! shares, and dav1d (through ffmpeg), which it does not. Sharing tables means
//! a mistake could agree with itself, so dav1d is what proves the stream
//! conforms; PSNR against the source proves the picture survived.
#![cfg(feature = "decode")]

use std::process::Command;

use threers_codec::av1::Av1Encoder;
use threers_codec::decode::av1::Av1Decoder;

/// Gradients, hard edges, a flat area and noise, so every block size and both
/// skip and heavily coded blocks occur.
fn picture(w: usize, h: usize) -> [Vec<u8>; 3] {
    let mut seed = 0x1234567u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 24) as i32
    };
    let luma = |x: usize, y: usize, n: i32| -> u8 {
        let v = if x < w / 3 {
            (x * 200 / w + y * 40 / h) as i32 + 16
        } else if y < h / 3 {
            if (x / 12 + y / 12) % 2 == 0 {
                200
            } else {
                50
            }
        } else if x < 2 * w / 3 {
            120
        } else {
            100 + (n % 48)
        };
        v.clamp(0, 255) as u8
    };
    let mut y = vec![0u8; w * h];
    for (i, p) in y.iter_mut().enumerate() {
        *p = luma(i % w, i / w, noise());
    }
    let (cw, ch) = (w / 2, h / 2);
    let u = (0..cw * ch)
        .map(|i| (64 + (i % cw) * 128 / cw) as u8)
        .collect();
    let v = (0..cw * ch)
        .map(|i| (200 - (i / cw) * 120 / ch) as u8)
        .collect();
    [y, u, v]
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let se: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i64 - y as i64).pow(2) as u64)
        .sum();
    if se == 0 {
        return 99.0;
    }
    10.0 * (255.0f64 * 255.0 * a.len() as f64 / se as f64).log10()
}

fn ours(tu: &[u8]) -> Vec<Vec<u8>> {
    let mut dec = Av1Decoder::new();
    let frames = dec.decode(tu).expect("our decoder rejected the stream");
    assert_eq!(frames.len(), 1);
    let f = frames[0].to_decoded();
    [&f.y, &f.u, &f.v]
        .iter()
        .map(|p| p.iter().map(|&v| v as u8).collect())
        .collect()
}

#[test]
fn a_key_frame_decodes_and_resembles_its_source() {
    for (w, h, q) in [
        (64, 64, 60),
        (128, 96, 40),
        (200, 136, 80),
        (72, 40, 20),
        (8, 8, 100),
        (320, 8, 60),
    ] {
        let [y, u, v] = picture(w, h);
        let mut enc = Av1Encoder::new(w, h, q).unwrap();
        let tu = enc.encode(&y, &u, &v).unwrap();
        let planes = ours(&tu);
        let p = psnr(&planes[0], &y);
        eprintln!(
            "{w}x{h} q{q}: {} bytes of {}, Y {p:.1} dB, U {:.1}, V {:.1}",
            tu.len(),
            w * h * 3 / 2,
            psnr(&planes[1], &u),
            psnr(&planes[2], &v)
        );
        assert!(p > 28.0, "{w}x{h} q{q}: {p:.1} dB");
    }
}

#[test]
fn dav1d_decodes_exactly_what_we_decode() {
    let have = Command::new("ffmpeg")
        .args(["-hide_banner", "-decoders"])
        .output();
    if !have.is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("libdav1d")) {
        eprintln!("skipping: ffmpeg without libdav1d");
        return;
    }
    for (w, h, q) in [(128, 96, 40), (200, 136, 80), (72, 40, 20)] {
        let [y, u, v] = picture(w, h);
        let mut enc = Av1Encoder::new(w, h, q).unwrap();
        let tu = enc.encode(&y, &u, &v).unwrap();
        let path =
            std::env::temp_dir().join(format!("threers_av1enc_{}_{w}.obu", std::process::id()));
        std::fs::write(&path, &tu).unwrap();
        let out = Command::new("ffmpeg")
            .args(["-v", "error", "-c:v", "libdav1d", "-f", "obu", "-i"])
            .arg(&path)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(
            out.status.success(),
            "dav1d: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let mine: Vec<u8> = ours(&tu).concat();
        assert_eq!(out.stdout.len(), mine.len(), "{w}x{h}");
        assert!(out.stdout == mine, "{w}x{h}: dav1d and our decoder differ");
    }
}

#[test]
fn several_frames_are_independent_units() {
    let [y, u, v] = picture(64, 64);
    let mut enc = Av1Encoder::new(64, 64, 60).unwrap();
    let (a, b) = (
        enc.encode(&y, &u, &v).unwrap(),
        enc.encode(&y, &u, &v).unwrap(),
    );
    let mut dec = Av1Decoder::new();
    assert_eq!(dec.decode(&a).unwrap().len(), 1);
    assert_eq!(dec.decode(&b).unwrap().len(), 1);
    assert!(
        b.len() < a.len(),
        "only the first carries the sequence header"
    );
}

/// Frames muxed into WebM and read back by a third party: a clip, not just a
/// picture, and `decode_video` on the same file.
#[test]
fn a_webm_clip_plays() {
    use threers_codec::webm::{mux_webm, WebmCodec, WebmFrame, WebmParams};
    let (w, h) = (96, 64);
    let mut enc = Av1Encoder::new(w, h, 60).unwrap();
    let units: Vec<Vec<u8>> = (0..12)
        .map(|i| {
            let [mut y, u, v] = picture(w, h);
            y.rotate_left(i * 3);
            enc.encode(&y, &u, &v).unwrap()
        })
        .collect();
    let frames: Vec<WebmFrame> = units
        .iter()
        .enumerate()
        .map(|(i, data)| WebmFrame {
            data,
            alpha: None,
            timecode: i as u64 * 40,
            keyframe: true,
        })
        .collect();
    let file = mux_webm(&WebmParams {
        width: w as u32,
        height: h as u32,
        codec: WebmCodec::Av1,
        timecode_scale_ns: 1_000_000,
        frames: &frames,
    });
    let mut count = 0;
    threers_codec::decode::decode_video(&file, |_, f| {
        assert_eq!((f.width as usize, f.height as usize), (w, h));
        count += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(count, 12);
}

#[test]
fn awkward_sizes_are_refused() {
    assert!(Av1Encoder::new(30, 32, 40).is_err());
    assert!(Av1Encoder::new(32, 32, 0).is_err());
}
