//! H.264 with P-frames (`InterEncoder`): every picture a decoder makes of
//! the stream is the encoder's own reconstruction, sample for sample —
//! threers' decoder and, when installed, ffmpeg — and the stream is much
//! smaller than all-intra.

#![cfg(all(feature = "native-codec", feature = "video"))]

use std::process::Command;

use threers::codec::decode::NativeDecoder;
use threers::codec::h264::{CompressedEncoder, InterEncoder, Yuv420Frame};
use threers::codec::mp4_demux::TrackCodec;

/// A scene that moves: a texture panning by `(dx, dy)` samples a frame
/// (odd amounts, so chroma lands between samples), a square moving the
/// other way, and from frame `cut` on, a different scene.
fn frame(w: u32, h: u32, t: u32, cut: u32) -> Yuv420Frame {
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    let (pan_x, pan_y) = (3 * t as i32, t as i32);
    let (sq_x, sq_y) = (w as i32 - 30 - 5 * t as i32, 10 + 2 * t as i32);
    let other = t >= cut;
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let (u, v) = (x + pan_x, y + pan_y);
            let i = ((y as u32 * w + x as u32) * 4) as usize;
            let mut px = if other {
                [((x * 7 + y * 3) % 256) as u8, 40, ((y * 5) % 256) as u8]
            } else {
                let checker = ((u / 6) + (v / 6)) % 2 == 0;
                let wave = (((u as f32 * 0.21).sin() + (v as f32 * 0.13).cos()) * 50.0 + 120.0) as u8;
                if checker { [wave, 90, 200] } else { [30, wave, 60] }
            };
            if (sq_x..sq_x + 20).contains(&x) && (sq_y..sq_y + 20).contains(&y) {
                px = [240, 240, 20];
            }
            rgba[i..i + 4].copy_from_slice(&[px[0], px[1], px[2], 255]);
        }
    }
    Yuv420Frame::from_rgba(w, h, &rgba)
}

/// The top-left `w × h` of a coded-size plane.
fn crop(p: &[u8], stride: u32, w: u32, h: u32) -> Vec<u8> {
    (0..h).flat_map(|y| p[(y * stride) as usize..][..w as usize].to_vec()).collect()
}

struct Encoded {
    avcc: Vec<u8>,
    samples: Vec<(Vec<u8>, bool)>,
    /// Display-size planes per picture, as the encoder reconstructed them.
    recon: Vec<[Vec<u8>; 3]>,
}

fn encode(w: u32, h: u32, n: u32, cut: u32, qp: i32, keyint: u32) -> Encoded {
    let mut enc = InterEncoder::new(w, h, qp).keyframe_interval(keyint).threads(4);
    let mut out = Encoded { avcc: enc.avcc(), samples: vec![], recon: vec![] };
    for t in 0..n {
        let pic = enc.encode(&frame(w, h, t, cut));
        out.samples.push((pic.sample, pic.keyframe));
        let r = enc.reconstruction().unwrap();
        out.recon.push([
            crop(&r.y, r.coded_width, w, h),
            crop(&r.u, r.coded_width / 2, w / 2, h / 2),
            crop(&r.v, r.coded_width / 2, w / 2, h / 2),
        ]);
    }
    out
}

fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[test]
fn threers_decoder_reproduces_the_reconstruction() {
    for (w, h) in [(176u32, 144u32), (120, 90)] {
        let e = encode(w, h, 24, 14, 26, 30);
        assert!(e.samples[0].1, "the first picture is a keyframe");
        assert!(!e.samples[1].1 && !e.samples[13].1, "then P pictures");
        assert!(e.samples[14].1, "the cut is coded as a keyframe");
        assert!(!e.samples[15].1);

        let mut dec = NativeDecoder::from_config(TrackCodec::H264, &e.avcc, w, h).unwrap();
        for (i, (sample, _)) in e.samples.iter().enumerate() {
            dec.push(sample, i as i64, i as i64, 1).unwrap_or_else(|err| panic!("{w}×{h} picture {i}: {err}"));
            let f = dec.take_frame(i as i64).unwrap_or_else(|| panic!("picture {i} not decoded"));
            for (plane, (got, want)) in [&f.y, &f.u, &f.v].iter().zip(&e.recon[i]).enumerate() {
                let got: Vec<u8> = got.iter().map(|&s| s as u8).collect();
                assert!(
                    got == *want,
                    "{w}×{h} picture {i} plane {plane} differs first at {:?}",
                    first_difference(&got, want)
                );
            }
        }
    }
}

#[test]
fn p_frames_are_much_smaller_than_all_intra() {
    let (w, h, n) = (176u32, 144u32, 30u32);
    let e = encode(w, h, n, u32::MAX, 26, 100);
    let inter: usize = e.samples.iter().map(|(s, _)| s.len()).sum();
    let intra_enc = CompressedEncoder::new(w, h, 26).deblock(true);
    let intra: usize = (0..n).map(|t| intra_enc.encode_sample(&frame(w, h, t, u32::MAX)).len()).sum();
    eprintln!("all-intra {intra} bytes, with P-frames {inter} bytes ({:.0}%)", 100.0 * inter as f64 / intra as f64);
    assert!(inter * 2 < intra, "P-frames: {inter} bytes, all-intra: {intra}");
}

#[test]
fn a_still_scene_is_almost_all_skip() {
    let (w, h) = (160u32, 96u32);
    let still = frame(w, h, 0, u32::MAX);
    let mut enc = InterEncoder::new(w, h, 26);
    let key = enc.encode(&still).sample.len();
    let p: Vec<usize> = (0..5).map(|_| enc.encode(&still).sample.len()).collect();
    assert!(p.iter().all(|&s| s < 40), "keyframe {key} bytes, then {p:?}");
}

#[test]
fn keyframe_interval_and_forcing() {
    let (w, h) = (64u32, 48u32);
    let mut enc = InterEncoder::new(w, h, 30).keyframe_interval(4);
    let keys: Vec<bool> = (0..9).map(|t| enc.encode(&frame(w, h, t, u32::MAX)).keyframe).collect();
    assert_eq!(keys, [true, false, false, false, true, false, false, false, true]);
    enc.force_keyframe();
    assert!(enc.encode(&frame(w, h, 9, u32::MAX)).keyframe);
}

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg").arg("-version").output().map(|o| o.status.success()).unwrap_or(false)
}

#[test]
fn ffmpeg_decodes_the_same_pictures() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not installed; skipped");
        return;
    }
    let (w, h) = (120u32, 90u32);
    let e = encode(w, h, 20, 12, 28, 30);
    // Annex B: the avcC's parameter sets, then every sample's NALs.
    let mut stream = Vec::new();
    let mut at = 6;
    let mut sets = Vec::new();
    let take = |at: &mut usize| {
        let len = u16::from_be_bytes([e.avcc[*at], e.avcc[*at + 1]]) as usize;
        *at += 2 + len;
        e.avcc[*at - len..*at].to_vec()
    };
    sets.push(take(&mut at));
    at += 1;
    sets.push(take(&mut at));
    sets.push(take(&mut at));
    for s in sets {
        stream.extend_from_slice(&[0, 0, 0, 1]);
        stream.extend_from_slice(&s);
    }
    for (sample, _) in &e.samples {
        let mut i = 0;
        while i < sample.len() {
            let len = u32::from_be_bytes(sample[i..i + 4].try_into().unwrap()) as usize;
            stream.extend_from_slice(&[0, 0, 0, 1]);
            stream.extend_from_slice(&sample[i + 4..i + 4 + len]);
            i += 4 + len;
        }
    }
    let dir = std::env::temp_dir().join(format!("threers-h264-inter-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (src, out) = (dir.join("p.264"), dir.join("p.yuv"));
    std::fs::write(&src, &stream).unwrap();
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(&src)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    let yuv = std::fs::read(&out).unwrap();
    let size = (w * h + 2 * (w / 2) * (h / 2)) as usize;
    assert_eq!(yuv.len(), size * e.recon.len(), "ffmpeg decoded every picture");
    for (i, r) in e.recon.iter().enumerate() {
        let want: Vec<u8> = r.concat();
        let got = &yuv[i * size..][..size];
        assert!(got == want.as_slice(), "picture {i} differs from ffmpeg's first at {:?}", first_difference(got, &want));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Smooth texture drifting by `0.75, 0.25` samples a frame: only
/// fractional vectors predict it well.
fn drifting(w: u32, h: u32, t: u32) -> Yuv420Frame {
    let (ox, oy) = (0.75 * t as f64, 0.25 * t as f64);
    let rgba: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let (x, y) = ((i % w) as f64 + ox, (i / w) as f64 + oy);
            let v = 128.0 + 60.0 * (x * 0.31).sin() * (y * 0.23).cos() + 40.0 * ((x + y) * 0.11).sin();
            let u = 128.0 + 50.0 * (x * 0.07 - y * 0.05).cos();
            [v as u8, u as u8, (255.0 - v) as u8, 255]
        })
        .collect();
    Yuv420Frame::from_rgba(w, h, &rgba)
}

#[test]
fn sub_sample_motion_decodes_exactly_and_pays() {
    let (w, h, n) = (160u32, 96u32, 16u32);
    let mut enc = InterEncoder::new(w, h, 24).keyframe_interval(100).threads(2);
    let avcc = enc.avcc();
    let mut dec = NativeDecoder::from_config(TrackCodec::H264, &avcc, w, h).unwrap();
    let mut p_bytes = 0;
    for t in 0..n {
        let pic = enc.encode(&drifting(w, h, t));
        if !pic.keyframe {
            p_bytes += pic.sample.len();
        }
        dec.push(&pic.sample, t as i64, t as i64, 1).unwrap();
        let f = dec.take_frame(t as i64).unwrap();
        let r = enc.reconstruction().unwrap();
        let want = crop(&r.y, r.coded_width, w, h);
        let got: Vec<u8> = f.y.iter().map(|&s| s as u8).collect();
        assert!(got == want, "picture {t} luma differs first at {:?}", first_difference(&got, &want));
        for (plane, rec) in [(&f.u, &r.u), (&f.v, &r.v)] {
            let want = crop(rec, r.coded_width / 2, w / 2, h / 2);
            let got: Vec<u8> = plane.iter().map(|&s| s as u8).collect();
            assert!(got == want, "picture {t} chroma differs first at {:?}", first_difference(&got, &want));
        }
    }
    // Whole-sample prediction of a 0.75-sample drift leaves a residual in
    // every macroblock; quarter samples remove most of it.
    let raw = (w * h * 3 / 2) as usize * (n as usize - 1);
    eprintln!("sub-sample drift: P-frames {p_bytes} bytes, {:.1}% of raw", 100.0 * p_bytes as f64 / raw as f64);
    assert!(p_bytes * 25 < raw, "P-frames {p_bytes} bytes for {raw} raw");
}

#[test]
fn cavlc_p_slices_still_decode_exactly() {
    use threers::codec::h264::EntropyMode;
    let (w, h) = (120u32, 90u32);
    let mut enc = InterEncoder::new(w, h, 26).entropy(EntropyMode::Cavlc).threads(2);
    let mut dec = NativeDecoder::from_config(TrackCodec::H264, &enc.avcc(), w, h).unwrap();
    for t in 0..12 {
        let pic = enc.encode(&frame(w, h, t, 8));
        dec.push(&pic.sample, t as i64, t as i64, 1).unwrap();
        let f = dec.take_frame(t as i64).unwrap();
        let r = enc.reconstruction().unwrap();
        let want = crop(&r.y, r.coded_width, w, h);
        let got: Vec<u8> = f.y.iter().map(|&s| s as u8).collect();
        assert!(got == want, "picture {t} differs first at {:?}", first_difference(&got, &want));
    }
}
