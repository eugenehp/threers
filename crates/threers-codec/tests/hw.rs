//! `hw::Mp4Encoder` on every backend this build has, decoded back by both
//! decoders: the hardware and software paths must agree on what they made.

use threers_codec::decode::{Backend, Decoder};
use threers_codec::hw::{EncodeBackend, EncodeOptions, Mp4Encoder};
use threers_codec::mp4_demux::demux_video;

const W: u32 = 320;
const H: u32 = 240;
const FRAMES: u32 = 8;

/// Smooth, moving content: easy to code, so any decoded frame far from it is
/// a pipeline fault rather than encoder quality.
fn frame(i: u32) -> Vec<u8> {
    let mut rgba = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let p = ((y * W + x) * 4) as usize;
            rgba[p] = ((x + i * 4) % 256) as u8;
            rgba[p + 1] = ((y * 255) / H) as u8;
            rgba[p + 2] = 128;
            rgba[p + 3] = 255;
        }
    }
    rgba
}

fn psnr(rgba: &[u8], bgra: &[u8]) -> f64 {
    let mut se = 0.0f64;
    for (s, d) in rgba.chunks_exact(4).zip(bgra.chunks_exact(4)) {
        for (a, b) in [(s[0], d[2]), (s[1], d[1]), (s[2], d[0])] {
            let e = a as f64 - b as f64;
            se += e * e;
        }
    }
    let mse = se / (rgba.len() / 4 * 3) as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

fn encode(backend: EncodeBackend) -> Vec<u8> {
    let opts = EncodeOptions { fps: 24, quality: 0.8, ..Default::default() };
    let mut enc = Mp4Encoder::with_backend(W, H, opts, backend).unwrap();
    assert_eq!(enc.backend(), backend);
    for i in 0..FRAMES {
        enc.push_rgba(&frame(i)).unwrap();
    }
    enc.finish().unwrap()
}

fn decode_and_score(mp4: &[u8], backend: Backend) -> Vec<f64> {
    let track = demux_video(mp4).unwrap();
    assert_eq!((track.width, track.height), (W, H));
    assert_eq!(track.samples.len(), FRAMES as usize);
    let mut dec = Decoder::with_backend(&track, backend).unwrap();
    // Samples go in in decode order; pictures come out by presentation time.
    for s in &track.samples {
        dec.push(&mp4[s.offset as usize..][..s.size as usize], s.pts, s.dts, s.duration as i64)
            .unwrap();
    }
    track
        .presentation_order()
        .into_iter()
        .enumerate()
        .map(|(i, idx)| {
            let bgra = dec.take(track.samples[idx].pts).expect("a picture per sample");
            psnr(&frame(i as u32), &bgra)
        })
        .collect()
}

#[test]
fn every_encode_backend_round_trips_through_every_decoder() {
    let encoders = [EncodeBackend::Native, EncodeBackend::VideoToolbox, EncodeBackend::Nvenc, EncodeBackend::Vaapi];
    let decoders = [Backend::Native, Backend::VideoToolbox, Backend::Nvdec, Backend::Vaapi];
    let mut ran = 0;
    for e in encoders.into_iter().filter(|b| b.available()) {
        let mp4 = encode(e);
        for d in decoders.into_iter().filter(|b| b.available()) {
            let scores = decode_and_score(&mp4, d);
            let worst = scores.iter().cloned().fold(f64::INFINITY, f64::min);
            assert!(worst > 30.0, "{e:?} -> {d:?}: PSNR {scores:?}");
            eprintln!("{e:?} -> {d:?}: worst PSNR {worst:.1} dB");
            ran += 1;
        }
    }
    assert!(ran >= 1);
}

#[test]
fn the_default_encoder_is_the_preferred_backend() {
    let enc = Mp4Encoder::new(W, H, EncodeOptions::default()).unwrap();
    assert_eq!(enc.backend(), EncodeBackend::preferred());
}

/// Hardware and software decoders handed the same stream must produce the
/// same picture, give or take the colour conversion each does on the way out.
#[test]
fn hardware_and_native_decoders_agree() {
    let Some(hw_backend) = Backend::HARDWARE.into_iter().find(|b| b.available()) else {
        return;
    };
    let mp4 = encode(EncodeBackend::Native);
    let track = demux_video(&mp4).unwrap();
    let mut hw = Decoder::with_backend(&track, hw_backend).unwrap();
    let mut sw = Decoder::with_backend(&track, Backend::Native).unwrap();
    for s in &track.samples {
        let bytes = &mp4[s.offset as usize..][..s.size as usize];
        hw.push(bytes, s.pts, s.dts, s.duration as i64).unwrap();
        sw.push(bytes, s.pts, s.dts, s.duration as i64).unwrap();
        let (a, b) = (hw.take(s.pts).unwrap(), sw.take(s.pts).unwrap());
        let mut as_rgba = b.clone();
        for p in as_rgba.chunks_exact_mut(4) {
            p.swap(0, 2);
        }
        assert!(psnr(&as_rgba, &a) > 40.0, "decoders disagree at pts {}", s.pts);
    }
}
