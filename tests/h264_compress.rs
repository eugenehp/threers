//! External-decoder conformance for the compressed H.264 intra encoder.
//!
//! The encoder returns the reconstruction a conformant decoder must reproduce
//! bit-for-bit, so decoding with ffmpeg and comparing to it is the real test —
//! the CAVLC tables and the intra formulas are transcribed constants, and a
//! round-trip through the encoder's own view of them proves nothing.
//!
//! The tests build up: flat content exercises the least machinery, and detail
//! exercises the coefficient coder that flat content never reaches.
//!
//! # State
//!
//! Conformant: ffmpeg decodes to the encoder's own reconstruction bit-for-bit
//! across sizes, QPs and content, including the randomised sweep in
//! [`fuzz_random_content_is_conformant`].
//!
//! Six bugs were found by these tests plus targeted probing against ffmpeg, and
//! the last three are the interesting ones because each was invisible to
//! everything except an external decoder:
//!
//! 1. `total_zeros` counted the high-frequency zeros beyond the last
//!    coefficient — a DC-only block claimed 15 instead of 0.
//! 2. The in-loop deblocking filter was left enabled, so the encoder's
//!    reconstruction drifted from the decoder's at every block edge, and more so
//!    at high QP where the filter is stronger.
//! 3. `CT1`'s Kraft sum exceeded 1, which is impossible for a prefix code.
//! 4. Two `CT1` `coeff_token` entries were transposed.
//! 5. Levels were not bounded by what the *inverse transform* can carry. A
//!    dequantized coefficient is representable well past the point where the
//!    transform's first-stage sums are, and decoders hold those in 16 bits.
//! 6. Two `total_zeros` entries at `tzVlcIndex = 12` were transposed. This one
//!    caused both of the gaps that were open before it was found — chroma AC
//!    across macroblocks *and* steep high-frequency luma — because both are just
//!    ways of reaching a 12-coefficient block.
//!
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use threers::codec::h264::compress::CompressedEncoder;
use threers::codec::h264::params::EntropyMode;
use threers::codec::hevc::Yuv420Frame;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Encode one frame and return `(encoder reconstruction, ffmpeg's decode)`,
/// both cropped to display size and covering all three planes.
fn encode_and_decode(frame: &Yuv420Frame, qp: i32) -> (Vec<u8>, Vec<u8>) {
    encode_and_decode_with(CompressedEncoder::new(frame.width, frame.height, qp), frame)
}

/// As [`encode_and_decode`], but with an encoder the caller has configured.
fn encode_and_decode_with(mut enc: CompressedEncoder, frame: &Yuv420Frame) -> (Vec<u8>, Vec<u8>) {
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let uniq = format!(
        "{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let (w, h) = (frame.width as usize, frame.height as usize);

    let (au, recon) = enc.encode_frame(frame);

    let dir = std::env::temp_dir();
    let src = dir.join(format!("threers-h264-{uniq}.264"));
    let dst = dir.join(format!("threers-h264-{uniq}.yuv"));
    std::fs::write(&src, &au).unwrap();
    let ok = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-f", "h264", "-i"])
        .arg(&src)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dst)
        .status()
        .expect("run ffmpeg")
        .success();
    assert!(ok, "ffmpeg could not decode the stream at all");
    let decoded = std::fs::read(&dst).unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);

    let cw = recon.coded_width as usize;
    let cwc = cw / 2;
    let mut expect = Vec::with_capacity(w * h * 3 / 2);
    for r in 0..h {
        expect.extend_from_slice(&recon.y[r * cw..r * cw + w]);
    }
    for plane in [&recon.u, &recon.v] {
        for r in 0..h / 2 {
            expect.extend_from_slice(&plane[r * cwc..r * cwc + w / 2]);
        }
    }
    assert_eq!(
        decoded.len(),
        expect.len(),
        "decoded {} bytes, expected {}",
        decoded.len(),
        expect.len()
    );
    (expect, decoded)
}

fn assert_conformant(frame: &Yuv420Frame, qp: i32, what: &str) {
    let (recon, decoded) = encode_and_decode(frame, qp);
    let diffs = recon.iter().zip(&decoded).filter(|(a, b)| a != b).count();
    let worst = recon
        .iter()
        .zip(&decoded)
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);
    assert_eq!(
        diffs, 0,
        "{what} (qp {qp}): decoder != encoder reconstruction — \
         {diffs}/{} samples differ, largest by {worst}",
        recon.len()
    );
}

/// Build a frame. Chroma is whatever `f` returns for the co-sited luma sample.
fn frame_from(w: u32, h: u32, f: impl Fn(u32, u32) -> (u8, u8, u8)) -> Yuv420Frame {
    let mut y = vec![0u8; (w * h) as usize];
    let mut u = vec![0u8; (w / 2 * (h / 2)) as usize];
    let mut v = vec![0u8; (w / 2 * (h / 2)) as usize];
    for yy in 0..h {
        for xx in 0..w {
            y[(yy * w + xx) as usize] = f(xx, yy).0;
        }
    }
    for yy in 0..h / 2 {
        for xx in 0..w / 2 {
            let (_, cb, cr) = f(xx * 2, yy * 2);
            u[(yy * (w / 2) + xx) as usize] = cb;
            v[(yy * (w / 2) + xx) as usize] = cr;
        }
    }
    Yuv420Frame {
        width: w,
        height: h,
        y,
        u,
        v,
        alpha: None,
    }
}

/// A flat frame: every residual quantizes to zero, so this exercises the
/// macroblock layer and prediction with the coefficient coder idle.
#[test]
fn flat_frame_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    let f = frame_from(32, 32, |_, _| (120, 128, 128));
    assert_conformant(&f, 26, "flat 32x32");
}

/// A gradient: a few low-frequency coefficients per block, so `coeff_token`,
/// the level coder and `total_zeros` all run, but sparsely.
#[test]
fn gradient_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    let f = frame_from(32, 32, |x, y| {
        ((x * 4 + y * 2) as u8, (128 + x / 4) as u8, 128u8)
    });
    for qp in [22, 30, 38] {
        assert_conformant(&f, qp, "gradient 32x32");
    }
}

/// Dense detail across several macroblocks — the case that fills blocks with
/// coefficients, drives `nC` above the first table's range, and exercises the
/// run and level coding properly.
#[test]
fn detailed_frame_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    let f = frame_from(64, 48, |x, y| {
        let luma = ((x * 37 + y * 91) % 256) as u8 ^ ((x * y) % 61) as u8;
        (luma, ((x * 13) % 256) as u8, ((y * 29) % 256) as u8)
    });
    for qp in [18, 26, 34] {
        assert_conformant(&f, qp, "dense detail 64x48");
    }
}

/// Strong directional structure, which pushes the mode search onto the
/// directional predictors rather than DC.
#[test]
fn directional_content_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    for (name, f) in [
        ("vertical stripes", &(|x: u32, _: u32| {
            (if x.is_multiple_of(3) { 30u8 } else { 220u8 }, 128u8, 128u8)
        }) as &dyn Fn(u32, u32) -> (u8, u8, u8)),
        ("horizontal stripes", &|_: u32, y: u32| {
            (if y.is_multiple_of(3) { 30u8 } else { 220u8 }, 128u8, 128u8)
        }),
    ] {
        let frame = frame_from(48, 48, f);
        assert_conformant(&frame, 24, name);
    }
}

/// Sizes that are not a whole number of macroblocks exercise the cropping the
/// SPS frame-crop signals.
#[test]
fn non_macroblock_aligned_sizes_are_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    for &(w, h) in &[(20u32, 12u32), (36, 28), (64, 34)] {
        let f = frame_from(w, h, |x, y| {
            (
                ((x * 11 + y * 7) % 256) as u8,
                (100 + x % 50) as u8,
                (150 - y % 50) as u8,
            )
        });
        assert_conformant(&f, 28, &format!("{w}x{h}"));
    }
}

/// The whole point: a compressed frame must be dramatically smaller than the
/// `I_PCM` path it replaces.
#[test]
fn compressed_is_far_smaller_than_pcm() {
    let (w, h) = (128u32, 96u32);
    let f = frame_from(w, h, |x, y| {
        (
            ((x * 3 + y) % 256) as u8,
            (128 + x % 20) as u8,
            (128 + y % 20) as u8,
        )
    });
    let mut enc = CompressedEncoder::new(w, h, 28);
    let (au, _) = enc.encode_frame(&f);

    let raw = (w * h * 3 / 2) as usize;
    assert!(
        au.len() * 4 < raw,
        "compressed {} bytes against {raw} raw — not compressing",
        au.len()
    );
}


/// The one known non-conformance, kept as a precise reproducer.
///
/// Luma of any content is conformant, and so is chroma whose residual is
/// constant or DC-only. With chroma AC coefficients spanning more than one
/// macroblock the parse desyncs; ffmpeg reports `total_coeff=16`, which is
/// impossible for a 15-coefficient chroma block, at the first block whose `nC`
/// reaches the `nC >= 8` fixed-length code.
///
/// What has been ruled out by probing every case against ffmpeg:
///
/// * `coeff_token` for every `(TrailingOnes, TotalCoeff)` in all four tables,
///   for both luma (16-coefficient) and chroma AC (15-coefficient) blocks;
/// * `total_zeros` and `run_before` for single, paired and scattered
///   coefficients at every scan position;
/// * chroma DC, alone and together with AC;
/// * `cbp_chroma` alternating between 1 and 2 across macroblocks;
/// * `nC` varying across blocks and across macroblocks.
///
/// Clamping `nC` to 7 changes which cases fail rather than fixing any, so the
/// fixed-length path is not wrong on its own. The remaining suspect is the
/// neighbour derivation feeding `nC` for chroma AC across a macroblock edge.
#[test]
fn chroma_ac_across_macroblocks_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    // Two macroblocks side by side; flat luma so only chroma is in play.
    let f = frame_from(32, 16, |x, y| {
        (120, (100 + (x * 13 + y * 5) % 100) as u8, (90 + (x * 7 + y * 21) % 120) as u8)
    });
    assert_conformant(&f, 32, "chroma AC over two macroblocks");
}


/// Steep, wrapping diagonal content: large coefficients and many of them.
///
/// Fails with flat chroma, so it is a second, separate gap from the chroma one
/// — the luma coefficient coder holds up for every forced pattern probed
/// against ffmpeg (every `(TrailingOnes, TotalCoeff)` in all four tables,
/// scattered runs, levels through the escape), so what this content reaches
/// that those do not is not yet identified.
#[test]
fn steep_diagonal_content_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    let f = frame_from(48, 48, |x, y| (((x + y) * 17 % 256) as u8, 128u8, 128u8));
    for qp in [18, 24, 30, 36] {
        assert_conformant(&f, qp, "steep diagonal");
    }
}


/// Randomised end-to-end sweep: content, size and QP all varied.
///
/// The targeted tests above each reach one path. This is what catches the
/// combinations — bug 6 above only appeared with a 12-coefficient block whose
/// levels varied enough to drive the `suffixLength` adaptation, which no
/// hand-written case happened to produce.
#[test]
fn fuzz_random_content_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    // Deterministic xorshift so any failure is reproducible from the seed.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for &(w, h) in &[(16u32, 16u32), (32, 16), (48, 32), (20, 12), (36, 28)] {
        for &qp in &[14, 26, 38, 50] {
            let f = frame_from(w, h, |_, _| (0, 0, 0));
            // Fill with noise rather than structure: structure tends to pick one
            // prediction mode and stay there, which narrows what gets exercised.
            let mut f = f;
            for v in f.y.iter_mut() {
                *v = (next() % 256) as u8;
            }
            for v in f.u.iter_mut() {
                *v = (next() % 256) as u8;
            }
            for v in f.v.iter_mut() {
                *v = (next() % 256) as u8;
            }
            assert_conformant(&f, qp, &format!("random {w}x{h}"));
        }
    }
}

/// Randomised *smooth* content, which is a different blind spot from noise.
///
/// `fuzz_random_content_is_conformant` fills every plane with noise, and noise
/// drives every coded block pattern to 15 — so the context derivations that key
/// off a *partially* coded macroblock are never taken. A CABAC bug lived
/// exactly there: the chroma DC context asked whether the neighbour had a
/// chroma block at all instead of whether that block carried coefficients, and
/// the two only differ when chroma AC is coded over a zero DC. Noise never
/// produces that; a gradient always does.
///
/// So this sweep randomises the *smoothness* rather than the samples: ramps,
/// low-frequency sinusoids and near-flat fields, at sizes and QPs where blocks
/// fall on both sides of the quantizer's threshold.
#[test]
fn fuzz_smooth_content_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    let mut state = 0x5DEE_CE66_D2C1_1B3Fu64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for &(w, h) in &[(48u32, 32u32), (64, 48), (32, 32), (80, 16), (36, 28)] {
        for &qp in &[16, 24, 30, 36, 44] {
            // A slope and a coarse ripple per plane, with independently random
            // amplitudes so some planes go flat while others do not.
            let p: Vec<i32> = (0..9).map(|_| (next() % 9) as i32).collect();
            let f = frame_from(w, h, |x, y| {
                let s = |o: usize, k: i32| {
                    let (xi, yi) = (x as i32, y as i32);
                    128 + p[o] * xi / k + p[o + 1] * yi / k
                        - p[o + 2] * ((xi / 7 + yi / 5) % 3)
                };
                (
                    s(0, 3).clamp(0, 255) as u8,
                    s(3, 5).clamp(0, 255) as u8,
                    s(6, 4).clamp(0, 255) as u8,
                )
            });
            assert_conformant(&f, qp, &format!("smooth {w}x{h} amp {p:?}"));
        }
    }
}

/// Conformance with a *varying* quantizer.
///
/// Adaptive quantization is off by default (it measured worse than a uniform
/// QP — see `docs/codec-benchmark.md`), but the per-macroblock quantizer it
/// rides on is a real part of the bitstream and stays reachable through
/// `CompressedEncoder::aq`. Turning it on here keeps `mb_qp_delta` exercised,
/// which matters because the element is prediction-coded: each delta is
/// relative to the previous macroblock that *sent* one, so a macroblock with no
/// residual has to leave the predictor alone. Getting that wrong desynchronises
/// the quantizer without desynchronising the bitstream, which is exactly the
/// kind of error only an external decoder catches.
#[test]
fn varying_quantizer_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    // Flat tiles next to busy ones, so the QP swings hard and often, and so
    // that plenty of macroblocks code nothing at all and must be skipped over.
    let f = frame_from(96, 64, |x, y| {
        let busy = ((x / 17) + (y / 13)) % 2 == 0;
        let v = if busy {
            ((x * 53) ^ (y * 97)) as u8
        } else {
            (120 + (x / 24)) as u8
        };
        (v, v.wrapping_add(30), v.wrapping_sub(20))
    });
    for &strength in &[0.5f32, 1.5, 3.0] {
        for qp in [20, 28, 36] {
            let enc = CompressedEncoder::new(f.width, f.height, qp).aq(strength);
            let (expect, decoded) = encode_and_decode_with(enc, &f);
            let n = expect.iter().zip(&decoded).filter(|(a, b)| a != b).count();
            assert_eq!(
                n, 0,
                "aq {strength} qp {qp}: {n}/{} samples differ",
                expect.len()
            );
        }
    }
}

/// The CAVLC path, which CABAC replaced as the default.
///
/// It is still reachable — Baseline profile is the widest-compatibility output
/// this encoder can produce — so it still has to be conformant, including with
/// a varying quantizer, where the two coders binarise `mb_qp_delta` completely
/// differently: `se(v)` for CAVLC, unary over the same mapping for CABAC.
#[test]
fn cavlc_path_is_still_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    let f = frame_from(64, 48, |x, y| {
        let busy = ((x / 11) + (y / 9)) % 2 == 0;
        let v = if busy {
            ((x * 29) ^ (y * 71)) as u8
        } else {
            (100 + (x / 16)) as u8
        };
        (v, v.wrapping_add(45), v.wrapping_sub(15))
    });
    for &aq in &[0.0f32, 2.0] {
        for qp in [18, 27, 36, 45] {
            let enc = CompressedEncoder::new(f.width, f.height, qp)
                .entropy(EntropyMode::Cavlc)
                .aq(aq);
            let (expect, decoded) = encode_and_decode_with(enc, &f);
            let n = expect.iter().zip(&decoded).filter(|(a, b)| a != b).count();
            assert_eq!(n, 0, "cavlc aq {aq} qp {qp}: {n}/{} differ", expect.len());
        }
    }
}
