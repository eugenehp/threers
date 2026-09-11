//! External-decoder conformance for the HEVC residual coder.
//!
//! The encoder returns the reconstruction a conformant decoder must reproduce
//! bit-for-bit, so decoding with ffmpeg and comparing to it is the real test.
//! `tests/hevc_compress.rs` does this too, but only on smooth images — and
//! smooth content quantizes to so few coefficients that it never reaches the
//! path that was broken.
//!
//! # What these caught
//!
//! Two wrong bytes in `src/codec/hevc/tables.rs`: `transIdxLps[28]` was 23
//! (spec: 22) and `rangeTabLps[31][0]` was 28 (spec: 29). A CABAC context only
//! reaches state 28 or 31 after a long one-sided run of bins, so only detailed
//! blocks got there — and the encode↔decode round-trip test in `cabac.rs` shares
//! the tables, so both sides agreed with each other while disagreeing with the
//! standard. It took an external decoder plus content dense enough to drive the
//! states that far.
//!
//! Keep these pointed at *detailed* content. That is the whole point of them.

use std::f64::consts::PI;
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use threers::codec::hevc::{CompressedEncoder, Yuv420Frame};

const N: usize = 16;

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A 16×16 frame whose luma is a sum of DCT basis functions and whose chroma is
/// exactly neutral, so the chroma residual is identically zero.
fn frame_from_dct(coeffs: &[((usize, usize), f64)]) -> Yuv420Frame {
    let mut px = [[0.0f64; N]; N];
    for &((u, v), amp) in coeffs {
        let cu = if u == 0 { (1.0 / N as f64).sqrt() } else { (2.0 / N as f64).sqrt() };
        let cv = if v == 0 { (1.0 / N as f64).sqrt() } else { (2.0 / N as f64).sqrt() };
        for (y, row) in px.iter_mut().enumerate() {
            for (x, p) in row.iter_mut().enumerate() {
                *p += amp
                    * cu
                    * cv
                    * (PI * (2.0 * x as f64 + 1.0) * u as f64 / (2.0 * N as f64)).cos()
                    * (PI * (2.0 * y as f64 + 1.0) * v as f64 / (2.0 * N as f64)).cos();
            }
        }
    }
    let y: Vec<u8> = px
        .iter()
        .flat_map(|row| row.iter().map(|&p| (128.0 + p).round().clamp(0.0, 255.0) as u8))
        .collect();
    let chroma = vec![128u8; (N / 2) * (N / 2)];
    Yuv420Frame {
        width: N as u32,
        height: N as u32,
        y,
        u: chroma.clone(),
        v: chroma,
        alpha: None,
    }
}

/// Encode one frame with the residual path and return `(encoder reconstruction,
/// ffmpeg's decode)`, both as the luma plane at display size.
fn encode_and_decode(frame: &Yuv420Frame, qp: i32) -> (Vec<u8>, Vec<u8>) {
    let mut enc = CompressedEncoder::new(N as u32, N as u32, qp).residual(true);
    let (au, recon) = enc.encode_frame(frame);

    // Unique per call: cargo runs the tests in this binary on parallel threads,
    // and a shared path means they overwrite each other's streams.
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let uniq = format!(
        "{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let dir = std::env::temp_dir();
    let src = dir.join(format!("threers-resid-{uniq}.265"));
    let dst = dir.join(format!("threers-resid-{uniq}.yuv"));
    std::fs::write(&src, &au).unwrap();
    let ok = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-f", "hevc", "-i"])
        .arg(&src)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dst)
        .status()
        .expect("run ffmpeg")
        .success();
    assert!(ok, "ffmpeg failed to decode the stream at all");
    let decoded = std::fs::read(&dst).unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);

    // The reconstruction is at CTB-aligned size; crop to display size like the
    // conformance window makes a decoder do. Here they are equal (16 == CTB).
    let cw = recon.coded_width as usize;
    let luma: Vec<u8> = (0..N).flat_map(|r| recon.y[r * cw..r * cw + N].to_vec()).collect();
    (luma, decoded[..N * N].to_vec())
}

fn assert_conformant(frame: &Yuv420Frame, qp: i32, what: &str) {
    let (recon, decoded) = encode_and_decode(frame, qp);
    let diffs: Vec<i32> = recon
        .iter()
        .zip(&decoded)
        .map(|(&a, &b)| b as i32 - a as i32)
        .collect();
    let worst = diffs.iter().map(|d| d.abs()).max().unwrap_or(0);
    let n_diff = diffs.iter().filter(|&&d| d != 0).count();
    assert_eq!(
        worst, 0,
        "{what} (qp {qp}): decoder != encoder reconstruction — \
         {n_diff}/{} samples differ, largest by {worst}",
        recon.len()
    );
}

#[test]
fn two_coefficient_reproducer() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    // The minimal case delta-debugging reduced the original failure to: one
    // coefficient in sub-block (3,3) (scan index 15, the last) and one in
    // sub-block (3,0) (scan index 9). Either alone was fine even when broken;
    // it took both to drive a context into the mistranscribed state.
    let frame = frame_from_dct(&[((14, 15), 600.0), ((12, 0), -500.0)]);
    assert_conformant(&frame, 27, "two coefficients, sub-blocks 15 and 9");
}

#[test]
fn single_coefficient_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    // The control: every single-coefficient block passes, which is why the bug
    // needs at least two coded sub-blocks to show up.
    for &(u, v) in &[(14usize, 15usize), (12, 0), (8, 0), (0, 12), (3, 3)] {
        let frame = frame_from_dct(&[((u, v), 600.0)]);
        assert_conformant(&frame, 27, &format!("single coefficient at ({u},{v})"));
    }
}

#[test]
fn luma_only_grayscale_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    // Chroma is neutral, so cbf_cb/cbf_cr are 0 and no chroma residual is coded.
    // This failing too is what ruled out the "dense 8x8 chroma" reading the
    // module docs used to carry: the luma path was equally affected.
    let detailed: Vec<((usize, usize), f64)> = (0..N)
        .step_by(3)
        .flat_map(|u| (0..N).step_by(5).map(move |v| ((u, v), 300.0)))
        .collect();
    for qp in [12, 27, 34] {
        assert_conformant(&frame_from_dct(&detailed), qp, "grayscale, luma path only");
    }
}

/// Randomised search: the harness that found the reproducers above. Kept because
/// a minimised case only pins the one path it happens to take.
#[test]
fn fuzz_random_coefficient_sets() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    // Deterministic xorshift so a failure is reproducible from the seed alone.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut failures = 0;
    for _ in 0..120 {
        let k = 2 + (next() % 8) as usize;
        let coeffs: Vec<((usize, usize), f64)> = (0..k)
            .map(|_| {
                let u = (next() % N as u64) as usize;
                let v = (next() % N as u64) as usize;
                let sign = if next() % 2 == 0 { 1.0 } else { -1.0 };
                let amp = 50.0 + (next() % 850) as f64;
                ((u, v), sign * amp)
            })
            .collect();
        let (recon, decoded) = encode_and_decode(&frame_from_dct(&coeffs), 27);
        if recon != decoded {
            failures += 1;
        }
    }
    assert_eq!(failures, 0, "{failures}/120 random coefficient sets were non-conformant");
}


/// Strongly directional content across several CTBs.
///
/// This is the case the DCT-basis tests above miss. Splitting the transform tree
/// brings 8x8 luma and 4x4 chroma blocks into play, and at those sizes HEVC
/// picks the coefficient scan from the prediction mode (§7.4.9.11) — which in
/// turn changes a `sig_coeff_flag` context offset at 8x8 and transposes the
/// signalled last-significant position under a vertical scan. Content that never
/// selects a near-horizontal or near-vertical mode exercises none of it, and two
/// bugs lived there until a real render caught them.
#[test]
fn directional_multi_ctb_content_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    // 64x64 = 16 CTBs, so cross-CTB neighbour availability is exercised too.
    let (w, h) = (64usize, 64usize);
    for (name, f) in [
        // Near-vertical structure -> horizontal scan.
        ("vertical stripes", &(|x: usize, y: usize| {
            (if x.is_multiple_of(3) { 40 } else { 210 }) + (y % 5) as i32
        }) as &dyn Fn(usize, usize) -> i32),
        // Near-horizontal structure -> vertical scan.
        ("horizontal stripes", &|x: usize, y: usize| {
            (if y.is_multiple_of(3) { 30 } else { 200 }) + (x % 7) as i32
        }),
        // Diagonal, plus fine detail to keep coefficients dense.
        ("diagonal detail", &|x: usize, y: usize| {
            ((x + y) * 9 % 256) as i32 ^ ((x * y) % 17) as i32
        }),
    ] {
        let y_plane: Vec<u8> = (0..h)
            .flat_map(|yy| (0..w).map(move |xx| f(xx, yy).clamp(0, 255) as u8))
            .collect();
        // Chroma carries structure too, so the 4x4 chroma scan is exercised.
        let chroma: Vec<u8> = (0..(h / 2))
            .flat_map(|yy| {
                (0..(w / 2)).map(move |xx| ((f(xx * 2, yy * 2) / 2 + 64).clamp(0, 255)) as u8)
            })
            .collect();
        let frame = Yuv420Frame {
            width: w as u32,
            height: h as u32,
            y: y_plane,
            u: chroma.clone(),
            v: chroma,
            alpha: None,
        };
        for qp in [18, 27, 36] {
            let (recon, decoded) = encode_and_decode_size(&frame, qp, w, h);
            let n_diff = recon.iter().zip(&decoded).filter(|(a, b)| a != b).count();
            assert_eq!(
                n_diff, 0,
                "{name} (qp {qp}): {n_diff}/{} samples differ",
                recon.len()
            );
        }
    }
}

/// [`encode_and_decode`] for a frame of any size (the helper above is pinned to
/// a single 16x16 CTB).
fn encode_and_decode_size(
    frame: &Yuv420Frame,
    qp: i32,
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<u8>) {
    encode_and_decode_aq(frame, qp, w, h, 0.0)
}

/// As [`encode_and_decode_size`], with an adaptive-quantization strength.
fn encode_and_decode_aq(
    frame: &Yuv420Frame,
    qp: i32,
    w: usize,
    h: usize,
    aq: f32,
) -> (Vec<u8>, Vec<u8>) {
    encode_and_decode_tiled(frame, qp, w, h, aq, (1, 1))
}

/// As [`encode_and_decode_aq`], with a tile grid.
fn encode_and_decode_tiled(
    frame: &Yuv420Frame,
    qp: i32,
    w: usize,
    h: usize,
    aq: f32,
    tiles: (u32, u32),
) -> (Vec<u8>, Vec<u8>) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEQ: AtomicU32 = AtomicU32::new(1000);
    let uniq = format!(
        "{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let mut enc = CompressedEncoder::new(w as u32, h as u32, qp)
        .residual(true)
        .aq(aq)
        .tiles(tiles.0, tiles.1);
    let (au, recon) = enc.encode_frame(frame);

    let dir = std::env::temp_dir();
    let src = dir.join(format!("threers-dir-{uniq}.265"));
    let dst = dir.join(format!("threers-dir-{uniq}.yuv"));
    std::fs::write(&src, &au).unwrap();
    let ok = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error", "-f", "hevc", "-i"])
        .arg(&src)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dst)
        .status()
        .expect("run ffmpeg")
        .success();
    assert!(ok, "ffmpeg failed to decode");
    let decoded = std::fs::read(&dst).unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);

    // Crop the CTB-aligned reconstruction to display size, all three planes.
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
    (expect, decoded)
}

/// Randomised content over a picture that is neither CTB-aligned nor uniform.
///
/// A 64-sample coding tree block only earns its keep if the quadtree can pick a
/// size per region, and the interesting cases are the ones a tidy test picture
/// never reaches: coding units that hang off the right and bottom edges, where
/// the split is forced and unsignalled; `split_cu_flag` contexts derived from
/// neighbours at a different depth; and a mode-prediction candidate that now
/// comes from another coding unit inside the same CTB rather than always
/// defaulting to DC.
///
/// 200x136 is three-and-a-bit CTBs across and two-and-a-bit down, and the
/// content mixes flat regions (which want one big unit) with noise (which wants
/// small ones), so both sides of every decision get taken.
#[test]
fn quadtree_over_unaligned_random_content_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    let (w, h) = (200usize, 136usize);
    let mut state = 0x2545_F491u32;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };

    for trial in 0..6 {
        let plane = |w: usize, h: usize, rnd: &mut dyn FnMut() -> u32| -> Vec<u8> {
            let mut out = vec![0u8; w * h];
            // Flat tiles, gradients and noise side by side, in tiles that are
            // deliberately not aligned to any block size.
            for y in 0..h {
                for x in 0..w {
                    let tile = ((x / 37) + (y / 29)) % 3;
                    out[y * w + x] = match tile {
                        0 => 140,
                        1 => ((x * 3 + y * 2) % 256) as u8,
                        _ => (rnd() >> 24) as u8,
                    };
                }
            }
            out
        };
        let y_plane = plane(w, h, &mut rnd);
        let u_plane = plane(w / 2, h / 2, &mut rnd);
        let v_plane = plane(w / 2, h / 2, &mut rnd);
        let frame = Yuv420Frame {
            width: w as u32,
            height: h as u32,
            y: y_plane,
            u: u_plane,
            v: v_plane,
            alpha: None,
        };
        let qp = [14, 22, 27, 32, 37, 45][trial];
        let (recon, decoded) = encode_and_decode_size(&frame, qp, w, h);
        let n_diff = recon.iter().zip(&decoded).filter(|(a, b)| a != b).count();
        let worst = recon
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap_or(0);
        assert_eq!(
            n_diff,
            0,
            "trial {trial} (qp {qp}): {n_diff}/{} samples differ, largest by {worst}",
            recon.len()
        );
    }
}

/// Conformance with a quantizer that changes between coding tree blocks.
///
/// `cu_qp_delta` is prediction-coded and it does not appear where you would
/// expect: not in the coding unit's header, but in the first transform unit
/// anywhere below it that has something to code — and not at all if none does,
/// in which case the decoder's `QpY` is the predictor rather than the value the
/// encoder used. A group that codes nothing therefore must not move the
/// predictor, and a group that codes only *chroma* must still carry the delta.
///
/// Adaptive quantization is off by default, so without this test nothing would
/// ever send a non-zero delta.
#[test]
fn per_ctb_quantizer_is_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    // Flat regions next to busy ones at a scale that does not line up with the
    // 64-sample coding tree, so the quantizer swings and plenty of blocks come
    // out empty.
    let (w, h) = (200usize, 136usize);
    let f = |x: usize, y: usize| -> i32 {
        if ((x / 43) + (y / 37)).is_multiple_of(2) {
            130 + (x as i32 / 50)
        } else {
            (((x * 61) ^ (y * 113)) & 0xFF) as i32
        }
    };
    let frame = Yuv420Frame {
        width: w as u32,
        height: h as u32,
        y: (0..h)
            .flat_map(|yy| (0..w).map(move |xx| f(xx, yy).clamp(0, 255) as u8))
            .collect(),
        u: (0..h / 2)
            .flat_map(|yy| (0..w / 2).map(move |xx| (f(xx * 2, yy * 2) / 2 + 60).clamp(0, 255) as u8))
            .collect(),
        v: (0..h / 2)
            .flat_map(|yy| (0..w / 2).map(move |xx| (200 - f(xx * 2, yy * 2) / 3).clamp(0, 255) as u8))
            .collect(),
        alpha: None,
    };
    for &aq in &[1.0f32, 3.0] {
        for qp in [18, 27, 36, 44] {
            let (recon, decoded) = encode_and_decode_aq(&frame, qp, w, h, aq);
            let n = recon.iter().zip(&decoded).filter(|(a, b)| a != b).count();
            assert_eq!(n, 0, "aq {aq} qp {qp}: {n}/{} samples differ", recon.len());
        }
    }
}

/// Tiles, decoded by ffmpeg.
///
/// A tile is a picture within a picture: intra prediction stops at its edge and
/// the entropy coder restarts, which is what lets tiles be encoded on separate
/// threads. That also means almost everything can go wrong somewhere a
/// round-trip would never look — the coding tree blocks are scanned in tile
/// order rather than picture order, each tile's substream has to be terminated
/// with `end_of_subset_one_bit` rather than the end-of-slice flag, and the
/// slice header has to carry byte offsets into the concatenation *counted after
/// emulation prevention*, or the decoder starts the second tile in the middle of
/// the first.
///
/// The grids below include one that does not divide evenly, so the right and
/// bottom tiles are clipped, and a 1xN and Nx1 to catch a transposed column and
/// row count — which would otherwise produce a perfectly valid picture of the
/// wrong shape.
#[test]
fn tiled_pictures_are_conformant() {
    if !have_ffmpeg() {
        eprintln!("skipping: ffmpeg not found");
        return;
    }
    let (w, h) = (320usize, 208usize);
    let f = |x: usize, y: usize| -> i32 {
        if ((x / 37) + (y / 29)).is_multiple_of(2) {
            120 + (x as i32 / 40)
        } else {
            (((x * 71) ^ (y * 131)) & 0xFF) as i32
        }
    };
    let frame = Yuv420Frame {
        width: w as u32,
        height: h as u32,
        y: (0..h)
            .flat_map(|yy| (0..w).map(move |xx| f(xx, yy).clamp(0, 255) as u8))
            .collect(),
        u: (0..h / 2)
            .flat_map(|yy| (0..w / 2).map(move |xx| (f(xx * 2, yy * 2) / 2 + 55).clamp(0, 255) as u8))
            .collect(),
        v: (0..h / 2)
            .flat_map(|yy| (0..w / 2).map(move |xx| (190 - f(xx * 2, yy * 2) / 3).clamp(0, 255) as u8))
            .collect(),
        alpha: None,
    };
    for &tiles in &[(2u32, 1u32), (1, 2), (2, 2), (3, 2), (5, 3)] {
        for qp in [20, 30, 40] {
            let (recon, decoded) = encode_and_decode_tiled(&frame, qp, w, h, 0.0, tiles);
            let n = recon.iter().zip(&decoded).filter(|(a, b)| a != b).count();
            assert_eq!(
                n,
                0,
                "tiles {tiles:?} qp {qp}: {n}/{} samples differ",
                recon.len()
            );
        }
    }
}
