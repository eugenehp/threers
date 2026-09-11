//! Frame-level parallelism must not change a single byte of output.
//!
//! Every frame of an all-intra stream is an IDR that carries nothing forward:
//! `compressed_slice_rbsp` takes `&HevcConfig`/`&H264Config` and the pixels, and
//! no `&mut self`. That is what makes encoding frames concurrently safe, and it
//! is a property worth testing rather than asserting, because it is exactly the
//! kind of thing a later change — a cached table, a reused buffer promoted to a
//! field — would quietly break.
//!
//! ```text
//! cargo test --features native-codec,parallel --test frame_parallel
//! ```
#![cfg(feature = "native-codec")]

use threers::codec::hevc::Yuv420Frame;

fn clip(w: u32, h: u32, n: usize) -> Vec<Yuv420Frame> {
    // Content that varies per frame, so a mix-up between frames would show.
    (0..n)
        .map(|k| {
            let f = |x: u32, y: u32| -> u8 {
                (((x * 7) ^ (y * 13) ^ (k as u32 * 61)).wrapping_mul(3) & 0xFF) as u8
            };
            Yuv420Frame {
                width: w,
                height: h,
                y: (0..h).flat_map(|y| (0..w).map(move |x| f(x, y))).collect(),
                u: (0..h / 2)
                    .flat_map(|y| (0..w / 2).map(move |x| f(x * 2, y * 2) / 2 + 40))
                    .collect(),
                v: (0..h / 2)
                    .flat_map(|y| (0..w / 2).map(move |x| 200 - f(x * 2, y * 2) / 3))
                    .collect(),
                alpha: None,
            }
        })
        .collect()
}

/// Both encoders, several frames, one call each: the muxed file must be the
/// same bytes however the work was scheduled.
#[test]
fn batched_encoding_matches_frame_by_frame() {
    let frames = clip(96, 64, 7);

    // The reference: the parameter sets plus each frame's slice, assembled the
    // way the encoder does it, one frame at a time in order.
    let hevc_enc = threers::codec::hevc::compress::CompressedEncoder::new(96, 64, 27);
    let serial: Vec<Vec<u8>> = frames.iter().map(|f| hevc_enc.encode_slice_au(f)).collect();
    let batched = threers::codec::hevc::compress::encode_mp4(96, 64, 10, 27, &frames);
    for (i, au) in serial.iter().enumerate() {
        assert!(
            contains(&batched, &au[4..]),
            "HEVC frame {i}'s slice is missing from the muxed output"
        );
    }

    let h264_enc = threers::codec::h264::compress::CompressedEncoder::new(96, 64, 27);
    let serial: Vec<Vec<u8>> = frames.iter().map(|f| h264_enc.encode_slice_au(f)).collect();
    let batched = threers::codec::h264::compress::encode_mp4(96, 64, 10, 27, &frames);
    for (i, au) in serial.iter().enumerate() {
        assert!(
            contains(&batched, &au[4..]),
            "H.264 frame {i}'s slice is missing from the muxed output"
        );
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    needle.len() <= haystack.len()
        && haystack.windows(needle.len()).any(|w| w == needle)
}

/// The same clip encoded on one thread and on many must produce identical
/// files.
///
/// This is the test that would catch a race or an ordering slip: the thread
/// count is varied at run time, so both paths go through the same code and only
/// the scheduling differs.
#[cfg(feature = "parallel")]
#[test]
fn thread_count_does_not_change_the_output() {
    let frames = clip(112, 80, 9);
    let run = |threads: usize| -> (Vec<u8>, Vec<u8>) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build pool");
        pool.install(|| {
            (
                threers::codec::hevc::compress::encode_mp4(112, 80, 10, 27, &frames),
                threers::codec::h264::compress::encode_mp4(112, 80, 10, 27, &frames),
            )
        })
    };
    let (hevc_1, h264_1) = run(1);
    for threads in [2, 4, 8] {
        let (hevc_n, h264_n) = run(threads);
        assert_eq!(
            hevc_1.len(),
            hevc_n.len(),
            "HEVC output length changed on {threads} threads"
        );
        assert!(hevc_1 == hevc_n, "HEVC output changed on {threads} threads");
        assert!(h264_1 == h264_n, "H.264 output changed on {threads} threads");
    }
}
