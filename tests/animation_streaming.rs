//! The browser MP4 paths consume frames lazily instead of collecting them.
//!
//! An RGBA frame is 4 bytes per pixel — 133 MB at 8K — so collecting a clip
//! before encoding it is the dominant memory cost in
//! [`encode_animation_rgba`]. The MP4 codecs do not need that, and now avoid it.
//! What matters is that avoiding it changed nothing else: same bytes, same
//! errors, same progress.

use threers::codec::{
    encode_animation_rgba, encode_animation_rgba_with_progress, AnimationEncodeError,
    AnimationEncodeOptions, AnimationExportPhase, BrowserCodec,
};

fn opts(codec: BrowserCodec, w: u32, h: u32) -> AnimationEncodeOptions {
    AnimationEncodeOptions {
        width: w,
        height: h,
        fps: 24,
        codec,
        ..Default::default()
    }
}

fn frame(w: u32, h: u32, t: u32) -> Vec<u8> {
    let mut v = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            v[i] = (x * 3 + t * 17) as u8;
            v[i + 1] = (y * 5) as u8;
            v[i + 2] = ((x ^ y) + t) as u8;
            v[i + 3] = 255;
        }
    }
    v
}

#[test]
fn lazy_and_eager_iterators_produce_the_same_bytes() {
    let (w, h, n) = (64u32, 48u32, 5u32);
    let collected: Vec<Vec<u8>> = (0..n).map(|t| frame(w, h, t)).collect();

    for codec in [BrowserCodec::Mp4, BrowserCodec::Mp4Hevc] {
        let from_vec = encode_animation_rgba(&opts(codec, w, h), collected.clone()).unwrap();
        // A lazy iterator that never materialises the whole clip.
        let from_lazy =
            encode_animation_rgba(&opts(codec, w, h), (0..n).map(|t| frame(w, h, t))).unwrap();
        assert_eq!(from_vec, from_lazy, "{codec:?}: lazy differs from collected");
        assert!(!from_vec.is_empty());
    }
}

#[test]
fn an_empty_clip_is_still_rejected() {
    for codec in [BrowserCodec::Mp4, BrowserCodec::Mp4Hevc] {
        let r = encode_animation_rgba(&opts(codec, 64, 48), Vec::<Vec<u8>>::new());
        assert!(
            matches!(r, Err(AnimationEncodeError::Empty)),
            "{codec:?}: expected Empty, got {r:?}",
            r = r.map(|b| b.len())
        );
    }
}

#[test]
fn a_wrong_sized_frame_is_still_rejected() {
    let (w, h) = (64u32, 48u32);
    for codec in [BrowserCodec::Mp4, BrowserCodec::Mp4Hevc] {
        // Second frame is short. The streaming path has to notice mid-flight.
        let frames = vec![frame(w, h, 0), vec![0u8; 10], frame(w, h, 2)];
        let r = encode_animation_rgba(&opts(codec, w, h), frames);
        match r {
            Err(AnimationEncodeError::FrameSize { frame, expected, got }) => {
                assert_eq!(frame, 1);
                assert_eq!(expected, (w * h * 4) as usize);
                assert_eq!(got, 10);
            }
            other => panic!("{codec:?}: expected FrameSize, got {:?}", other.map(|b| b.len())),
        }
    }
}

#[test]
fn odd_dimensions_are_rejected_for_both_mp4_codecs() {
    // 4:2:0 chroma needs even dimensions. This guard covered `Mp4` only until
    // `Mp4Hevc` was added beside it.
    for codec in [BrowserCodec::Mp4, BrowserCodec::Mp4Hevc] {
        let r = encode_animation_rgba(&opts(codec, 65, 48), vec![frame(65, 48, 0)]);
        assert!(
            matches!(r, Err(AnimationEncodeError::Dimension)),
            "{codec:?}: odd width should be rejected"
        );
    }
}

#[test]
fn progress_reaches_every_frame_and_finishes() {
    let (w, h, n) = (64u32, 48u32, 4u32);
    for codec in [BrowserCodec::Mp4, BrowserCodec::Mp4Hevc] {
        let mut seen = Vec::new();
        let mut done = false;
        let out = encode_animation_rgba_with_progress(
            &opts(codec, w, h),
            (0..n).map(|t| frame(w, h, t)),
            |p| {
                if matches!(p.phase, AnimationExportPhase::Encode) && p.frame > 0 {
                    seen.push(p.frame);
                }
                done |= matches!(p.phase, AnimationExportPhase::Done);
            },
        )
        .unwrap();
        assert!(!out.is_empty());
        assert_eq!(seen, (1..=n).collect::<Vec<_>>(), "{codec:?}: progress frames");
        assert!(done, "{codec:?}: never reported Done");
    }
}
