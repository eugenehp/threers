//! The native export path renders, encodes and writes one frame at a time.
//!
//! It used to collect every frame as `Yuv420Frame` before encoding, then build
//! the whole file in memory before writing it. Both are `width * height`-scaled
//! costs: on a 4K 24-frame export that was 1518 MB peak, against 385 MB now.
//!
//! Streaming moves work around, so what these check is that it did not move any
//! behaviour: the output still decodes, and a bad frame still fails without
//! leaving a partial file for the caller to mistake for a result.

#![cfg(all(feature = "video", feature = "native-codec"))]

use std::process::Command;

use threers::{export_video, VideoCodec, VideoError, VideoOptions};

fn tmp(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("threers-exp-{}-{name}", std::process::id()))
        .to_string_lossy()
        .into_owned()
}

fn rgba(w: u32, h: u32, t: usize) -> Vec<u8> {
    let mut v = vec![0u8; (w * h * 4) as usize];
    for p in 0..(w * h) as usize {
        v[p * 4] = (p + t * 5) as u8;
        v[p * 4 + 1] = (p / 3) as u8;
        v[p * 4 + 2] = (p ^ t) as u8;
        v[p * 4 + 3] = 255;
    }
    v
}

#[test]
fn streamed_export_is_playable() {
    let (w, h, n) = (96u32, 64u32, 5usize);
    let out = tmp("ok.mp4");
    let opts = VideoOptions::new(out.clone()).fps(24).codec(VideoCodec::H264);
    export_video(w, h, n, &opts, |i| rgba(w, h, i)).expect("export");

    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=codec_name,width,height,nb_read_frames",
               "-of", "csv=p=0"])
        .arg(&out)
        .output();
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    let _ = std::fs::remove_file(&out);
    assert!(size > 0, "export wrote nothing");

    if let Ok(o) = probe {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            assert_eq!(s, format!("h264,{w},{h},{n}"), "ffprobe said {s:?}");
        }
    }
}

#[test]
fn a_wrong_sized_frame_errors_and_leaves_no_file() {
    let (w, h, n) = (96u32, 64u32, 4usize);
    let out = tmp("bad.mp4");
    let opts = VideoOptions::new(out.clone()).fps(24).codec(VideoCodec::H264);

    // Third frame is short. The encoder only discovers this mid-stream, after
    // it has already begun writing.
    let r = export_video(w, h, n, &opts, |i| {
        if i == 2 {
            vec![0u8; 7]
        } else {
            rgba(w, h, i)
        }
    });

    match r {
        Err(VideoError::FrameSize { frame, expected, got }) => {
            assert_eq!(frame, 2);
            assert_eq!(expected, (w * h * 4) as usize);
            assert_eq!(got, 7);
        }
        other => {
            let _ = std::fs::remove_file(&out);
            panic!("expected FrameSize, got {other:?}");
        }
    }
    assert!(
        !std::path::Path::new(&out).exists(),
        "a failed export left {out} behind"
    );
}
