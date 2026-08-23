//! Validate the native soft-subtitle tracks end-to-end.
//!
//! The structural tests always run: they walk the container and assert the
//! subtitle track is present and well-formed. The round-trip tests need
//! `ffmpeg` on `PATH` and use it as an oracle — it must demux our subtitle
//! track back to the exact cue text and timings we put in. A malformed track
//! either fails to open or comes back with the wrong text.
#![cfg(all(feature = "native-codec", feature = "captions"))]

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use threers::captions::{CaptionTrack, Cue};
use threers::codec::hevc::hvcc::{build_hvcc, HvccArray, HvccProfile};
use threers::codec::hevc::{HevcEncoder, Yuv420Frame};
use threers::codec::mp4::{mux_hevc_with_captions, Mp4Params};
use threers::codec::vp9::encode_intra_gray;
use threers::codec::webm::{mux_webm_with_captions, WebmCodec, WebmFrame, WebmParams};

fn ffmpeg_ok() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn tmp(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = N.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "threers_captions_{}_{n}_{seq}_{name}",
        std::process::id()
    ))
}

/// The track both container tests mux.
fn sample_track() -> CaptionTrack {
    CaptionTrack::new()
        .language("en")
        .label("English")
        .cue(0.0, 1.0, "First caption")
        .cue(1.5, 2.5, "Second caption\non two lines")
}

/// Demux the first subtitle stream back out as SubRip text.
fn extract_subtitles(path: &std::path::Path) -> String {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(path)
        .args(["-map", "0:s:0", "-f", "srt", "-"])
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg failed to demux subtitles: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n")
}

// ---------------------------------------------------------------------------
// WebM / WebVTT
// ---------------------------------------------------------------------------

fn build_webm(captions: &CaptionTrack) -> Vec<u8> {
    let bitstream = encode_intra_gray(64, 64);
    let frames: Vec<WebmFrame> = (0..90)
        .map(|i| WebmFrame {
            data: &bitstream,
            alpha: None,
            timecode: i as u64 * 33,
            keyframe: true,
        })
        .collect();
    mux_webm_with_captions(
        &WebmParams {
            width: 64,
            height: 64,
            codec: WebmCodec::Vp9,
            timecode_scale_ns: 1_000_000,
            frames: &frames,
        },
        captions,
    )
}

#[test]
fn webm_carries_a_webvtt_track_entry() {
    let bytes = build_webm(&sample_track());
    assert!(
        bytes.windows(18).any(|w| w == b"D_WEBVTT/SUBTITLES"),
        "expected the WebM WebVTT codec id"
    );
    assert!(
        bytes.windows(3).any(|w| w == b"eng"),
        "expected the ISO-639-2 language"
    );
    assert!(
        bytes.windows(7).any(|w| w == b"English"),
        "expected the track label"
    );
    // Cue payloads are `identifier\nsettings\ntext`.
    assert!(bytes.windows(15).any(|w| w == b"\n\nFirst caption"));
}

#[test]
fn webm_without_captions_is_byte_identical_to_before() {
    use threers::codec::webm::mux_webm;
    let bitstream = encode_intra_gray(64, 64);
    let frames = [WebmFrame {
        data: &bitstream,
        alpha: None,
        timecode: 0,
        keyframe: true,
    }];
    let params = WebmParams {
        width: 64,
        height: 64,
        codec: WebmCodec::Vp9,
        timecode_scale_ns: 1_000_000,
        frames: &frames,
    };
    let plain = mux_webm(&params);
    // An empty caption track must not add a track entry or change the bytes.
    let empty = mux_webm_with_captions(&params, &CaptionTrack::new());
    assert_eq!(plain, empty, "empty captions must not alter the output");
}

#[test]
fn webm_webvtt_round_trips_through_ffmpeg() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = tmp("captions.webm");
    std::fs::write(&path, build_webm(&sample_track())).unwrap();
    let srt = extract_subtitles(&path);
    let _ = std::fs::remove_file(&path);

    assert!(srt.contains("First caption"), "{srt}");
    assert!(srt.contains("Second caption\non two lines"), "{srt}");
    assert!(srt.contains("00:00:00,000 --> 00:00:01,000"), "{srt}");
    assert!(srt.contains("00:00:01,500 --> 00:00:02,500"), "{srt}");
}

#[test]
fn webm_cue_settings_survive_the_round_trip() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let mut track = CaptionTrack::new().language("en");
    track.push(
        Cue::new(0.0, 1.0, "Positioned")
            .id("intro")
            .align(threers::captions::CaptionAlign::Left)
            .at_position(0.2),
    );
    let path = tmp("settings.webm");
    std::fs::write(&path, build_webm(&track)).unwrap();

    // `-c:s copy` keeps the identifier and settings side data intact.
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&path)
        .args(["-map", "0:s:0", "-c:s", "copy", "-f", "webvtt", "-"])
        .output()
        .expect("run ffmpeg");
    let _ = std::fs::remove_file(&path);
    let vtt = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(vtt.contains("intro"), "cue identifier lost: {vtt}");
    assert!(vtt.contains("align:left"), "cue settings lost: {vtt}");
    assert!(vtt.contains("position:20%"), "cue settings lost: {vtt}");
}

// ---------------------------------------------------------------------------
// MP4 / tx3g
// ---------------------------------------------------------------------------

/// Split an Annex-B byte stream into its NAL units (start codes removed).
fn split_annexb(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push((i, 3));
            i += 3;
        } else if i + 4 <= stream.len()
            && stream[i] == 0
            && stream[i + 1] == 0
            && stream[i + 2] == 0
            && stream[i + 3] == 1
        {
            starts.push((i, 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::new();
    for (n, &(off, len)) in starts.iter().enumerate() {
        let end = starts.get(n + 1).map(|&(o, _)| o).unwrap_or(stream.len());
        nals.push(stream[off + len..end].to_vec());
    }
    nals
}

/// Encode a few HEVC frames and mux them with `captions` into an MP4.
fn build_mp4(captions: &CaptionTrack) -> Vec<u8> {
    let (w, h) = (64u32, 64u32);
    let mut encoder = HevcEncoder::new(w, h);
    let (mut vps, mut sps, mut pps) = (Vec::new(), Vec::new(), Vec::new());
    let mut samples: Vec<Vec<u8>> = Vec::new();

    for i in 0..30u32 {
        let mut frame = Yuv420Frame::new(w, h);
        frame.y.fill((i * 8) as u8);
        frame.u.fill(128);
        frame.v.fill(128);
        let au = encoder.encode_frame(&frame);
        let mut sample = Vec::new();
        for nal in split_annexb(&au) {
            // Parameter sets belong in `hvcC`, not in the samples.
            match (nal[0] >> 1) & 0x3F {
                32 => vps = nal,
                33 => sps = nal,
                34 => pps = nal,
                _ => {
                    sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                    sample.extend_from_slice(&nal);
                }
            }
        }
        samples.push(sample);
    }

    let vps_arr = [vps];
    let sps_arr = [sps];
    let pps_arr = [pps];
    let hvcc = build_hvcc(
        &HvccProfile::main_420(120),
        &[
            HvccArray {
                nal_type: 32,
                complete: true,
                nals: &vps_arr,
            },
            HvccArray {
                nal_type: 33,
                complete: true,
                nals: &sps_arr,
            },
            HvccArray {
                nal_type: 34,
                complete: true,
                nals: &pps_arr,
            },
        ],
    );

    mux_hevc_with_captions(
        &Mp4Params {
            width: w,
            height: h,
            timescale: 600,
            frame_duration: 20, // 30 fps
            hvcc_payload: &hvcc,
            almo_payload: None,
            samples: &samples,
        },
        captions,
    )
}

#[test]
fn mp4_boxes_still_tile_with_a_text_track() {
    let out = build_mp4(&sample_track());
    let mut o = 0;
    let mut tags = Vec::new();
    while o + 8 <= out.len() {
        let size = u32::from_be_bytes([out[o], out[o + 1], out[o + 2], out[o + 3]]) as usize;
        tags.push(String::from_utf8_lossy(&out[o + 4..o + 8]).to_string());
        assert!(size >= 8 && o + size <= out.len(), "box {size} at {o}");
        o += size;
    }
    assert_eq!(o, out.len(), "top-level boxes tile the file");
    assert_eq!(tags, vec!["ftyp", "mdat", "moov"]);
    assert!(
        out.windows(4).any(|w| w == b"tx3g"),
        "expected a tx3g entry"
    );
    assert!(
        out.windows(4).any(|w| w == b"sbtl"),
        "expected a subtitle handler"
    );
    assert!(
        out.windows(4).any(|w| w == b"ftab"),
        "expected a font table"
    );
}

#[test]
fn mp4_without_captions_is_unchanged() {
    use threers::codec::mp4::mux_hevc;
    let params = Mp4Params {
        width: 32,
        height: 32,
        timescale: 600,
        frame_duration: 20,
        hvcc_payload: &[0u8; 24],
        almo_payload: None,
        samples: &[vec![0u8; 16]],
    };
    assert_eq!(
        mux_hevc(&params),
        mux_hevc_with_captions(&params, &CaptionTrack::new()),
        "empty captions must not alter the output"
    );
}

#[test]
fn mp4_tx3g_round_trips_through_ffmpeg() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = tmp("captions.mp4");
    std::fs::write(&path, build_mp4(&sample_track())).unwrap();
    let srt = extract_subtitles(&path);
    let _ = std::fs::remove_file(&path);

    assert!(srt.contains("First caption"), "{srt}");
    assert!(srt.contains("Second caption"), "{srt}");
    assert!(srt.contains("00:00:00,000 --> 00:00:01,000"), "{srt}");
    assert!(srt.contains("00:00:01,500 --> 00:00:02,500"), "{srt}");
}

#[test]
fn mp4_reports_a_selectable_subtitle_stream() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let path = tmp("probe.mp4");
    std::fs::write(&path, build_mp4(&sample_track())).unwrap();
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=codec_name:stream_tags=language",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&path)
        .output();
    let _ = std::fs::remove_file(&path);
    let Ok(out) = out else {
        eprintln!("skipping: ffprobe not on PATH");
        return;
    };
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("mov_text"),
        "expected a tx3g/mov_text stream: {text}"
    );
    assert!(
        text.contains("eng"),
        "expected the English language tag: {text}"
    );
}
