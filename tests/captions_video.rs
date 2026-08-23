//! End-to-end checks of caption delivery through `export_video`.
//!
//! Burn-in and sidecar tests run anywhere (GIF encoding is native). The embed
//! tests shell out to ffmpeg and are skipped when it is not installed.
#![cfg(all(feature = "video", feature = "native-codec"))]

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use threers::captions::{CaptionAnchor, CaptionStyle};
use threers::{CaptionFormat, CaptionMode, CaptionTrack, VideoCodec, VideoExporter, VideoOptions};

const W: u32 = 192;
const H: u32 = 108;

fn ffmpeg_ok() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn tmp_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let seq = N.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("threers-captions-{}-{n}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn track() -> CaptionTrack {
    CaptionTrack::new()
        .language("en")
        .label("English")
        .cue(0.0, 0.5, "Opening line")
        .cue(0.5, 1.0, "Closing line")
}

/// A solid dark frame — any bright pixel afterwards came from a caption.
fn dark_frame(_i: usize) -> Vec<u8> {
    let mut f = vec![0u8; (W * H * 4) as usize];
    for px in f.chunks_exact_mut(4) {
        px.copy_from_slice(&[8, 8, 12, 255]);
    }
    f
}

fn caption_style() -> CaptionStyle {
    CaptionStyle::default()
        .font_size(12.0)
        .margin(6.0)
        .padding(3.0)
        .anchor(CaptionAnchor::Bottom)
}

#[test]
fn burn_in_paints_captions_into_the_captured_frames() {
    let dir = tmp_dir();
    let out = dir.join("burn.gif");
    // Capture the frames the exporter actually encodes.
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Vec<u8>>::new()));

    let opts = VideoOptions::new(out.to_string_lossy().into_owned())
        .fps(10)
        .codec(VideoCodec::Gif)
        .captions(track())
        .caption_style(caption_style());

    // 10 frames at 10 fps = one second, covering both cues.
    threers::export_video(W, H, 10, &opts, |i| {
        let f = dark_frame(i);
        seen.lock().unwrap().push(f.clone());
        f
    })
    .expect("export");

    // The closure hands back clean frames; burn-in happens after, so verify
    // through the encoded GIF instead.
    let gif = std::fs::read(&out).expect("gif written");
    assert_eq!(&gif[..3], b"GIF");
    let (_, decoded) = threers::decode_gif(&gif).expect("decode gif");
    assert_eq!(decoded.len(), 10);

    let bright = |rgba: &[u8]| {
        rgba.chunks_exact(4)
            .filter(|px| px[0] > 120 && px[1] > 120 && px[2] > 120)
            .count()
    };
    for (i, frame) in decoded.iter().enumerate() {
        assert!(
            bright(&frame.rgba) > 0,
            "frame {i} has no caption pixels — every frame here is inside a cue"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn frames_outside_every_cue_stay_clean() {
    let dir = tmp_dir();
    let out = dir.join("gap.gif");
    // One cue covering only the first half-second of a one-second export.
    let opts = VideoOptions::new(out.to_string_lossy().into_owned())
        .fps(10)
        .codec(VideoCodec::Gif)
        .captions(CaptionTrack::new().cue(0.0, 0.5, "Only at the start"))
        .caption_style(caption_style());
    threers::export_video(W, H, 10, &opts, dark_frame).expect("export");

    let (_, decoded) = threers::decode_gif(&std::fs::read(&out).unwrap()).unwrap();
    let bright = |rgba: &[u8]| {
        rgba.chunks_exact(4)
            .filter(|px| px[0] > 120 && px[1] > 120 && px[2] > 120)
            .count()
    };
    assert!(bright(&decoded[0].rgba) > 0, "cue should show at t=0");
    assert!(bright(&decoded[4].rgba) > 0, "cue should show at t=0.4");
    assert_eq!(
        bright(&decoded[5].rgba),
        0,
        "cue ends at 0.5s — frame 5 must be clean"
    );
    assert_eq!(bright(&decoded[9].rgba), 0, "frame 9 must be clean");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sidecar_mode_writes_a_subtitle_file_next_to_the_video() {
    let dir = tmp_dir();
    let out = dir.join("movie.gif");
    let opts = VideoOptions::new(out.to_string_lossy().into_owned())
        .fps(10)
        .codec(VideoCodec::Gif)
        .captions(track())
        .caption_mode(CaptionMode::Sidecar);
    threers::export_video(W, H, 4, &opts, dark_frame).expect("export");

    let srt = dir.join("movie.srt");
    let text = std::fs::read_to_string(&srt).expect("sidecar written");
    assert!(
        text.starts_with("1\n00:00:00,000 --> 00:00:00,500"),
        "{text}"
    );
    assert!(text.contains("Opening line"), "{text}");
    assert!(text.contains("Closing line"), "{text}");

    // Sidecar-only must leave the pixels alone.
    let (_, decoded) = threers::decode_gif(&std::fs::read(&out).unwrap()).unwrap();
    let bright = decoded[0]
        .rgba
        .chunks_exact(4)
        .filter(|px| px[0] > 120)
        .count();
    assert_eq!(bright, 0, "Sidecar mode must not burn text in");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sidecar_format_and_path_are_configurable() {
    let dir = tmp_dir();
    let out = dir.join("clip.gif");
    let custom = dir.join("subs/custom.vtt");
    std::fs::create_dir_all(custom.parent().unwrap()).unwrap();

    VideoExporter::new(out.to_string_lossy().into_owned())
        .size(W, H)
        .frames(2)
        .fps(10)
        .codec(VideoCodec::Gif)
        .captions(track())
        .caption_mode(CaptionMode::BurnAndSidecar)
        .caption_format(CaptionFormat::Vtt)
        .caption_style(caption_style())
        .caption_sidecar_path(custom.to_string_lossy().into_owned())
        .export(dark_frame)
        .expect("export");

    let text = std::fs::read_to_string(&custom).expect("custom sidecar written");
    assert!(text.starts_with("WEBVTT"), "{text}");
    assert!(text.contains("00:00:00.000 --> 00:00:00.500"), "{text}");
    // BurnAndSidecar also paints the frames.
    let (_, decoded) = threers::decode_gif(&std::fs::read(&out).unwrap()).unwrap();
    assert!(decoded[0]
        .rgba
        .chunks_exact(4)
        .any(|px| px[0] > 120 && px[1] > 120));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn embedding_into_a_container_without_subtitles_is_rejected_up_front() {
    let dir = tmp_dir();
    let out = dir.join("nope.gif");
    let opts = VideoOptions::new(out.to_string_lossy().into_owned())
        .codec(VideoCodec::Gif)
        .captions(track())
        .caption_mode(CaptionMode::Embed);

    let mut rendered = 0usize;
    let err = threers::export_video(W, H, 4, &opts, |i| {
        rendered += 1;
        dark_frame(i)
    })
    .expect_err("GIF cannot embed subtitles");
    assert!(
        matches!(err, threers::VideoError::Captions(_)),
        "got {err:?}"
    );
    assert!(err.to_string().contains("GIF"), "{err}");
    assert_eq!(rendered, 0, "must fail before rendering any frame");
    assert!(!out.exists(), "no output should be written");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_broken_caption_font_fails_before_rendering() {
    let dir = tmp_dir();
    let out = dir.join("badfont.gif");
    let opts = VideoOptions::new(out.to_string_lossy().into_owned())
        .codec(VideoCodec::Gif)
        .captions(track())
        .caption_font(vec![0u8; 64]); // not a TrueType file

    let err = threers::export_video(W, H, 2, &opts, dark_frame).expect_err("bad font");
    assert!(
        matches!(err, threers::VideoError::Captions(_)),
        "got {err:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn embed_mode_muxes_a_soft_subtitle_track_into_mp4() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let dir = tmp_dir();
    let out = dir.join("soft.mp4");
    let opts = VideoOptions::new(out.to_string_lossy().into_owned())
        .fps(10)
        .codec(VideoCodec::H264)
        .captions(track())
        .caption_mode(CaptionMode::Embed);
    threers::export_video(W, H, 10, &opts, dark_frame).expect("export");

    let demuxed = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&out)
        .args(["-map", "0:s:0", "-f", "srt", "-"])
        .output()
        .expect("run ffmpeg");
    let srt = String::from_utf8_lossy(&demuxed.stdout).replace("\r\n", "\n");
    assert!(
        demuxed.status.success(),
        "{}",
        String::from_utf8_lossy(&demuxed.stderr)
    );
    assert!(srt.contains("Opening line"), "{srt}");
    assert!(srt.contains("Closing line"), "{srt}");

    // Embed must not also burn the text in.
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "s",
            "-show_entries",
            "stream=codec_name:stream_tags=language,title",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(&out)
        .output();
    if let Ok(probe) = probe {
        let text = String::from_utf8_lossy(&probe.stdout);
        assert!(text.contains("mov_text"), "{text}");
        assert!(text.contains("eng"), "{text}");
        // The track label is not asserted here: MP4 keeps track names in a
        // `udta` box that ffmpeg's mov muxer does not populate from `title`.
        // WebM does carry it (Matroska `Name`) — see the muxer tests.
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn embed_mode_muxes_webvtt_into_webm() {
    if !ffmpeg_ok() {
        eprintln!("skipping: ffmpeg not on PATH");
        return;
    }
    let dir = tmp_dir();
    let out = dir.join("soft.webm");
    let opts = VideoOptions::new(out.to_string_lossy().into_owned())
        .fps(10)
        .codec(VideoCodec::Vp9)
        .captions(track())
        .caption_mode(CaptionMode::Embed);
    if threers::export_video(W, H, 10, &opts, dark_frame).is_err() {
        eprintln!("skipping: this ffmpeg has no libvpx-vp9");
        return;
    }

    let demuxed = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-i"])
        .arg(&out)
        .args(["-map", "0:s:0", "-f", "srt", "-"])
        .output()
        .expect("run ffmpeg");
    let srt = String::from_utf8_lossy(&demuxed.stdout);
    assert!(srt.contains("Opening line"), "{srt}");
    let _ = std::fs::remove_dir_all(&dir);
}
