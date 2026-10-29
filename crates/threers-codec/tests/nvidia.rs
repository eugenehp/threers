//! NVDEC and NVENC on a machine that has them.
//!
//! NVDEC must reproduce the native decoder's planes exactly: both decode a
//! conforming stream, and decoding is deterministic, so any difference is a
//! bug in one of them (or in how the NVDEC side copies surfaces out). The
//! streams come from ffmpeg's software encoders, with B-frames where the
//! codec has them. NVENC's output must decode, in the native decoder, to
//! something close to what went in.
//!
//! Skipped without an NVIDIA GPU and driver; set `THREERS_REQUIRE_NVIDIA=1`
//! to fail instead, so a run on the GPU machine cannot pass by skipping.

/// The backends exist on 64-bit Linux and Windows only; elsewhere this
/// suite says so rather than passing with no tests.
#[cfg(not(all(any(target_os = "linux", target_os = "windows"), target_pointer_width = "64")))]
#[test]
fn nvidia_backends_are_linux_and_windows_only() {
    eprintln!("SKIP: NVENC/NVDEC are built on 64-bit Linux and Windows only");
}

#[cfg(all(any(target_os = "linux", target_os = "windows"), target_pointer_width = "64"))]
mod nv {

use std::path::PathBuf;
use std::process::Command;

use threers_codec::decode::{Backend, Decoder, DecodedFrame, NativeDecoder};
use threers_codec::hw::{EncodeBackend, EncodeOptions, Mp4Encoder};
use threers_codec::mkv_demux::{Mkv, TrackKind};
use threers_codec::mp4_demux::{demux_video, Sample, TrackCodec, VideoTrack};
use threers_codec::nvidia::{self, NvdecDecoder};

fn gpu() -> bool {
    if nvidia::available() {
        eprintln!("NVIDIA: {}", nvidia::device_name().unwrap_or_default());
        return true;
    }
    assert!(std::env::var_os("THREERS_REQUIRE_NVIDIA").is_none(), "THREERS_REQUIRE_NVIDIA is set but no NVIDIA GPU/driver loaded");
    eprintln!("SKIP: no NVIDIA GPU or driver");
    false
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("threers_nv_{}_{name}", std::process::id()))
}

/// Encode a lavfi test pattern with ffmpeg; `None` when the encoder is missing.
fn ffmpeg(name: &str, args: &[&str]) -> Option<Vec<u8>> {
    let out = tmp(name);
    let ok = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=352x288:rate=25", "-frames:v", "24"])
        .args(args)
        .arg(&out)
        // SVT-AV1 prints a banner whatever ffmpeg's log level is.
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let data = ok.then(|| std::fs::read(&out).ok()).flatten();
    let _ = std::fs::remove_file(&out);
    if data.is_none() {
        eprintln!("SKIP {name}: ffmpeg could not make it");
    }
    data
}

/// A VP9/AV1 track out of a WebM, as `(track, frame bytes)`.
fn webm_track(data: &[u8], codec: TrackCodec) -> (VideoTrack, Vec<Vec<u8>>) {
    let mkv = Mkv::parse(data).unwrap();
    let t = mkv.tracks.iter().find(|t| t.kind == TrackKind::Video).unwrap();
    let frames: Vec<_> = mkv.frames.iter().filter(|f| f.track == t.number).collect();
    let samples = frames
        .iter()
        .enumerate()
        .map(|(i, f)| Sample { offset: 0, size: f.data.len() as u32, duration: 1, dts: i as i64, pts: f.pts, sync: f.keyframe })
        .collect();
    let track = VideoTrack { codec, width: t.width, height: t.height, config: t.codec_private.clone(), timescale: 1000, samples };
    (track, frames.iter().map(|f| f.data.clone()).collect())
}

fn mp4_track(data: &[u8]) -> (VideoTrack, Vec<Vec<u8>>) {
    let track = demux_video(data).unwrap();
    let bytes = track.samples.iter().map(|s| data[s.offset as usize..][..s.size as usize].to_vec()).collect();
    (track, bytes)
}

fn same(a: &DecodedFrame, b: &DecodedFrame) -> bool {
    (a.width, a.height, a.bit_depth) == (b.width, b.height, b.bit_depth) && a.y == b.y && a.u == b.u && a.v == b.v
}

/// Decode every sample with both decoders; every picture must match.
fn compare(name: &str, track: &VideoTrack, samples: &[Vec<u8>]) {
    let mut hw = NvdecDecoder::new(track).unwrap_or_else(|e| panic!("{name}: NVDEC would not open: {e}"));
    let mut sw = NativeDecoder::new(track).unwrap();
    for (s, bytes) in track.samples.iter().zip(samples) {
        hw.push(bytes, s.pts, s.dts, s.duration as i64).unwrap_or_else(|e| panic!("{name}: NVDEC: {e}"));
        sw.push(bytes, s.pts, s.dts, s.duration as i64).unwrap();
    }
    let mut n = 0;
    for i in track.presentation_order() {
        let pts = track.samples[i].pts;
        let (a, b) = (hw.take_frame(pts), sw.take_frame(pts));
        let (a, b) = (a.unwrap_or_else(|| panic!("{name}: NVDEC has no picture at {pts}")), b.unwrap());
        assert!(same(&a, &b), "{name}: picture at pts {pts} differs ({}x{} {}-bit)", a.width, a.height, a.bit_depth);
        n += 1;
    }
    eprintln!("{name}: {n} pictures identical to the native decoder");
}

#[test]
fn nvdec_matches_the_native_decoder() {
    if !gpu() {
        return;
    }
    let cases: [(&str, &[&str], TrackCodec); 5] = [
        ("h264_b.mp4", &["-c:v", "libx264", "-bf", "3", "-pix_fmt", "yuv420p"], TrackCodec::H264),
        ("hevc_b.mp4", &["-c:v", "libx265", "-x265-params", "bframes=3:log-level=error", "-pix_fmt", "yuv420p", "-tag:v", "hvc1"], TrackCodec::Hevc),
        ("hevc10.mp4", &["-c:v", "libx265", "-x265-params", "log-level=error", "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"], TrackCodec::Hevc),
        ("vp9.webm", &["-c:v", "libvpx-vp9", "-b:v", "500k", "-pix_fmt", "yuv420p"], TrackCodec::Vp9),
        ("av1.webm", &["-c:v", "libsvtav1", "-preset", "8", "-pix_fmt", "yuv420p"], TrackCodec::Av1),
    ];
    for (name, args, codec) in cases {
        let Some(data) = ffmpeg(name, args) else { continue };
        let (track, samples) = if name.ends_with(".mp4") { mp4_track(&data) } else { webm_track(&data, codec) };
        assert_eq!(track.codec, codec, "{name}");
        compare(name, &track, &samples);
    }
}

#[test]
fn nvdec_is_the_preferred_decoder_here() {
    if !gpu() {
        return;
    }
    assert_eq!(Backend::preferred(), Backend::Nvdec);
    let Some(data) = ffmpeg("pref.mp4", &["-c:v", "libx264", "-pix_fmt", "yuv420p"]) else { return };
    let track = demux_video(&data).unwrap();
    assert_eq!(Decoder::new(&track).unwrap().backend(), Backend::Nvdec);
}

#[test]
fn nvenc_writes_an_mp4_the_native_decoder_reads() {
    if !gpu() {
        return;
    }
    let (w, h) = (640u32, 360u32);
    let frame = |i: u32| -> Vec<u8> {
        (0..w * h)
            .flat_map(|p| {
                let (x, y) = (p % w, p / w);
                [((x + i * 3) % 256) as u8, (y * 255 / h) as u8, 96, 255]
            })
            .collect()
    };
    for options in [
        EncodeOptions { fps: 30, quality: 0.8, ..Default::default() },
        EncodeOptions { fps: 30, bitrate: Some(2_000_000), realtime: true, ..Default::default() },
    ] {
        let mut enc = Mp4Encoder::with_backend(w, h, options, EncodeBackend::Nvenc).unwrap();
        assert_eq!(enc.backend(), EncodeBackend::Nvenc);
        for i in 0..40 {
            enc.push_rgba(&frame(i)).unwrap();
        }
        let mp4 = enc.finish().unwrap();
        let track = demux_video(&mp4).unwrap();
        assert_eq!((track.codec, track.width, track.height, track.samples.len()), (TrackCodec::Hevc, w, h, 40));
        let mut dec = NativeDecoder::new(&track).unwrap();
        let mut worst = f64::INFINITY;
        for (i, s) in track.samples.iter().enumerate() {
            dec.push(&mp4[s.offset as usize..][..s.size as usize], s.pts, s.dts, s.duration as i64).unwrap();
            let bgra = dec.take(s.pts).unwrap();
            let src = frame(i as u32);
            let mut se = 0.0;
            for (a, b) in src.chunks_exact(4).zip(bgra.chunks_exact(4)) {
                for (x, y) in [(a[0], b[2]), (a[1], b[1]), (a[2], b[0])] {
                    se += (x as f64 - y as f64).powi(2);
                }
            }
            let psnr = 10.0 * (255.0f64.powi(2) / (se / (w * h * 3) as f64).max(1e-9)).log10();
            worst = worst.min(psnr);
        }
        eprintln!("NVENC {options:?}: {} bytes, worst PSNR {worst:.1} dB", mp4.len());
        assert!(worst > 30.0, "NVENC output too far from the source: {worst:.1} dB");
        // And ffprobe agrees it is a 40-frame HEVC MP4, when it is installed.
        let path = tmp("nvenc.mp4");
        std::fs::write(&path, &mp4).unwrap();
        if let Ok(out) = Command::new("ffprobe")
            .args(["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,nb_read_frames,profile", "-of", "csv=p=0"])
            .arg(&path)
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout);
            eprintln!("ffprobe: {}", s.trim());
            assert!(s.starts_with("hevc,"), "ffprobe: {s}");
            assert!(s.trim().ends_with(",40"), "ffprobe: {s}");
        }
        let _ = std::fs::remove_file(&path);
    }
}
}
