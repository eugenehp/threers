//! VAAPI on a machine that has a VA driver.
//!
//! The VAAPI decoder must reproduce the native decoder's planes exactly: the
//! native decoder supplies the reference lists and picture order the GPU
//! decodes with, and decoding is deterministic, so any difference is a bug in
//! how they were handed over. The streams come from x264 with the features
//! that exercise that hand-over: B-frames and their lists, CAVLC and CABAC,
//! 8×8 transforms with scaling matrices, weighted prediction, multiple
//! slices, long-term references and field coding.
//!
//! Skipped without a VA driver; set `THREERS_REQUIRE_VAAPI=1` to fail
//! instead, so a run on the GPU machine cannot pass by skipping.

/// The backend exists on Linux only; elsewhere this suite says so rather
/// than passing with no tests.
#[cfg(not(target_os = "linux"))]
#[test]
fn vaapi_is_linux_only() {
    eprintln!("SKIP: VAAPI is built on Linux only");
}

#[cfg(target_os = "linux")]
mod va {
    use std::process::Command;

    use threers_codec::decode::{Backend, DecodedFrame, Decoder, NativeDecoder};
    use threers_codec::mp4_demux::demux_video;
    use threers_codec::vaapi::{self, VaapiDecoder};

    fn gpu() -> bool {
        if vaapi::available() {
            return true;
        }
        assert!(std::env::var_os("THREERS_REQUIRE_VAAPI").is_none(), "THREERS_REQUIRE_VAAPI is set but no VA driver decodes H.264 here");
        eprintln!("SKIP: no VA driver");
        false
    }

    /// An x264 stream of a lavfi pattern; `None` when ffmpeg cannot make it.
    fn x264(name: &str, src: &str, params: &str) -> Option<Vec<u8>> {
        encode(name, src, &["-pix_fmt", "yuv420p", "-c:v", "libx264", "-x264-params", params])
    }

    /// An x265 stream; `pix` picks 8- or 10-bit.
    fn x265(name: &str, src: &str, pix: &str, params: &str) -> Option<Vec<u8>> {
        let params = format!("log-level=error:{params}");
        encode(name, src, &["-pix_fmt", pix, "-c:v", "libx265", "-tag:v", "hvc1", "-x265-params", &params])
    }

    /// A libvpx VP9 stream in WebM.
    fn vpx(name: &str, src: &str, args: &[&str]) -> Option<Vec<u8>> {
        let mut all = vec!["-c:v", "libvpx-vp9"];
        all.extend_from_slice(args);
        encode(&format!("{name}.webm"), src, &all)
    }

    fn encode(name: &str, src: &str, args: &[&str]) -> Option<Vec<u8>> {
        let file = if name.contains('.') { name.to_string() } else { format!("{name}.mp4") };
        let out = std::env::temp_dir().join(format!("threers_va_{}_{file}", std::process::id()));
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", src, "-frames:v", "30"])
            .args(args)
            .arg(&out)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let data = ok.then(|| std::fs::read(&out).ok()).flatten();
        let _ = std::fs::remove_file(&out);
        if data.is_none() {
            eprintln!("SKIP {name}: ffmpeg/libx264 could not make it");
        }
        data
    }

    fn same(a: &DecodedFrame, b: &DecodedFrame) -> bool {
        (a.width, a.height, a.bit_depth) == (b.width, b.height, b.bit_depth) && a.y == b.y && a.u == b.u && a.v == b.v
    }

    /// The video track of a WebM and its frames' bytes, as an MP4 track
    /// would describe them.
    fn webm(data: &[u8]) -> (threers_codec::mp4_demux::VideoTrack, Vec<Vec<u8>>) {
        use threers_codec::mkv_demux::{Mkv, TrackKind};
        use threers_codec::mp4_demux::{Sample, TrackCodec, VideoTrack};
        let mkv = Mkv::parse(data).unwrap();
        let t = mkv.tracks.iter().find(|t| t.kind == TrackKind::Video).unwrap();
        let frames: Vec<_> = mkv.frames.iter().filter(|f| f.track == t.number).collect();
        let samples = frames
            .iter()
            .enumerate()
            .map(|(i, f)| Sample { offset: 0, size: f.data.len() as u32, duration: 1, dts: i as i64, pts: f.pts, sync: f.keyframe })
            .collect();
        let codec = if t.codec_id == "V_VP9" { TrackCodec::Vp9 } else { TrackCodec::Av1 };
        let track = VideoTrack { codec, width: t.width, height: t.height, config: t.codec_private.clone(), timescale: 1000, samples };
        (track, frames.iter().map(|f| f.data.clone()).collect())
    }

    /// [`compare`] for a WebM.
    fn compare_webm(name: &str, data: &[u8]) {
        let (track, frames) = webm(data);
        let mut hw = VaapiDecoder::new(&track).unwrap_or_else(|e| panic!("{name}: VAAPI would not open: {e}"));
        let mut sw = NativeDecoder::new(&track).unwrap();
        for (s, bytes) in track.samples.iter().zip(&frames) {
            hw.push(bytes, s.pts, s.dts, 1).unwrap_or_else(|e| panic!("{name}: VAAPI: {e}"));
            sw.push(bytes, s.pts, s.dts, 1).unwrap();
        }
        for s in &track.samples {
            let (a, b) = (hw.take_frame(s.pts), sw.take_frame(s.pts));
            match (a, b) {
                (Some(a), Some(b)) => assert!(same(&a, &b), "{name}: picture at pts {} differs ({}-bit)", s.pts, a.bit_depth),
                (None, None) => {}
                (a, b) => panic!("{name}: at pts {} VAAPI has {} and native {}", s.pts, a.is_some(), b.is_some()),
            }
        }
        eprintln!("{name}: {} frames identical to the native decoder", frames.len());
    }

    /// Decode `data` with VAAPI and natively; every picture must match.
    fn compare(name: &str, data: &[u8]) {
        let track = demux_video(data).unwrap();
        let mut hw = VaapiDecoder::new(&track).unwrap_or_else(|e| panic!("{name}: VAAPI would not open: {e}"));
        let mut sw = NativeDecoder::new(&track).unwrap();
        for s in &track.samples {
            let bytes = &data[s.offset as usize..][..s.size as usize];
            hw.push(bytes, s.pts, s.dts, s.duration as i64).unwrap_or_else(|e| panic!("{name}: VAAPI: {e}"));
            sw.push(bytes, s.pts, s.dts, s.duration as i64).unwrap();
        }
        for i in track.presentation_order() {
            let pts = track.samples[i].pts;
            let a = hw.take_frame(pts).unwrap_or_else(|| panic!("{name}: VAAPI has no picture at {pts}"));
            let b = sw.take_frame(pts).unwrap();
            assert!(same(&a, &b), "{name}: picture at pts {pts} differs ({}-bit)", a.bit_depth);
        }
        eprintln!("{name}: {} pictures identical to the native decoder", track.samples.len());
    }

    #[test]
    fn vaapi_h264_matches_the_native_decoder() {
        if !gpu() {
            return;
        }
        const SRC: &str = "testsrc2=size=640x360:rate=25";
        const MOVING: &str = "testsrc2=size=640x360:rate=25,rotate=a=t/2:c=black";
        let cases = [
            ("baseline", SRC, "profile=baseline:cabac=0:bframes=0"),
            ("main_b", SRC, "bframes=3:b-pyramid=normal"),
            ("high_8x8_cqm", MOVING, "bframes=2:8x8dct=1:cqm=jvt:ref=4"),
            ("weighted", "testsrc2=size=640x360:rate=25,fade=in:0:30", "bframes=2:weightp=2:weightb=1"),
            ("slices", MOVING, "slices=4:bframes=2"),
            ("cavlc_b", MOVING, "cabac=0:bframes=2"),
            ("interlaced", MOVING, "interlaced=1:bframes=2"),
        ];
        let mut ran = 0;
        for (name, src, params) in cases {
            let Some(data) = x264(name, src, params) else { continue };
            compare(name, &data);
            ran += 1;
        }
        assert!(ran > 0, "no stream could be made");
    }

    #[test]
    fn vaapi_hevc_matches_the_native_decoder() {
        if !gpu() {
            return;
        }
        const SRC: &str = "testsrc2=size=640x360:rate=25";
        const MOVING: &str = "testsrc2=size=640x360:rate=25,rotate=a=t/2:c=black";
        let cases = [
            ("hevc_b", SRC, "yuv420p", "bframes=3:b-pyramid=1"),
            ("hevc_moving", MOVING, "yuv420p", "bframes=4:ref=4"),
            ("hevc10", MOVING, "yuv420p10le", "bframes=2"),
            ("hevc_wpp", MOVING, "yuv420p", "wpp=1:bframes=2"),
            ("hevc_tiles", MOVING, "yuv420p", "wpp=0:bframes=2:tiles=2x2"),
            ("hevc_slices", MOVING, "yuv420p", "slices=3:bframes=2"),
            ("hevc_scaling", MOVING, "yuv420p", "scaling-list=default:bframes=2"),
            ("hevc_weighted", "testsrc2=size=640x360:rate=25,fade=in:0:30", "yuv420p", "weightp=1:weightb=1:bframes=2"),
            ("hevc_odd_size", "testsrc2=size=636x358:rate=25", "yuv420p", "bframes=2"),
        ];
        let mut ran = 0;
        for (name, src, pix, params) in cases {
            let Some(data) = x265(name, src, pix, params) else { continue };
            compare(name, &data);
            ran += 1;
        }
        assert!(ran > 0, "no stream could be made");
    }

    #[test]
    fn vaapi_vp9_matches_the_native_decoder() {
        if !gpu() {
            return;
        }
        if VaapiDecoder::new(&webm(&match vpx("probe", "testsrc2=size=320x240:rate=25", &["-b:v", "300k"]) { Some(d) => d, None => return }).0).is_err() {
            // A driver need not do every codec (Intel's free iHD build has no AV1).
            eprintln!("SKIP: the VA driver here does not decode VP9");
            return;
        }
        const MOVING: &str = "testsrc2=size=640x360:rate=25,rotate=a=t/2:c=black";
        let cases: [(&str, &[&str]); 6] = [
            // Alt-ref frames: hidden frames packed into superframes.
            ("vp9_altref", &["-b:v", "800k", "-auto-alt-ref", "1", "-lag-in-frames", "16"]),
            ("vp9_tiles", &["-b:v", "800k", "-tile-columns", "2", "-threads", "4"]),
            ("vp9_error_resilient", &["-b:v", "600k", "-error-resilient", "1"]),
            ("vp9_segments", &["-b:v", "600k", "-aq-mode", "3"]),
            ("vp9_10bit", &["-b:v", "800k", "-pix_fmt", "yuv420p10le", "-profile:v", "2"]),
            ("vp9_lossless", &["-lossless", "1"]),
        ];
        let mut ran = 0;
        for (name, args) in cases {
            let Some(data) = vpx(name, MOVING, args) else { continue };
            compare_webm(name, &data);
            ran += 1;
        }
        assert!(ran > 0, "no stream could be made");
    }

    #[test]
    fn vaapi_av1_matches_the_native_decoder() {
        if !gpu() {
            return;
        }
        let probe = |args: &[&str]| encode("av1_probe.webm", "testsrc2=size=320x240:rate=25", args);
        let Some(p) = probe(&["-c:v", "libsvtav1", "-preset", "10"]) else { return };
        if VaapiDecoder::new(&webm(&p).0).is_err() {
            // A driver need not do every codec (Intel's free iHD build has no AV1).
            eprintln!("SKIP: the VA driver here does not decode AV1");
            return;
        }
        const MOVING: &str = "testsrc2=size=640x360:rate=25,rotate=a=t/2:c=black";
        let svt = |extra: &'static str| -> Vec<&'static str> {
            let mut v = vec!["-c:v", "libsvtav1", "-preset", "8"];
            if !extra.is_empty() {
                v.extend(["-svtav1-params", extra]);
            }
            v
        };
        let cases: Vec<(&str, &str, Vec<&str>)> = vec![
            ("av1_svt", MOVING, svt("")),
            ("av1_svt_10bit", MOVING, [svt(""), vec!["-pix_fmt", "yuv420p10le"]].concat()),
            ("av1_svt_grain", MOVING, svt("film-grain=8")),
            ("av1_svt_tiles", MOVING, svt("tile-columns=2:tile-rows=1")),
            ("av1_svt_screen", "testsrc=size=640x360:rate=25", svt("scm=1")),
            ("av1_aom", MOVING, vec!["-c:v", "libaom-av1", "-cpu-used", "8", "-b:v", "600k", "-lag-in-frames", "10"]),
        ];
        let mut ran = 0;
        for (name, src, args) in cases {
            let Some(data) = encode(&format!("{name}.webm"), src, &args) else { continue };
            compare_webm(name, &data);
            ran += 1;
        }
        assert!(ran > 0, "no stream could be made");
        // The conformance vectors, when they are here.
        let dir = std::env::var("THREERS_AV1_CONFORMANCE").unwrap_or_else(|_| "/var/tmp/av1conf".into());
        let Ok(rd) = std::fs::read_dir(&dir) else { return };
        let mut paths: Vec<_> = rd.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "ivf")).collect();
        paths.sort();
        for p in paths {
            let data = std::fs::read(&p).unwrap();
            let (_, _, _, _, packets) = threers_codec::mkv_demux::parse_ivf(&data).unwrap();
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            use threers_codec::mp4_demux::{Sample, TrackCodec, VideoTrack};
            let first = threers_codec::decode::av1::Av1Decoder::new().decode(&packets[0].1).ok().and_then(|f| f.into_iter().next());
            let (w, h) = first.map_or((0, 0), |f| (f.pic.w as u32, f.pic.h as u32));
            let samples = packets.iter().enumerate().map(|(i, _)| Sample { offset: 0, size: 0, duration: 1, dts: i as i64, pts: i as i64, sync: i == 0 }).collect();
            let track = VideoTrack { codec: TrackCodec::Av1, width: w, height: h, config: Vec::new(), timescale: 1000, samples };
            let mut hw = match VaapiDecoder::new(&track) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("{name}: VAAPI refuses it ({e})");
                    continue;
                }
            };
            let mut sw = NativeDecoder::new(&track).unwrap();
            let mut n = 0;
            for (i, (_, pkt)) in packets.iter().enumerate() {
                if let Err(e) = hw.push(pkt, i as i64, i as i64, 1) {
                    eprintln!("{name}: VAAPI refuses frame {i} ({e})");
                    break;
                }
                sw.push(pkt, i as i64, i as i64, 1).unwrap();
                if let (Some(a), Some(b)) = (hw.take_frame(i as i64), sw.take_frame(i as i64)) {
                    assert!(same(&a, &b), "{name}: frame {i} differs");
                    n += 1;
                }
            }
            eprintln!("{name}: {n} frames identical to the native decoder");
        }
    }

    #[test]
    fn vaapi_encoder_writes_mp4s_the_native_decoder_reads() {
        if !gpu() {
            return;
        }
        use threers_codec::hw::{EncodeBackend, EncodeOptions, Mp4Encoder};
        if !EncodeBackend::Vaapi.available() {
            // A driver need not do every codec (Intel's free iHD build has no AV1).
            eprintln!("SKIP: the VA driver here does not encode HEVC");
            return;
        }
        // Sizes that are not whole 64×64 CTBs, so the driver must crop.
        for (w, h, options) in [
            (640u32, 360u32, EncodeOptions { fps: 30, quality: 0.8, ..Default::default() }),
            (1920, 1080, EncodeOptions { fps: 30, bitrate: Some(4_000_000), ..Default::default() }),
            (318, 238, EncodeOptions { fps: 25, quality: 0.7, ..Default::default() }),
        ] {
            let frame = |i: u32| -> Vec<u8> {
                (0..w * h)
                    .flat_map(|p| {
                        let (x, y) = (p % w, p / w);
                        [((x + i * 3) % 256) as u8, (y * 255 / h) as u8, ((x / 4 + y / 4) % 256) as u8, 255]
                    })
                    .collect()
            };
            let mut enc = Mp4Encoder::with_backend(w, h, options, EncodeBackend::Vaapi).unwrap();
            for i in 0..40 {
                enc.push_rgba(&frame(i)).unwrap();
            }
            let mp4 = enc.finish().unwrap();
            let track = demux_video(&mp4).unwrap();
            assert_eq!((track.width, track.height, track.samples.len()), (w, h, 40));
            let mut dec = NativeDecoder::new(&track).unwrap();
            let mut worst = f64::INFINITY;
            for (i, s) in track.samples.iter().enumerate() {
                dec.push(&mp4[s.offset as usize..][..s.size as usize], s.pts, s.dts, s.duration as i64).unwrap();
                let bgra = dec.take(s.pts).unwrap();
                let src = frame(i as u32);
                let se: f64 = src
                    .chunks_exact(4)
                    .zip(bgra.chunks_exact(4))
                    .map(|(a, b)| [(a[0], b[2]), (a[1], b[1]), (a[2], b[0])].iter().map(|&(x, y)| (x as f64 - y as f64).powi(2)).sum::<f64>())
                    .sum();
                worst = worst.min(10.0 * (255f64.powi(2) / (se / (w * h * 3) as f64).max(1e-9)).log10());
            }
            eprintln!("VAAPI {w}x{h} {options:?}: {} bytes, worst PSNR {worst:.1} dB", mp4.len());
            assert!(worst > 30.0, "{w}x{h}: {worst:.1} dB");
            let path = std::env::temp_dir().join(format!("threers_va_enc_{}_{w}.mp4", std::process::id()));
            std::fs::write(&path, &mp4).unwrap();
            if let Ok(out) = Command::new("ffprobe")
                .args(["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,nb_read_frames", "-of", "csv=p=0"])
                .arg(&path)
                .output()
            {
                let s = String::from_utf8_lossy(&out.stdout);
                assert_eq!(s.trim(), format!("hevc,{w},{h},40"), "ffprobe");
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn decoder_prefers_vaapi_here() {
        if !gpu() {
            return;
        }
        let Some(data) = x264("pref", "testsrc2=size=320x240:rate=25", "bframes=0") else { return };
        let track = demux_video(&data).unwrap();
        let d = Decoder::new(&track).unwrap();
        // NVDEC outranks VAAPI where both exist.
        let expect = if Backend::Nvdec.available() { Backend::Nvdec } else { Backend::Vaapi };
        assert_eq!(d.backend(), expect);
    }
}
