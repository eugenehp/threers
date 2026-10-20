//! The native H.264 / HEVC decoders, checked sample-for-sample.
//!
//! A decoder is only right if it reproduces exactly what the encoder
//! reconstructed — "looks fine" hides off-by-one rounding that drifts a long
//! GOP into mush. So every check here compares planes for equality against
//! an independent truth:
//!
//! - the crate's own encoders, whose reconstructions are already verified
//!   against ffmpeg (PCM streams must come back lossless);
//! - x264-encoded streams against ffmpeg's decode, and x265 streams against
//!   x265's own `--recon`, when those tools are installed. For HEVC the
//!   encoder's reconstruction is the reference rather than ffmpeg: ffmpeg was
//!   found to disagree with both x265 and libde265 on SAO at a CTU-16 slice
//!   (two chroma samples), and the encoder is the one that cannot be wrong.

#![cfg(all(feature = "native-codec", feature = "video"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use threers::codec::decode::{decode_annexb, DecodedFrame, NativeDecoder};
use threers::codec::hevc::Yuv420Frame;
use threers::codec::mp4_demux::{demux_video, TrackCodec};

/// A frame with edges, gradients and colour — every intra mode has
/// something to do.
fn test_frame(w: u32, h: u32, t: u32) -> Yuv420Frame {
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            let xs = x + 3 * t;
            let block = ((xs / 7) + (y / 5)).is_multiple_of(2);
            rgba[i] = if block { 230 } else { (xs * 255 / w) as u8 };
            rgba[i + 1] = if block { 30 } else { (y * 255 / h) as u8 };
            rgba[i + 2] = ((x * y + t * 17) % 251) as u8;
            rgba[i + 3] = 255;
        }
    }
    Yuv420Frame::from_rgba(w, h, &rgba)
}

/// The top-left `w × h` of a coded-size plane.
fn crop(plane: &[u8], stride: u32, w: u32, h: u32) -> Vec<u16> {
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| plane[(y * stride + x) as usize] as u16)
        .collect()
}

fn widen(p: &[u8]) -> Vec<u16> {
    p.iter().map(|&v| v as u16).collect()
}

fn assert_planes(f: &DecodedFrame, y: &[u16], u: &[u16], v: &[u16], what: &str) {
    assert_eq!(f.y.len(), y.len(), "{what}: luma size");
    let first = |a: &[u16], b: &[u16]| a.iter().zip(b).position(|(p, q)| p != q);
    assert!(f.y == y, "{what}: luma differs first at {:?}", first(&f.y, y));
    assert!(f.u == u, "{what}: Cb differs first at {:?}", first(&f.u, u));
    assert!(f.v == v, "{what}: Cr differs first at {:?}", first(&f.v, v));
}

/// Push every sample of an MP4 through `NativeDecoder`, returning frames in
/// presentation order.
fn decode_mp4(mp4: &[u8]) -> (TrackCodec, Vec<DecodedFrame>) {
    let track = demux_video(mp4).expect("demux");
    let mut dec = NativeDecoder::new(&track).expect("open decoder");
    let mut out = Vec::new();
    for s in &track.samples {
        let bytes = &mp4[s.offset as usize..][..s.size as usize];
        dec.push(bytes, s.pts, s.dts, s.duration as i64).expect("decode");
        let f = dec.take_frame(s.pts).expect("a picture per sample");
        out.push((s.pts, f));
    }
    out.sort_by_key(|(p, _)| *p);
    (track.codec, out.into_iter().map(|(_, f)| f).collect())
}

// ----------------------------------------------------------------------
// Against the crate's own encoders
// ----------------------------------------------------------------------

#[test]
fn h264_pcm_mp4_decodes_lossless() {
    // Odd size: exercises the conformance crop on both axes.
    let (w, h) = (66, 50);
    let frames: Vec<_> = (0..3).map(|t| test_frame(w, h, t)).collect();
    let mp4 = threers::codec::h264::encode_mp4(w, h, 30, &frames);
    let (codec, out) = decode_mp4(&mp4);
    assert_eq!(codec, TrackCodec::H264);
    assert_eq!(out.len(), frames.len());
    for (i, (f, src)) in out.iter().zip(&frames).enumerate() {
        assert_eq!((f.width, f.height), (w, h));
        assert_planes(f, &widen(&src.y), &widen(&src.u), &widen(&src.v), &format!("frame {i}"));
    }
}

#[test]
fn h264_compressed_matches_the_encoders_reconstruction() {
    use threers::codec::h264::{CompressedEncoder, EntropyMode};
    for mode in [EntropyMode::Cavlc, EntropyMode::Cabac] {
        for deblock in [false, true] {
            let (w, h) = (80, 48);
            let mut enc = CompressedEncoder::new(w, h, 30).entropy(mode).deblock(deblock);
            for t in 0..2 {
                let frame = test_frame(w, h, t);
                let (au, rec) = enc.encode_frame(&frame);
                let what = format!("{mode:?} deblock={deblock} frame {t}");
                // Parameter sets ride on the first access unit only; a
                // repeat is harmless, so give every frame its own.
                let mut stream = Vec::new();
                for ps in [enc.sps(), enc.pps()] {
                    stream.extend_from_slice(&[0, 0, 0, 1]);
                    stream.extend_from_slice(ps);
                }
                stream.extend_from_slice(&au);
                let out = decode_annexb(TrackCodec::H264, &stream).expect(&what);
                assert_eq!(out.len(), 1, "{what}");
                let cw = rec.coded_width;
                assert_planes(
                    &out[0],
                    &crop(&rec.y, cw, w, h),
                    &crop(&rec.u, cw / 2, w / 2, h / 2),
                    &crop(&rec.v, cw / 2, w / 2, h / 2),
                    &what,
                );
            }
        }
    }
}

#[test]
fn hevc_pcm_decodes_lossless() {
    use threers::codec::hevc::HevcEncoder;
    let (w, h) = (66, 66);
    let mut enc = HevcEncoder::new(w, h);
    let frame = test_frame(w, h, 1);
    let au = enc.encode_frame(&frame);
    let out = decode_annexb(TrackCodec::Hevc, &au).expect("decode");
    assert_eq!(out.len(), 1);
    assert_eq!((out[0].width, out[0].height), (w, h));
    assert_planes(&out[0], &widen(&frame.y), &widen(&frame.u), &widen(&frame.v), "HEVC PCM");
}

#[test]
fn hevc_compressed_with_tiles_matches_the_encoders_reconstruction() {
    use threers::codec::hevc::CompressedEncoder;
    // Tiles are the one HEVC tool x265 cannot produce; this is their test.
    for (cols, rows) in [(1, 1), (2, 1), (2, 2), (3, 2)] {
        for deblock in [false, true] {
            // Wide enough for three 256-sample columns: HEVC's minimum.
            let (w, h) = (832, 200);
            let mut enc = CompressedEncoder::new(w, h, 28)
                .residual(true)
                .deblock(deblock)
                .tiles(cols, rows);
            assert_eq!(enc.config().tile_grid, (cols, rows), "grid was clamped");
            let frame = test_frame(w, h, 2);
            let (au, rec) = enc.encode_frame(&frame);
            let what = format!("tiles {cols}x{rows} deblock={deblock}");
            let out = decode_annexb(TrackCodec::Hevc, &au).expect(&what);
            let cw = rec.coded_width;
            assert_planes(
                &out[0],
                &crop(&rec.y, cw, w, h),
                &crop(&rec.u, cw / 2, w / 2, h / 2),
                &crop(&rec.v, cw / 2, w / 2, h / 2),
                &what,
            );
        }
    }
}

#[test]
fn tiles_below_the_minimum_size_are_refused() {
    use threers::codec::hevc::CompressedEncoder;
    // 200 samples is too narrow for two 256-sample columns; 136 is tall
    // enough for two 64-sample rows.
    let enc = CompressedEncoder::new(200, 136, 28).tiles(2, 2);
    assert_eq!(enc.config().tile_grid, (1, 2));
}

#[test]
fn hevc_mp4_round_trips_through_native_decoder() {
    let (w, h) = (96, 64);
    let frames: Vec<_> = (0..4).map(|t| test_frame(w, h, t)).collect();
    let mp4 = threers::codec::hevc::encode_compressed_mp4(w, h, 30, 26, &frames);
    let (codec, out) = decode_mp4(&mp4);
    assert_eq!(codec, TrackCodec::Hevc);
    assert_eq!(out.len(), 4);
    for f in &out {
        assert_eq!((f.width, f.height), (w, h));
        // BGRA out, like VideoToolbox.
        assert_eq!(f.to_bgra().len(), (w * h * 4) as usize);
    }
}

#[test]
fn transparent_hevc_decodes_its_colour_layer() {
    use threers::codec::hevc::TransparentEncoder;
    let (w, h) = (64, 64);
    let mut enc = TransparentEncoder::new(w, h);
    let color = test_frame(w, h, 0);
    let alpha: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
    enc.encode_frame(&color, &alpha);
    let mov = enc.finish_mov(30);
    // The alpha layer is nuh_layer_id 1, which a base-layer decoder skips.
    let (_, out) = decode_mp4(&mov);
    assert_eq!(out.len(), 1);
    assert_planes(&out[0], &widen(&color.y), &widen(&color.u), &widen(&color.v), "colour layer");
}

#[test]
fn take_returns_bgra_and_drops_earlier_frames() {
    let (w, h) = (32, 32);
    let frames: Vec<_> = (0..3).map(|t| test_frame(w, h, t)).collect();
    let mp4 = threers::codec::h264::encode_mp4(w, h, 30, &frames);
    let track = demux_video(&mp4).unwrap();
    let mut dec = NativeDecoder::new(&track).unwrap();
    assert_eq!(dec.size(), (w, h));
    for s in &track.samples {
        dec.push(&mp4[s.offset as usize..][..s.size as usize], s.pts, s.dts, 0).unwrap();
    }
    assert_eq!(dec.queued(), 3);
    let last = track.samples.last().unwrap().pts;
    let bgra = dec.take(last).expect("last frame");
    assert_eq!(bgra.len(), (w * h * 4) as usize);
    assert_eq!(dec.queued(), 0, "taking a later frame drops the ones before it");
}

// ----------------------------------------------------------------------
// Against external encoders (skipped when the tools are missing)
// ----------------------------------------------------------------------

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success() || !o.stdout.is_empty() || !o.stderr.is_empty())
        .unwrap_or(false)
}

fn tmp(name: &str) -> PathBuf {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("threers_dec_{}_{n}_{name}", std::process::id()))
}

fn run(cmd: &mut Command) {
    let out = cmd.output().expect("spawn");
    assert!(out.status.success(), "{:?}: {}", cmd, String::from_utf8_lossy(&out.stderr));
}

fn raw_frames(frames: &[DecodedFrame]) -> Vec<u8> {
    frames.iter().flat_map(|f| f.to_raw_yuv()).collect()
}

/// Lavfi source with motion, so inter prediction is doing real work.
const SRC: &str = "testsrc2=size=176x144:rate=25";

#[test]
fn x264_streams_match_ffmpeg() {
    if !have("ffmpeg") {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let configs: &[(&str, &str)] = &[
        ("baseline", "-profile:v baseline"),
        ("main", "-profile:v main -x264-params bframes=3"),
        ("high", "-x264-params bframes=4:b-pyramid=normal:weightp=2:ref=4"),
        ("temporal", "-x264-params direct=temporal:bframes=3"),
        ("cavlc_b", "-x264-params cabac=0:bframes=2:8x8dct=1"),
        ("slices", "-x264-params slices=3:partitions=all"),
        ("cqm", "-x264-params cqm=jvt"),
    ];
    for (name, args) in configs {
        let mp4 = tmp(&format!("{name}.mp4"));
        let yuv = tmp(&format!("{name}.yuv"));
        let encoded = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", SRC, "-frames:v", "12", "-pix_fmt", "yuv420p"])
            .args(["-c:v", "libx264"])
            .args(args.split_whitespace())
            .arg(&mp4)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !encoded {
            eprintln!("skipping {name}: no libx264");
            return;
        }
        run(Command::new("ffmpeg").args(["-v", "error", "-y", "-i"]).arg(&mp4).args([
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        ]).arg(&yuv));
        let (_, frames) = decode_mp4(&std::fs::read(&mp4).unwrap());
        let reference = std::fs::read(&yuv).unwrap();
        assert!(raw_frames(&frames) == reference, "x264 {name}: differs from ffmpeg");
        let _ = std::fs::remove_file(&mp4);
        let _ = std::fs::remove_file(&yuv);
    }
}

#[test]
fn x265_streams_match_the_encoders_reconstruction() {
    if !have("x265") || !have("ffmpeg") {
        eprintln!("skipping: needs x265 and ffmpeg");
        return;
    }
    let src = tmp("src.yuv");
    run(Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i", SRC, "-frames:v", "12", "-pix_fmt", "yuv420p", "-f", "rawvideo"])
        .arg(&src));
    let configs: &[(&str, &str)] = &[
        ("default", ""),
        ("ctu16", "--ctu 16 --min-cu-size 8"),
        ("slow_amp", "--preset slow --amp --rect"),
        ("nowpp", "--no-wpp"),
        ("slices", "--slices 3"),
        ("tools", "--tskip --cu-lossless --scaling-list default --constrained-intra"),
        ("weighted", "--weightp --weightb --ref 4 --bframes 6 --b-pyramid"),
        ("opengop", "--keyint 6 --open-gop"),
    ];
    for (name, args) in configs {
        let bs = tmp(&format!("{name}.265"));
        let rec = tmp(&format!("{name}.rec.yuv"));
        run(Command::new("x265")
            .args(["--log-level", "error", "--input"])
            .arg(&src)
            .args(["--input-res", "176x144", "--fps", "25"])
            .args(args.split_whitespace())
            .arg("--recon")
            .arg(&rec)
            .arg("-o")
            .arg(&bs));
        let frames = decode_annexb(TrackCodec::Hevc, &std::fs::read(&bs).unwrap()).expect(name);
        let reference = std::fs::read(&rec).unwrap();
        assert!(raw_frames(&frames) == reference, "x265 {name}: differs from x265's reconstruction");
        for p in [&bs, &rec] {
            let _ = std::fs::remove_file(p as &Path);
        }
    }
    let _ = std::fs::remove_file(&src);
}

/// Against HM, the HEVC reference decoder (JCT-VC's `TAppDecoder`). Not
/// packaged anywhere, so opt-in: set `THREERS_HM_DECODER` to the binary.
/// Covers what x265's `--recon` cannot — it writes 8-bit recon files whatever
/// the coded depth — and the crate's own tiled streams.
#[test]
fn hevc_streams_match_hm() {
    let Ok(hm) = std::env::var("THREERS_HM_DECODER") else {
        eprintln!("skipping: THREERS_HM_DECODER not set");
        return;
    };
    let mut streams: Vec<(String, Vec<u8>)> = Vec::new();
    {
        use threers::codec::hevc::CompressedEncoder;
        let (w, h) = (832, 200);
        for deblock in [false, true] {
            let mut enc = CompressedEncoder::new(w, h, 28).residual(true).deblock(deblock).tiles(3, 2);
            streams.push((format!("tiles_deblock_{deblock}"), enc.encode_frame(&test_frame(w, h, 3)).0));
        }
    }
    if have("x265") && have("ffmpeg") {
        let src = tmp("hm_src.yuv");
        run(Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", SRC, "-frames:v", "8", "-pix_fmt", "yuv420p", "-f", "rawvideo"])
            .arg(&src));
        for (name, args) in [("main10", "--output-depth 10"), ("main12", "--output-depth 12 --weightp"), ("ctu16", "--ctu 16 --min-cu-size 8")] {
            let bs = tmp(&format!("{name}.265"));
            run(Command::new("x265")
                .args(["--log-level", "error", "--input"])
                .arg(&src)
                .args(["--input-res", "176x144", "--fps", "25"])
                .args(args.split_whitespace())
                .arg("-o")
                .arg(&bs));
            streams.push((name.to_string(), std::fs::read(&bs).unwrap()));
            let _ = std::fs::remove_file(&bs);
        }
        let _ = std::fs::remove_file(&src);
    }
    for (name, bs) in streams {
        let input = tmp(&format!("{name}.hm.265"));
        let output = tmp(&format!("{name}.hm.yuv"));
        std::fs::write(&input, &bs).unwrap();
        // `-d 0`: write samples at the coded bit depth.
        run(Command::new(&hm).arg("-b").arg(&input).arg("-o").arg(&output).args(["-d", "0"]));
        let frames = decode_annexb(TrackCodec::Hevc, &bs).expect(&name);
        let reference = std::fs::read(&output).unwrap();
        assert!(raw_frames(&frames) == reference, "{name}: differs from HM");
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
    }
}

/// The JCT-VC HEVC conformance bitstreams (as ffmpeg's FATE suite ships
/// them: `rsync -rltLW rsync://fate-suite.ffmpeg.org/fate-suite/hevc-conformance/ dir`),
/// each decoded by HM and by us. Opt-in: needs `THREERS_HEVC_CONFORMANCE` (the
/// directory) and `THREERS_HM_DECODER`.
///
/// Streams using tools this decoder refuses (4:2:2/4:4:4, range extensions)
/// must fail with an error rather than decode wrongly; streams HM itself
/// cannot decode are skipped.
#[test]
fn hevc_conformance_suite_matches_hm() {
    let (Ok(dir), Ok(hm)) = (
        std::env::var("THREERS_HEVC_CONFORMANCE"),
        std::env::var("THREERS_HM_DECODER"),
    ) else {
        eprintln!("skipping: THREERS_HEVC_CONFORMANCE / THREERS_HM_DECODER not set");
        return;
    };
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("bit" | "bin")))
        .collect();
    entries.sort();
    let (mut passed, mut refused, mut skipped) = (0, 0, 0);
    let mut failures = Vec::new();
    for path in &entries {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let out = tmp(&format!("{name}.hm.yuv"));
        let ok = Command::new(&hm)
            .arg("-b")
            .arg(path)
            .arg("-o")
            .arg(&out)
            .args(["-d", "0"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !ok {
            skipped += 1;
            let _ = std::fs::remove_file(&out);
            continue;
        }
        let reference = std::fs::read(&out).unwrap_or_default();
        let _ = std::fs::remove_file(&out);
        match decode_annexb(TrackCodec::Hevc, &std::fs::read(path).unwrap()) {
            Err(e) if e.contains("not supported") => refused += 1,
            Err(e) => failures.push(format!("{name}: {e}")),
            Ok(frames) if raw_frames(&frames) == reference => passed += 1,
            Ok(_) => failures.push(format!("{name}: differs from HM")),
        }
    }
    eprintln!("{passed} match HM, {refused} refused as unsupported, {skipped} skipped (HM failed)");
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
