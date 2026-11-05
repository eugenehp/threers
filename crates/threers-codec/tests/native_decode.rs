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

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use threers_codec::decode::{decode_annexb, decode_video, DecodedFrame, NativeDecoder};
use threers_codec::hevc::Yuv420Frame;
use threers_codec::mp4_demux::{demux_video, TrackCodec};

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
    assert!(
        f.y == y,
        "{what}: luma differs first at {:?}",
        first(&f.y, y)
    );
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
        dec.push(bytes, s.pts, s.dts, s.duration as i64)
            .expect("decode");
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
    let mp4 = threers_codec::h264::encode_mp4(w, h, 30, &frames);
    let (codec, out) = decode_mp4(&mp4);
    assert_eq!(codec, TrackCodec::H264);
    assert_eq!(out.len(), frames.len());
    for (i, (f, src)) in out.iter().zip(&frames).enumerate() {
        assert_eq!((f.width, f.height), (w, h));
        assert_planes(
            f,
            &widen(&src.y),
            &widen(&src.u),
            &widen(&src.v),
            &format!("frame {i}"),
        );
    }
}

#[test]
fn h264_compressed_matches_the_encoders_reconstruction() {
    use threers_codec::h264::{CompressedEncoder, EntropyMode};
    for mode in [EntropyMode::Cavlc, EntropyMode::Cabac] {
        for deblock in [false, true] {
            let (w, h) = (80, 48);
            let mut enc = CompressedEncoder::new(w, h, 30)
                .entropy(mode)
                .deblock(deblock);
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
    use threers_codec::hevc::HevcEncoder;
    let (w, h) = (66, 66);
    let mut enc = HevcEncoder::new(w, h);
    let frame = test_frame(w, h, 1);
    let au = enc.encode_frame(&frame);
    let out = decode_annexb(TrackCodec::Hevc, &au).expect("decode");
    assert_eq!(out.len(), 1);
    assert_eq!((out[0].width, out[0].height), (w, h));
    assert_planes(
        &out[0],
        &widen(&frame.y),
        &widen(&frame.u),
        &widen(&frame.v),
        "HEVC PCM",
    );
}

#[test]
fn hevc_compressed_with_tiles_matches_the_encoders_reconstruction() {
    use threers_codec::hevc::CompressedEncoder;
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
    use threers_codec::hevc::CompressedEncoder;
    // 200 samples is too narrow for two 256-sample columns; 136 is tall
    // enough for two 64-sample rows.
    let enc = CompressedEncoder::new(200, 136, 28).tiles(2, 2);
    assert_eq!(enc.config().tile_grid, (1, 2));
}

#[test]
fn hevc_mp4_round_trips_through_native_decoder() {
    let (w, h) = (96, 64);
    let frames: Vec<_> = (0..4).map(|t| test_frame(w, h, t)).collect();
    let mp4 = threers_codec::hevc::encode_compressed_mp4(w, h, 30, 26, &frames);
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
    use threers_codec::hevc::TransparentEncoder;
    let (w, h) = (64, 64);
    let mut enc = TransparentEncoder::new(w, h);
    let color = test_frame(w, h, 0);
    let alpha: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
    enc.encode_frame(&color, &alpha);
    let mov = enc.finish_mov(30);
    // The alpha layer is nuh_layer_id 1, which a base-layer decoder skips.
    let (_, out) = decode_mp4(&mov);
    assert_eq!(out.len(), 1);
    assert_planes(
        &out[0],
        &widen(&color.y),
        &widen(&color.u),
        &widen(&color.v),
        "colour layer",
    );
}

#[test]
fn take_returns_bgra_and_drops_earlier_frames() {
    let (w, h) = (32, 32);
    let frames: Vec<_> = (0..3).map(|t| test_frame(w, h, t)).collect();
    let mp4 = threers_codec::h264::encode_mp4(w, h, 30, &frames);
    let track = demux_video(&mp4).unwrap();
    let mut dec = NativeDecoder::new(&track).unwrap();
    assert_eq!(dec.size(), (w, h));
    for s in &track.samples {
        dec.push(
            &mp4[s.offset as usize..][..s.size as usize],
            s.pts,
            s.dts,
            0,
        )
        .unwrap();
    }
    assert_eq!(dec.queued(), 3);
    let last = track.samples.last().unwrap().pts;
    let bgra = dec.take(last).expect("last frame");
    assert_eq!(bgra.len(), (w * h * 4) as usize);
    assert_eq!(
        dec.queued(),
        0,
        "taking a later frame drops the ones before it"
    );
}

/// Range-extension chroma QP offset lists with three entries, (6, 6),
/// (-4, 3) and (2, -5), chosen per 16x16 group, in a 4:2:2 10-bit I/B stream.
/// No conformance stream has more than one entry, so this one comes from HM
/// 19's encoder with two `setChromaQpOffsetListEntry` lines added to
/// `TEncTop.cpp` (it cycles the index by block position):
/// `TAppEncoder -c encoder_randomaccess_main.cfg -wdt 416 -hgt 240 -f 9
/// --InputBitDepth=10 --InternalBitDepth=10 --InputChromaFormat=422
/// --Profile=main_422_10 --MaxCUChromaQpAdjustmentDepth=2 --QP=22`.
/// The hash is FNV-1a 64 of HM's own decode (`TAppDecoder -d 0`).
#[test]
fn chroma_qp_offset_lists_match_hm() {
    fixture_matches_hm("chroma_qp_offset_list_422_10.bin", 9, 0x5b4fefbfe7f3bf67);
}

/// Range-extension tools no conformance stream combines: every CU lossless
/// with implicit (intra) and explicit (inter) RDPCM, 4×4 rotation, the
/// transform-skip significance context and persistent Rice adaptation.
/// Lossless implicit-RDPCM blocks also drop the intra edge filters, and
/// rotate as transform-skipped ones do — HM's behaviour, not ffmpeg's.
/// From HM 19: `TAppEncoder -c encoder_randomaccess_main.cfg -wdt 176
/// -hgt 144 -f 9 --InputChromaFormat=444 --Profile=main_444
/// --TransquantBypassEnable=1 --CUTransquantBypassFlagForce=1
/// --CostMode=lossless --ImplicitResidualDPCM=1 --ExplicitResidualDPCM=1
/// --ResidualRotation=1 --SingleSignificanceMapContext=1
/// --GolombRiceParameterAdaptation=1 --TransformSkip=1`, on testsrc2.
#[test]
fn rext_lossless_rdpcm_matches_hm() {
    fixture_matches_hm("rext_lossless_rdpcm_444_8.bin", 9, 0xdd43567023c76f75);
}

/// High-precision weighted-prediction offsets, which only matter with
/// weighted prediction — and no RExt conformance stream uses it. A 10-bit
/// fade: `TAppEncoder -c encoder_randomaccess_main.cfg -wdt 176 -hgt 144
/// -f 9 --InputBitDepth=10 --InternalBitDepth=10 --InputChromaFormat=444
/// --Profile=main_444_10 --WeightedPredP=1 --WeightedPredB=1
/// --HighPrecisionPredictionWeighting=1 --QP=27`.
#[test]
fn high_precision_weighted_prediction_matches_hm() {
    fixture_matches_hm("rext_high_precision_wp_444_10.bin", 9, 0xcda29f427a595ca0);
}

/// Decode `tests/support/hevc/<name>` and compare the FNV-1a 64 hash of its
/// raw output with that of HM's decode.
fn fixture_matches_hm(name: &str, frames: usize, hm_hash: u64) {
    let bs = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/support/hevc")
            .join(name),
    )
    .unwrap();
    let out = decode_annexb(TrackCodec::Hevc, &bs).expect(name);
    assert_eq!(out.len(), frames, "{name}");
    let hash = raw_frames(&out)
        .iter()
        .fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        });
    assert_eq!(hash, hm_hash, "{name}: differs from HM's decode");
}

#[test]
fn decode_video_reads_mp4_with_times() {
    let (w, h) = (66, 50);
    let frames: Vec<_> = (0..3).map(|t| test_frame(w, h, t)).collect();
    let mp4 = threers_codec::h264::encode_mp4(w, h, 30, &frames);
    let mut out = Vec::new();
    decode_video(&mp4, |t, f| {
        out.push((t, f));
        Ok(())
    })
    .expect("decode");
    assert_eq!(out.len(), 3);
    for (i, ((t, f), src)) in out.iter().zip(&frames).enumerate() {
        let t = t.expect("an MP4 has timing");
        assert!((t - i as f64 / 30.0).abs() < 1e-9, "frame {i} at {t}");
        assert_planes(
            f,
            &widen(&src.y),
            &widen(&src.u),
            &widen(&src.v),
            &format!("frame {i}"),
        );
    }
}

#[test]
fn decode_video_tells_annexb_codecs_apart() {
    use threers_codec::hevc::HevcEncoder;
    let (w, h) = (66, 66);
    let frame = test_frame(w, h, 1);
    let hevc = HevcEncoder::new(w, h).encode_frame(&frame);
    let mut h264 = threers_codec::h264::CompressedEncoder::new(w, h, 30);
    let (au, _) = h264.encode_frame(&frame);
    let mut h264_stream = Vec::new();
    for ps in [h264.sps(), h264.pps()] {
        h264_stream.extend_from_slice(&[0, 0, 0, 1]);
        h264_stream.extend_from_slice(ps);
    }
    h264_stream.extend_from_slice(&au);
    for (what, stream) in [("HEVC", &hevc), ("H.264", &h264_stream)] {
        let mut out = Vec::new();
        decode_video(stream, |t, f| {
            out.push((t, f));
            Ok(())
        })
        .expect(what);
        assert_eq!(out.len(), 1, "{what}");
        assert_eq!(out[0].0, None, "{what}: elementary streams have no timing");
        assert_eq!((out[0].1.width, out[0].1.height), (w, h), "{what}");
    }
    assert!(
        decode_video(&[0, 0, 1, 0x09, 0xf0], |_, _| Ok(())).is_err(),
        "no parameter sets"
    );
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
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("threers_dec_{}_{n}_{name}", std::process::id()))
}

fn run(cmd: &mut Command) {
    let out = cmd.output().expect("spawn");
    assert!(
        out.status.success(),
        "{:?}: {}",
        cmd,
        String::from_utf8_lossy(&out.stderr)
    );
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
        (
            "high",
            "-x264-params bframes=4:b-pyramid=normal:weightp=2:ref=4",
        ),
        ("temporal", "-x264-params direct=temporal:bframes=3"),
        ("cavlc_b", "-x264-params cabac=0:bframes=2:8x8dct=1"),
        ("slices", "-x264-params slices=3:partitions=all"),
        ("cqm", "-x264-params cqm=jvt"),
    ];
    for (name, args) in configs {
        let mp4 = tmp(&format!("{name}.mp4"));
        let yuv = tmp(&format!("{name}.yuv"));
        let encoded = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                SRC,
                "-frames:v",
                "12",
                "-pix_fmt",
                "yuv420p",
            ])
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
        run(Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(&mp4)
            .args([
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
            ])
            .arg(&yuv));
        let (_, frames) = decode_mp4(&std::fs::read(&mp4).unwrap());
        let reference = std::fs::read(&yuv).unwrap();
        assert!(
            raw_frames(&frames) == reference,
            "x264 {name}: differs from ffmpeg"
        );
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
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            SRC,
            "-frames:v",
            "12",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
        ])
        .arg(&src));
    let configs: &[(&str, &str)] = &[
        ("default", ""),
        ("ctu16", "--ctu 16 --min-cu-size 8"),
        ("slow_amp", "--preset slow --amp --rect"),
        ("nowpp", "--no-wpp"),
        ("slices", "--slices 3"),
        (
            "tools",
            "--tskip --cu-lossless --scaling-list default --constrained-intra",
        ),
        (
            "weighted",
            "--weightp --weightb --ref 4 --bframes 6 --b-pyramid",
        ),
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
        assert!(
            raw_frames(&frames) == reference,
            "x265 {name}: differs from x265's reconstruction"
        );
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
        use threers_codec::hevc::CompressedEncoder;
        let (w, h) = (832, 200);
        for deblock in [false, true] {
            let mut enc = CompressedEncoder::new(w, h, 28)
                .residual(true)
                .deblock(deblock)
                .tiles(3, 2);
            streams.push((
                format!("tiles_deblock_{deblock}"),
                enc.encode_frame(&test_frame(w, h, 3)).0,
            ));
        }
    }
    if have("x265") && have("ffmpeg") {
        for (name, pix_fmt, args) in [
            ("main10", "yuv420p", "--output-depth 10"),
            ("main12", "yuv420p", "--output-depth 12 --weightp"),
            ("ctu16", "yuv420p", "--ctu 16 --min-cu-size 8"),
            // 4:2:2 has no conformance stream without other RExt tools.
            ("main422_10", "yuv422p10le", "--input-csp i422 --input-depth 10 --output-depth 10 --profile main422-10"),
            ("main444_8", "yuv444p", "--input-csp i444 --profile main444-8"),
            // NxN intra gives four chroma modes per CU, coded in 4:4:4 only.
            (
                "main444_12_nxn_bframes",
                "yuv444p12le",
                "--input-csp i444 --input-depth 12 --output-depth 12 --profile main444-12 --ctu 16 --min-cu-size 8 --bframes 3 --weightp",
            ),
        ] {
            let src = tmp(&format!("{name}_src.yuv"));
            run(Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-f", "lavfi", "-i", SRC, "-frames:v", "8", "-pix_fmt", pix_fmt, "-f", "rawvideo"])
                .arg(&src));
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
            let _ = std::fs::remove_file(&src);
        }
    }
    for (name, bs) in streams {
        let input = tmp(&format!("{name}.hm.265"));
        let output = tmp(&format!("{name}.hm.yuv"));
        std::fs::write(&input, &bs).unwrap();
        // `-d 0`: write samples at the coded bit depth.
        run(Command::new(&hm)
            .arg("-b")
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .args(["-d", "0"]));
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
/// must fail with an error rather than decode wrongly. HM 19 asserts on a few
/// older stream revisions (PS_A_VIDYO_3 reads its pre-v2 extension data as a
/// range extension); those are checked against ffmpeg instead, when it is
/// installed, and skipped only if ffmpeg cannot decode them either.
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
    let ffmpeg = have("ffmpeg");
    let (mut passed, mut refused, mut by_ffmpeg, mut skipped) = (0, 0, 0, 0);
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
            let _ = std::fs::remove_file(&out);
            match ffmpeg.then(|| ffmpeg_check(path, &name)).flatten() {
                None => skipped += 1,
                Some(Ok(())) => by_ffmpeg += 1,
                Some(Err(e)) => failures.push(e),
            }
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
    eprintln!(
        "{passed} match HM, {refused} refused as unsupported, {by_ffmpeg} match ffmpeg (HM failed), \
         {skipped} skipped (HM and ffmpeg both failed)"
    );
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// EXT_A_ericsson_3 sets the pre-version-2 extension bits on a Main stream,
/// and HM and ffmpeg both reject it. Its reissue, _4, clears them and is
/// checked against HM above; the two must decode to the same pictures.
/// Opt-in: needs `THREERS_HEVC_CONFORMANCE`.
#[test]
fn hevc_pre_v2_extension_data_is_ignored() {
    let Ok(dir) = std::env::var("THREERS_HEVC_CONFORMANCE") else {
        eprintln!("skipping: THREERS_HEVC_CONFORMANCE not set");
        return;
    };
    let decode = |name: &str| {
        let data = std::fs::read(Path::new(&dir).join(name)).expect(name);
        raw_frames(&decode_annexb(TrackCodec::Hevc, &data).expect(name))
    };
    let (old, reissue) = (
        decode("EXT_A_ericsson_3.bit"),
        decode("EXT_A_ericsson_4.bit"),
    );
    assert!(!reissue.is_empty());
    assert!(
        old == reissue,
        "EXT_A_ericsson_3 decodes differently from its reissue _4"
    );
}

/// The JVT conformance streams (`rsync -rltLW
/// rsync://fate-suite.ffmpeg.org/fate-suite/h264-conformance/`) against ffmpeg.
/// They use what no encoder here does: long-term references, MMCO 2-6, POC
/// type 1, CABAC `I_PCM`, field pictures and MBAFF. FMO and SP/SI are refused
/// by design, and those are printed. Opt-in: set `THREERS_H264_CONFORMANCE`.
#[test]
fn h264_conformance_suite_matches_ffmpeg() {
    let Ok(dir) = std::env::var("THREERS_H264_CONFORMANCE") else {
        eprintln!("skipping: THREERS_H264_CONFORMANCE not set");
        return;
    };
    assert!(have("ffmpeg"), "the suite needs ffmpeg as its reference");
    fn streams(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                streams(&path, out);
            } else if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("264" | "26l" | "avc" | "h264" | "jsv" | "jvt")
            ) {
                out.push(path);
            }
        }
    }
    let mut entries = Vec::new();
    streams(Path::new(&dir), &mut entries);
    entries.sort();
    let (mut passed, mut refused) = (0, Vec::new());
    let mut failures = Vec::new();
    for path in &entries {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let data = std::fs::read(path).unwrap();
        let frames = match decode_annexb(TrackCodec::H264, &data) {
            Ok(frames) if !frames.is_empty() => frames,
            Ok(_) => {
                failures.push(format!("{name}: no frames"));
                continue;
            }
            Err(e) => {
                refused.push(format!("{name}: {e}"));
                continue;
            }
        };
        let f = &frames[0];
        // ffmpeg outputs monochrome as 4:2:0, and asking for `gray` would
        // convert its range: take that and keep the luma.
        let mono = f.chroma == threers_codec::decode::ChromaFormat::Monochrome;
        let base = match f.chroma {
            threers_codec::decode::ChromaFormat::Yuv422 => "yuv422p",
            threers_codec::decode::ChromaFormat::Yuv444 => "yuv444p",
            _ => "yuv420p",
        };
        let pix_fmt = if f.bit_depth == 8 {
            base.to_string()
        } else {
            format!("{base}{}le", f.bit_depth)
        };
        let out = tmp(&format!("{name}.h264.yuv"));
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "h264", "-i"])
            .arg(path)
            .args([
                "-f",
                "rawvideo",
                "-pix_fmt",
                &pix_fmt,
                "-fps_mode",
                "passthrough",
            ])
            .arg(&out)
            .output()
            .is_ok_and(|o| o.status.success());
        let mut reference = std::fs::read(&out).unwrap_or_default();
        let _ = std::fs::remove_file(&out);
        let ours = raw_frames(&frames);
        if mono {
            let luma = ours.len() / frames.len();
            reference = reference
                .chunks(luma * 3 / 2)
                .flat_map(|f| &f[..luma])
                .copied()
                .collect();
        }
        if !ok {
            failures.push(format!("{name}: ffmpeg failed"));
        } else if ours == reference {
            passed += 1;
        } else {
            let frame = ours.len() / frames.len();
            let at = ours
                .iter()
                .zip(&reference)
                .position(|(a, b)| a != b)
                .map(|i| i / frame);
            failures.push(format!(
                "{name}: {} frames, ffmpeg {}, first difference in frame {at:?}",
                frames.len(),
                reference.len() / frame,
            ));
        }
    }
    eprintln!(
        "{} streams: {passed} match ffmpeg, {} refused",
        entries.len(),
        refused.len()
    );
    for r in &refused {
        eprintln!("  refused {r}");
    }
    assert!(passed > 0);
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A stream HM could not decode, against ffmpeg: `None` if ffmpeg cannot
/// either. A weaker reference than HM (it has its own SAO bug at CTU 16),
/// but where it decodes and we refuse, we are the one that is wrong.
fn ffmpeg_check(path: &Path, name: &str) -> Option<Result<(), String>> {
    let ours = decode_annexb(TrackCodec::Hevc, &std::fs::read(path).unwrap());
    let pix_fmt = match &ours {
        Ok(frames) if !frames.is_empty() => {
            let f = &frames[0];
            let base = if f.chroma == threers_codec::decode::ChromaFormat::Monochrome {
                "gray"
            } else {
                "yuv420p"
            };
            if f.bit_depth == 8 {
                base.to_string()
            } else {
                format!("{base}{}le", f.bit_depth)
            }
        }
        _ => "yuv420p".to_string(),
    };
    let out = tmp(&format!("{name}.ff.yuv"));
    let ok = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(path)
        .args([
            "-f",
            "rawvideo",
            "-pix_fmt",
            &pix_fmt,
            "-fps_mode",
            "passthrough",
        ])
        .arg(&out)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let reference = std::fs::read(&out).unwrap_or_default();
    let _ = std::fs::remove_file(&out);
    if !ok || reference.is_empty() {
        return None;
    }
    Some(match ours {
        Err(e) => Err(format!(
            "{name}: ffmpeg decodes it (HM failed), we fail: {e}"
        )),
        Ok(frames) if raw_frames(&frames) == reference => Ok(()),
        Ok(_) => Err(format!("{name}: differs from ffmpeg (HM failed)")),
    })
}

/// Picture jobs must change nothing: `decode_video` from a thread outside the
/// pool uses them, and from a pool thread cannot, so the two runs are the
/// threaded and the serial decoder on the same bytes.
#[cfg(feature = "parallel")]
#[test]
fn picture_jobs_match_serial_decoding() {
    if !have("ffmpeg") {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    type Decoded = Vec<(Option<f64>, Vec<u8>)>;
    let decode = |bytes: &[u8]| -> Decoded {
        let mut out = Vec::new();
        threers_codec::decode::decode_video(bytes, |t, f| {
            out.push((t, f.to_raw_yuv()));
            Ok(())
        })
        .unwrap();
        out
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for (name, args) in [
        ("h264", "-c:v libx264 -bf 3 -refs 4"),
        (
            "hevc",
            "-c:v libx265 -x265-params log-level=error:bframes=4:b-pyramid=1",
        ),
    ] {
        let path = tmp(&format!("jobs_{name}.mp4"));
        let made = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                SRC,
                "-frames:v",
                "90",
                "-pix_fmt",
                "yuv420p",
            ])
            .args(args.split(' '))
            .arg(&path)
            .status()
            .is_ok_and(|s| s.success());
        if !made {
            eprintln!("skipping {name}: ffmpeg could not make it");
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let threaded = decode(&bytes);
        let serial = pool.install(|| decode(&bytes));
        assert_eq!(threaded.len(), 90, "{name}");
        assert!(
            threaded == serial,
            "{name}: picture jobs changed the output"
        );
    }
}

/// B-frames reach `decode_video` in decode order; `decode_video_in_order`
/// must show them by time, and stop when asked.
#[test]
fn decode_video_in_order_sorts_b_frames_and_stops() {
    let path = std::env::temp_dir().join(format!("threers_order_{}.mp4", std::process::id()));
    let ok = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=25",
            "-frames:v",
            "60",
        ])
        .args([
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-bf", "3", "-g", "30",
        ])
        .arg(&path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("SKIP: needs ffmpeg with libx264");
        return;
    }
    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let mut raw = Vec::new();
    threers_codec::decode::decode_video(&data, |t, _| {
        raw.push(t.unwrap());
        Ok(())
    })
    .unwrap();
    assert!(
        raw.windows(2).any(|w| w[1] < w[0]),
        "the stream has no reordering to test"
    );
    let mut times = Vec::new();
    threers_codec::decode::decode_video_in_order(&data, |t, _| {
        times.push(t.unwrap());
        Ok(true)
    })
    .unwrap();
    let mut want = raw.clone();
    want.sort_by(f64::total_cmp);
    assert_eq!(times, want);
    let mut n = 0;
    threers_codec::decode::decode_video_in_order(&data, |_, _| {
        n += 1;
        Ok(n < 5)
    })
    .unwrap();
    assert_eq!(n, 5);
}
