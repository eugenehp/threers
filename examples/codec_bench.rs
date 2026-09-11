//! Encode a raw `yuv420p` clip with each pure-Rust encoder and report bytes and
//! wall time as JSON on stdout.
//!
//! This is the native arm of the codec benchmark. `scripts/codec_bench.py`
//! prepares the clip, runs the ffmpeg and VideoToolbox arms, decodes every
//! result back and computes the quality metrics; this binary only encodes, so
//! the timings here are the encoder and nothing else.
//!
//! Input is planar `yuv420p` — the same bytes the ffmpeg arms are fed — rather
//! than RGBA, so no arm pays a colour-conversion cost the others avoid and the
//! comparison is codec-to-codec.
//!
//! ```text
//! codec_bench --input clip.yuv --size 1280x1440 --fps 24 \
//!             --encoder hevc-intra --qp 26 --out out.mp4
//! ```

use std::time::Instant;

use threers::codec::h264::{encode_compressed_mp4, encode_mp4, split_annexb};
use threers::codec::hevc::{
    build_hvcc, encode_compressed_mp4_streaming, CompressedEncoder, HevcEncoder, HvccArray,
    HvccProfile, Yuv420Frame,
};
use threers::codec::mp4::{mux_hevc, Mp4Params};

fn main() {
    let args = Args::parse();
    let frames = if args.encoder == "hevc-stream" {
        Vec::new()
    } else {
        read_yuv(&args)
    };
    eprintln!(
        "codec_bench: {} frames {}x{} via {}",
        frames.len(),
        args.width,
        args.height,
        args.encoder
    );

    let t0 = Instant::now();
    let bytes = match args.encoder.as_str() {
        "h264-pcm" => encode_mp4(args.width, args.height, args.fps, &frames),
        "hevc-pcm" => encode_hevc_pcm(&args, &frames),
        "h264-native" => encode_compressed_mp4(args.width, args.height, args.fps, args.qp, &frames),
        "hevc-intra" => encode_hevc_intra(&args, &frames, false),
        "hevc-intra-residual" => encode_hevc_intra(&args, &frames, true),
        // The library's own export path, which batches frames so they can be
        // encoded concurrently. `hevc-intra-residual` above drives the encoder
        // a frame at a time and so never exercises it.
        "hevc-mp4" => encode_hevc_mp4(&args, &frames),
        // Same encoder, but pulling frames from disk one at a time — the shape
        // that matters at 8K, where holding every source frame is gigabytes.
        "hevc-stream" => {
            let (w, h) = (args.width as usize, args.height as usize);
            let frame_len = w * h + 2 * (w / 2) * (h / 2);
            let n = std::fs::metadata(&args.input).expect("stat input").len() as usize / frame_len;
            let mut file = std::fs::File::open(&args.input).expect("open input");
            eprintln!("codec_bench: streaming {n} frames of {w}x{h}");
            encode_compressed_mp4_streaming(args.width, args.height, args.fps, args.qp, n, None, |_| {
                read_one(&mut file, args.width, args.height)
            })
        }
        other => {
            eprintln!("unknown encoder `{other}`");
            std::process::exit(2);
        }
    };
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    std::fs::write(&args.out, &bytes).expect("write output");
    println!(
        "{{\"encoder\":\"{}\",\"qp\":{},\"bytes\":{},\"encode_ms\":{:.1},\"frames\":{}}}",
        args.encoder,
        args.qp,
        bytes.len(),
        ms,
        frames.len()
    );
}

/// PCM HEVC: every CTB carries raw samples. Lossless in YUV, and the size floor
/// this benchmark exists to move.
fn encode_hevc_pcm(args: &Args, frames: &[Yuv420Frame]) -> Vec<u8> {
    let mut enc = HevcEncoder::new(args.width, args.height);
    let level = enc.config().level_idc;
    let stream: Vec<Vec<u8>> = frames.iter().map(|f| enc.encode_frame(f)).collect();
    mux_annexb(args, &stream, level)
}

/// Transform-coded intra HEVC. `residual` adds coefficient coding on top of the
/// prediction-only path.
///
/// With `--recon` set, the encoder's own reconstruction is written out too. A
/// conformant decoder must reproduce it bit-for-bit, so comparing that file to
/// ffmpeg's decode separates "the bitstream is wrong" from "the encoder's
/// choices are bad" — two failures that look identical in a PSNR column.
/// The library export path, optionally tiled.
fn encode_hevc_mp4(args: &Args, frames: &[Yuv420Frame]) -> Vec<u8> {
    use threers::codec::hevc::compress::CompressedEncoder;
    let enc = CompressedEncoder::new(args.width, args.height, args.qp)
        .tiles(args.tiles.0, args.tiles.1);
    let level = enc.config().level_idc;
    let (vps, sps, pps) = enc.parameter_sets();
    let mut stream = vec![[vec![0, 0, 0, 1], vps, vec![0, 0, 0, 1], sps, vec![0, 0, 0, 1], pps].concat()];
    stream.extend(threers::codec::map_frames_pub(frames, |f| enc.encode_slice_au(f)));
    mux_annexb(args, &stream, level)
}

fn encode_hevc_intra(args: &Args, frames: &[Yuv420Frame], residual: bool) -> Vec<u8> {
    let mut enc = CompressedEncoder::new(args.width, args.height, args.qp).residual(residual);
    let level = enc.config().level_idc;
    let mut recon_out = Vec::new();
    let stream: Vec<Vec<u8>> = frames
        .iter()
        .map(|f| {
            let (au, recon) = enc.encode_frame(f);
            if !args.recon.is_empty() {
                // Crop the CTB-aligned reconstruction back to display size, the
                // way the conformance window makes a decoder do.
                let (cw, cwc) = (recon.coded_width as usize, (recon.coded_width / 2) as usize);
                let (w, h) = (args.width as usize, args.height as usize);
                for row in 0..h {
                    recon_out.extend_from_slice(&recon.y[row * cw..row * cw + w]);
                }
                for plane in [&recon.u, &recon.v] {
                    for row in 0..h / 2 {
                        recon_out.extend_from_slice(&plane[row * cwc..row * cwc + w / 2]);
                    }
                }
            }
            au
        })
        .collect();
    if !args.recon.is_empty() {
        std::fs::write(&args.recon, &recon_out).expect("write reconstruction");
    }
    mux_annexb(args, &stream, level)
}

/// Split each access unit's Annex-B NALs, hoist the parameter sets into `hvcC`,
/// and length-prefix the slice NALs into MP4 samples.
fn mux_annexb(args: &Args, access_units: &[Vec<u8>], level_idc: u8) -> Vec<u8> {
    let (mut vps, mut sps, mut pps) = (Vec::new(), Vec::new(), Vec::new());
    let mut samples: Vec<Vec<u8>> = Vec::with_capacity(access_units.len());

    for au in access_units {
        let mut sample = Vec::new();
        for nal in split_annexb(au) {
            // HEVC NAL header is 2 bytes; nal_unit_type is bits 1..7 of byte 0.
            let kind = (nal[0] >> 1) & 0x3F;
            match kind {
                32 => vps.push(nal),
                33 => sps.push(nal),
                34 => pps.push(nal),
                _ => {
                    sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                    sample.extend_from_slice(&nal);
                }
            }
        }
        samples.push(sample);
    }

    let hvcc = build_hvcc(
        &HvccProfile::main_420(level_idc),
        &[
            HvccArray {
                nal_type: 32,
                complete: true,
                nals: &vps,
            },
            HvccArray {
                nal_type: 33,
                complete: true,
                nals: &sps,
            },
            HvccArray {
                nal_type: 34,
                complete: true,
                nals: &pps,
            },
        ],
    );

    let timescale = 600u32;
    mux_hevc(&Mp4Params {
        width: args.width,
        height: args.height,
        timescale,
        frame_duration: timescale / args.fps.max(1),
        hvcc_payload: &hvcc,
        almo_payload: None,
        samples: &samples,
    })
}

/// Read the next planar `yuv420p` frame from an open file.
fn read_one(file: &mut std::fs::File, width: u32, height: u32) -> Yuv420Frame {
    use std::io::Read;
    let (w, h) = (width as usize, height as usize);
    let (cw, ch) = (w / 2, h / 2);
    let mut y = vec![0u8; w * h];
    let mut u = vec![0u8; cw * ch];
    let mut v = vec![0u8; cw * ch];
    file.read_exact(&mut y).expect("read luma");
    file.read_exact(&mut u).expect("read Cb");
    file.read_exact(&mut v).expect("read Cr");
    Yuv420Frame {
        width,
        height,
        y,
        u,
        v,
        alpha: None,
    }
}

/// Read the whole planar `yuv420p` file into frames.
fn read_yuv(args: &Args) -> Vec<Yuv420Frame> {
    let raw = std::fs::read(&args.input).expect("read input clip");
    let (w, h) = (args.width as usize, args.height as usize);
    let (cw, ch) = (w / 2, h / 2);
    let frame_len = w * h + 2 * cw * ch;
    assert!(
        raw.len().is_multiple_of(frame_len),
        "clip is {} bytes, not a whole number of {w}x{h} yuv420p frames ({frame_len} each)",
        raw.len()
    );

    raw.chunks_exact(frame_len)
        .map(|c| Yuv420Frame {
            width: args.width,
            height: args.height,
            y: c[..w * h].to_vec(),
            u: c[w * h..w * h + cw * ch].to_vec(),
            v: c[w * h + cw * ch..].to_vec(),
            alpha: None,
        })
        .collect()
}

struct Args {
    input: String,
    out: String,
    width: u32,
    height: u32,
    fps: u32,
    qp: i32,
    encoder: String,
    recon: String,
    tiles: (u32, u32),
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            input: String::new(),
            out: "out.mp4".into(),
            width: 0,
            height: 0,
            fps: 30,
            qp: 26,
            encoder: "hevc-intra".into(),
            recon: String::new(),
            tiles: (1, 1),
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let val = || argv.get(i + 1).cloned().unwrap_or_default();
            match argv[i].as_str() {
                "--input" => a.input = val(),
                "--out" => a.out = val(),
                "--fps" => a.fps = val().parse().expect("--fps"),
                "--qp" => a.qp = val().parse().expect("--qp"),
                "--encoder" => a.encoder = val(),
                "--recon" => a.recon = val(),
                "--tiles" => {
                    let s = val();
                    let (c, r) = s.split_once('x').expect("--tiles CxR");
                    a.tiles = (c.parse().expect("cols"), r.parse().expect("rows"));
                }
                "--size" => {
                    let s = val();
                    let (w, h) = s.split_once('x').expect("--size WxH");
                    a.width = w.parse().expect("width");
                    a.height = h.parse().expect("height");
                }
                other => {
                    eprintln!("unknown flag `{other}`");
                    std::process::exit(2);
                }
            }
            i += 2;
        }
        assert!(!a.input.is_empty(), "--input is required");
        assert!(a.width > 0 && a.height > 0, "--size WxH is required");
        a
    }
}
