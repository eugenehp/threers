//! `decode::av1` against dav1d (through `ffmpeg -c:v libdav1d`), every shown
//! frame byte for byte.
//!
//! Two sources of streams, each skipped when unavailable:
//!
//! - every `.ivf` / `.obu` under `$THREERS_AV1_CONFORMANCE` (default
//!   `~/tools/av1conf`): libaom's `av1-1-b8-*` / `av1-1-b10-*` test vectors
//!   (listed in libaom's `test/test_vectors.cc`, each at
//!   `https://storage.googleapis.com/aom-test-data/<name>`) and ffmpeg's FATE
//!   AV1 samples;
//! - streams SVT-AV1 encodes here, chosen for paths the vectors miss:
//!   screen content with intra block copy on (asserted), 10-bit, and 4x2
//!   tiles (asserted) for the tile-parallel decode under `parallel`.
//!
//! ffmpeg is run with `-noautoscale`: without it, frames after a resolution
//! change come back scaled to the first frame's size, and a correct decode of
//! a `sizeup` / `switch_frame` stream compares as a mismatch.

use std::path::{Path, PathBuf};
use std::process::Command;

use threers_codec::decode::av1::{obu::Layout, split_annexb, Av1Decoder, Av1Frame};
use threers_codec::mkv_demux::parse_ivf;

/// Whether ffmpeg has `lib` as an encoder or a decoder (ffmpeg answers one
/// listing per run, so ask for each).
fn ffmpeg_with(lib: &str) -> bool {
    ["-encoders", "-decoders"].iter().any(|list| {
        Command::new("ffmpeg")
            .args(["-hide_banner", list])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains(lib))
            .unwrap_or(false)
    })
}

/// Every shown frame as raw planes, as `ffmpeg -f rawvideo` writes them.
fn decode(data: &[u8]) -> Result<Vec<u8>, String> {
    let annexb = !data.starts_with(b"DKIF") && !data.starts_with(&[0x12, 0x00]);
    let packets: Vec<Vec<u8>> = if data.starts_with(b"DKIF") {
        parse_ivf(data)?.4.into_iter().map(|(_, d)| d).collect()
    } else if annexb {
        split_annexb(data)?
            .into_iter()
            .map(|t| t.to_vec())
            .collect()
    } else {
        vec![data.to_vec()]
    };
    let mut dec = Av1Decoder::new();
    let mut out = Vec::new();
    for p in &packets {
        let frames = if annexb {
            dec.decode_annexb(p)?
        } else {
            dec.decode(p)?
        };
        for f in frames {
            let pic = &f.pic;
            let planes = if pic.layout == Layout::I400 { 1 } else { 3 };
            for pl in 0..planes {
                let (ssh, ssv) = pic.ss(pl);
                let w = ((pic.w + ssh) >> ssh) as usize;
                let h = ((pic.h + ssv) >> ssv) as usize;
                let s = pic.pstride(pl);
                for y in 0..h {
                    let row = &pic.planes[pl][y * s..y * s + w];
                    if pic.bpc == 8 {
                        out.extend(row.iter().map(|&v| v as u8));
                    } else {
                        out.extend(row.iter().flat_map(|&v| v.to_le_bytes()));
                    }
                }
            }
        }
    }
    Ok(out)
}

fn dav1d(path: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-c:v", "libdav1d", "-i"])
        .arg(path)
        .args(["-noautoscale", "-f", "rawvideo", "-"])
        .output()
        .expect("ffmpeg");
    assert!(out.status.success(), "dav1d failed on {}", path.display());
    out.stdout
}

fn check(paths: &[PathBuf]) {
    let mut bad = Vec::new();
    for p in paths {
        let ours = decode(&std::fs::read(p).unwrap());
        match ours {
            Ok(ours) if ours == dav1d(p) => {}
            Ok(_) => bad.push(format!("{}: differs from dav1d", p.display())),
            Err(e) => bad.push(format!("{}: {e}", p.display())),
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} streams:\n{}",
        bad.len(),
        paths.len(),
        bad.join("\n")
    );
    eprintln!("{} streams match dav1d", paths.len());
}

#[test]
fn conformance_vectors_match_dav1d() {
    if !ffmpeg_with("libdav1d") {
        eprintln!("SKIP: no ffmpeg with libdav1d");
        return;
    }
    let dir = std::env::var("THREERS_AV1_CONFORMANCE").unwrap_or_else(|_| {
        format!(
            "{}/tools/av1conf",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    paths.retain(|p| matches!(p.extension().and_then(|x| x.to_str()), Some("ivf" | "obu")));
    paths.sort();
    if paths.is_empty() {
        eprintln!("SKIP: no AV1 streams under {dir}");
        return;
    }
    check(&paths);
}

/// Every sample of `frames`, with each picture's size and depth.
fn samples(frames: &[Av1Frame]) -> Vec<u16> {
    let mut out = Vec::new();
    for f in frames {
        let pic = &f.pic;
        out.extend([pic.w as u16, pic.h as u16, pic.bpc as u16]);
        for pl in 0..if pic.layout == Layout::I400 { 1 } else { 3 } {
            let (ssh, ssv) = pic.ss(pl);
            let (w, h, s) = (
                ((pic.w + ssh) >> ssh) as usize,
                ((pic.h + ssv) >> ssv) as usize,
                pic.pstride(pl),
            );
            for y in 0..h {
                out.extend_from_slice(&pic.planes[pl][y * s..y * s + w]);
            }
        }
    }
    out
}

const BATCH_THREADS: [usize; 4] = [2, 3, 6, 12];

/// `decode_batch` must give what decoding one unit at a time gives, however
/// frames land on threads. Run under `parallel`, or it only tests the
/// fallback.
#[test]
fn batch_decode_matches_serial() {
    let dir = std::env::var("THREERS_AV1_CONFORMANCE").unwrap_or_else(|_| {
        format!(
            "{}/tools/av1conf",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    paths.retain(|p| p.extension().and_then(|x| x.to_str()) == Some("ivf"));
    paths.sort();
    if paths.is_empty() {
        eprintln!("SKIP: no AV1 streams under {dir}");
        return;
    }
    let mut bad = Vec::new();
    for p in &paths {
        let data = std::fs::read(p).unwrap();
        let units: Vec<Vec<u8>> = parse_ivf(&data)
            .unwrap()
            .4
            .into_iter()
            .map(|(_, d)| d)
            .collect();
        let refs: Vec<&[u8]> = units.iter().map(|u| &u[..]).collect();
        let mut dec = Av1Decoder::new();
        let serial: Vec<Av1Frame> = refs.iter().flat_map(|u| dec.decode(u).unwrap()).collect();
        let want = samples(&serial);
        for mode in BATCH_THREADS {
            match Av1Decoder::new().decode_batch(&refs, mode) {
                Ok(f) if samples(&f) == want => {}
                Ok(_) => bad.push(format!("{}: {mode} threads differs", p.display())),
                Err(e) => bad.push(format!("{}: {mode} threads: {e}", p.display())),
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} failures:\n{}",
        bad.len(),
        bad.join("\n")
    );
    eprintln!(
        "{} streams match serial in {} modes",
        paths.len(),
        BATCH_THREADS.len()
    );
}

/// A damaged stream must end in an answer, never a hang: later frames wait
/// on earlier ones, so one that fails has to release them.
#[test]
fn batch_decode_ends_on_damaged_streams() {
    let dir = std::env::var("THREERS_AV1_CONFORMANCE").unwrap_or_else(|_| {
        format!(
            "{}/tools/av1conf",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let Ok(data) = std::fs::read(Path::new(&dir).join("av1-1-b8-23-film_grain-50.ivf")) else {
        eprintln!("SKIP: no AV1 streams under {dir}");
        return;
    };
    let units: Vec<Vec<u8>> = parse_ivf(&data)
        .unwrap()
        .4
        .into_iter()
        .map(|(_, d)| d)
        .collect();
    let mut x = 12345u32;
    for round in 0..40 {
        let mut damaged = units.clone();
        for _ in 0..4 {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            let u = &mut damaged[(x >> 8) as usize % units.len()];
            let at = (x >> 4) as usize % u.len();
            if round % 2 == 0 {
                u[at] ^= 0x55;
            } else {
                u.truncate(at + 1);
            }
        }
        let refs: Vec<&[u8]> = damaged.iter().map(|u| &u[..]).collect();
        for mode in BATCH_THREADS {
            let _ = Av1Decoder::new().decode_batch(&refs, mode);
        }
    }
}

/// `decode_video` runs AV1 through the concurrent path: the frames and their
/// times must come out as one-at-a-time decoding gives them, in order.
#[test]
fn decode_video_keeps_av1_frames_and_times_in_order() {
    if !ffmpeg_with("libsvtav1") {
        eprintln!("SKIP: needs ffmpeg with libsvtav1");
        return;
    }
    let dir = std::env::temp_dir().join(format!("threers_av1v_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ivf = dir.join("v.ivf");
    let ok = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=25",
            "-frames:v",
            "40",
        ])
        .args(["-pix_fmt", "yuv420p", "-c:v", "libsvtav1", "-preset", "8"])
        .arg(&ivf)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "SVT-AV1 failed");
    let data = std::fs::read(&ivf).unwrap();
    let parsed = parse_ivf(&data).unwrap();
    let (num, den) = parsed.3;
    let mut dec = Av1Decoder::new();
    let mut want = Vec::new();
    for (pts, p) in &parsed.4 {
        for f in dec.decode(p).unwrap() {
            want.push((*pts as f64 * num as f64 / den as f64, f.to_decoded().y));
        }
    }
    assert_eq!(want.len(), 40);
    let mp4 = dir.join("v.mp4");
    let ok = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(&ivf)
        .args(["-c", "copy"])
        .arg(&mp4)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "mp4 remux failed");
    for file in [&ivf, &mp4] {
        let mut got = Vec::new();
        threers_codec::decode::decode_video(&std::fs::read(file).unwrap(), |t, f| {
            got.push((t.unwrap(), f.y));
            Ok(())
        })
        .unwrap();
        assert_eq!(got.len(), want.len(), "{}", file.display());
        let mut prev = -1.0;
        for (i, ((t, y), (wt, wy))) in got.iter().zip(&want).enumerate() {
            assert!(y == wy, "{} frame {i} differs", file.display());
            assert!(
                (t - wt).abs() < 1e-3,
                "{} frame {i}: time {t}, wanted {wt}",
                file.display()
            );
            assert!(*t > prev, "{} frame {i} out of order", file.display());
            prev = *t;
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn svt_av1_streams_match_dav1d() {
    if !ffmpeg_with("libdav1d") || !ffmpeg_with("libsvtav1") {
        eprintln!("SKIP: needs ffmpeg with libsvtav1 and libdav1d");
        return;
    }
    let dir = std::env::temp_dir().join(format!("threers_av1_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cases = [
        // Screen-content tools on, which allows palette and intra block copy.
        (
            "scm.ivf",
            "testsrc=size=352x288:rate=10",
            "yuv420p",
            "scm=1:keyint=10",
        ),
        (
            "b10.ivf",
            "testsrc2=size=320x240:rate=15",
            "yuv420p10le",
            "keyint=30",
        ),
        // 4x2 tiles, which `parallel` decodes concurrently. 640 wide is ten
        // 64x64 superblocks, so the tile edges split 128x128 filter-mask
        // entries between two tiles.
        (
            "tiles.ivf",
            "testsrc2=size=640x360:rate=15",
            "yuv420p",
            "tile-columns=2:tile-rows=1:keyint=30",
        ),
        (
            "tiles10.ivf",
            "testsrc2=size=640x360:rate=15",
            "yuv420p10le",
            "tile-columns=2:tile-rows=1:keyint=30",
        ),
    ];
    let mut paths = Vec::new();
    for (name, src, pix, params) in cases {
        let path = dir.join(name);
        let ok = Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                src,
                "-frames:v",
                "20",
            ])
            .args(["-pix_fmt", pix, "-c:v", "libsvtav1", "-preset", "8"])
            .args(["-svtav1-params", params])
            .arg(&path)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "SVT-AV1 failed on {name}");
        paths.push(path);
    }
    check(&paths);
    // The screen-content stream is only worth having if it reaches the intra
    // block copy path, which runs on frames that allow it.
    let data = std::fs::read(&paths[0]).unwrap();
    let mut dec = Av1Decoder::new();
    let ibc = parse_ivf(&data)
        .unwrap()
        .4
        .iter()
        .flat_map(|(_, p)| dec.decode(p).unwrap())
        .filter(|f| f.hdr.allow_intrabc)
        .count();
    eprintln!("{ibc} frames allow intra block copy");
    assert!(ibc > 0, "SVT-AV1 did not enable intra block copy");
    // The tiled streams likewise, if they are to reach the tile-parallel path.
    for p in &paths[2..] {
        let data = std::fs::read(p).unwrap();
        let mut dec = Av1Decoder::new();
        let tiled = parse_ivf(&data)
            .unwrap()
            .4
            .iter()
            .flat_map(|(_, p)| dec.decode(p).unwrap())
            .all(|f| f.hdr.tiling.cols * f.hdr.tiling.rows > 1);
        assert!(tiled, "SVT-AV1 did not tile {}", p.display());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
