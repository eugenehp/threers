//! `decode::av1` against dav1d (through `ffmpeg -c:v libdav1d`), every shown
//! frame byte for byte.
//!
//! Two sources of streams, each skipped when unavailable:
//!
//! - every `.ivf` / `.obu` under `$THREERS_AV1_CONFORMANCE` (default
//!   `/Users/Shared/tribev2-ref/media/av1conf`): libaom's `av1-1-b8-*` /
//!   `av1-1-b10-*` test vectors and ffmpeg's FATE AV1 samples;
//! - streams SVT-AV1 encodes here, chosen for paths the vectors miss:
//!   screen content with intra block copy on (asserted), and 10-bit.
//!
//! ffmpeg is run with `-noautoscale`: without it, frames after a resolution
//! change come back scaled to the first frame's size, and a correct decode of
//! a `sizeup` / `switch_frame` stream compares as a mismatch.

use std::path::{Path, PathBuf};
use std::process::Command;

use threers_codec::decode::av1::{obu::Layout, split_annexb, Av1Decoder};
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
        split_annexb(data)?.into_iter().map(|t| t.to_vec()).collect()
    } else {
        vec![data.to_vec()]
    };
    let mut dec = Av1Decoder::new();
    let mut out = Vec::new();
    for p in &packets {
        let frames = if annexb { dec.decode_annexb(p)? } else { dec.decode(p)? };
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
    assert!(bad.is_empty(), "{} of {} streams:\n{}", bad.len(), paths.len(), bad.join("\n"));
    eprintln!("{} streams match dav1d", paths.len());
}

#[test]
fn conformance_vectors_match_dav1d() {
    if !ffmpeg_with("libdav1d") {
        eprintln!("SKIP: no ffmpeg with libdav1d");
        return;
    }
    let dir = std::env::var("THREERS_AV1_CONFORMANCE")
        .unwrap_or_else(|_| "/Users/Shared/tribev2-ref/media/av1conf".into());
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
        ("scm.ivf", "testsrc=size=352x288:rate=10", "yuv420p", "scm=1:keyint=10"),
        ("b10.ivf", "testsrc2=size=320x240:rate=15", "yuv420p10le", "keyint=30"),
    ];
    let mut paths = Vec::new();
    for (name, src, pix, params) in cases {
        let path = dir.join(name);
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", src, "-frames:v", "20"])
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
    let ibc = parse_ivf(&data).unwrap().4.iter().flat_map(|(_, p)| dec.decode(p).unwrap()).filter(|f| f.hdr.allow_intrabc).count();
    eprintln!("{ibc} frames allow intra block copy");
    assert!(ibc > 0, "SVT-AV1 did not enable intra block copy");
    let _ = std::fs::remove_dir_all(&dir);
}
